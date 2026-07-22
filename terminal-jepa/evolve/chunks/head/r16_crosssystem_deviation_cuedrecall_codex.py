"""R16 head chunk: cross-system deviation coding plus cued recall.

Predictive-coding and efficient-coding accounts of cortex emphasize transmitting
prediction errors, not raw sensations: subtract a context prior, then allocate
capacity to the residual that distinguishes the current stimulus. The stalled
cat margin has the same shape: retrieve-by-command supplies the cross-system
"typical" file body, but the world model must add this system's deviation.

This train-only head keeps the champion cued-recall probe, then adds a second
linear probe and output-margin aux. For every batch it mines high command-cosine,
close-but-not-duplicate cross-row observations as a detached lexical prototype;
the hidden state must reconstruct and classify z_obs - prototype, and the model's
own prediction is softly required to be closer to the true observation than to
that prototype by a residual-scaled margin. Eval forward is untouched.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r16_crosssystem_deviation_cuedrecall_codex"
DESCRIPTION = (
    "Train-only predictive-error head: preserves the R12 cued-recall binding probe, "
    "then mines high-cmd-similarity, close-but-distinct cross-row observations as a "
    "detached lexical/system prototype. A second linear probe must encode this row's "
    "system deviation (true obs minus prototype) with reconstruction + InfoNCE, while "
    "the main prediction gets a small residual-scaled margin forcing it closer to the "
    "true obs than to the prototype. Targets are labels only; eval forward is untouched."
)

_DEFAULTS = {
    "row_frac": 0.5,
    "aux_weight": 1.0,
    "ramp_steps": 400,
    "recall_weight": 1.0,
    "recall_top_q": 0.35,
    "recall_s_floor": 0.20,
    "recall_cos_weight": 0.08,
    "recall_mse_weight": 0.012,
    "recall_norm_weight": 0.015,
    "recall_nce_weight": 0.10,
    "recall_nce_temp": 0.07,
    "recall_dup_thresh": 0.999,
    "dev_weight": 1.0,
    "dev_delay": 100,
    "dev_ramp_steps": 700,
    "dev_top_q": 0.40,
    "cmd_floor": 0.78,
    "cmd_pow": 2.0,
    "proto_k": 6,
    "lam_frac": 0.65,
    "dup_delta": 0.04,
    "mass_floor": 1e-4,
    "dev_floor": 0.025,
    "dev_cos_weight": 0.055,
    "dev_mse_weight": 0.008,
    "dev_norm_weight": 0.010,
    "dev_nce_weight": 0.060,
    "dev_nce_temp": 0.08,
    "dev_dup_thresh": 0.995,
    "proto_margin_weight": 0.045,
    "proto_margin_frac": 0.35,
    "proto_margin_tau": 0.15,
}

_EPS = 1e-8
_NEG = -1.0e4


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _huber_abs(x, beta=0.2):
    a = x.abs()
    b = float(beta)
    return torch.where(a < b, 0.5 * a * a / b, a - 0.5 * b)


def _interleave_layout_ok(b):
    for k in ("tok", "types", "key_pad", "tgt", "cmd_mask"):
        if k not in b:
            return False
    tok, types, key_pad, tgt, cmd_mask = b["tok"], b["types"], b["key_pad"], b["tgt"], b["cmd_mask"]
    if tok.dim() != 3 or tgt.dim() != 3 or types.dim() != 2 or key_pad.dim() != 2:
        return False
    if tok.shape[:2] != types.shape or tok.shape[:2] != key_pad.shape:
        return False
    if cmd_mask.shape != tgt.shape[:2] or tok.shape[0] != tgt.shape[0] or tok.shape[2] != tgt.shape[2]:
        return False
    if tok.shape[1] != 2 * tgt.shape[1]:
        return False
    live = ~key_pad.bool()
    if not bool(live.any().item()):
        return False
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    if even.numel() == 0 or odd.numel() == 0:
        return False
    return bool((even == 0).all().item()) and bool((odd == 1).all().item())


def _probe_hidden_dim(net, D):
    was_training = net.training
    try:
        net.eval()
        with torch.no_grad():
            tok = torch.zeros(1, 4, int(D))
            types = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
            key_pad = torch.zeros(1, 4, dtype=torch.bool)
            out = net(tok, types, key_pad)
        if not (isinstance(out, tuple) and len(out) >= 2):
            return None
        h = out[1]
        if not torch.is_tensor(h) or h.dim() != 3 or h.shape[1] != 4:
            return None
        d = int(h.shape[-1])
        return d if d > 0 else None
    except Exception:
        return None
    finally:
        if was_training:
            net.train()


def _to_obs_space(pred, prev_obs, net):
    tmod = getattr(net, "target_module", None)
    if tmod is None:
        return pred
    shape = pred.shape
    d = shape[-1]
    return tmod.to_obs(pred.reshape(-1, d), prev_obs.reshape(-1, d)).reshape(shape)


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params or {})
    cfg["D"] = int(D)
    cfg["_step"] = 0
    d = _probe_hidden_dim(net, D)
    if d is None:
        cfg["_disabled"] = True
        return cfg
    recall_probe = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, int(D)))
    dev_probe = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, int(D)))
    nn.init.zeros_(recall_probe[1].bias)
    nn.init.zeros_(dev_probe[1].bias)
    net.r16_recall_probe = recall_probe
    net.r16_dev_probe = dev_probe
    cfg["_disabled"] = False
    return cfg


def _mine_recall_anchors(cmd, valid):
    B, maxn, _ = cmd.shape
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))
    ii = torch.arange(maxn, device=cmd.device)
    past = ii[None, :, None] > ii[None, None, :]
    allowed = past & valid[:, None, :].expand(B, maxn, maxn)
    sim = sim.masked_fill(~allowed, -2.0)
    s, a = sim.max(dim=2)
    eligible = valid & (s > -1.5)
    return s, a, eligible


def _recall_loss(cfg, h_cmd, cmd, obs, valid, probe):
    zero = h_cmd.new_zeros(())
    if float(cfg["recall_weight"]) <= 0.0:
        return zero
    B, maxn, _ = cmd.shape
    if maxn < 2:
        return zero
    with torch.no_grad():
        s, a, eligible = _mine_recall_anchors(cmd, valid)
        s = torch.nan_to_num(s, nan=-2.0)
        flat = eligible.reshape(-1)
        if not bool(flat.any().item()):
            return zero
        idx = flat.nonzero(as_tuple=False).squeeze(1)
        s_flat = s.reshape(-1)[idx]
        keep_n = max(1, int(math.ceil(float(cfg["recall_top_q"]) * idx.numel())))
        top = torch.topk(s_flat, keep_n).indices
        idx, s_flat = idx[top], s_flat[top]
        floor = s_flat > float(cfg["recall_s_floor"])
        if not bool(floor.any().item()):
            return zero
        idx, s_flat = idx[floor], s_flat[floor]
        b_sel = idx // maxn
        j_sel = idx % maxn
        a_sel = a.reshape(-1)[idx]
        w = (s_flat - float(cfg["recall_s_floor"])).clamp_min(0.0) + 1e-4
        w = w / w.sum().clamp_min(_EPS)

    p = torch.nan_to_num(probe(h_cmd[b_sel, j_sel]), nan=0.0, posinf=1e4, neginf=-1e4)
    z = torch.nan_to_num(obs[b_sel, a_sel].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    pu, zu = _unit(p), _unit(z)
    cos_err = (w * (1.0 - (pu * zu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (p - z).pow(2).mean(dim=-1)).sum()
    pn = p.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    zn = z.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    norm_err = (w * _huber_abs(torch.log(pn) - torch.log(zn), beta=0.2)).sum()
    total = (
        float(cfg["recall_cos_weight"]) * cos_err
        + float(cfg["recall_mse_weight"]) * mse_err
        + float(cfg["recall_norm_weight"]) * norm_err
    )
    N = p.shape[0]
    if N >= 2 and float(cfg["recall_nce_weight"]) > 0.0:
        logits = (pu @ zu.transpose(0, 1)) / max(float(cfg["recall_nce_temp"]), 1e-4)
        with torch.no_grad():
            sim = (zu @ zu.transpose(0, 1)).clamp(-1.0, 1.0)
            eye = torch.eye(N, dtype=torch.bool, device=p.device)
            dup = (sim > float(cfg["recall_dup_thresh"])) & ~eye
        logits = logits.masked_fill(dup, _NEG)
        ce = F.cross_entropy(logits, torch.arange(N, device=p.device), reduction="none")
        total = total + float(cfg["recall_nce_weight"]) * (w * ce).sum()
    return float(cfg["recall_weight"]) * total


def _system_deviation_loss(cfg, h_cmd, pred_cmd, cmd, obs, valid, probe, net):
    zero = h_cmd.new_zeros(())
    if float(cfg["dev_weight"]) <= 0.0:
        return zero
    B, maxn, D = obs.shape
    M = B * maxn
    if M < 2 or int(valid.sum().item()) < 2 or pred_cmd.shape[-1] != D:
        return zero

    h_flat = h_cmd.reshape(M, h_cmd.shape[-1])
    pred_flat = pred_cmd.reshape(M, D)
    cmd_flat = cmd.reshape(M, D)
    obs_flat = obs.reshape(M, D)
    valid_flat = valid.reshape(M).bool()
    row_id = torch.arange(B, device=obs.device).unsqueeze(1).expand(B, maxn).reshape(M)
    step_id = torch.arange(maxn, device=obs.device).unsqueeze(0).expand(B, maxn).reshape(M)

    with torch.no_grad():
        c = _unit(torch.nan_to_num(cmd_flat, nan=0.0, posinf=1e4, neginf=-1e4))
        o = torch.nan_to_num(obs_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        valid_pair = valid_flat[:, None] & valid_flat[None, :]
        valid_pair = valid_pair & (row_id[:, None] != row_id[None, :])
        valid_pair = valid_pair & ~torch.eye(M, dtype=torch.bool, device=obs.device)
        if not bool(valid_pair.any().item()):
            return zero

        csim = (c @ c.transpose(0, 1)).clamp(-1.0, 1.0)
        o2 = (o * o).sum(dim=-1, keepdim=True)
        odist = (o2 + o2.transpose(0, 1) - 2.0 * (o @ o.transpose(0, 1))).clamp_min(0.0) / float(D)
        mean_dist = odist[valid_pair].mean().clamp_min(_EPS)
        lam = (float(cfg["lam_frac"]) * mean_dist).clamp_min(_EPS)
        floor = float(cfg["cmd_floor"])
        sim_gate = ((csim - floor) / max(1.0 - floor, 1e-6)).clamp(0.0, 1.0)
        sim_gate = sim_gate.pow(float(cfg["cmd_pow"]))
        dup_delta = max(float(cfg["dup_delta"]), 1e-6)
        ring = torch.exp(-odist / lam) * (1.0 - torch.exp(-odist / dup_delta))
        weights = sim_gate * ring * valid_pair.to(o.dtype)

        k = min(max(1, int(cfg["proto_k"])), M)
        if k < M:
            kth = torch.topk(weights, k, dim=1).values[:, -1:].clamp_min(0.0)
            weights = torch.where((weights > 0.0) & (weights >= kth), weights, torch.zeros_like(weights))
        mass = weights.sum(dim=1)
        eligible = valid_flat & (step_id > 0) & (mass > float(cfg["mass_floor"]))
        if not bool(eligible.any().item()):
            return zero

        proto = (weights @ o) / mass.clamp_min(_EPS).unsqueeze(1)
        dev = o - proto
        dev_energy = dev.pow(2).mean(dim=-1)
        eligible = eligible & (dev_energy > float(cfg["dev_floor"]))
        if not bool(eligible.any().item()):
            return zero

        idx = eligible.nonzero(as_tuple=False).squeeze(1)
        score = (mass / (mass + float(cfg["mass_floor"]))).clamp(0.0, 1.0) * dev_energy.sqrt()
        score_i = score[idx]
        keep_n = max(1, int(math.ceil(float(cfg["dev_top_q"]) * idx.numel())))
        top = torch.topk(score_i, keep_n).indices
        idx = idx[top]
        score_i = score_i[top]
        w = score_i + 1e-4
        w = w / w.sum().clamp_min(_EPS)
        dev_t = dev[idx].detach()
        proto_t = proto[idx].detach()
        true_t = o[idx].detach()
        dev_e = dev_energy[idx].detach()

    p = torch.nan_to_num(probe(h_flat[idx]), nan=0.0, posinf=1e4, neginf=-1e4)
    dclean = torch.nan_to_num(dev_t, nan=0.0, posinf=1e4, neginf=-1e4)
    pu, du = _unit(p), _unit(dclean)
    cos_err = (w * (1.0 - (pu * du).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (p - dclean).pow(2).mean(dim=-1)).sum()
    pn = p.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    dn = dclean.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    norm_err = (w * _huber_abs(torch.log(pn) - torch.log(dn), beta=0.2)).sum()
    total = (
        float(cfg["dev_cos_weight"]) * cos_err
        + float(cfg["dev_mse_weight"]) * mse_err
        + float(cfg["dev_norm_weight"]) * norm_err
    )

    N = p.shape[0]
    if N >= 2 and float(cfg["dev_nce_weight"]) > 0.0:
        logits = (pu @ du.transpose(0, 1)) / max(float(cfg["dev_nce_temp"]), 1e-4)
        with torch.no_grad():
            rsim = (du @ du.transpose(0, 1)).clamp(-1.0, 1.0)
            eye = torch.eye(N, dtype=torch.bool, device=p.device)
            dup = (rsim > float(cfg["dev_dup_thresh"])) & ~eye
        logits = logits.masked_fill(dup, _NEG)
        ce = F.cross_entropy(logits, torch.arange(N, device=p.device), reduction="none")
        total = total + float(cfg["dev_nce_weight"]) * (w * ce).sum()

    if float(cfg["proto_margin_weight"]) > 0.0:
        prev = torch.cat([obs.new_zeros(B, 1, D), obs[:, :-1]], dim=1).reshape(M, D)
        pred_obs = torch.nan_to_num(
            _to_obs_space(pred_flat[idx], prev[idx], net),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        d_true = (pred_obs - true_t).pow(2).mean(dim=-1)
        d_proto = (pred_obs - proto_t).pow(2).mean(dim=-1)
        margin = float(cfg["proto_margin_frac"]) * dev_e.clamp_min(0.0)
        tau = max(float(cfg["proto_margin_tau"]), 1e-4)
        hinge = tau * F.softplus((d_true + margin - d_proto) / tau)
        total = total + float(cfg["proto_margin_weight"]) * (w * hinge).sum()

    return float(cfg["dev_weight"]) * total


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    recall_probe = getattr(net, "r16_recall_probe", None)
    dev_probe = getattr(net, "r16_dev_probe", None)
    if recall_probe is None or dev_probe is None:
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    recall_ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    dev_ramp = _smoothstep((cfg["_step"] - float(cfg["dev_delay"])) / max(1.0, float(cfg["dev_ramp_steps"])))
    if recall_ramp <= 0.0 and dev_ramp <= 0.0:
        return 0.0

    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 2:
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    rows = torch.randperm(B, device=tok.device)[:nrows]
    tok_s, types_s, pad_s = tok[rows], types[rows], key_pad[rows]
    valid = cmd_mask[rows]
    cmd = tok_s[:, 0::2][:, :maxn]
    obs = tok_s[:, 1::2][:, :maxn]

    out = net(tok_s, types_s, pad_s)
    if not (isinstance(out, tuple) and len(out) >= 2):
        return 0.0
    pred_full, h = out[0], out[1]
    if not torch.is_tensor(h) or h.dim() != 3 or h.shape[1] != tok_s.shape[1]:
        return 0.0
    if not torch.is_tensor(pred_full) or pred_full.dim() != 3 or pred_full.shape[:2] != tok_s.shape[:2]:
        return 0.0
    h_cmd = h[:, 0::2][:, :maxn]
    pred_cmd = pred_full[:, 0::2][:, :maxn]

    total = h_cmd.new_zeros(())
    if recall_ramp > 0.0:
        total = total + recall_ramp * _recall_loss(cfg, h_cmd, cmd, obs, valid, recall_probe)
    if dev_ramp > 0.0:
        total = total + dev_ramp * _system_deviation_loss(cfg, h_cmd, pred_cmd, cmd, obs, valid, dev_probe, net)

    total = float(cfg["aux_weight"]) * total
    if not bool(torch.isfinite(total).item()):
        return 0.0
    return total


def leak_safe(mod, params):
    """Forward is untouched. Recall anchors are strictly earlier within a trajectory;
    system-deviation prototypes and residuals are detached labels mined from other
    rows by command/observation geometry, never inputs to the causal net. The only
    eval-visible gradients are ordinary training gradients on hidden states/predictions.
    """
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS if k != "proto_k"}
        proto_k = int(p["proto_k"])
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        0.0 < vals["row_frac"] <= 1.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        vals["recall_weight"] >= 0.0,
        0.0 < vals["recall_top_q"] <= 1.0,
        -1.0 <= vals["recall_s_floor"] < 1.0,
        vals["recall_cos_weight"] >= 0.0,
        vals["recall_mse_weight"] >= 0.0,
        vals["recall_norm_weight"] >= 0.0,
        vals["recall_nce_weight"] >= 0.0,
        vals["recall_nce_temp"] > 0.0,
        0.0 < vals["recall_dup_thresh"] <= 1.0,
        vals["dev_weight"] >= 0.0,
        vals["dev_delay"] >= 0.0,
        vals["dev_ramp_steps"] >= 1.0,
        0.0 < vals["dev_top_q"] <= 1.0,
        -1.0 <= vals["cmd_floor"] < 1.0,
        vals["cmd_pow"] >= 0.0,
        proto_k >= 1,
        vals["lam_frac"] > 0.0,
        vals["dup_delta"] > 0.0,
        vals["mass_floor"] > 0.0,
        vals["dev_floor"] >= 0.0,
        vals["dev_cos_weight"] >= 0.0,
        vals["dev_mse_weight"] >= 0.0,
        vals["dev_norm_weight"] >= 0.0,
        vals["dev_nce_weight"] >= 0.0,
        vals["dev_nce_temp"] > 0.0,
        0.0 < vals["dev_dup_thresh"] <= 1.0,
        vals["proto_margin_weight"] >= 0.0,
        vals["proto_margin_frac"] >= 0.0,
        vals["proto_margin_tau"] > 0.0,
    ]
    return all(checks)
