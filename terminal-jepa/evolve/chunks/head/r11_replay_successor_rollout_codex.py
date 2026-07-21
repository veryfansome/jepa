"""R11 head chunk: hippocampal-replay rollout consistency V2.

Idea: keep eval forward untouched, but train on short internally generated
successor chains rather than R10's single write. The cross-domain lens is
hippocampal replay / multiscale predictive maps: memory and planning are trained
by replayed future trajectories, not only one-step teacher-forced transitions
(Momennejad 2024, https://arxiv.org/abs/2401.09491; Peyrache 2022,
https://arxiv.org/abs/2205.02665).

Mechanism: on a scheduled row/window sample, predict and write 2 or 3 consecutive
observation tokens, then supervise the next prediction under that imagined
context. Loss is obs-space cosine + log-norm + small MSE, plus a command-weighted
hard-negative cosine hinge so imagined listings separate from sibling-like
in-batch foils. Causality is preserved: each written obs is produced at its own
command position from prior tokens only; aux is train-only.
"""

import math

import torch
import torch.nn.functional as F

NAME = "r11_replay_successor_rollout_codex"
DESCRIPTION = (
    "Train-only hippocampal-replay rollout V2 head: unchanged eval forward, but the "
    "aux samples short command windows, writes 2-3 self-predicted obs tokens back into "
    "the stream, then supervises successor predictions in obs space with cosine, norm, "
    "small MSE, and command-weighted hard-negative discrimination against sibling-like "
    "in-batch foils. Scheduled probability/weight and gradient-scaled feedback protect "
    "the content retrieval margin while targeting write-policy planning fidelity."
)

_DEFAULTS = {
    "row_frac": 0.25,
    "p_min": 0.25,
    "p_max": 0.80,
    "aux_weight": 1.0,
    "cos_weight": 0.10,
    "mse_weight": 0.018,
    "norm_weight": 0.020,
    "disc_weight": 0.045,
    "disc_margin": 0.06,
    "disc_temp": 0.07,
    "cmd_gamma": 2.0,
    "cmd_penalty": 0.35,
    "grad_scale": 0.40,
    "ramp_steps": 600,
    "max_feedbacks": 3,
    "h3_after": 0.35,
}

_EPS = 1e-8


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    return cfg


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


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
    even_live = live[:, 0::2]
    odd_live = live[:, 1::2]
    even = types[:, 0::2][even_live]
    odd = types[:, 1::2][odd_live]
    if even.numel() == 0 or odd.numel() == 0:
        return False
    return bool((even == 0).all().item()) and bool((odd == 1).all().item())


def _eligible_starts(cmd_mask, feedbacks):
    B, maxn = cmd_mask.shape
    if maxn < feedbacks + 1:
        return cmd_mask.new_zeros(B, 0)
    out = cmd_mask[:, : maxn - feedbacks].clone()
    for j in range(1, feedbacks + 1):
        out = out & cmd_mask[:, j : maxn - feedbacks + j]
    return out


def _step_mask(nrows, maxn, starts, offset, device):
    m = torch.zeros(nrows, maxn, dtype=torch.bool, device=device)
    m[torch.arange(nrows, device=device), starts + int(offset)] = True
    return m


def _to_obs_space(pred, prev_obs, net):
    tmod = getattr(net, "target_module", None)
    if tmod is None:
        return pred
    shape = pred.shape
    d = shape[-1]
    return tmod.to_obs(pred.reshape(-1, d), prev_obs.reshape(-1, d)).reshape(shape)


def _grad_scaled(x, scale):
    s = float(scale)
    if s >= 1.0:
        return x
    if s <= 0.0:
        return x.detach()
    return x.detach() + s * (x - x.detach())


def _disc_loss(pred_obs, true_obs, cmd_emb, cfg):
    n = pred_obs.shape[0]
    if n < 2:
        return pred_obs.new_zeros(())
    sim = _unit(pred_obs) @ _unit(true_obs).transpose(0, 1)
    pos = sim.diagonal()
    with torch.no_grad():
        csim = (_unit(cmd_emb) @ _unit(cmd_emb).transpose(0, 1)).clamp(-1.0, 1.0)
        csim = ((csim + 1.0) * 0.5).clamp_min(0.0).pow(float(cfg["cmd_gamma"]))
        penalty = (1.0 - csim) * float(cfg["cmd_penalty"])
        eye = torch.eye(n, dtype=torch.bool, device=pred_obs.device)
    neg = (sim - penalty).masked_fill(eye, -1.0e4)
    hard = neg.max(dim=1).values
    temp = max(float(cfg["disc_temp"]), 1e-4)
    z = (hard + float(cfg["disc_margin"]) - pos) / temp
    return F.softplus(z).mean() * temp


def _obs_losses(pred_obs, true_obs, cmd_emb, cfg):
    pred_obs = torch.nan_to_num(pred_obs, nan=0.0, posinf=1e4, neginf=-1e4)
    true_obs = torch.nan_to_num(true_obs, nan=0.0, posinf=1e4, neginf=-1e4)
    cos_err = (1.0 - (_unit(pred_obs) * _unit(true_obs)).sum(dim=-1).clamp(-1.0, 1.0)).mean()
    mse_err = (pred_obs - true_obs).pow(2).mean()
    pn = pred_obs.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    tn = true_obs.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    norm_err = _huber_abs(torch.log(pn) - torch.log(tn), beta=0.2).mean()
    loss = (
        float(cfg["cos_weight"]) * cos_err
        + float(cfg["mse_weight"]) * mse_err
        + float(cfg["norm_weight"]) * norm_err
    )
    if float(cfg["disc_weight"]) > 0.0:
        loss = loss + float(cfg["disc_weight"]) * _disc_loss(pred_obs, true_obs, cmd_emb, cfg)
    return loss


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or float(cfg.get("aux_weight", 0.0)) == 0.0:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    p_keep = float(cfg["p_min"]) + (float(cfg["p_max"]) - float(cfg["p_min"])) * ramp
    if p_keep <= 0.0:
        return 0.0

    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    tgt, cmd_mask = batch["tgt"], batch["cmd_mask"].bool()
    B, maxn, D = tgt.shape
    if maxn < 3:
        return 0.0

    npre = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    rows = torch.randperm(B, device=device)[:npre]
    cm_pre = cmd_mask[rows]

    feedbacks = 2
    if int(cfg["max_feedbacks"]) >= 3:
        h3_after = float(cfg["h3_after"])
        p3 = 0.0 if ramp <= h3_after else (ramp - h3_after) / max(1e-6, 1.0 - h3_after)
        if p3 > 0.0 and float(torch.rand((), device=device).item()) < p3:
            feedbacks = 3

    elig = _eligible_starts(cm_pre, feedbacks)
    if elig.numel() == 0 or not bool(elig.any().item()):
        feedbacks = 2
        elig = _eligible_starts(cm_pre, feedbacks)
    if elig.numel() == 0:
        return 0.0

    row_ok = elig.any(dim=1)
    row_keep = row_ok & (torch.rand(npre, device=device) < p_keep)
    if not bool(row_keep.any().item()):
        return 0.0

    rows = rows[row_keep]
    elig = elig[row_keep]
    tok_s, types_s, pad_s = tok[rows], types[rows], key_pad[rows]
    tgt_s = tgt[rows]
    nrows = tok_s.shape[0]

    starts = torch.rand(elig.shape, device=device).masked_fill(~elig, -1.0).argmax(dim=1).long()
    cmd_tok = tok_s[:, 0::2][:, :maxn]
    current_obs = tok_s[:, 1::2][:, :maxn]
    tok_cur = torch.stack([cmd_tok, current_obs], dim=2).reshape(nrows, 2 * maxn, D)

    total = tok_cur.new_zeros(())
    wsum = 0.0
    for j in range(feedbacks):
        pred_full, _ = net(tok_cur, types_s, pad_s)
        cmd_pred = pred_full[:, 0::2][:, :maxn]
        prev_obs = torch.cat([current_obs.new_zeros(nrows, 1, D), current_obs[:, :-1]], dim=1)
        pred_obs_all = torch.nan_to_num(_to_obs_space(cmd_pred, prev_obs, net), nan=0.0, posinf=1e4, neginf=-1e4)

        m_j = _step_mask(nrows, maxn, starts, j, device)
        step_w = float(j + 1) / float(feedbacks + 1)
        total = total + step_w * _obs_losses(pred_obs_all[m_j], tgt_s[m_j], cmd_tok[m_j], cfg)
        wsum += step_w

        feed = _grad_scaled(pred_obs_all, float(cfg["grad_scale"]))
        current_obs = torch.where(m_j.unsqueeze(-1), feed, current_obs)
        tok_cur = torch.stack([cmd_tok, current_obs], dim=2).reshape(nrows, 2 * maxn, D)

    pred_full, _ = net(tok_cur, types_s, pad_s)
    cmd_pred = pred_full[:, 0::2][:, :maxn]
    prev_obs = torch.cat([current_obs.new_zeros(nrows, 1, D), current_obs[:, :-1]], dim=1)
    pred_obs_all = torch.nan_to_num(_to_obs_space(cmd_pred, prev_obs, net), nan=0.0, posinf=1e4, neginf=-1e4)

    m_f = _step_mask(nrows, maxn, starts, feedbacks, device)
    total = total + _obs_losses(pred_obs_all[m_f], tgt_s[m_f], cmd_tok[m_f], cfg)
    wsum += 1.0

    out = float(cfg["aux_weight"]) * ramp * total / max(wsum, 1e-6)
    if not bool(torch.isfinite(out).item()):
        return 0.0
    return out


def leak_safe(mod, params):
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
        max_feedbacks = int(p["max_feedbacks"])
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        0.0 < vals["row_frac"] <= 1.0,
        0.0 <= vals["p_min"] <= vals["p_max"] <= 1.0,
        vals["aux_weight"] >= 0.0,
        vals["cos_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["norm_weight"] >= 0.0,
        vals["disc_weight"] >= 0.0,
        vals["disc_margin"] >= 0.0,
        vals["disc_temp"] > 0.0,
        vals["cmd_gamma"] >= 0.0,
        vals["cmd_penalty"] >= 0.0,
        0.0 <= vals["grad_scale"] <= 1.0,
        vals["ramp_steps"] >= 1.0,
        max_feedbacks in (2, 3),
        0.0 <= vals["h3_after"] < 1.0,
    ]
    return all(checks)
