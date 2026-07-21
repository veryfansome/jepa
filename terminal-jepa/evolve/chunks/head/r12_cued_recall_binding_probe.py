"""R12 head chunk: cued-recall binding probe (hippocampal pattern completion).

Idea: the v2 collection policy makes ~60% of head/tail/stat targets and ~50% of
grep queries HISTORY-LINKED — the file being re-queried appeared earlier in the
same trajectory. The main loss never tells the trunk to *retrieve* that earlier
observation; it only supervises the final view. This aux adds a cued-recall
term: at each command position whose best earlier same-trajectory anchor (by
frozen cmd-embedding cosine — the same-file re-query) is strong, a LINEAR probe
from the trunk hidden state must (a) reconstruct the anchor's earlier
observation embedding and (b) discriminate it among the other anchors in the
batch (InfoNCE). Gradients flow into the trunk's memory pathway, forcing the
state at a grep/tail/head re-query to carry the file's earlier-seen content —
the in-context binding that the grep (+.326) and tail (+.347) margins demand,
which nearest-command retrieval supplies lexically but the model must supply
in-state to beat it. Cross-domain lens: hippocampal cued recall / pattern
completion (CA3 attractor completion from a partial cue; Neunuebel & Knierim
2014; Rolls 2013 front. syst. neurosci.). Eval forward is untouched; the aux
target is strictly PAST data (anchor index < query index), so it is causal and
leak-free by construction; hard-zero fallback on any non-v2 / non-interleaved
layout.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r12_cued_recall_binding_probe"
DESCRIPTION = (
    "Train-only cued-recall binding aux: eval forward untouched. Mines, per trajectory, "
    "each command's best EARLIER anchor by frozen cmd-embedding cosine (the same-file "
    "re-query structure v2's linked head/tail/grep steps create), keeps the top-quantile "
    "strongest cues, and trains a linear probe from the trunk hidden state at the query "
    "to reconstruct (cosine+mse+log-norm) and discriminate (duplicate-masked InfoNCE) the "
    "anchor's earlier observation. Backprop into the trunk teaches it to hold the file's "
    "earlier-seen content in-state at re-query time — aimed at the grep-hit and tail "
    "margins where within-trajectory content binding is the diagnosed bottleneck."
)

_DEFAULTS = {
    "row_frac": 0.5,      # fraction of batch rows the aux forward runs on
    "top_q": 0.35,        # keep this fraction of eligible queries, by anchor cosine
    "s_floor": 0.20,      # loose absolute floor on anchor cosine (relative top_q is the gate)
    "cos_weight": 0.08,   # probe->anchor cosine reconstruction
    "mse_weight": 0.012,  # small metric anchor
    "norm_weight": 0.015, # log-norm match (keeps probe scale honest)
    "nce_weight": 0.10,   # cued-recall discrimination among in-batch anchors
    "nce_temp": 0.07,
    "dup_thresh": 0.999,  # anchors this similar are false negatives -> masked in NCE
    "aux_weight": 1.0,
    "ramp_steps": 400,    # smoothstep ramp so early training is main-loss-dominated
}

_EPS = 1e-8


# ------------------------------------------------------------------ helpers

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
    """Hard-zero guard: only run on the single-vector cmd/obs interleave (v2 default
    stream). Any other layout (multi-vector, missing keys, shape drift) -> aux off."""
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
    """Infer the trunk hidden width by a tiny eval-mode forward (net is still on CPU at
    wrap time; forward must return (pred [B,L,D], h [B,L,d]))."""
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


# ------------------------------------------------------------------ contract

def wrap(net, D, **params):
    """No forward re-point, no module cycle: register ONLY a linear recall probe on the
    net (so the genome's optimizer and .to(device) pick it up) and return a config dict.
    A linear probe (LN + Linear) keeps the pressure on the TRUNK: the anchor content must
    be linearly present in the hidden state, the probe cannot compute it itself."""
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    d = _probe_hidden_dim(net, D)
    if d is None:
        cfg["_disabled"] = True
        return cfg
    probe = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, int(D)))
    nn.init.zeros_(probe[1].bias)
    net.r12_recall_probe = probe  # registered submodule; NOT a wrapper around net
    cfg["_disabled"] = False
    return cfg


def _mine_anchors(cmd, obs, valid):
    """Per row, for each command position j, the most cmd-similar EARLIER position
    i < j (the within-trajectory re-query anchor). Returns (sim s [B,maxn],
    anchor index a [B,maxn], eligible [B,maxn])."""
    B, maxn, _ = cmd.shape
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))                      # [B, maxn(j), maxn(i)]
    ii = torch.arange(maxn, device=cmd.device)
    past = ii[None, :, None] > ii[None, None, :]                 # anchor strictly earlier
    allowed = past & valid[:, None, :].expand(B, maxn, maxn)     # anchor must be a live step
    sim = sim.masked_fill(~allowed, -2.0)
    s, a = sim.max(dim=2)
    eligible = valid & (s > -1.5)                                # has at least one live earlier step
    return s, a, eligible


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    probe = getattr(net, "r12_recall_probe", None)
    if probe is None:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 2:
        return 0.0

    # -- row subsample (cost control: ONE extra forward on a fraction of rows) --
    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    rows = torch.randperm(B, device=device)[:nrows]
    tok_s, types_s, pad_s = tok[rows], types[rows], key_pad[rows]
    valid = cmd_mask[rows]

    cmd = tok_s[:, 0::2][:, :maxn]                               # standardized z_cmd  [nr,maxn,D]
    obs = tok_s[:, 1::2][:, :maxn]                               # standardized z_obs  [nr,maxn,D]

    with torch.no_grad():
        s, a, eligible = _mine_anchors(cmd, obs, valid)
        s = torch.nan_to_num(s, nan=-2.0)
        flat = eligible.reshape(-1)
        if not bool(flat.any().item()):
            return 0.0
        idx = flat.nonzero(as_tuple=False).squeeze(1)            # eligible (row, j) pairs
        s_flat = s.reshape(-1)[idx]
        # relative gate: top-q strongest cues in this batch, plus a loose absolute floor
        keep_n = max(1, int(math.ceil(float(cfg["top_q"]) * idx.numel())))
        top = torch.topk(s_flat, keep_n).indices
        idx, s_flat = idx[top], s_flat[top]
        floor = s_flat > float(cfg["s_floor"])
        if not bool(floor.any().item()):
            return 0.0
        idx, s_flat = idx[floor], s_flat[floor]
        b_sel = idx // maxn                                       # row within the subsample
        j_sel = idx % maxn                                        # query step
        a_sel = a.reshape(-1)[idx]                                # anchor step (< j_sel)
        w = (s_flat - float(cfg["s_floor"])).clamp_min(0.0) + 1e-4
        w = w / w.sum()

    # -- trunk forward on the subsample; probe the hidden state at the query cmd --
    out = net(tok_s, types_s, pad_s)
    if not (isinstance(out, tuple) and len(out) >= 2):
        return 0.0
    h = out[1]
    if not torch.is_tensor(h) or h.dim() != 3 or h.shape[1] != tok_s.shape[1]:
        return 0.0
    h_cmd = h[:, 0::2][:, :maxn]                                 # [nr, maxn, d]
    hq = h_cmd[b_sel, j_sel]                                     # [N, d]
    p = probe(hq)                                                # [N, D] recalled anchor obs
    p = torch.nan_to_num(p, nan=0.0, posinf=1e4, neginf=-1e4)
    z_a = obs[b_sel, a_sel].detach()                             # [N, D] PAST observation (i<j)
    z_a = torch.nan_to_num(z_a, nan=0.0, posinf=1e4, neginf=-1e4)

    # (a) reconstruction: cosine + small mse + log-norm match, cue-strength weighted
    pu, au = _unit(p), _unit(z_a)
    cos_err = (w * (1.0 - (pu * au).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (p - z_a).pow(2).mean(dim=-1)).sum()
    pn = p.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    an = z_a.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    norm_err = (w * _huber_abs(torch.log(pn) - torch.log(an), beta=0.2)).sum()
    total = (
        float(cfg["cos_weight"]) * cos_err
        + float(cfg["mse_weight"]) * mse_err
        + float(cfg["norm_weight"]) * norm_err
    )

    # (b) cued recall as discrimination: pick YOUR anchor among the batch's anchors.
    # Constant hidden states cannot minimize this (loss >= log N at any constant output),
    # so the aux is anti-collapse by construction. Near-identical anchors (the same file
    # cat twice, empty-output steps) are false negatives -> masked.
    N = p.shape[0]
    if N >= 2 and float(cfg["nce_weight"]) > 0.0:
        logits = (pu @ au.transpose(0, 1)) / max(float(cfg["nce_temp"]), 1e-4)
        with torch.no_grad():
            aa = (au @ au.transpose(0, 1)).clamp(-1.0, 1.0)
            eye = torch.eye(N, dtype=torch.bool, device=p.device)
            dup = (aa > float(cfg["dup_thresh"])) & ~eye
        logits = logits.masked_fill(dup, -1.0e4)
        ce = F.cross_entropy(logits, torch.arange(N, device=p.device), reduction="none")
        total = total + float(cfg["nce_weight"]) * (w * ce).sum()

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    """The aux branch reads ONLY past tokens (anchor index strictly < query index) and
    never re-points net.forward, so the eval path is bit-identical to the wrapped arch.
    Validate the params are finite and in range; reject anything else."""
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        0.0 < vals["row_frac"] <= 1.0,
        0.0 < vals["top_q"] <= 1.0,
        -1.0 <= vals["s_floor"] < 1.0,
        vals["cos_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["norm_weight"] >= 0.0,
        vals["nce_weight"] >= 0.0,
        vals["nce_temp"] > 0.0,
        0.0 < vals["dup_thresh"] <= 1.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
