"""R16 head chunk: system-deviation typicality probe, composed with cued recall.

Idea (information-theoretic): the fitness is a MARGIN over retrieve-by-cmd, and
retrieve-by-cmd predicts (approximately) the command-conditioned TYPICAL
observation from the train pool. So the only component of the target that can
ever carry margin is the DEVIATION of this system's observation from the
cross-system typical for that command — exactly where the stalled cat margin
lives (same path, different body per system; retrieve-by-cmd top1 .531 there).
The main MSE spreads capacity isotropically over typical+deviation; nothing
tells the trunk that the deviation direction is the paying one.

Mechanism: at each command position, mine the in-batch CROSS-TRAJECTORY
command group (same standardized cmd embedding, other rows — the sysblock
batcher's uniform half supplies cross-system rows), form a content-deduplicated
"typical" observation (near-identical member observations are down-weighted to
count once, approximating a per-system mean rather than a per-row mean), and
train a LINEAR probe from the trunk hidden state at the command to (a)
reconstruct the deviation z_obs - typical and (b) discriminate its own
deviation among the batch's deviations (weighted, duplicate-masked InfoNCE).
Only genuinely system-variant steps are supervised (group must exist, relative
deviation norm gated + top-quantile). The trunk can only predict the deviation
from HISTORY (what this trajectory already revealed about this system), so the
gradient forces system identity to be carried in-state as a structured
deviation code — complementing, not duplicating, the champion cued-recall probe
(which binds within-trajectory episodic content; this term binds cross-system
schema-vs-episode structure). Cross-domain lens: complementary learning
systems — neocortical schema vs hippocampal episodic specifics (McClelland,
McNaughton & O'Reilly 1995) — and hippocampal/CA1 match-mismatch novelty
comparison against a schema prediction (Lisman & Grace 2005); predictive-coding
residual coding (Rao & Ballard 1999): represent what deviates from expectation,
not the expectation itself.

Composition: this module EMBEDS the champion cued-recall binding aux verbatim
(same defaults, renamed params rec_*) and shares its single extra subsampled
forward pass, so the new signal costs no additional trunk forward. Eval forward
untouched; recall targets are strictly past (anchor < query); deviation targets
use only in-batch TRAIN data as supervision targets (never as eval input), so
the eval path stays bit-identical and leak-free. Hard-zero fallback on any
non-interleaved layout; NaN-guarded; both InfoNCE terms are >= log N at any
constant hidden state, so the aux is anti-collapse by construction.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r16_sysdev_typicality_probe"
DESCRIPTION = (
    "Train-only composed aux, eval forward untouched: (1) the champion cued-recall binding "
    "probe verbatim (within-trajectory episodic binding); (2) a NEW system-deviation probe — "
    "per command position, build the content-deduplicated cross-trajectory typical "
    "observation for that command from the batch, and train a linear probe on the trunk "
    "hidden state to reconstruct and discriminate THIS system's deviation from that typical. "
    "Supervises exactly the margin-carrying component (what retrieve-by-cmd cannot supply), "
    "targeting the stalled system-variant cat margin. Both probes share one subsampled "
    "forward; deviation supervision only fires on gated, genuinely system-variant steps."
)

_DEFAULTS = {
    # shared
    "row_frac": 0.5,       # fraction of batch rows the one aux forward runs on
    "ramp_steps": 400,     # smoothstep ramp so early training is main-loss-dominated
    "aux_weight": 1.0,
    # cued-recall branch (champion r12 defaults, verbatim)
    "rec_top_q": 0.35,
    "rec_s_floor": 0.20,
    "rec_cos_weight": 0.08,
    "rec_mse_weight": 0.012,
    "rec_norm_weight": 0.015,
    "rec_nce_weight": 0.10,
    "rec_nce_temp": 0.07,
    "rec_dup_thresh": 0.999,
    # system-deviation branch (new)
    "dev_cmd_thresh": 0.985,   # cross-row steps this cmd-similar form the command group
    "dev_obs_dup": 0.995,      # member observations this similar count once in the typical
    "dev_min_group": 2.0,      # need >= this many raw cross-row members
    "dev_rel_floor": 0.05,     # ||dev|| / ||obs|| floor: skip system-invariant steps
    "dev_top_q": 0.6,          # keep this fraction of gated steps, by relative deviation
    "dev_max_queries": 320.0,  # hard cap on supervised steps per aux call
    "dev_cos_weight": 0.06,    # probe->deviation cosine reconstruction
    "dev_mse_weight": 0.008,   # small metric anchor
    "dev_nce_weight": 0.08,    # discriminate own deviation among batch deviations
    "dev_nce_temp": 0.07,
    "dev_dup_thresh": 0.999,   # near-identical deviations are false negatives -> masked
}

_EPS = 1e-8
_MAX_STEPS = 2048  # soft cap on subsampled step count (bounds the no_grad sim matrices)


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
    """No forward re-point, no module cycle: register TWO linear probes on the net (so the
    genome's optimizer and .to(device) pick them up) and return a config dict. Linear
    probes (LN + Linear) keep the pressure on the TRUNK: the recalled content and the
    system-deviation code must be linearly present in the hidden state — the probes cannot
    compute them themselves."""
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    d = _probe_hidden_dim(net, D)
    if d is None:
        cfg["_disabled"] = True
        return cfg
    recall = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, int(D)))
    nn.init.zeros_(recall[1].bias)
    net.r16_recall_probe = recall
    sysdev = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, int(D)))
    nn.init.zeros_(sysdev[1].bias)
    net.r16_sysdev_probe = sysdev
    cfg["_disabled"] = False
    return cfg


def _mine_anchors(cmd, valid):
    """Cued-recall mining (champion, verbatim mechanism): per row, for each command
    position j, the most cmd-similar EARLIER position i < j. Returns (sim s [B,maxn],
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


def _mine_sysdev(cmd, obs, valid, cfg):
    """System-deviation mining (no_grad): for each valid step q, the cross-row same-command
    group; a content-deduplicated typical observation; the deviation target. Returns
    (sel flat indices [N], dev targets [N,D], weights [N]) or None."""
    nr, maxn, D = obs.shape
    M = nr * maxn
    dev_dtype = obs.dtype
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4)).reshape(M, D)
    ob = torch.nan_to_num(obs, nan=0.0, posinf=1e4, neginf=-1e4).reshape(M, D)
    ou = _unit(ob)
    vflat = valid.reshape(M)
    if not bool(vflat.any().item()):
        return None
    row_id = torch.arange(nr, device=obs.device).repeat_interleave(maxn)

    same_cmd = (cu @ cu.transpose(0, 1)) > float(cfg["dev_cmd_thresh"])
    cross = row_id[:, None] != row_id[None, :]
    mem = same_cmd & cross & vflat[None, :] & vflat[:, None]
    cnt = mem.sum(dim=1)
    if int(cnt.max().item()) < int(cfg["dev_min_group"]):
        return None

    # Content dedup: a member counts 1/(number of near-identical-obs members in the same
    # group), so the typical approximates a per-SYSTEM mean (the batcher's same-image
    # block half cannot drag the typical toward the query's own content).
    obsdup = ((ou @ ou.transpose(0, 1)) > float(cfg["dev_obs_dup"])).to(dev_dtype)
    memf = mem.to(dev_dtype)
    dupcnt = memf @ obsdup                                       # dupcnt[q,k] over group q
    W = memf / dupcnt.clamp_min(1.0)
    Wsum = W.sum(dim=1, keepdim=True)
    typical = (W @ ob) / Wsum.clamp_min(_EPS)
    dev = ob - typical

    obs_norm = ob.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
    rel = dev.pow(2).sum(dim=-1).clamp_min(0.0).sqrt() / obs_norm
    rel = torch.nan_to_num(rel, nan=0.0)
    elig = vflat & (cnt >= int(cfg["dev_min_group"])) & (rel > float(cfg["dev_rel_floor"]))
    if not bool(elig.any().item()):
        return None
    idx = elig.nonzero(as_tuple=False).squeeze(1)
    rel_e = rel[idx]
    keep_n = max(1, int(math.ceil(float(cfg["dev_top_q"]) * idx.numel())))
    keep_n = min(keep_n, int(cfg["dev_max_queries"]), idx.numel())
    top = torch.topk(rel_e, keep_n).indices
    sel = idx[top]
    w = rel_e[top].clamp_min(1e-4)
    w = w / w.sum()
    return sel, dev[sel], w


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    recall_probe = getattr(net, "r16_recall_probe", None)
    sysdev_probe = getattr(net, "r16_sysdev_probe", None)
    if recall_probe is None or sysdev_probe is None:
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

    # -- row subsample (cost control: ONE extra forward shared by both branches) --
    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    nrows = min(nrows, max(1, _MAX_STEPS // max(1, maxn)))
    rows = torch.randperm(B, device=device)[:nrows]
    tok_s, types_s, pad_s = tok[rows], types[rows], key_pad[rows]
    valid = cmd_mask[rows]

    cmd = tok_s[:, 0::2][:, :maxn]                               # standardized z_cmd  [nr,maxn,D]
    obs = tok_s[:, 1::2][:, :maxn]                               # standardized z_obs  [nr,maxn,D]

    # ---------------- no_grad mining for BOTH branches ----------------
    rec_sel = None
    with torch.no_grad():
        s, a, eligible = _mine_anchors(cmd, valid)
        s = torch.nan_to_num(s, nan=-2.0)
        flat = eligible.reshape(-1)
        if bool(flat.any().item()):
            idx = flat.nonzero(as_tuple=False).squeeze(1)
            s_flat = s.reshape(-1)[idx]
            keep_n = max(1, int(math.ceil(float(cfg["rec_top_q"]) * idx.numel())))
            top = torch.topk(s_flat, keep_n).indices
            idx, s_flat = idx[top], s_flat[top]
            floor = s_flat > float(cfg["rec_s_floor"])
            if bool(floor.any().item()):
                idx, s_flat = idx[floor], s_flat[floor]
                b_sel = idx // maxn
                j_sel = idx % maxn
                a_sel = a.reshape(-1)[idx]
                w_rec = (s_flat - float(cfg["rec_s_floor"])).clamp_min(0.0) + 1e-4
                w_rec = w_rec / w_rec.sum()
                rec_sel = (b_sel, j_sel, a_sel, w_rec)
        dev_sel = _mine_sysdev(cmd, obs, valid, cfg)

    if rec_sel is None and dev_sel is None:
        return 0.0

    # -- the ONE shared trunk forward on the subsample --
    out = net(tok_s, types_s, pad_s)
    if not (isinstance(out, tuple) and len(out) >= 2):
        return 0.0
    h = out[1]
    if not torch.is_tensor(h) or h.dim() != 3 or h.shape[1] != tok_s.shape[1]:
        return 0.0
    h_cmd = h[:, 0::2][:, :maxn]                                 # [nr, maxn, d]

    total = h_cmd.sum() * 0.0  # typed zero on the right device/graph

    # ---------------- branch 1: cued-recall binding (champion, verbatim) ----------------
    if rec_sel is not None:
        b_sel, j_sel, a_sel, w = rec_sel
        hq = h_cmd[b_sel, j_sel]                                 # [N, d]
        p = recall_probe(hq)                                     # [N, D] recalled anchor obs
        p = torch.nan_to_num(p, nan=0.0, posinf=1e4, neginf=-1e4)
        z_a = obs[b_sel, a_sel].detach()                         # [N, D] PAST observation (i<j)
        z_a = torch.nan_to_num(z_a, nan=0.0, posinf=1e4, neginf=-1e4)

        pu, au = _unit(p), _unit(z_a)
        cos_err = (w * (1.0 - (pu * au).sum(dim=-1).clamp(-1.0, 1.0))).sum()
        mse_err = (w * (p - z_a).pow(2).mean(dim=-1)).sum()
        pn = p.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
        an = z_a.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
        norm_err = (w * _huber_abs(torch.log(pn) - torch.log(an), beta=0.2)).sum()
        total = total + (
            float(cfg["rec_cos_weight"]) * cos_err
            + float(cfg["rec_mse_weight"]) * mse_err
            + float(cfg["rec_norm_weight"]) * norm_err
        )
        N = p.shape[0]
        if N >= 2 and float(cfg["rec_nce_weight"]) > 0.0:
            logits = (pu @ au.transpose(0, 1)) / max(float(cfg["rec_nce_temp"]), 1e-4)
            with torch.no_grad():
                aa = (au @ au.transpose(0, 1)).clamp(-1.0, 1.0)
                eye = torch.eye(N, dtype=torch.bool, device=p.device)
                dup = (aa > float(cfg["rec_dup_thresh"])) & ~eye
            logits = logits.masked_fill(dup, -1.0e4)
            ce = F.cross_entropy(logits, torch.arange(N, device=p.device), reduction="none")
            total = total + float(cfg["rec_nce_weight"]) * (w * ce).sum()

    # ---------------- branch 2: system-deviation typicality (NEW) ----------------
    if dev_sel is not None:
        sel, dev_t, w = dev_sel
        d_hid = h_cmd.shape[-1]
        hq = h_cmd.reshape(-1, d_hid)[sel]                       # [N, d]
        p = sysdev_probe(hq)                                     # [N, D] predicted deviation
        p = torch.nan_to_num(p, nan=0.0, posinf=1e4, neginf=-1e4)
        dev_t = torch.nan_to_num(dev_t.detach(), nan=0.0, posinf=1e4, neginf=-1e4)

        pu, du = _unit(p), _unit(dev_t)
        cos_err = (w * (1.0 - (pu * du).sum(dim=-1).clamp(-1.0, 1.0))).sum()
        mse_err = (w * (p - dev_t).pow(2).mean(dim=-1)).sum()
        total = total + (
            float(cfg["dev_cos_weight"]) * cos_err
            + float(cfg["dev_mse_weight"]) * mse_err
        )
        N = p.shape[0]
        if N >= 2 and float(cfg["dev_nce_weight"]) > 0.0:
            logits = (pu @ du.transpose(0, 1)) / max(float(cfg["dev_nce_temp"]), 1e-4)
            with torch.no_grad():
                dd = (du @ du.transpose(0, 1)).clamp(-1.0, 1.0)
                eye = torch.eye(N, dtype=torch.bool, device=p.device)
                dup = (dd > float(cfg["dev_dup_thresh"])) & ~eye
            logits = logits.masked_fill(dup, -1.0e4)
            ce = F.cross_entropy(logits, torch.arange(N, device=p.device), reduction="none")
            total = total + float(cfg["dev_nce_weight"]) * (w * ce).sum()

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    """Neither branch touches the eval path: no forward re-point, probes are extra
    submodules used only inside aux_loss. Recall targets are strictly PAST tokens
    (anchor index < query index); deviation targets are in-batch TRAIN observations used
    only as supervision targets, never as model input — the eval forward is bit-identical
    to the wrapped arch. Validate the params are finite and in range; reject anything else."""
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
        vals["ramp_steps"] >= 1.0,
        vals["aux_weight"] >= 0.0,
        0.0 < vals["rec_top_q"] <= 1.0,
        -1.0 <= vals["rec_s_floor"] < 1.0,
        vals["rec_cos_weight"] >= 0.0,
        vals["rec_mse_weight"] >= 0.0,
        vals["rec_norm_weight"] >= 0.0,
        vals["rec_nce_weight"] >= 0.0,
        vals["rec_nce_temp"] > 0.0,
        0.0 < vals["rec_dup_thresh"] <= 1.0,
        0.0 < vals["dev_cmd_thresh"] < 1.0,
        0.0 < vals["dev_obs_dup"] <= 1.0,
        vals["dev_min_group"] >= 1.0,
        0.0 <= vals["dev_rel_floor"] < 1.0,
        0.0 < vals["dev_top_q"] <= 1.0,
        vals["dev_max_queries"] >= 1.0,
        vals["dev_cos_weight"] >= 0.0,
        vals["dev_mse_weight"] >= 0.0,
        vals["dev_nce_weight"] >= 0.0,
        vals["dev_nce_temp"] > 0.0,
        0.0 < vals["dev_dup_thresh"] <= 1.0,
    ]
    return all(checks)
