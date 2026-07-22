"""R16 head chunk: perirhinal view-invariant file-object code + cued-recall stack.

Idea: head/tail/cat/grep of one path are VIEWS of one underlying file object. The
champion's cued-recall probe (kept verbatim here — this impl is a strict stack on it,
sharing the same subsampled trunk forward) forces the hidden state to CONTAIN the
earlier-seen observation; it never asks the trunk's PREDICTIONS for different views
of the same file to be mutually consistent. This impl adds a view-consistency branch:
a small shared codec C (LN+Linear, obs-space D -> code_dim) is trained — on REAL
observation embeddings only — to give the same code to two different-view
observations of one file within a trajectory (InfoNCE across the batch's mined
pairs), i.e. a view-invariant, file-and-system-specific identity code. The trunk is
then pulled so that its prediction at a later view carries the SAME code as the
earlier actually-observed view (cosine in code space, target detached). That is
precisely the component the stalled cat margin needs: retrieve-by-cmd emits one
output per command string across systems, while a prediction whose identity code is
bound to THIS trajectory's earlier-seen file body separates system-variant files.
Cross-domain lens: view-invariant object identity in perirhinal/IT cortex upstream of
hippocampal pattern completion (invariant single-unit object codes, Quiroga et al.
2005; perirhinal object-identity vs hippocampal episodic binding, Murray & Bussey
1999) — the champion implements the hippocampal half; this adds the perirhinal half.
Also: multi-view invariance learning (SimCLR, Chen et al. 2020; alignment/uniformity,
Wang & Isola 2020) with views given for free by the shell (head/tail/cat/grep), and
slow-feature identity codes across temporally-linked views (Wiskott & Sejnowski 2002).

Design against the R10/R11 aux inversion mode (proxy gains inverting at full budget).
Those heads shaped the prediction with MODEL-GENERATED targets (rollout feedback,
the model's own velocity field) — self-referential pressure that compounds over a
4x-longer run. Here, by construction: (1) every codec training signal and every
alignment target is a function of REAL past observations only — no model output is
ever a target; (2) the alignment pull is TRUTH-GATED per pair by the agreement of
the two real observations' codes (if the data does not support "same file", the pair
exerts ~no pull); (3) the new branch ANNEALS to a floor after step 1200 (proxy=1000
never sees the decay; at full budget the late phase reduces toward the champion's
exact recipe, so main-loss geometry finalizes the prediction); (4) the pull is
cosine-only in a low-dim identity subspace and cannot move prediction norms. The
recall branch keeps the champion's proven full-strength schedule untouched.
Eval forward untouched; anchors strictly earlier (i<j) so causal and leak-free;
hard-zero fallback off the v2 interleave layout; InfoNCE terms make constant codes /
constant predictions non-minimizing (anti-collapse by construction).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r16_perirhinal_viewcode_consistency"
DESCRIPTION = (
    "Train-only stack: champion cued-recall binding probe (verbatim, same schedule) + a "
    "perirhinal-style view-invariant file-object code. A shared codec on obs space is "
    "trained by InfoNCE over mined within-trajectory same-file view pairs (real "
    "observations only) to emit a view-invariant, file/system-specific identity code; the "
    "trunk's prediction at the later view is pulled (cosine, detached target, truth-gated "
    "by real-obs code agreement, annealed after step 1200) to carry the earlier-observed "
    "view's code. Aimed at the system-variant cat bodies and the weak tail view, where "
    "predictions must be bound to this trajectory's earlier-seen content."
)

_DEFAULTS = {
    # ---- recall branch: champion r12_cued_recall_binding_probe defaults, unchanged ----
    "row_frac": 0.5,      # fraction of batch rows the shared aux forward runs on
    "top_q": 0.35,        # keep this fraction of eligible recall queries, by anchor cosine
    "s_floor": 0.20,      # loose absolute floor on recall anchor cosine
    "cos_weight": 0.08,   # probe->anchor cosine reconstruction
    "mse_weight": 0.012,  # small metric anchor
    "norm_weight": 0.015, # log-norm match
    "nce_weight": 0.10,   # cued-recall discrimination among in-batch anchors
    "nce_temp": 0.07,
    "dup_thresh": 0.999,  # near-identical anchors are false negatives -> masked
    "aux_weight": 1.0,
    "ramp_steps": 400,    # smoothstep ramp so early training is main-loss-dominated
    # ---- view-code branch (new) ----
    "v_floor": 0.20,      # cmd-cosine floor for a candidate same-file pair
    "v_dup": 0.985,       # cmd-cosine ceiling: above this it is an exact re-query, not a new view
    "v_q": 0.30,          # keep this fraction of in-band pairs, by cmd cosine
    "v_max_pairs": 192,   # hard cap on pairs per batch (cost control)
    "align_weight": 0.10, # pull pred's code toward the earlier real view's code
    "code_weight": 0.08,  # codec InfoNCE on real-obs view pairs
    "code_temp": 0.08,
    "code_dim": 128,      # identity-code width
    "gate_pow": 1.0,      # sharpening of the truth gate
    "anneal_start": 1200, # steps before the view branch starts decaying
    "anneal_len": 800,    # smoothstep decay length
    "anneal_floor": 0.35, # residual view-branch weight after decay
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


def _anneal(step, start, length, floor):
    """1.0 until `start`, then smoothstep decay to `floor` over `length` steps. Proxy
    (1000 steps) never enters the decay; at full budget the late phase is champion-like."""
    if step <= start:
        return 1.0
    t = _smoothstep((step - start) / max(1.0, length))
    return 1.0 - (1.0 - float(floor)) * t


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
    """No forward re-point, no module cycle: register ONLY (a) the champion's linear
    recall probe on the hidden state and (b) a small obs-space codec (LN + Linear
    D->code_dim) shared by real observations and predictions, then return a config
    dict. Both are registered submodules so the genome's optimizer and .to(device)
    pick them up. Linear maps keep the pressure on the TRUNK: the anchor content must
    be linearly present in the hidden state, and the identity code must be linearly
    present in the prediction — neither module can compute it itself."""
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_recall_off"] = True
    cfg["_view_off"] = True
    d = _probe_hidden_dim(net, D)
    if d is not None:
        probe = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, int(D)))
        nn.init.zeros_(probe[1].bias)
        net.r16_recall_probe = probe   # registered submodule; NOT a wrapper around net
        cfg["_recall_off"] = False
    code_dim = max(8, int(cfg["code_dim"]))
    codec = nn.Sequential(nn.LayerNorm(int(D)), nn.Linear(int(D), code_dim))
    nn.init.zeros_(codec[1].bias)
    net.r16_view_codec = codec         # registered submodule; obs-space input, so no h dep
    cfg["_view_off"] = False
    return cfg


def _mine_anchors(sim, valid):
    """Recall mining (champion): per row, for each command position j, the most
    cmd-similar EARLIER live position i < j. sim is the precomputed [B,maxn,maxn]
    cmd-cosine matrix. Returns (s [B,maxn], anchor a [B,maxn], eligible [B,maxn])."""
    B, maxn, _ = sim.shape
    ii = torch.arange(maxn, device=sim.device)
    past = ii[None, :, None] > ii[None, None, :]                 # anchor strictly earlier
    allowed = past & valid[:, None, :].expand(B, maxn, maxn)     # anchor must be a live step
    s, a = sim.masked_fill(~allowed, -2.0).max(dim=2)
    eligible = valid & (s > -1.5)                                # has at least one live earlier step
    return s, a, eligible


def _mine_view_pairs(sim, valid, cfg):
    """View-pair mining: (b, j, i) with i<j, both live, cmd cosine in the BAND
    (v_floor, v_dup) — similar enough to be the same file, below the exact-re-query
    ceiling so the pair carries a genuine view change (cat vs head/tail/grep, or a
    re-phrasing). Keeps the top v_q fraction by cosine, capped at v_max_pairs.
    Returns (b_v, j_v, i_v, w_v) or None."""
    B, maxn, _ = sim.shape
    ii = torch.arange(maxn, device=sim.device)
    past = ii[None, :, None] > ii[None, None, :]
    live_pair = valid[:, :, None] & valid[:, None, :]
    band = past & live_pair & (sim > float(cfg["v_floor"])) & (sim < float(cfg["v_dup"]))
    idx3 = band.nonzero(as_tuple=False)                          # [M, 3] rows of (b, j, i)
    M = idx3.shape[0]
    if M == 0:
        return None
    svals = sim[band]
    keep_n = min(int(cfg["v_max_pairs"]), max(1, int(math.ceil(float(cfg["v_q"]) * M))))
    keep_n = min(keep_n, M)
    top = torch.topk(svals, keep_n).indices
    sel, s_sel = idx3[top], svals[top]
    w = (s_sel - float(cfg["v_floor"])).clamp_min(0.0) + 1e-4
    w = w / w.sum()
    return sel[:, 0], sel[:, 1], sel[:, 2], w


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    if bool(cfg.get("_recall_off", True)) and bool(cfg.get("_view_off", True)):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    step = cfg["_step"]
    ramp = _smoothstep(step / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 2:
        return 0.0

    # -- row subsample (cost control: ONE shared extra forward on a fraction of rows) --
    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    rows = torch.randperm(B, device=device)[:nrows]
    tok_s, types_s, pad_s = tok[rows], types[rows], key_pad[rows]
    valid = cmd_mask[rows]

    cmd = tok_s[:, 0::2][:, :maxn]                               # standardized z_cmd  [nr,maxn,D]
    obs = tok_s[:, 1::2][:, :maxn]                               # standardized z_obs  [nr,maxn,D]

    # -- one cmd-cosine matrix feeds BOTH minings --
    with torch.no_grad():
        cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
        sim = torch.nan_to_num(torch.bmm(cu, cu.transpose(1, 2)), nan=-2.0)

        recall_sel = None
        if not bool(cfg.get("_recall_off", True)):
            s, a, eligible = _mine_anchors(sim, valid)
            flat = eligible.reshape(-1)
            if bool(flat.any().item()):
                idx = flat.nonzero(as_tuple=False).squeeze(1)    # eligible (row, j) pairs
                s_flat = s.reshape(-1)[idx]
                keep_n = max(1, int(math.ceil(float(cfg["top_q"]) * idx.numel())))
                top = torch.topk(s_flat, keep_n).indices
                idx, s_flat = idx[top], s_flat[top]
                floor = s_flat > float(cfg["s_floor"])
                if bool(floor.any().item()):
                    idx, s_flat = idx[floor], s_flat[floor]
                    w_r = (s_flat - float(cfg["s_floor"])).clamp_min(0.0) + 1e-4
                    recall_sel = (idx // maxn, idx % maxn, a.reshape(-1)[idx], w_r / w_r.sum())

        view_sel = None
        if not bool(cfg.get("_view_off", True)):
            view_sel = _mine_view_pairs(sim, valid, cfg)

    if recall_sel is None and view_sel is None:
        return 0.0

    # -- shared trunk forward on the subsample --
    out = net(tok_s, types_s, pad_s)
    if not (isinstance(out, tuple) and len(out) >= 2):
        return 0.0
    pred_full, h = out[0], out[1]
    total = tok_s.new_zeros(())

    # ================= recall branch: champion r12 mechanism, verbatim =================
    probe = getattr(net, "r16_recall_probe", None)
    if recall_sel is not None and probe is not None and torch.is_tensor(h) \
            and h.dim() == 3 and h.shape[1] == tok_s.shape[1]:
        b_sel, j_sel, a_sel, w = recall_sel
        h_cmd = h[:, 0::2][:, :maxn]                             # [nr, maxn, d]
        hq = h_cmd[b_sel, j_sel]                                 # [N, d]
        p = probe(hq)                                            # [N, D] recalled anchor obs
        p = torch.nan_to_num(p, nan=0.0, posinf=1e4, neginf=-1e4)
        z_a = obs[b_sel, a_sel].detach()                         # [N, D] PAST observation (i<j)
        z_a = torch.nan_to_num(z_a, nan=0.0, posinf=1e4, neginf=-1e4)

        # (a) reconstruction: cosine + small mse + log-norm match, cue-strength weighted
        pu, au = _unit(p), _unit(z_a)
        cos_err = (w * (1.0 - (pu * au).sum(dim=-1).clamp(-1.0, 1.0))).sum()
        mse_err = (w * (p - z_a).pow(2).mean(dim=-1)).sum()
        pn = p.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
        an = z_a.pow(2).sum(dim=-1).clamp_min(_EPS).sqrt()
        norm_err = (w * _huber_abs(torch.log(pn) - torch.log(an), beta=0.2)).sum()
        total = total + (
            float(cfg["cos_weight"]) * cos_err
            + float(cfg["mse_weight"]) * mse_err
            + float(cfg["norm_weight"]) * norm_err
        )

        # (b) cued recall as discrimination: pick YOUR anchor among the batch's anchors.
        # Constant hidden states cannot minimize this (loss >= log N at constant output).
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

    # ============= view-code branch: perirhinal identity-code consistency =============
    codec = getattr(net, "r16_view_codec", None)
    if view_sel is not None and codec is not None and torch.is_tensor(pred_full) \
            and pred_full.dim() == 3 and pred_full.shape[:2] == tok_s.shape[:2]:
        b_v, j_v, i_v = view_sel[0], view_sel[1], view_sel[2]
        w_v = view_sel[3]
        pred_cmd = pred_full[:, 0::2][:, :maxn]                  # trunk's per-step prediction
        pq = torch.nan_to_num(pred_cmd[b_v, j_v], nan=0.0, posinf=1e4, neginf=-1e4)
        z_i = torch.nan_to_num(obs[b_v, i_v], nan=0.0, posinf=1e4, neginf=-1e4).detach()
        z_j = torch.nan_to_num(obs[b_v, j_v], nan=0.0, posinf=1e4, neginf=-1e4).detach()

        c_pred = _unit(codec(pq))                                # code of the trunk's prediction
        c_i = _unit(codec(z_i))                                  # code of the EARLIER real view
        c_j = _unit(codec(z_j))                                  # code of the current real view

        # truth gate: only pull when the REAL observations' codes agree that the two
        # steps show the same file. A mis-mined pair (different files) exerts ~no pull.
        with torch.no_grad():
            gate = (c_i * c_j).sum(dim=-1).clamp(0.0, 1.0).pow(float(cfg["gate_pow"]))

        wl = _anneal(step, float(cfg["anneal_start"]), float(cfg["anneal_len"]),
                     float(cfg["anneal_floor"]))

        # (c) alignment: the prediction at the later view must CARRY the earlier
        # actually-observed view's identity code (target detached: real data only).
        if float(cfg["align_weight"]) > 0.0:
            align = (w_v * gate * (1.0 - (c_pred * c_i.detach()).sum(dim=-1).clamp(-1.0, 1.0))).sum()
            total = total + wl * float(cfg["align_weight"]) * align

        # (d) codec training: InfoNCE over REAL-obs view pairs — same file's other view
        # is the positive, other pairs' earlier views are negatives. Trains C to be
        # view-invariant yet file/system-discriminative; constant codes give >= log N.
        Nv = int(z_i.shape[0])
        if Nv >= 2 and float(cfg["code_weight"]) > 0.0:
            logits = (c_j @ c_i.transpose(0, 1)) / max(float(cfg["code_temp"]), 1e-4)
            with torch.no_grad():
                zi_u = _unit(z_i)
                zj_u = _unit(z_j)
                eye = torch.eye(Nv, dtype=torch.bool, device=z_i.device)
                # false negatives: another pair whose earlier view is (near-)identical to
                # this pair's earlier OR current view is the same file -> masked.
                dup = ((zi_u @ zi_u.transpose(0, 1)) > float(cfg["dup_thresh"])) & ~eye
                dup = dup | ((zj_u @ zi_u.transpose(0, 1)) > float(cfg["dup_thresh"])) & ~eye
            logits = logits.masked_fill(dup, -1.0e4)
            ce = F.cross_entropy(logits, torch.arange(Nv, device=z_i.device), reduction="none")
            total = total + wl * float(cfg["code_weight"]) * (w_v * ce).sum()

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not torch.is_tensor(out_loss):
        return 0.0
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    """The aux branch reads ONLY real past/current observations as targets (anchor index
    strictly < query index; the current step's obs is the main loss's own label), never
    feeds any future token into the forward, and never re-points net.forward — the eval
    path is bit-identical to the wrapped arch. Validate params are finite and in range."""
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
        -1.0 <= vals["v_floor"] < 1.0,
        0.0 < vals["v_dup"] <= 1.0,
        vals["v_dup"] > vals["v_floor"],
        0.0 < vals["v_q"] <= 1.0,
        vals["v_max_pairs"] >= 1.0,
        vals["align_weight"] >= 0.0,
        vals["code_weight"] >= 0.0,
        vals["code_temp"] > 0.0,
        vals["code_dim"] >= 8.0,
        vals["gate_pow"] > 0.0,
        vals["anneal_start"] >= 0.0,
        vals["anneal_len"] >= 1.0,
        0.0 <= vals["anneal_floor"] <= 1.0,
    ]
    return all(checks)