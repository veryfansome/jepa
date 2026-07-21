"""head chunk: TRUTH-GATED TEMPORAL STRAIGHTENING of the prediction trajectory.

Cross-domain principle (temporal straightening for latent planning, arXiv:2603.12231):
regularizing CONSECUTIVE LATENT VELOCITIES to align provably conditions distance-based
planning objectives — on PointMaze it lifts planning success 44%->94% without touching
the dynamics loss. Our planning instrument (evolve/path_battery.py) is exactly such an
objective: horizon-2 write-policy imagination ranked by SUM-OF-COSINE to a goal
embedding. Its measured failure mode is per-decision discrimination at the final
approach (remaining-depth 1: 0.42-0.51), where the imagined next-ls must separate the
goal's listing from up to 237 siblings while lagging the real-obs field (imagined-vs-
real ls ~3446 sqL2 vs 1436 random-pair) — directionally erratic imagination in a field
whose ground truth already ranks the on-path child first on 64/100 goals.

Shell trajectories are NOT physical motion, so blanket straightening is wrong: a
`cat`/`uname` jump genuinely turns, and forcing it straight would fight the main loss.
But DESCENDING A DIRECTORY PATH IS A STRAIGHT WALK IN A GOOD GEOMETRY (cwd echoes and
listings extend monotonically down one branch), and that is precisely the regime the
battery scores. Rather than hand-coding a verb heuristic over frozen embeddings, we
gate on the data: the aux measures the alignment of the TRUE next-obs velocities
(labels the aux already owns) and applies the straightening pressure only where the
real walk is itself straight in the frozen e5 field.

Mechanism (train-only, forward untouched). One extra teacher-forced pass; at cmd
positions map predictions to obs space (exact inverse for a LEARNED target, identity
for the champion), then for every valid triple (t, t+1, t+2):

    v_t = pred_{t+1} - pred_t          (prediction velocity, grads flow)
    u_t = tgt_{t+1}  - tgt_t           (true velocity, detached labels)
    g_t = relu((cos(u_t, u_{t+1}) - tau) / (1 - tau)) ** gamma      (gate, detached)
    L   = w * sum_t g_t * (1 - cos(v_t, v_{t+1})) / sum_t g_t

Why the content-verb margin survives: the pressure is second-order (directions of
velocity differences, never positions), small (w=0.1 vs the champion's tolerated 0.05
MSE anchor), and vanishes exactly where the true trajectory turns — where active it
pushes predictions toward curvature the retrieval targets already have, so it cannot
pull them off the manifold the rank metric lives on. The main contrastive loss is
untouched on the full batch.

Contract / safety:
  * wrap() adds NO modules and does NOT re-point net.forward (no module cycle, no
    forward recursion; the leakage probe runs the arch's own causal forward).
  * Causal: one ordinary pass of the causal net; the loss couples predictions at
    different positions, but no future token ever enters any input. True obs appear
    only as detached loss labels/gates.
  * Anti-collapse: a constant prediction gives v = 0, cos(0,0) = 0 under clamped
    norms, so every gated triple pays the FULL penalty g_t * 1 — collapse is
    penalized, never rewarded; identical true obs (u = 0) auto-close the gate.
  * NaN-safe: clamped norms, finiteness checks, hard-zero fallbacks (maxn < 3, no
    gated triple, non-interleave stream, degenerate batch).
"""

import math

import torch

NAME = "r11_truthgated_velocity_straightening"
DESCRIPTION = (
    "Temporal straightening (2603.12231) adapted to shell walks: penalize misalignment "
    "of consecutive prediction velocities at cmd positions, gated per-triple by the "
    "alignment of the TRUE obs velocities — straighten imagination exactly where the "
    "real walk (directory descent) is straight in the frozen field, gate off cat/uname "
    "jumps automatically. Train-only aux; forward untouched; conditions the battery's "
    "sum-of-cosine planning objective."
)

_AUX_W = 0.1       # weight (champion tolerates a 0.05 MSE anchor without margin damage)
_GATE_TAU = 0.0    # gate threshold on true-velocity cosine (0 = any forward alignment)
_GATE_POW = 2.0    # gate sharpness: emphasize strongly straight ground-truth segments
_ROW_FRAC = 1.0    # fraction of batch rows the aux touches (1 extra forward per step)
_EPS = 1e-8


def wrap(net, D, aux_weight=_AUX_W, gate_tau=_GATE_TAU, gate_pow=_GATE_POW,
         row_frac=_ROW_FRAC):
    """No modules, no forward re-pointing: the arch's readout is untouched, so the aux
    composes with ANY arch. Returns a plain config dict as head_state."""
    return {"w": float(aux_weight), "tau": float(gate_tau), "pow": float(gate_pow),
            "frac": float(row_frac), "D": int(D)}


def _interleave_layout_ok(b):
    """The cmd-position indexing assumes the single-vector interleave: tok[2t]=cmd_t,
    tok[2t+1]=obs_t, tgt[:,t]=z_obs_t. Verify cheaply; other streams -> aux off."""
    tok, types, tgt = b["tok"], b["types"], b["tgt"]
    if tok.dim() != 3 or tgt.dim() != 3 or tok.shape[1] != 2 * tgt.shape[1]:
        return False
    live = ~b["key_pad"]
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    return bool((even == 0).all()) and bool((odd == 1).all())


def _cos(a, b):
    """Row-wise cosine with clamped norms (zero vectors -> cosine 0, never NaN)."""
    num = (a * b).sum(-1)
    den = a.norm(dim=-1).clamp_min(_EPS) * b.norm(dim=-1).clamp_min(_EPS)
    return num / den


def aux_loss(head_state, batch, net, device):
    """Truth-gated velocity-straightening term (train only). Strictly causal: one
    teacher-forced pass of the causal net; true observations enter only as DETACHED
    gate statistics and never as inputs."""
    hs = head_state
    if hs is None or hs["w"] == 0.0:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0
    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    tgt, cmd_mask = batch["tgt"], batch["cmd_mask"]
    B, maxn, D = tgt.shape
    if maxn < 3:
        return 0.0

    # Optional row subsample (compute knob; default = full batch, 1 extra forward).
    if hs["frac"] < 1.0:
        nrows = max(1, int(math.ceil(B * hs["frac"])))
        rows = torch.randperm(B, device=device)[:nrows]
        tok, types, key_pad = tok[rows], types[rows], key_pad[rows]
        tgt, cmd_mask = tgt[rows], cmd_mask[rows]

    # Teacher-forced pass (grad ON): predictions at every cmd position.
    pred_full, _ = net(tok, types, key_pad)
    cmd_pred = pred_full[:, 0::2][:, :maxn]                    # [B, maxn, D]

    # Map predictions -> obs space (exact inverse for a LEARNED target, identity for
    # the champion), mirroring plan_env.imagine_candidate / path_battery's to_obs.
    tmod = getattr(net, "target_module", None)
    if tmod is not None:
        prev = torch.cat([torch.zeros_like(tgt[:, :1]), tgt[:, :-1]], dim=1)
        b_, m_, d_ = cmd_pred.shape
        obs_pred = tmod.to_obs(cmd_pred.reshape(-1, d_), prev.reshape(-1, d_)).reshape(b_, m_, d_)
    else:
        obs_pred = cmd_pred

    # Velocities of the predicted and true obs trajectories.
    v = obs_pred[:, 1:] - obs_pred[:, :-1]                     # [B, maxn-1, D] (grads)
    u = (tgt[:, 1:] - tgt[:, :-1]).detach()                    # [B, maxn-1, D] (labels)

    # Valid triples: cmd_mask at t, t+1, t+2.
    trip = cmd_mask[:, :-2] & cmd_mask[:, 1:-1] & cmd_mask[:, 2:]   # [B, maxn-2]
    if not trip.any():
        return 0.0

    # Gate: alignment of the TRUE consecutive velocities (detached), thresholded.
    cos_u = _cos(u[:, :-1], u[:, 1:])                          # [B, maxn-2]
    tau = hs["tau"]
    gate = torch.relu((cos_u - tau) / max(1.0 - tau, _EPS)).pow(hs["pow"])
    gate = gate * trip.float()
    gsum = gate.sum()
    if float(gsum) <= _EPS:
        return 0.0

    # Straightening penalty on the PREDICTION velocities, gate-weighted mean.
    cos_v = _cos(v[:, :-1], v[:, 1:])                          # [B, maxn-2]
    pen = (gate * (1.0 - cos_v)).sum() / gsum.clamp_min(_EPS)
    if not torch.isfinite(pen):
        return 0.0
    return hs["w"] * pen


def leak_safe(mod, params):
    """Certify no-future-leakage:
      - forward is UNTOUCHED (wrap adds nothing), so the stream's leakage_ok probe
        runs the arch's own causal forward unchanged;
      - the aux runs one ordinary causal pass; predictions at different positions are
        coupled only inside the LOSS, never fed back as inputs;
      - true observations enter only as detached gate statistics / loss labels in
        aux_loss, which is absent from forward() and thus dropped at eval.
    Validate params: finite non-negative weight, tau in [-1, 1), pow > 0, frac in (0, 1]."""
    w = float(params.get("aux_weight", _AUX_W))
    tau = float(params.get("gate_tau", _GATE_TAU))
    p = float(params.get("gate_pow", _GATE_POW))
    f = float(params.get("row_frac", _ROW_FRAC))
    return (w >= 0.0 and math.isfinite(w) and -1.0 <= tau < 1.0 and p > 0.0
            and math.isfinite(p) and 0.0 < f <= 1.0)
