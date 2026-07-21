"""head chunk: REACHABILITY-SHAPED PREDICTIONS — time-contrastive horizon ranking aux.

Cross-domain principle (goal-conditioned RL / quasimetric value shaping, arXiv:2601.00844;
time-contrastive networks, Sermanet et al. arXiv:1704.06888): planning by latent distance
only works if DISTANCE ENCODES REACHABILITY. The R11 battery ranks candidate actions by
cosine(imagined next-obs, goal-obs) — a future observation several steps ahead — so the
binding requirement is not one-step fidelity but that the prediction's cosine field be
MONOTONE IN TIME-TO-REACH: an imagined latent should be cosine-closer to near futures
than to far futures, and, dually, latents imagined closer to a future obs should be
cosine-closer to it than latents imagined farther back. The champion's contrastive
objective shapes pred-vs-own-target ranking against in-batch negatives, but places NO
constraint on how a prediction orders the OTHER observations of its own trajectory —
exactly the geometry the battery's sum-of-cosine-to-goal readout consumes, and exactly
where Stage-2 located the loss (remaining-depth-1 decisions, 0.42-0.51: the goal's own
listing vs its siblings').

Mechanism (train-only, forward untouched, ZERO new parameters — unlike the registered
r7_normres_multihorizon_aux, which bolts on an aux Linear and regresses future obs by
MSE, this shapes the MAIN prediction's cosine geometry by ranking, adds no modules, and
is scale-invariant so it cannot fight the norm calibration of any stacked mechanism):
  1. One causal pass on a row-subsample: pred_t = net(tok)[cmd pos 2t], mapped to obs
     space via target_module.to_obs when a learned target is attached (the battery's own
     readout path), identity otherwise.
  2. Per row, the cosine matrix S[t, u] = cos(pred_t, z_obs_u) over the sequence's TRUE
     observations (labels only, never inputs). For every horizon pair 0 <= j1 < j2 <= K
     and every valid anchor, two smooth hinges enforce the reachability ordering:
       forward  (imagination fan-out): S[t, t+j1]     > S[t, t+j2]     + m
       backward (progress-to-goal):    S[u-j1, u]     > S[u-j2, u]     + m
     i.e. L = w * mean tau*softplus((s_far - s_near + m)/tau). The j1=0 terms make the
     own-target the argmax of each prediction's cosine row (reinforcing the retrieval
     fitness, G1) and make each observation best-explained by its own step's prediction;
     the j1>=1 terms are the new signal: a monotone "temporal potential" along the path,
     the discrete analogue of value-function shaping — on tree-structured filesystem
     walks, futures reached sooner must score closer, so at a fan-out the on-path
     child's horizon-2 imagination (whose true continuation contains the goal soon)
     cosine-dominates off-path siblings (whose continuation reaches it later or never).

Contract / safety:
  * wrap() adds NO modules, does NOT re-point net.forward (no cycle / recursion
    hazards; the stream's leakage probe runs the arch's own forward unchanged). Returns
    a plain config dict as head_state.
  * Strictly causal: pred_t is the net's causal output at position 2t; all future obs
    z_{t+j} enter only as LOSS LABELS from batch["tgt"], never as inputs.
  * Anti-collapse-safe: the aux is a ranking over FROZEN, varying targets — a constant
    prediction makes every cosine row identical, which cannot satisfy the orderings
    (the same obs is 'near' for one anchor and 'far' for another), and the untouched
    main contrastive loss retains full anti-collapse pressure. Cosine is norm-invariant,
    so the aux exerts zero collapse-ward pull on prediction scale.
  * NaN-safe: clamped norms, .any() guards, isfinite check, hard-zero fallbacks;
    non-interleave stream layouts switch the aux off.
"""

import math

import torch
import torch.nn.functional as F

NAME = "r11_reachability_horizon_rank_aux"
DESCRIPTION = (
    "Parameter-free time-contrastive reachability shaping of the MAIN prediction: on a "
    "row-subsample, rank each causal prediction's cosine to its trajectory's true future "
    "observations monotonically in horizon (near > far, forward and goal-anchored views, "
    "smooth hinge). Makes latent cosine distance encode time-to-reach — the exact readout "
    "the horizon-2 sum-of-cosine planner uses — with no new modules, forward untouched, "
    "aux dropped at eval."
)

_K_MAX = 4      # largest horizon in the ranking pairs (pairs 0<=j1<j2<=K)
_MARGIN = 0.05  # cosine margin per ordering constraint
_TAU = 0.10     # softplus temperature (smooth hinge)
_AUX_W = 0.10   # modest weight: shapes, never replaces, the champion signal
_ROW_FRAC = 0.5 # fraction of batch rows the aux touches (~0.5 extra fwd/step)


def wrap(net, D, k_max=_K_MAX, margin=_MARGIN, tau=_TAU, aux_weight=_AUX_W,
         row_frac=_ROW_FRAC):
    """No modules, no forward re-pointing: the arch's readout is untouched (composes
    with ANY arch, including non-per-position-pred mechanisms). Returns a config dict
    as head_state for aux_loss."""
    return {"k": int(k_max), "m": float(margin), "tau": float(tau),
            "w": float(aux_weight), "frac": float(row_frac), "D": int(D)}


def _interleave_layout_ok(b):
    """The cmd-position extraction assumes the single-vector interleave: tok[2t]=cmd_t,
    tok[2t+1]=obs_t, tgt[:,t]=z_obs_t. Verify cheaply; other streams -> aux off."""
    tok, types, tgt = b["tok"], b["types"], b["tgt"]
    if tok.dim() != 3 or tgt.dim() != 3 or tok.shape[1] != 2 * tgt.shape[1]:
        return False
    live = ~b["key_pad"]
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    return bool((even == 0).all()) and bool((odd == 1).all())


def aux_loss(head_state, batch, net, device):
    """Time-contrastive horizon-ranking term (train only). Strictly causal: every
    ranked prediction is the net's causal output; true future obs appear only as
    LOSS LABELS, never as inputs."""
    hs = head_state
    if hs is None or hs["w"] == 0.0 or hs["k"] < 1:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0
    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    tgt, cmd_mask = batch["tgt"], batch["cmd_mask"]
    B, maxn, D = tgt.shape
    if maxn < 2:
        return 0.0

    # Row subsample: aux compute ~= 1 forward on frac of rows.
    nrows = max(1, int(math.ceil(B * hs["frac"])))
    rows = torch.randperm(B, device=device)[:nrows]
    tok_s, types_s, pad_s = tok[rows], types[rows], key_pad[rows]
    tgt_s, cm = tgt[rows], cmd_mask[rows]

    # Causal predictions at each cmd position (grad ON).
    pred_full, _ = net(tok_s, types_s, pad_s)               # [n, L, D]
    cmd_pred = pred_full[:, 0::2][:, :maxn]                 # [n, maxn, D]

    # Map prediction -> obs space (identity for the champion; exact inverse for a
    # LEARNED target, mirroring the battery's imagine path).
    tmod = getattr(net, "target_module", None)
    if tmod is not None:
        prev = torch.cat([torch.zeros_like(tgt_s[:, :1]), tgt_s[:, :-1]], dim=1)
        n_, m_, d_ = cmd_pred.shape
        cmd_pred = tmod.to_obs(cmd_pred.reshape(-1, d_), prev.reshape(-1, d_)).reshape(n_, m_, d_)

    # Cosine field S[b, t, u] = cos(pred_t, z_obs_u) over TRUE observations (labels).
    pred_n = cmd_pred / cmd_pred.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt_n = tgt_s / tgt_s.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    S = torch.einsum("bmd,bnd->bmn", pred_n, tgt_n)         # [n, maxn, maxn]

    m, tau, K = hs["m"], hs["tau"], hs["k"]
    total = S.new_zeros(())
    count = 0
    for j2 in range(1, min(K, maxn - 1) + 1):
        far = S.diagonal(offset=j2, dim1=1, dim2=2)         # [n, maxn-j2]: S[t, t+j2]
        L2 = maxn - j2
        for j1 in range(0, j2):
            near = S.diagonal(offset=j1, dim1=1, dim2=2)    # [n, maxn-j1]: S[t, t+j1]
            # FORWARD view (shared pred anchor t): S[t,t+j1] should beat S[t,t+j2].
            vf = cm[:, :L2] & cm[:, j1:j1 + L2] & cm[:, j2:]
            if vf.any():
                h = tau * F.softplus((far[vf] - near[:, :L2][vf] + m) / tau)
                total = total + h.sum(); count += int(vf.sum())
            # BACKWARD view (shared obs anchor u): S[u-j1,u] should beat S[u-j2,u].
            vb = cm[:, j2 - j1:maxn - j1] & cm[:, :L2] & cm[:, j2:]
            if vb.any():
                h = tau * F.softplus((far[vb] - near[:, j2 - j1:][vb] + m) / tau)
                total = total + h.sum(); count += int(vb.sum())
    if count == 0:
        return 0.0
    out = hs["w"] * (total / count)
    if not torch.isfinite(out):
        return 0.0
    return out


def leak_safe(mod, params):
    """Certify no-future-leakage:
      - forward is UNTOUCHED (wrap adds nothing), so the stream's leakage_ok probe
        runs the arch's own causal forward unchanged;
      - every ranked prediction is the net's causal output at position 2t (a function
        of tokens <= 2t); future observations enter only as loss labels inside
        aux_loss, which is absent from forward() and thus dropped at eval.
    Validate params: k_max >= 1 integer, margin/tau finite with tau > 0, weight
    finite non-negative, row_frac in (0, 1]."""
    k = int(params.get("k_max", _K_MAX))
    mg = float(params.get("margin", _MARGIN))
    tau = float(params.get("tau", _TAU))
    w = float(params.get("aux_weight", _AUX_W))
    f = float(params.get("row_frac", _ROW_FRAC))
    return (k >= 1 and math.isfinite(mg) and mg >= 0.0 and math.isfinite(tau)
            and tau > 0.0 and math.isfinite(w) and w >= 0.0 and 0.0 < f <= 1.0)
