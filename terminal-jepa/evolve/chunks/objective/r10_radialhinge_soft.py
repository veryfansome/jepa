"""WEIGHT VARIANT of r10_radial_hinge_calibrated_contrastive — softer calibration: LAMBDA_RAD 0.5, LAMBDA_PLACE 0.03 (recover ranking margin, accept equilibrium ratio ~1.2)."""
"""objective chunk: R10 CALIBRATED free-energy contrastive — champion geometry + two
self-annealing absolute-placement terms (radial log-norm match + placement hinge).

THE MEASURED DEFECT THIS FIXES (R10 prereg): the champion's row-softmax over
-dist2/tau is INVARIANT to each prediction's own squared norm (a per-row constant),
so nothing pays for absolute placement -> norm_ratio ||pred||^2/||true||^2 ~ 4.7,
matched sqL2 ~2020 > random TRUE-pair ~1430, and latent-MPC planning fails. The
0.05 MSE anchor is too weak to oppose the norm inflation that sharpens the softmax.

MECHANISM — keep the champion listwise term VERBATIM, add two calibration terms
that act on the axes the softmax cannot see, and whose gradients ANNEAL TO ~ZERO
once the prediction is on-manifold (this is why it is not the failed mse+contrastive
blend: the regression pressure is gated, not constant):

  1. RADIAL log-norm match (fixes norm_ratio exactly):
         L_rad = mean_i ( log r_i )^2,  r_i = (||p_i||^2 + 0.01*||t_i||^2) / ||t_i||^2
     Scale-free, gradient purely along p_i (radial), hence ORTHOGONAL to the
     direction/ranking geometry the contrastive term trains. Minimized at
     ||p_i|| ~ ||t_i|| per row -> batch norm_ratio -> 1 (gate G2 wants [0.8,1.5]).
     The 0.01*||t||^2 floor keeps the gradient bounded and -> 0 smoothly as
     ||p|| -> 0 (a fresh net's tiny-norm phase cannot blow up).
     Note pure distance-minimization at imperfect cosine c has its radial optimum
     at ||p|| = c*||t|| (ratio c^2 ~ 0.49 at c=0.70 — would UNDERSHOOT the gate),
     so an explicit norm-matching term is needed; a hinge alone is not enough.

  2. PLACEMENT hinge (fixes matched sqL2; makes training pay for absolute placement):
         L_place = mean_i softplus( (d_true_i - rho) / tau_c )
     with d_true_i the SAME precision-weighted per-dim-mean squared L2 the champion
     softmax uses (diag of dist2), and rho a DETACHED radius = KAPPA * (mean
     off-diagonal tgt-tgt distance in the same weighted geometry) — i.e. "be closer
     to your own target than half the typical distance between two true
     observations", the literal G2 criterion embedded as a smooth constraint.
     While off-manifold (champion today: d_true ~2.6/dim vs rho ~0.9/dim) this is a
     near-linear precision-weighted regression pull; once inside the radius the
     sigmoid gate turns it off EXPONENTIALLY and the champion geometry governs
     alone. The calibrated equilibrium (norm_ratio ~1, cos ~0.7 -> d_true ~0.6/dim)
     sits INSIDE the radius, so at convergence this term contributes ~nothing.

WHY THE RANKING MARGIN SURVIVES (G1): the champion's precision-weighted focal
listwise term is untouched at full strength throughout; the radial term's gradient
is orthogonal to direction; the hinge's gradient is exactly the precision-weighted
error direction (the same signal as the champion's own MSE anchor, adaptively
gated); and the train-time ranking geometry (dist2 includes the varying ||t_j||^2
term) co-moves with the eval's squared-L2 retrieval at ANY prediction norm, so the
model re-equilibrates its inner products to the calibrated norm rather than losing
the ranking. At the calibrated fixed point the total loss reduces to the champion
loss + epsilon. Verified at stationarity in an adversarial scale-only toy
(direction frozen at cos~0.68, contrastive inflation live): champion equilibrium
norm_ratio 1.68 (fails [0.8,1.5]); this loss 1.10 (passes), matched 508 < 1511
rand-pair, top-1 identical at 1.000.

Contract / safety:
  * Pure function of (pred, tgt); precision weights and rho DETACHED; no state,
    no in-place edits. Elementwise + two [n,n] matmuls (fast).
  * NaN-safe: eps floors inside 1/MSE, the ratio, and rho; dist2 clamp_min(0);
    n < 2 -> anchor + radial only (no contrastive / hinge).
  * Anti-collapse: constant pred -> identical logit rows, softmax cannot place
    mass on the diagonal (champion argument unchanged); MSE(const, varying tgt)
    strictly positive; the radial term is strictly positive for any constant pred
    vs varying ||t_i|| (and huge for pred ~ 0); the hinge is >= 0 and detached-rho
    offers no escape. Collapse cannot minimize the loss.
"""

import torch
import torch.nn.functional as F

NAME = "radial_hinge_calibrated_precision_contrastive"
DESCRIPTION = (
    "Champion free-energy precision-weighted focal listwise L2 contrastive kept verbatim, plus two "
    "SELF-ANNEALING absolute-placement terms the row-softmax is blind to: (1) a radial log-norm match "
    "(log||p||^2 - log||t||^2)^2 whose gradient is orthogonal to ranking direction and drives "
    "norm_ratio -> 1, and (2) a placement hinge softplus((d_true - rho)/tau_c) in the same "
    "precision-weighted geometry with a detached radius rho = 0.5 * mean inter-target distance — "
    "regression pressure exists only while predictions are off-manifold and gates off exponentially "
    "once matched distance beats the random-true-pair scale, so the ranking geometry is recovered "
    "intact at the calibrated fixed point."
)

# ---- champion constants (unchanged) ----
_TEMP = 0.25       # softmax temperature on mean-1-normalized per-dim-mean sqL2
_GAMMA = 1.0       # focal focus on not-yet-#1 rows
_ANCHOR = 0.05     # small MSE anchor (kept: anti-collapse + placement seed)
_BETA = 0.5        # precision temper Pi_d^BETA
_EPS = 1e-2        # MSE floor inside the precision
_WMIN, _WMAX = 0.25, 4.0

# ---- new calibration constants ----
_LAMBDA_RAD = 0.5    # radial log-norm match weight. Verified at stationarity in an adversarial
                     # scale-only toy (direction frozen at cos~0.68, contrastive inflation live):
                     # 0.25 -> equilibrium ratio 1.23, 1.0 -> 1.11, 3.0 -> 1.05; top-1 1.000 at all.
_LAMBDA_PLACE = 0.03  # placement hinge weight
_TAU_C = 0.25        # hinge gate sharpness (same scale as _TEMP; per-dim distances are O(1))
_KAPPA = 0.5         # radius = KAPPA * mean off-diag tgt-tgt per-dim weighted distance
_REPS = 1e-2         # smooth ratio floor: r = (||p||^2 + REPS*||t||^2)/||t||^2, so the radial
                     # gradient (prop. to log(r)/(||p||^2+REPS*||t||^2)) stays BOUNDED and -> 0
                     # smoothly as ||p|| -> 0 (a fresh net's tiny-norm phase cannot blow up).
_NEPS = 1e-6         # absolute floor inside the log-norms


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee.
    mse_anchor = ((pred - tgt) ** 2).mean()

    # Radial calibration: per-row log-norm-ratio match (works for any n >= 1).
    p_sq = (pred * pred).sum(dim=1)                            # [n]
    with torch.no_grad():
        t_sq_det = (tgt * tgt).sum(dim=1).clamp_min(_NEPS)     # [n] (tgt carries no grad anyway)
    ratio = (p_sq + _REPS * t_sq_det) / t_sq_det               # [n], floored smoothly
    radial = ratio.log().pow(2).mean()

    if n < 2:
        return mse_anchor + _LAMBDA_RAD * radial

    # --- Free-energy precision: Pi_d = 1 / Var(error_d), batch-estimated, DETACHED. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)                # [d]
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)                  # tempered precision
        w = w / w.mean().clamp_min(1e-12)                      # mean 1 (match eval scale)
        w = w.clamp(_WMIN, _WMAX)                              # band
        w = w / w.mean().clamp_min(1e-12)                      # re-normalize
        sw = w.sqrt().unsqueeze(0)                             # [1, d]

    # Precision-weighted per-dim-mean squared L2 (champion geometry).
    pw = pred * sw                                             # [n, d]
    tw = tgt * sw                                              # [n, d]
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)                 # [n, 1]
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)                 # [n, 1]
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())            # [n, n]
    dist2 = dist2.clamp_min(0.0) / float(d)

    # --- Champion listwise term (verbatim). ---
    logits = -dist2 / _TEMP
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    # --- Placement hinge: be within rho of YOUR OWN target, in the same geometry. ---
    with torch.no_grad():
        # typical inter-target distance in the precision-weighted space (off-diagonal mean)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)   # [n, n]
        rho = (_KAPPA * tt.sum() / (n * (n - 1))).clamp_min(_EPS)                  # scalar, detached
    d_true = dist2.diagonal()                                                       # [n], carries grad
    place = F.softplus((d_true - rho) / _TAU_C).mean()

    return (listwise + _ANCHOR * mse_anchor
            + _LAMBDA_RAD * radial + _LAMBDA_PLACE * place)
