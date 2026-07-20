"""objective chunk: champion free-energy contrastive + RADIAL Wasserstein-2 recalibration.

THE R10 DEFECT, RESTATED IN COORDINATES
---------------------------------------
Write pred_i = r_i * u_i (radius r_i = ||pred_i||, unit direction u_i). The champion's row
softmax over -dist^2/tau is invariant to r_i's additive contribution (a per-row constant),
and via the -2 r_i u_i . tgt_j term a larger r_i only SHARPENS the row's logits -- a free
per-row inverse temperature. So training pressure on the radius is one-sided (inflate to
sharpen; focal saturation is the only brake) while ALL ranking signal lives in u_i. Result:
r_i drifts to ~2.2x the true radius (norm_ratio 4.7) with cosine 0.702 intact -- rank-good,
placement-broken, latent-MPC blocked.

THE FIX: DISTRIBUTION-MATCH THE RADIAL COORDINATE, LEAVE THE ANGULAR GEOMETRY ALONE
-----------------------------------------------------------------------------------
The gradient of r_i = ||p_i|| w.r.t. p_i is exactly u_i, so any loss term that is a function
of radii only exerts PURELY RADIAL force: it cannot rotate u_i and therefore cannot fight the
champion's ranking geometry at first order. We add two such terms (radii normalized by
sqrt(D) so they are O(1) in the standardized space):

1. Per-example robust radial regression  smooth_l1(r_i/sqrt(D), ||t_i||/sqrt(D)):
   calibrates each prediction's radius to ITS OWN target's radius -- this is what the
   matched-sqL2 gate literally measures (matched = r_p^2 + r_t^2 - 2 r_p r_t cos, minimized
   over r_p at r_p = r_t cos).

2. Radial 1-D Wasserstein-2 (quantile) matching  mean((sort r_p - sort r_t)^2):
   the exact optimal-transport distance between the two radius POPULATIONS. Per-example
   regression alone shrinks predicted radii toward conditional means (variance collapse in
   the radial marginal); the sorted-quantile term restores the spread so the predicted
   radius DISTRIBUTION matches the true one -- absolute latent distances (what a planner
   compares across candidates) stay meaningful in the tails, not just on average.

WHY THE RANKING MARGIN SHOULD SURVIVE (the G1 argument)
-------------------------------------------------------
(a) Radial terms exert zero torque on directions; the listwise/precision/focal machinery that
    earned +0.6260 is byte-identical here. (b) Train/eval geometry stays matched: the eval
    ranks by ||p - t_j||^2 = ||t_j||^2 - 2 p.t_j + const, and the training softmax ranks with
    the SAME pred the eval sees -- at the calibrated radius during training, so the direction
    re-optimizes for the calibrated geometry rather than the inflated one. (c) The inflation
    was never load-bearing signal: it is a per-row temperature, and the champion's own
    temperature comment ("gap d_true ~0.1-1 vs foil ~2") was written assuming ON-manifold
    per-dim-mean distances -- calibration moves training INTO the regime the constants were
    tuned for. (d) The equilibrium is self-limiting: the sharpening gain saturates (focal
    weight (1-p_true) -> 0 on solved rows) while the radial cost grows quadratically, so the
    optimizer settles at r_p ~ r_t instead of drifting.

CONTRACT / SAFETY
-----------------
* Pure function of (pred, tgt); no state, no in-place edits. Precision weights detached as in
  the champion; target radii carry no grad (tgt is data); torch.sort is differentiable w.r.t.
  the pred radii. Cost adds two norms + one sort: negligible next to the [n,n] matmuls.
* NaN-safe: norms via clamp_min(1e-8); everything else inherits the champion's floors/clamps.
  n < 2 -> MSE anchor + per-example radial term only (quantile matching needs a population).
* Anti-collapse: constant pred c -> the champion listwise part is pinned >= log-ish away from
  its minimum exactly as before (identical rows, off-diagonal mass) and MSE(const, varying
  tgt) > 0. The radial terms cannot rescue collapse: with all r_i equal, the W2 term equals
  the variance of the true radii (> 0) plus offset, and smooth_l1(|c| vs varying ||t_i||) is
  minimized but positive -- both add nonnegative amounts a collapsed pred cannot remove.
"""

import torch
import torch.nn.functional as F

NAME = "radial_w2_free_energy_contrastive"
DESCRIPTION = (
    "Champion free-energy precision-weighted L2 listwise objective, unchanged, plus two purely "
    "RADIAL calibration terms whose gradients act only along each prediction's unit direction "
    "(zero torque on the ranking geometry): a per-example robust radius regression "
    "smooth_l1(||pred_i||, ||tgt_i||)/sqrt(D) closing the matched-sqL2 gap, and an exact 1-D "
    "Wasserstein-2 (sorted-quantile) match between the predicted and true radius populations "
    "so the radial DISTRIBUTION (not just its mean) lands on-manifold for latent-MPC."
)

_TEMP = 0.25       # champion: temperature on per-dim-mean sqL2 logits
_GAMMA = 1.0       # champion: focal focus on not-yet-#1 rows
_ANCHOR = 0.05     # champion: absolute-placement MSE anchor (also anti-collapse)
_BETA = 0.5        # champion: precision temper Pi_d^BETA
_EPS = 1e-2        # champion: MSE floor inside the precision
_WMIN, _WMAX = 0.25, 4.0  # champion: band on mean-1 precision weights
_LAM_RAD = 1.0     # per-example radial regression weight (normalized radii are O(1))
_LAM_W2 = 0.5      # radial quantile (1-D W2) population-matching weight
_RAD_BETA = 1.0    # smooth_l1 transition: quadratic within +-1 normalized-radius units


def loss(pred, tgt):
    n, d = pred.shape
    inv_sqrt_d = 1.0 / float(d) ** 0.5

    # Absolute-placement anchor + anti-collapse guarantee (champion, unchanged).
    mse_anchor = ((pred - tgt) ** 2).mean()

    # --- RADIAL coordinates (normalized by sqrt(D) -> O(1) in standardized space). ---
    r_pred = pred.norm(dim=1).clamp_min(1e-8) * inv_sqrt_d          # [n], carries grad
    with torch.no_grad():
        r_true = tgt.norm(dim=1).clamp_min(1e-8) * inv_sqrt_d       # [n], data

    # (1) Per-example robust radial regression: pull each radius to its own target's radius.
    rad_pair = F.smooth_l1_loss(r_pred, r_true, beta=_RAD_BETA)

    if n < 2:
        return mse_anchor + _LAM_RAD * rad_pair

    # (2) Radial 1-D Wasserstein-2: exact OT between the radius populations via quantiles.
    w2_rad = ((torch.sort(r_pred).values - torch.sort(r_true).values) ** 2).mean()

    # --- Free-energy precision (champion, unchanged): Pi_d = 1/Var(error_d), DETACHED. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)                 # [d]
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)                   # tempered precision
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        sw = w.sqrt().unsqueeze(0)                              # [1, d]

    # Precision-weighted per-dim-mean squared L2 (champion, unchanged).
    pw = pred * sw
    tw = tgt * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)                  # [n, 1]
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)                  # [n, 1]
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())             # [n, n]
    dist2 = dist2.clamp_min(0.0) / float(d)

    logits = -dist2 / _TEMP
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)

    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)

    listwise = (focal * nll).mean()
    return listwise + _ANCHOR * mse_anchor + _LAM_RAD * rad_pair + _LAM_W2 * w2_rad
