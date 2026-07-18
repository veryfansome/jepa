"""objective chunk: Free-Energy PRECISION geometry + MUTUAL-PROXIMITY hubness correction.

TWO CROSS-DOMAIN PRINCIPLES, FUSED (never combined before in this search)
--------------------------------------------------------------------------
1. Free-energy / predictive-coding precision (Friston; Rao-Ballard; Millidge et al.
   arXiv:2107.12979). A Gaussian generative model's variational free energy is a sum of
   squared prediction errors each scaled by its PRECISION Pi_d = 1/Var(error_d) -- the
   inverse error variance of that channel. Precisions down-weight unreliable dims. Applied
   to the retrieval geometry (the incumbent best objective, r6_free_energy_*): scale the
   per-dim-mean squared-L2 by a detached, tempered, mean-1, banded precision so the same-verb
   ranking is decided by SIGNAL dims (small error to truth, large to a foil) not NOISE dims.
   A proper-scoring-rule reading (variogram / patch energy score, arXiv:2407.00650) confirms
   inverse-variance weights on squared-distance terms are a principled re-weighting, not a hack.

2. Mutual Proximity hubness correction (Schnitzer, Flexer, Schedl, Widmer; JMLR 13, 2012).
   The eval draws 63 same-verb foils from the POOL OF TRUE observations. In 768-d e5 space that
   pool is HUB-structured: a few central same-verb observations are near many predictions and
   repeatedly out-rank the true target, flipping top-1. r6_csls_hub_listwise already tried a
   ONE-SIDED additive column boost (CSLS/local-scaling: a candidate's k-NN density). Mutual
   Proximity is the SYMMETRIC secondary distance MP(i,j) that asks BOTH "is tgt_j close to
   pred_i relative to pred_i's OWN neighborhood?" (row) AND "is pred_i close to tgt_j relative
   to tgt_j's OWN neighborhood?" (column). This two-sided, per-endpoint standardization repairs
   the asymmetry hubs create -- a globally-central candidate looks near everyone (large z_col)
   and is demoted -- which the one-sided CSLS boost and anchor-proximity weighting cannot capture.

WHY THE FUSION SHOULD BEAT EITHER ALONE (epistasis argument)
------------------------------------------------------------
Precision fixes WHICH DIMENSIONS decide the ranking; mutual proximity fixes WHICH CANDIDATES
compete after that geometry is set. They act on orthogonal axes (dimensions vs candidates), so
combining them is not redundant. Crucially MP is computed INSIDE the precision-weighted space
(the tgt-tgt neighborhood distances use the same sqrt(Pi) scaling), so the hub estimate is
denoised too -- a hub in the raw space that is a noise-dim artefact is not treated as a hub. We
keep the metric-exact scaffold of the incumbents (per-dim-mean squared L2, row-only direction,
T=0.25, detached focal top-1 reweighting, small MSE anchor) and change ONLY the effective
distance fed to the softmax: eff = (1-MP)*d_prec + MP*d_mp.

CONTRACT / SAFETY
-----------------
* Pure function of (pred, tgt). Precisions and all MP scales (row/col means & stds) are DETACHED
  empirical-Bayes / secondary-distance estimates -- gradients flow only through the primary
  distances, never the preconditioner. No file/network/global state; no in-place edits of inputs.
  All ops are elementwise + two [n,n] matmuls (fast on MPS); same order as the incumbents.
* NaN-safe: MSE floored by eps inside the precision; dist2 clamp_min(0); neighborhood stds
  clamp_min; weights banded and re-normalized; n<2 -> MSE anchor only. n=2 uses the (n-1)=1
  neighborhood safely (std floored).
* Anti-collapse: for a CONSTANT prediction c, every row's primary distance vector -||c-tgt_j||^2
  is IDENTICAL across rows i. The row z-score (subtract each row's own mean) makes z_row's
  cross-column pattern identical across rows, z_col standardizes columns identically across rows,
  so the blended logits are the SAME across rows while the label for row i is column i. The mean
  focal-NLL is then a cross-entropy of a per-row one-hot against a FIXED column distribution,
  >= log(n) by Jensen and un-minimizable (equality needs all ||c-tgt_j||^2 equal, impossible for
  constant c vs varying tgt). The precision weighting and MP correction are detached and finite so
  they add no collapse escape, and MSE(const, varying tgt) is strictly positive. Collapse cannot win.
"""

import torch
import torch.nn.functional as F

NAME = "free_energy_mutual_proximity_listwise"
DESCRIPTION = (
    "Row-only (metric-direction) focal-listwise loss on per-dim-mean squared-L2 logits whose "
    "geometry is FREE-ENERGY PRECISION-WEIGHTED (detached mean-1 banded Pi_d = 1/MSE_d, Rao-"
    "Ballard/Friston), with the effective candidate distance further passed through a MUTUAL-"
    "PROXIMITY hubness correction (Schnitzer et al. JMLR 2012): a symmetric two-sided secondary "
    "distance that standardizes each pred->tgt distance by BOTH the query's and the candidate's "
    "own neighborhood distance distribution, demoting central same-verb hubs that flip held-out "
    "top-1. Blends raw precision geometry with the MP distance; detached focal + small MSE anchor."
)

_TEMP = 0.25       # on the effective per-dim-mean sqL2; gap d_true(~0.1-1) vs foil(~2) is O(1)
_GAMMA = 1.0       # focal focus on not-yet-#1 examples (the ones that decide top-1)
_ANCHOR = 0.05     # absolute-placement anchor (metric is shift-sensitive); also anti-collapse
_BETA = 0.5        # precision temper: w_d = Pi_d^BETA (0 -> uniform geometry; 1 -> full precision)
_EPS = 1e-2        # MSE floor inside precision (standardized dims have MSE ~0.1-2)
_WMIN, _WMAX = 0.25, 4.0   # band on mean-1 weights so no single dim dominates the geometry
_MP = 0.5          # blend: eff = (1-MP)*d_prec + MP*d_mp  (0 -> pure free-energy incumbent)
_SDFLOOR = 1e-2    # floor on neighborhood std for the mutual-proximity z-scores


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee (positive for constant pred vs varying tgt).
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # --- Free-energy precision: Pi_d = 1 / Var(error_d), estimated from the batch, DETACHED. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)                # [d] per-dim mean squared error
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)                  # [d] tempered precision
        w = w / w.mean().clamp_min(1e-12)                      # mean 1 (match eval scale)
        w = w.clamp(_WMIN, _WMAX)                              # band: no single dim dominates
        w = w / w.mean().clamp_min(1e-12)                      # re-normalize to mean 1
        sw = w.sqrt().unsqueeze(0)                             # [1, d]; scale both sides by sqrt(Pi)

    # Precision-weighted per-dim-mean squared L2 (metric geometry on sqrt(Pi)-scaled vectors).
    pw = pred * sw                                             # [n, d]
    tw = tgt * sw                                              # [n, d]
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)                 # [n, 1]
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)                 # [n, 1]
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())           # [n, n] pred_i vs tgt_j
    dist2 = dist2.clamp_min(0.0) / float(d)                    # per-dim mean (metric normalization)

    # --- Mutual-Proximity secondary distance (symmetric hubness correction), DETACHED scales. ---
    with torch.no_grad():
        # column (candidate) neighborhood scale: tgt_j's distance distribution to OTHER targets,
        # computed IN THE PRECISION-WEIGHTED SPACE so the hub estimate is denoised too.
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)   # [n, n] tgt-tgt
        eye = torch.eye(n, device=pred.device, dtype=torch.bool)
        tt0 = tt.masked_fill(eye, 0.0)
        denom = float(n - 1)
        col_mu = tt0.sum(dim=1) / denom                        # [n] mean dist of tgt_j to other tgts
        col_var = (tt0.pow(2).sum(dim=1) / denom - col_mu.pow(2)).clamp_min(0.0)
        col_sd = col_var.sqrt().clamp_min(_SDFLOOR)            # [n]
        # row (query) neighborhood scale: pred_i's distance distribution over all candidates.
        row_mu = dist2.mean(dim=1)                             # [n]
        row_sd = dist2.var(dim=1, unbiased=False).clamp_min(0.0).sqrt().clamp_min(_SDFLOOR)  # [n]

    # Two standardized views (detached scales -> grad flows through dist2 only):
    #   z_row: is tgt_j close to pred_i relative to pred_i's OWN neighborhood?
    #   z_col: is pred_i close to tgt_j relative to tgt_j's OWN neighborhood?
    z_row = (dist2 - row_mu[:, None]) / row_sd[:, None]        # [n, n]
    z_col = (dist2 - col_mu[None, :]) / col_sd[None, :]        # [n, n]
    d_mp = 0.5 * (z_row + z_col)                               # symmetric mutual-proximity distance

    eff = (1.0 - _MP) * dist2 + _MP * d_mp                     # blend raw precision + MP geometry

    # Metric direction only: pred_i ranked against candidate TRUE targets by (negative) eff distance.
    logits = -eff / _TEMP
    labels = torch.arange(n, device=pred.device)

    logp = F.log_softmax(logits, dim=1)                        # [n, n]
    nll = -logp.gather(1, labels[:, None]).squeeze(1)          # [n]

    # Detached focal weight: concentrate gradient on rows whose true target is not yet closest.
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)

    listwise = (focal * nll).mean()
    return listwise + _ANCHOR * mse_anchor

