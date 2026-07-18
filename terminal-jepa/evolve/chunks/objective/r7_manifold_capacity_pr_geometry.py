"""objective chunk: MANIFOLD-CAPACITY geometry loss (neural population geometry lens).

Cross-domain principle (neural population geometry / manifold-capacity theory --
Chung, Cohen, Lee, Sompolinsky, "Separability and geometry of object manifolds in
deep neural networks", PMC7005295):
  The linear separability of a set of class-manifolds is governed by their manifold
  classification capacity alpha_c ~ alpha_Ball(R_M, D_M), which DECREASES as each
  manifold's effective RADIUS R_M and effective DIMENSION D_M grow, and INCREASES as
  the manifold CENTERS spread apart. Dimension reduction is the primary driver of
  improved separability across successive network layers.

WHY this might raise the held-out content-verb margin:
  The eval is literally a manifold-separability test: rank the true next-obs against
  SAME-VERB foils by squared L2. The true target is retrieved iff the prediction lands
  inside its own target's tight neighborhood and clear of the neighboring same-verb
  targets. We translate capacity maximization directly into the loss on top of the
  metric-matched L2 listwise contrastive core (the champion geometry, unchanged):

    (a) RADIUS + DIMENSION of the prediction-ERROR manifold. Let R = pred - tgt be the
        per-step error cloud, centered. Its capacity-relevant geometry is summarized by
        the eigen-spectrum {lambda} of the error covariance. The participation-ratio
        (effective) dimension is D_eff = (sum lambda)^2 / sum lambda^2, and the squared
        radius is proportional to sum lambda. Crucially, for the CENTERED error matrix
        R (n x d) with Gram G = R R^T (n x n), the nonzero eigenvalues of the covariance
        equal those of G, so
            sum lambda   = tr(G)                 (total error variance)
            sum lambda^2 = ||G||_F^2             (spectral second moment)
            D_eff        = tr(G)^2 / ||G||_F^2   (participation ratio)
        -- computed with ONLY a matmul + trace + Frobenius norm, NO eigendecomposition
        (eigvalsh / svdvals are unimplemented on Apple MPS). Shrinking radius*(1+D_eff)
        collapses the error manifold to a small, LOW-dimensional cloud so it does not
        spill across neighboring same-verb targets -- the JEPA "abstract away
        unpredictable detail" bet expressed as geometry.

    (b) CENTER-SPREAD (VICReg-style variance HINGE, arXiv:2105.04906). Capacity rises
        when distinct manifolds' centers are far apart. We keep the predictions from
        under-spreading relative to the (detached) target spread with a one-sided hinge
        relu(1 - Var(pred)/Var(tgt)); this preserves separated centers AND is a second
        anti-collapse guard.

  This is capacity maximization -- distinct from per-dimension precision reweighting
  (free_energy) and per-feature decorrelation (Barlow/VICReg covariance): the driver is
  the participation-ratio DIMENSION of the joint error cloud, a batch-geometry quantity.

Contract / safety:
  * Pure function of (pred, tgt); no file/network/global state; no in-place edits of
    inputs (every derived tensor is a fresh allocation). Ops are matmul + elementwise
    only -> fast on MPS, no eig/svd.
  * NaN-safe: dist2 clamped >= 0; Frobenius norm floored by eps; spread ratio floored;
    n < 2 -> MSE anchor only.
  * Anti-collapse: the capacity signal is driven by the RESIDUAL pred - tgt, so a
    constant prediction yields residual ~ -tgt (large radius, high effective dimension,
    under-spread) -> the capacity term and the spread hinge are BOTH maximal; and a
    constant pred makes every row's contrastive logit vector identical, pinning the
    listwise NLL away from its minimum; and MSE(const, varying tgt) > 0. Collapse
    cannot minimize the loss along any of the three terms.
"""

import torch
import torch.nn.functional as F

NAME = "manifold_capacity_pr_geometry"
DESCRIPTION = (
    "Metric-matched L2 listwise contrastive core plus a NEURAL-POPULATION-GEOMETRY "
    "capacity regularizer: shrink the effective RADIUS and PARTICIPATION-RATIO DIMENSION "
    "D_eff=(tr G)^2/||G||_F^2 of the centered prediction-error cloud (Gram-only, no "
    "eigendecomposition, MPS-native), plus a VICReg-style center-spread hinge that keeps "
    "distinct predictions from under-spreading vs the detached target spread. Raises "
    "same-verb manifold separability = the retrieval capacity the eval scores. Focal "
    "top-1 reweighting + small MSE anchor for absolute placement and anti-collapse."
)

_TEMP = 0.25          # on the per-dim-mean squared L2; gap d_true(~0.1-1) vs foil(~2) is O(1)
_GAMMA = 1.0          # focal focus on not-yet-#1 rows (the ones that decide top-1)
_ANCHOR = 0.05        # absolute-placement anchor (metric is shift-sensitive); anti-collapse
_LAMBDA_CAP = 0.25    # weight on the radius*(1+D_eff) capacity term (error-manifold shrink)
_LAMBDA_SPREAD = 0.05 # weight on the center-spread hinge (keep manifold centers apart)
_EPS = 1e-6           # floors for Frobenius norm / spread ratio


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee (positive for const pred vs varying tgt).
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # ---- Metric-matched L2 listwise contrastive core (champion geometry, unchanged) ----
    ps = (pred * pred).sum(dim=1, keepdim=True)            # [n, 1]
    ts = (tgt * tgt).sum(dim=1, keepdim=True)              # [n, 1]
    dist2 = ps + ts.t() - 2.0 * (pred @ tgt.t())           # [n, n] squared L2 (eval geometry)
    dist2 = dist2.clamp_min(0.0) / float(d)                # per-dim mean (metric normalization)

    logits = -dist2 / _TEMP                                # closer target => higher logit
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)                    # [n, n]
    nll = -logp.gather(1, labels[:, None]).squeeze(1)      # [n]

    with torch.no_grad():                                  # detached focal top-1 reweighting
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    # ---- Manifold-capacity geometry: radius + participation-ratio dim of the ERROR cloud ----
    # Centered error matrix R = (pred - tgt) - mean_step(pred - tgt); Gram G = R R^T (n x n PSD).
    # Nonzero eigenvalues of the error covariance == eigenvalues of G, so we read the whole
    # spectrum summary off G via trace and Frobenius norm -- NO eig/svd (MPS-native).
    resid = pred - tgt
    resid = resid - resid.mean(dim=0, keepdim=True)        # fresh tensor; center the cloud
    gram = resid @ resid.t()                               # [n, n] PSD Gram
    tr = torch.diagonal(gram).sum()                        # sum lambda  (total error variance)
    fro2 = (gram * gram).sum()                             # sum lambda^2 (spectral 2nd moment)

    radius2 = tr / (float(n - 1) * float(d))               # mean per-dim residual variance (scale)
    pr_dim = (tr * tr) / (fro2 + _EPS)                     # (sum l)^2 / sum l^2 = effective dim
    pr_frac = pr_dim / float(min(n, d))                    # in (0, 1]; low => concentrated error
    # small, LOW-dimensional error manifold => low capacity penalty (more separable same-verb).
    cap_penalty = radius2 * (1.0 + pr_frac)

    # ---- Center-spread hinge (VICReg-style): keep distinct predictions from clumping ----
    # Preserves separated manifold CENTERS (capacity numerator) and is a 2nd anti-collapse guard.
    with torch.no_grad():
        tgt_c = tgt - tgt.mean(dim=0, keepdim=True)
        tgt_spread = (tgt_c * tgt_c).sum() / (float(n - 1) * float(d)) + _EPS  # detached scale
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    pred_spread = (pred_c * pred_c).sum() / (float(n - 1) * float(d))
    spread_pen = F.relu(1.0 - pred_spread / tgt_spread)    # only fires when under-spread

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_CAP * cap_penalty + _LAMBDA_SPREAD * spread_pen

