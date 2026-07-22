"""objective chunk: CROSS-SYSTEM TEMPLATE-ERROR WHITENING — champion free-energy
precision-weighted focal listwise L2 with ring-reweighted negatives, whose anti-retrieval
push is replaced by a two-sided REGRESSION penalty that drives the prediction ERROR to
zero ALONG the pooled cross-system template->truth axis (a Kalman-innovation / common-mode
translation targeting cat's system-variant file bodies).

THE CAT STALL, RE-DIAGNOSED (the standing R14 diagnosis). cat's margin is frozen across two
champion generations while every other content verb rose. cat has the highest retrieve-by-cmd
baseline (.531): the corpus already returns the correct file for every path whose content is
IDENTICAL across systems. The entire REMAINING cat margin therefore lives in SYSTEM-VARIANT
file bodies — same path (/etc/os-release, distro configs), DIFFERENT content per system. For
those, retrieve-by-cmd returns a WRONG-BUT-CLOSE sibling (the same file from another distro),
and the world model can only win by predicting the SPECIFIC variant THIS system has.

WHY THE FAILURE IS SHRINKAGE, NOT CORRELATED ERROR (the subtlety that kills the naive fix).
The tempting \"cross-system consistency\" move is to decorrelate the prediction ERRORS across
systems that share a path (\"don't make the same template-error everywhere\"). But work it out:
write t_i = mu_C + delta_i, where mu_C is the shared cross-system template (centroid of the
same-path family) and delta_i is system i's specific deviation. A predictor that regresses to
the template outputs pred_i ~ mu_C + rho*delta_i with shrinkage rho<1, so its error is
e_i = pred_i - t_i = -(1-rho)*delta_i — PROPORTIONAL AND ANTI-ALIGNED TO THAT SYSTEM'S OWN
DEVIATION. Two different systems' deviations delta_i, delta_j point in DIFFERENT
system-specific directions, so the errors are already spread (nearly uncorrelated) — literal
pairwise error-decorrelation attacks a failure that is not there. The real signature of the
failure is per-example: the error e_i lies ALONG the axis u_i = delta_i/||delta_i|| (the
template->truth direction), because the model captures only a fraction rho of the deviation.

THE MECHANISM — WHITEN THE INNOVATION ALONG THE INFORMATIVE AXIS. In optimal filtering the
innovation (one-step prediction error) of a correctly specified model is WHITE and, crucially,
carries NO remaining projection onto the predictable signal directions; a residual correlation
between the innovation and a known regressor is the textbook diagnostic of an under-fit /
over-shrunk filter (Kalman 1960; innovations whiteness test, Mehra 1970, IEEE TAC; Kailath's
innovations approach). We translate exactly that test into the loss. For each example we pool
its same-path family via the champion's band-pass CONFUSABILITY RING on target-target
distances (near-identical targets excluded as the identical-across-systems FALSE negatives
retrieve-by-cmd already gets right; far targets excluded as easy), estimate the SHARED template
by pooling ACROSS SYSTEMS
    c_i   = sum_j q_ij t_j          (the cross-system template centroid; q from the ring)
    u_i   = (t_i - c_i)/||t_i - c_i||           (the template->truth axis, DETACHED)
and drive the prediction's position along that axis exactly to truth:
    s_i   = (pred_i . u_i - c_i . u_i) / ||t_i - c_i||     (0 at template, 1 at truth)
    L_tw  = gate_i * f_i * (s_i - 1)^2
This is the eval's own within-family decision variable — truth sits at s=1, the confusable
siblings cluster near s=0 (the template), and the squared-L2 perpendicular-bisector boundary is
at s=0.5. A prediction that COPIES the template (pure shrinkage) gives s=0 and the maximal
penalty; a prediction that resolves the exact variant gives s=1 and ~0 penalty.

WHY TWO-SIDED SQUARED, NOT A ONE-SIDED HINGE. A boundary hinge (be past s=0.5) only asks the
prediction to WIN the ranking; the two-sided regression penalty asks it to land ON the specific
variant, which additionally beats the true target's own cross-system scatter and calibrates
absolute placement along the axis. It is the \"innovation must be zero along the informative
direction\" condition, not merely \"innovation on the right side\". A detached focus weight
f_i ~ ||t_i - c_i||^2 (mean-1 over active rows, banded) concentrates the term on the LARGE-
deviation families — exactly cat's system-variant bodies, where the template->truth gap is
biggest and the residual margin lives — and leaves tight (near-identical) families alone.

SHARING STATISTICAL STRENGTH. c_i is a POOLED estimate of the shared template over every system
that shares the path in the batch, so a system with few in-batch neighbors borrows the template
from its siblings (an empirical-Bayes / shrinkage estimator used here as the REFERENCE the model
must beat, not as the model's own output). The sysblock batcher makes this bite: it packs a
trajectory's cmd positions into the flattened batch, and the same-system block + uniform
remainder supply the same-path cross-system observations that populate the ring.

The retrieval backbone is UNCHANGED from the champion (free-energy precision-weighted focal
listwise L2 with ring importance-weighting of in-batch negatives, which lifts the other verbs);
this term REPLACES the champion's isotropic repulsion hinge, sharpening the anti-template push
onto the cat system-variant residual instead of spending it on decision-irrelevant dimensions.

Contract / safety:
  * Pure function of (pred, tgt). Ring, q, gate, c_i, u_i, ||delta||, focus f all DETACHED; the
    only grad path in L_tw is (pred_i . u_i), linear in pred. No state, no in-place edits.
  * NaN-safe: eps floors in precision, lambda, ring mass, ||delta||, focus normalization;
    dist2/tt clamp_min(0); n<2 -> MSE anchor only.
  * Anti-collapse: a constant pred cannot minimize the listwise NLL (log a is target-only and
    the diagonal cannot be favored) NOR L_tw (u_i, c_i.u_i, ||delta|| are target-only and point
    in many directions, so no single pred vector puts every s_i at 1 -> the squared shortfall is
    strictly positive) NOR the MSE anchor (constant vs varying tgt > 0). Collapse cannot minimize
    the loss.
"""

import torch
import torch.nn.functional as F

NAME = "template_error_whitening"
DESCRIPTION = (
    "Champion free-energy precision-weighted focal listwise L2 contrastive with ring-reweighted "
    "in-batch negatives, whose anti-retrieval push is replaced by a Kalman-innovation TEMPLATE-"
    "ERROR WHITENING term: for each example it pools the shared cross-system template c_i from "
    "its same-path confusability family, forms the template->truth axis u_i, and drives the "
    "prediction's normalized position s_i along u_i to exactly 1 via a two-sided squared "
    "regression penalty (s_i-1)^2 — whitening the innovation along the informative direction so "
    "the model resolves the SPECIFIC system-variant file body instead of regressing to the "
    "cross-distro template, focused (detached) on the large-deviation families where cat's "
    "system-variant margin lives."
)

# ---- champion constants (unchanged) ----
_TEMP = 0.25       # softmax temperature on mean-1-normalized per-dim-mean sqL2
_GAMMA = 1.0       # focal focus on not-yet-#1 rows
_ANCHOR = 0.05     # small MSE anchor (anti-collapse + absolute placement)
_BETA = 0.5        # precision temper Pi_d^BETA
_EPS = 1e-2        # MSE floor inside the precision
_WMIN, _WMAX = 0.25, 4.0

# ---- ring (same-path confusable family) constants ----
_KAPPA = 4.0       # peak extra negative mass on a maximally confusable pair
_DELTA = 0.05      # per-dim sq-dist below which two targets are the SAME answer (false neg -> ring~0)
_LAM_FRAC = 0.5    # confusability kernel scale = _LAM_FRAC * mean off-diag target-target dist

# ---- template-error whitening (cross-system anti-shrinkage) constants ----
_LAMBDA_TW = 0.1   # weight of the template-whitening term (small; the listwise does the bulk)
_GEPS = 1e-3       # ring-mass floor inside the row gate
_DNEPS = 1e-4      # floor on the template->truth axis norm (guards near-duplicate centroids)
_FMAX = 4.0        # cap on the detached large-deviation focus weight (mean-1 over active rows)


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee.
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

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
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())           # [n, n]
    dist2 = dist2.clamp_min(0.0) / float(d)

    # --- Confusability ring on TARGET-TARGET distances (all DETACHED). ---
    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)  # [n, n]
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        confus = torch.exp(-tt / lam)                          # 1 at identical -> 0 far
        dupmask = 1.0 - torch.exp(-tt / _DELTA)                # ~0 identical -> ~1 distinct
        ring = (confus * dupmask).masked_fill(eye, 0.0)        # [n, n] band-pass, zero diag

        # Importance weights for the listwise negatives (off-diag mean 1 per row).
        a_raw = 1.0 + _KAPPA * ring
        row_mean = a_raw.masked_fill(eye, 0.0).sum(dim=1, keepdim=True) / (n - 1)
        a = (a_raw / row_mean.clamp_min(1e-6)).masked_fill(eye, 1.0)
        log_a = a.clamp_min(1e-6).log()                        # [n, n]

        # Same-path family distribution + smooth row gate for the whitening term.
        mass = ring.sum(dim=1)                                 # [n]
        q = ring / mass.clamp_min(1e-6).unsqueeze(1)           # [n, n], rows ~1 (or ~0)
        gate = mass / (mass + _GEPS)                           # [n] in [0,1): ~0 if no family

        # POOLED cross-system TEMPLATE centroid and template->truth axis (target-only, detached).
        c = q @ tgt                                            # [n, d] shared template (pooled)
        delta = tgt - c                                        # [n, d] axis: template -> truth
        dn = delta.norm(dim=1).clamp_min(_DNEPS)               # [n]
        u = delta / dn.unsqueeze(1)                            # [n, d] unit template->truth axis
        c_proj = (c * u).sum(dim=1)                            # [n] projection of the template

        # Large-deviation focus: up-weight the system-variant families (big template->truth gap),
        # mean-1 over ACTIVE (gated) rows, banded. Detached.
        dn2 = dn * dn                                          # [n]
        gsum = gate.sum().clamp_min(1e-6)
        mean_dn2 = (gate * dn2).sum() / gsum
        focus = (dn2 / mean_dn2.clamp_min(1e-6)).clamp(0.0, _FMAX)  # [n], mean~1 on active rows

    # --- Champion listwise term with importance-weighted negatives. ---
    logits = -dist2 / _TEMP + log_a
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    # --- Template-error whitening: drive s_i (position along template->truth axis) to 1. ---
    # s_i = 0 at the pooled template, 1 at truth; ONLY grad path is (pred . u), linear in pred.
    p_proj = (pred * u).sum(dim=1)                             # [n], carries grad
    s = (p_proj - c_proj) / dn                                 # [n] scale-free position along axis
    tw_term = (gate * focus * (s - 1.0) ** 2).mean()

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_TW * tw_term
