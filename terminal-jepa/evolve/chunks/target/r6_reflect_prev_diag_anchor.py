"""target chunk: pure per-example Householder reflection mapping the previous-observation
direction onto a FIXED shared diagonal anchor (the normalized ones vector).

Motivation. The retrieval eval is fixed: to_obs must invert back to the raw next-obs z_obs, and
foils are ranked by squared L2 in that raw space. So a target transform can only change what is
learned by reshaping the TRAINING objective's geometry. Two families were already probed and lost:
delta / partial-residual (z_obs - alpha*z_prev) are per-example TRANSLATIONS, and any fixed
diagonal / coordinate-wise reweight is self-anti-aligned (training emphasizes a coordinate as
g'(z)^2 while eval amplifies its error as 1/g' -> opposite emphasis). The one family that keeps the
true-pair distance EXACTLY eval-aligned yet still reshapes the in-batch NEGATIVE geometry is a
per-example ISOMETRY.

Design. Build a Householder reflection H(z_prev) that maps the unit previous-observation direction
u_prev = z_prev/||z_prev|| onto a fixed unit anchor a = ones/sqrt(d), shared across every step and
every system. H is orthogonal and self-inverse, so:
    make_target(z_obs, z_prev) = H(z_prev) @ z_obs
    to_obs(pred,  z_prev)      = H(z_prev) @ pred        # same H, self-inverse -> exact recon
H fixes the (d-1)-dim subspace orthogonal to v = u_prev - a and only rotates the 2D plane
span{u_prev, a}. Hence the innovation / content subspace that discriminates same-verb foils is
preserved, while the along-previous "same banner / same cwd / same-system" component of every step
is canonicalized onto ONE shared direction a -> in the contrastive negatives it becomes a
common-mode offset that partially cancels between candidates, sharpening the innovation-driven
(content) separation the content-verb retrieval needs.

Unlike the concurrent Householder-prev target, this uses NO translation term (no -||z_prev||*e0
shift, which would reintroduce the failed delta-family), reflects onto the DIAGONAL anchor rather
than a single coordinate axis (spreading the shared component across dims instead of dumping it into
coord 0), and is a strict norm-preserving isometry.

Safety. No learned params, no train stats, no state; pure functions of (z_obs, z_prev). z_prev is
causally available (zeros at t=0 -> gate to identity). Being an isometry it leaves the objective's
anti-collapse property intact (a constant prediction still cannot match varied targets), and
to_obs(zeros, z_prev) == zeros so the predict-mean calibration guard is unaffected. NaN-safe via
clamp_min on both the norm and the reflection denominator, with an identity gate when z_prev is ~0
or already aligned with the anchor (v ~ 0).
"""

import torch

NAME = "reflect_prev_to_diag_anchor"
DESCRIPTION = (
    "Pure per-example Householder reflection mapping the previous-obs direction onto a fixed "
    "shared diagonal anchor (normalized ones); canonicalizes the history-shared subspace into a "
    "common reference frame with NO translation term. Self-inverse -> exact reconstruction; "
    "strict isometry so positive-pair eval geometry is preserved while negatives are reshaped."
)

_EPS = 1e-12


def _reflect(x, z_prev):
    """Apply H(z_prev) to x, where H reflects unit(z_prev) onto the unit diagonal anchor
    a = ones/sqrt(d). H is orthogonal and self-inverse, so this same function both applies and
    inverts the target transform. Identity-gated where z_prev is ~0 (t=0) or already aligned
    with the anchor (v ~ 0)."""
    d = z_prev.shape[-1]
    norm = z_prev.norm(dim=-1, keepdim=True)                       # [n,1]
    u = z_prev / norm.clamp_min(_EPS)                             # unit prev direction
    a = torch.full_like(z_prev, 1.0 / (float(d) ** 0.5))         # unit diagonal anchor
    v = u - a                                                     # Householder axis (maps u<->a)
    v_norm2 = (v * v).sum(dim=-1, keepdim=True)                   # [n,1]
    coef = 2.0 * (x * v).sum(dim=-1, keepdim=True) / v_norm2.clamp_min(_EPS)
    reflected = x - coef * v
    active = (norm > _EPS) & (v_norm2 > _EPS)                     # else identity (leak-free)
    return torch.where(active, reflected, x)


def make_target(z_obs, z_prev):
    # What the model is trained to predict: z_obs in the prev-canonicalized reference frame.
    return _reflect(z_obs, z_prev)


def to_obs(pred, z_prev):
    # Reconstruct the predicted next-obs for retrieval: same reflection inverts exactly.
    return _reflect(pred, z_prev)

