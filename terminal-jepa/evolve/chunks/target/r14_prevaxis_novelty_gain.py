"""target chunk (R14, changed-context retry of the residual/delta family): a per-sample
ANISOTROPIC NOVELTY GAIN around the z_prev axis — an exactly invertible rank-1 stretch of
the next-obs embedding that amplifies the component ORTHOGONAL to the previous observation
(the "novel/surprising" content) and leaves the component ALONG z_prev (the copy-predictable
shared structure) at unit gain.

CROSS-DOMAIN LENS — PREDICTIVE CODING / ERROR AMPLIFICATION (Rao & Ballard 1999; Friston's
free-energy). Cortex does not transmit the raw sensory vector; it transmits the PREDICTION
ERROR — the part of the input not explained by the top-down prediction — and it transmits that
error with HIGH GAIN (precision-weighting) while suppressing the already-predicted component.
On a shell trace, z_prev is the best single-vector top-down prediction of the next observation
(the copy_prev baseline is strong precisely because the next obs shares the previous banner /
cwd / near-duplicate file body). The projection of z_obs onto the z_prev direction is that
copy-predictable shared mode; the ORTHOGONAL residual is the genuinely new content — for cat on
a system-variant /etc/os-release, exactly the distro-specific lines that decide the retrieval
margin; for tail, the last-K-lines window that differs from the previous view. Predictive coding
says: put the representational GAIN on that residual, not on the shared mode.

THE MAP (exactly invertible for ANY z_prev, any gain g).
  u = z_prev / ||z_prev||                       (unit prev-axis; causal — z_prev is the obs token
                                                 immediately before the command)
  par(z)  = (z . u) u                            (component along the shared/copy-predictable axis)
  perp(z) = z - par(z)                           (the novel/surprise residual)
  make_target:  T(z_obs) = par(z_obs) + g * perp(z_obs) = z_obs + (g-1) * perp(z_obs)
  to_obs:       Z(pred)  = par(pred)  + (1/g) * perp(pred) = pred - (1 - 1/g) * perp(pred)
T is the symmetric linear operator with eigenvalue 1 along u and g on the (D-1) directions
orthogonal to u; its inverse has eigenvalues 1 and 1/g, so Z(T(z)) == z EXACTLY for any g, any u
(rank-1 diagonalization in the {u, u^perp} basis). At the first step z_prev = 0 (||z_prev|| ~ 0):
u is undefined, so BOTH functions fall back to the identity (perp := 0), and because make_target
and to_obs branch on the SAME z_prev the inverse stays exact there too.

WHY THIS SHOULD RAISE THE CONTENT-VERB MARGIN (and why the v2 context can flip the delta family).
  * The model is trained to predict IN the gained space. A prediction error e decomposes into a
    parallel part e_par and an orthogonal part e_perp; in target space its squared L2 cost is
    ||e_par||^2 + g^2 ||e_perp||^2 — the objective's gradient is weighted g^2 more on the NOVEL
    subspace than on the shared mode. The champion antiretrieval-ring objective already spends
    its contrastive budget on retrieval-CONFUSABLE target pairs; those pairs (the same config
    file across two systems, a file vs its own earlier body) differ almost entirely in the
    z_prev-orthogonal residual — so the two mechanisms COMPOSE: the ring picks the confusable
    negatives, the novelty gain sharpens the axis along which they are confusable. This is the
    exact residual the delta/partial-residual family targeted, but here the residual is
    AMPLIFIED (weighted up) rather than SUBTRACTED (removed) — subtraction only re-centers the
    target and lowers variance uniformly; the anisotropic gain concentrates loss pressure on the
    discriminative directions while leaving the easy shared mode alone. Fixed delta failed on the
    v1 mse/contrastive surface; on the v2 confusability-ring surface the amplified-residual form
    is aligned with what the objective is already trying to separate.
  * The inverse is CONTRACTIVE on the novel subspace (gain 1/g < 1): any residual novel-direction
    prediction error is DAMPED on reconstruction, never amplified — like radial companding's
    contractive expand, there is no reconstruction blow-up tail, and the shared mode (gain 1) is
    reconstructed as-is. The eval always inverts the TRUE pair exactly and ranks in the fixed obs
    space, so the same-verb angular structure the margin lives on is untouched for the truth; only
    the model's training-time capacity allocation is reshaped.

SAFETY / CONTRACT.
  * PURE functions of (z_obs/pred, z_prev) only — no learned params, no train stats, no state.
    g is a fixed geometry constant (like a temperature); it is NOT trained, so it cannot be pushed
    toward the dishonest g<1 direction (which would shrink the novel content to make targets easier
    to fit). Strictly causal: uses only z_prev.
  * Exact reconstruction by construction (rank-1 eigen-inverse) for any batch, any g > 0.
  * Anti-collapse INTACT: T is an invertible per-sample linear map, so a constant/zero prediction
    still cannot match varied gained targets; to_obs(zeros, z_prev) == zeros (par/perp of 0 are 0),
    so the predict-mean calibration guard (computed target-independent in raw obs space) is
    unaffected.
  * NaN-safe: the only division is z_prev / ||z_prev|| with the norm clamped and gated by an
    active-row mask (inactive rows -> identity, no division used); g and 1/g are finite constants;
    no logs/exps of data values. index-free, broadcast-only -> autograd-safe (make_target is on the
    graph via z_obs; the u/mask/perp construction from z_prev is a constant w.r.t. the prediction).
"""

import torch

NAME = "r14_prevaxis_novelty_gain"
DESCRIPTION = (
    "Predictive-coding novelty gain: exactly invertible rank-1 anisotropic stretch of the next-obs "
    "embedding around the z_prev axis — make_target amplifies the z_prev-ORTHOGONAL residual (the "
    "novel/surprise content) by g and keeps the z_prev-parallel copy-predictable mode at unit gain; "
    "to_obs applies the exact eigen-inverse (gain 1/g on the residual, contractive -> no blow-up). "
    "Concentrates the objective's loss gradient g^2 on the discriminative subspace that the "
    "antiretrieval-ring confusable pairs (cat system-variant bodies, tail windows) differ along. "
    "Fixed geometry constant (no learned params); identity fallback at step 0 (z_prev = 0)."
)

_G = 2.0        # novelty gain on the z_prev-orthogonal residual (eigenvalue g; inverse 1/g).
_INV_G = 1.0 / _G
_EPS = 1e-8     # norm floor / active-row threshold for the prev axis.


def _perp(x, z_prev):
    """Component of x orthogonal to the z_prev direction, per row. Rows whose z_prev has ~0 norm
    (step 0) contribute zero perp -> the caller's map degenerates to the identity there."""
    n = z_prev.norm(dim=-1, keepdim=True)                 # [., 1]
    active = (n > _EPS).to(x.dtype)                       # 1.0 on real prev rows, 0.0 at step 0
    u = z_prev / n.clamp_min(_EPS)                        # unit prev axis (safe divide)
    proj = (x * u).sum(dim=-1, keepdim=True) * u          # (x . u) u  = parallel component
    perp = (x - proj) * active                            # novel residual; zeroed on step-0 rows
    return perp


def make_target(z_obs, z_prev):
    # Amplify the novel (z_prev-orthogonal) residual by g; leave the shared axis at unit gain.
    # T(z) = z + (g - 1) * perp(z). Step-0 rows: perp = 0 -> T = z_obs (identity).
    return z_obs + (_G - 1.0) * _perp(z_obs, z_prev)


def to_obs(pred, z_prev):
    # Exact eigen-inverse: contract the novel residual by 1/g. Z(y) = y - (1 - 1/g) * perp(y).
    # Step-0 rows: perp = 0 -> Z = pred (identity). Contractive on the residual -> no blow-up.
    return pred - (1.0 - _INV_G) * _perp(pred, z_prev)