"""LEARNED target chunk: an exactly-ORTHOGONAL learned rotation Q of the next-obs embedding,
built as a composition of L Givens-rotation layers over FIXED random coordinate pairings.

  make_target(z_obs, z_prev) = Q @ z_obs      # trained jointly with the predictor
  to_obs(pred,  z_prev)      = Q^T @ pred      # Q orthogonal -> exact inverse; eval in obs space

WHY THIS IS NOT DIAGONAL-REDUNDANT WITH PRECISION-WEIGHTING (the flat-retry problem).
  The free-energy objective precision-weights per-dim errors by Pi_d = 1/MSE_d -- a DIAGONAL
  preconditioner in the fixed obs basis. A diagonal TARGET gate (the prior tgt_space_diag_gate,
  -0.002) is the SAME kind of operator, so the two commute and are redundant -> flat. A learned
  ROTATION is exactly the operator a diagonal weighting cannot express. Training in the rotated
  space makes the effective error metric  Q^T diag(Pi) Q  -- a FULL (non-diagonal) precision
  matrix, i.e. the ZCA-whitening form D*Lambda^-1/2*D^T (Ermolov et al.; "Whitening Consistently
  Improves SSL" arXiv:2408.07519). Shell observations have strongly CORRELATED error dimensions
  (shared banner / cwd / system boilerplate co-varies across many dims); a diagonal precision can
  only down-weight individual axes, never a correlated noise MODE. Q can rotate that mode onto a
  few axes where the objective's precision then suppresses it, sharpening same-verb separation on
  the decorrelated content directions. (BRo-JEPA arXiv:2606.01372: an orthogonal latent operator
  captures inter-dimension coupling a diagonal reweight cannot, while preserving norms/angles.)

PARAMETERIZATION (exactly orthogonal, identity at init, cheap).
  A single Givens layer rotates disjoint coordinate PAIRS by learnable angles theta (init 0):
      (x_i, x_j) -> (cos t * x_i - sin t * x_j,  sin t * x_i + cos t * x_j).
  This is an exact rotation (norm-preserving) for any theta, and its inverse is the same layer
  with theta negated. We compose L such layers over L different FIXED random pairings; the
  composition is dense (a coordinate spreads over 2^L dims) yet still exactly orthogonal, and its
  inverse applies the layers in reverse order with negated angles. All angles init 0 -> Q == I
  exactly at init (this genome starts bit-equal to the identity-target champion and only departs
  if the rotated basis lowers the precision-weighted contrastive loss). Cost is O(L*n*D), no
  matmul over a DxD matrix, no matrix inverse -- MPS-friendly.

HONESTY / SAFETY.
  * LEARNED extension: LEARNED=True; make(D)->nn.Module with make_target/to_obs/reg. Params
    (the angle vectors) are registered on the net and trained by the genome's optimizer; the
    pairings are FIXED buffers (no train stats, no per-batch state, no data leakage). z_prev is
    unused (a pure global rotation) -> strictly causal, and to_obs is the exact analytic inverse.
  * Anti-collapse INTACT: Q is a bijective isometry, so a constant prediction still cannot match
    varied rotated targets, and reconstruction re-amplifies any hidden error -- a degenerate
    rotation cannot win the retrieval eval, which ranks true z_obs vs same-verb foils by squared
    L2 in the FIXED obs space (Q orthogonal => that distance is preserved exactly for the true
    pair). to_obs(zeros, z_prev) == zeros, so the predict-mean calibration guard is unaffected.
  * reg() is a tiny angle-magnitude L2 that (a) keeps the identity-at-init prior weakly, and
    (b) discourages the rotation from drifting for no benefit; lam is small so it does not pin Q.
  * NaN-safe: cos/sin are bounded; index_select/index_copy build new tensors (no in-place autograd
    hazard). No divisions, no norms in the transform path.
"""

import torch
import torch.nn as nn

NAME = "r7_learned_givens_rotation_decorrelate"
DESCRIPTION = (
    "Learned exactly-orthogonal target rotation Q = composition of L Givens layers over fixed "
    "random coordinate pairings (angles init 0 -> identity at init). make_target = Q z_obs, "
    "to_obs = Q^T pred (exact inverse). Supplies the OFF-DIAGONAL basis rotation that the "
    "free-energy objective's diagonal precision weighting cannot represent, composing into a full "
    "precision metric (ZCA-whitening structure) -- escapes the diagonal-redundancy that flattened "
    "the diag-gate retry. Isometry: positive-pair eval geometry preserved; anti-collapse intact."
)

LEARNED = True

_L = 6            # number of Givens layers; coordinate spreads over 2^L=64 dims -> dense rotation
_REG_LAM = 1e-4   # weak angle-magnitude prior (keeps identity-at-init; does not pin the rotation)
_SEED = 20250717  # fixes the random pairings (a structural choice, not a train statistic)


class GivensRotationTarget(nn.Module):
    """Exactly-orthogonal learned rotation of the target, as a product of L Givens layers over
    fixed random coordinate pairings. Orthogonal for ANY angles -> to_obs is the exact inverse."""

    def __init__(self, dim, num_layers=_L):
        super().__init__()
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        r = self.dim // 2                       # number of disjoint pairs per layer
        self.r = r
        g = torch.Generator().manual_seed(_SEED)
        # Fixed (non-learned) random pairings; registered as buffers so they move with .to(device)
        # and are part of state, but carry NO gradient and NO train statistics.
        for l in range(self.num_layers):
            perm = torch.randperm(self.dim, generator=g)
            self.register_buffer(f"i_idx_{l}", perm[:r].clone(), persistent=True)
            self.register_buffer(f"j_idx_{l}", perm[r:2 * r].clone(), persistent=True)
        # Learnable angles, init EXACTLY 0 -> every layer is the identity at init -> Q == I.
        self.theta = nn.ParameterList(
            [nn.Parameter(torch.zeros(r)) for _ in range(self.num_layers)]
        )

    def _layer(self, x, l, inv=False):
        """Apply Givens layer l (or its inverse) to x [..., dim]. Builds new tensors (autograd-safe)."""
        i_idx = getattr(self, f"i_idx_{l}")
        j_idx = getattr(self, f"j_idx_{l}")
        t = self.theta[l]
        c = torch.cos(t)
        s = torch.sin(t)
        if inv:
            s = -s                              # inverse rotation = negate the angle
        xi = x.index_select(-1, i_idx)
        xj = x.index_select(-1, j_idx)
        ni = c * xi - s * xj
        nj = s * xi + c * xj
        out = x.index_copy(-1, i_idx, ni)       # returns a new tensor (out-of-place)
        out = out.index_copy(-1, j_idx, nj)
        return out

    def _forward_rot(self, x):
        for l in range(self.num_layers):
            x = self._layer(x, l, inv=False)
        return x

    def _inverse_rot(self, x):
        for l in range(self.num_layers - 1, -1, -1):
            x = self._layer(x, l, inv=True)
        return x

    def make_target(self, z_obs, z_prev):
        # What the model is trained to predict: z_obs rotated into the learned decorrelating basis.
        return self._forward_rot(z_obs)

    def to_obs(self, pred, z_prev):
        # Reconstruct the predicted next-obs for retrieval: exact inverse rotation Q^T.
        return self._inverse_rot(pred)

    def reg(self):
        # Weak angle-magnitude prior: keeps the identity-at-init bias and discourages drift for no
        # benefit. Small lam so it never pins Q away from a useful rotation.
        s = 0.0
        for t in self.theta:
            s = s + (t * t).sum()
        return _REG_LAM * s


def make(dim):
    return GivensRotationTarget(dim)

