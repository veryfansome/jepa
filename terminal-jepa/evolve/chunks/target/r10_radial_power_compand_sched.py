"""LEARNED target chunk (R10-CALIBRATION): scheduled radial power COMPANDING of the
next-obs embedding around the standardization shell — a norm-standardizing target space
whose exact inverse squashes the contrastive norm inflation back onto the data manifold.

THE DEFECT THIS ATTACKS (Stage-2 review, r10 prereg): the champion's contrastive
row-softmax over -dist^2/tau is invariant to each prediction's own squared norm (a
row-constant), so the only norm pressure is the weak 0.05 MSE anchor; the trained trunk
settles at ||pred||^2/||true||^2 ~ 4.7 — rankable but off-manifold, killing latent-MPC.

MECHANISM — an exactly invertible pure-radial bijection with pivot mu = sqrt(D):
  compand (make_target):  T(z) = z * (r/mu)^(s-1),   r = ||z||   =>  ||T(z)|| = mu*(r/mu)^s
  expand  (to_obs):       Z(y) = y * (R/mu)^(1/s-1), R = ||y||   =>  ||Z(y)|| = mu*(R/mu)^(1/s)
  (log-radius is scaled by s around log mu; direction untouched; Z(T(z)) == z exactly.)
Per-dim standardization makes E||z||^2 = D on train BY CONSTRUCTION, so mu = sqrt(D) is
the data shell, not a guess (measured geo-mean radius 27.28 vs sqrt(768) = 27.71, and
std(log r) ~ 0.125 => the warp stays moderate on real radii).

WHY THIS CALIBRATES: the trunk predicts IN the companded space, where the loss geometry
(contrastive + anchor) reaches whatever radial inflation k* it reaches; the eval maps the
prediction back with the exact inverse, which takes k* -> k*^(1/s). The champion's
measured k*^2 ~ 4.7 becomes ~ 4.7^(1/s) in obs space (s=5 => ~1.36, inside the [0.8,1.5]
gate; the residual target-space inflation grows mildly with s, so s=6 lands ~1.46).
Equivalently, the inverse-map Jacobian makes an obs-space radial (norm) error cost ~s^2 x
more than an angular error of the same obs size — norm calibration is no longer a gradient
orphan. Toy verification (linear predictor, the ACTUAL champion loss, 768-d clustered
targets with the measured log-normal radii): identity -> norm_ratio 4.78 (the real defect
is 4.7 — the toy reproduces it); s=4 -> 1.66, s=5 -> 1.52, s=6 -> 1.46, with in-batch
obs-space top-1 UNCHANGED (0.971 at every s) and matched sqL2 dropping from 2x rand-pair
to 0.76x. The inverse is contractive (exponent 1/s), so any stray prediction radius is
damped toward the shell, never amplified — no reconstruction blow-up tail.

WHY THE RANKING SURVIVES: the map is a smooth monotone reparametrization of the radius
only — every vector keeps its exact direction, and on the data shell (r ~ mu) the
tangential Jacobian is ~1, so the same-verb angular structure the margin lives on is
untouched. Training ranks candidates contrastively in the SAME companded space the trunk
predicts in, so train and eval stay metric-matched through the bijection. Radial contrast
between foils is AMPLIFIED s-fold in the target space (norm carries content size), which
the uniform-L2 eval rewards once the radius is calibrated.

SCHEDULE (identity at init, adiabatic warp): s(t) ramps 1 -> S_FINAL over the first
RAMP_STEPS training calls (a non-persistent step buffer; s=1 is the exact identity), so
the genome starts bit-close to the identity-target champion and deforms the geometry
adiabatically; the ramp ends well inside both proxy (1000) and full (4000) budgets, after
which the instance geometry EQUALS the fixed module-level geometry below.

FROZEN-INSTRUMENT COMPATIBILITY (load-bearing): calib_bench/plan_env strict-load the
checkpoint into a bare arch net and call module-level to_obs. Therefore (a) the learned
module keeps its ONLY state (the ramp counter) in a persistent=False buffer -> its
state_dict is EMPTY, so `net.load_state_dict(ck["state_dict"])` on a bare arch net still
works; (b) this module ALSO exposes pure module-level make_target/to_obs at the final
geometry s = S_FINAL, which is exactly the trained module's post-ramp state -> the frozen
eval reconstructs in precisely the space the trunk converged in.

SAFETY: pure radial map, z_prev unused -> strictly causal, no leakage. NaN-safe: radii
clamped, zero vectors map to zero (0 * (0/eps) = 0), no log/exp of data values.
Anti-collapse intact: bijection on the data range -> a constant or zero prediction cannot
match varied reconstructed targets; to_obs(zeros) == zeros keeps the predict-mean guard.
reg() = 0.0 (the schedule is not gradient-trained; a loss-trained radial gain would be
pushed toward compression — the dishonest direction — so the gain is a fixed geometry
constant, like a temperature).
"""

import math

import torch
import torch.nn as nn

NAME = "r10_radial_power_compand_sched"
DESCRIPTION = (
    "Norm-standardizing target space: exactly invertible radial power companding around the "
    "standardization shell mu = sqrt(D) — make_target scales each next-obs by (r/mu)^(s-1) "
    "(log-radius amplified s-fold, direction untouched), to_obs applies the exact inverse, so "
    "the contrastive norm inflation k collapses to k^(1/s) on reconstruction. s ramps 1 -> 6 "
    "over 400 steps (exact identity at init); the only state is a non-persistent step counter "
    "(empty state_dict -> frozen calib_bench/plan_env instruments load and invert exactly)."
)

LEARNED = True

S_FINAL = 6.0      # final radial gain; toy equilibrium: s=5 -> nr ~1.52 (gate edge), s=6 -> ~1.46
RAMP_STEPS = 400   # identity -> full warp over the first 400 training calls (< proxy budget)
_EPS = 1e-8


def _radial_power(x, s, mu):
    """x * (||x||/mu)^(s-1): radius r -> mu*(r/mu)^s, direction preserved. Exact-zero-safe."""
    r = x.norm(dim=-1, keepdim=True)
    ratio = (r / mu).clamp_min(_EPS)
    r_new = mu * ratio.pow(s)
    return x * (r_new / r.clamp_min(_EPS))


def _mu(x):
    return math.sqrt(float(x.shape[-1]))


# ---- module-level PURE functions: the final (post-ramp) geometry. Used by the frozen ----
# ---- calib_bench / plan_env instruments; identical to the trained instance's state.  ----

def make_target(z_obs, z_prev):
    return _radial_power(z_obs, S_FINAL, _mu(z_obs))


def to_obs(pred, z_prev):
    return _radial_power(pred, 1.0 / S_FINAL, _mu(pred))


# ---- LEARNED-contract module (harness training path): same map with the s(t) ramp. ----

class RadialCompandTarget(nn.Module):
    """Scheduled radial power compander. persistent=False => empty state_dict (the frozen
    instruments strict-load checkpoints into a bare arch net; the counter must not leak
    into the checkpoint). After RAMP_STEPS training calls, s == S_FINAL == the module-level
    pure functions above, so every eval path reconstructs in the exact trained geometry."""

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("_step", torch.zeros(()), persistent=False)

    def _s(self):
        frac = (self._step / float(RAMP_STEPS)).clamp(0.0, 1.0)
        return 1.0 + (S_FINAL - 1.0) * float(frac)

    def make_target(self, z_obs, z_prev):
        s = self._s()
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self._step += 1.0
        return _radial_power(z_obs, s, _mu(z_obs))

    def to_obs(self, pred, z_prev):
        return _radial_power(pred, 1.0 / self._s(), _mu(pred))

    def reg(self):
        return 0.0


def make(dim):
    return RadialCompandTarget(dim)
