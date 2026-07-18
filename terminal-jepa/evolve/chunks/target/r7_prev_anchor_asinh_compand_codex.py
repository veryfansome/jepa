"""Previous-anchored radial companding target.

Predict an observation-space point that stays essentially identical to z_obs for small
previous-relative changes, but asinh-compresses large jumps away from z_prev. This keeps
the successful identity target geometry for near-repeat shell observations while reducing
the variance and loss leverage of rare large command-induced moves; to_obs exactly expands
the radial displacement back for L2 retrieval.
"""

import torch

NAME = "r7_prev_anchor_asinh_compand"
DESCRIPTION = (
    "Predict z_prev plus an asinh-companded radial delta from z_prev; "
    "small deltas remain near identity, large jumps are compressed and exactly expanded."
)

_EPS = 1e-8
_SCALE_FRAC = 0.5


def _local_scale(z_prev):
    prev_norm = z_prev.norm(dim=-1, keepdim=True)
    fallback = (
        torch.sqrt(torch.as_tensor(z_prev.shape[-1], dtype=z_prev.dtype, device=z_prev.device))
        * _SCALE_FRAC
    )
    return torch.where(prev_norm > _EPS, prev_norm * _SCALE_FRAC, fallback.expand_as(prev_norm))


def _asinh_compand(delta, scale):
    radius = delta.norm(dim=-1, keepdim=True)
    safe_radius = radius.clamp_min(_EPS)
    warped_radius = scale * torch.asinh(radius / scale)
    return delta * (warped_radius / safe_radius)


def _sinh_expand(warped, scale):
    radius = warped.norm(dim=-1, keepdim=True)
    safe_radius = radius.clamp_min(_EPS)
    unwarped_radius = scale * torch.sinh(radius / scale)
    return warped * (unwarped_radius / safe_radius)


def make_target(z_obs, z_prev):
    return z_prev + _asinh_compand(z_obs - z_prev, _local_scale(z_prev))


def to_obs(pred, z_prev):
    return z_prev + _sinh_expand(pred - z_prev, _local_scale(z_prev))
