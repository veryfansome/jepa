"""target chunk: predict a Householder-canonical residual relative to the previous observation.

Use a per-example orthogonal Householder reflection (computed only from z_prev) that maps z_prev
onto the first coordinate axis, then predict the residual in that canonical frame:
    H(z_prev) @ z_obs - ||z_prev|| e_0.
This keeps the transform exactly invertible while concentrating the predictable "same as previous
observation" component into one coordinate, leaving the model to predict a lower-variance innovation
vector in a z_prev-aligned coordinate system. Because H is orthogonal and self-inverse, reconstruction
returns to the original embedding geometry used by L2 retrieval.
"""

import torch

NAME = "householder_prev_residual"
DESCRIPTION = (
    "Reflect into a z_prev-canonical frame, predict H(z_prev)@z_obs - ||z_prev||e0, "
    "then invert with the same Householder reflection."
)


_EPS = 1e-12


def _householder_apply(x, z_prev):
    norm = z_prev.norm(dim=-1, keepdim=True)
    e0 = torch.zeros_like(z_prev)
    e0[..., :1] = 1.0

    sign = torch.where(z_prev[..., :1] >= 0, 1.0, -1.0)
    v = z_prev + sign * norm * e0
    v_norm2 = (v * v).sum(dim=-1, keepdim=True)

    reflected = x - 2.0 * v * ((x * v).sum(dim=-1, keepdim=True) / v_norm2.clamp_min(_EPS))
    return torch.where(norm > _EPS, reflected, x)


def make_target(z_obs, z_prev):
    prev_norm = z_prev.norm(dim=-1, keepdim=True)
    axis_prev = torch.zeros_like(z_prev)
    axis_prev[..., :1] = prev_norm
    return _householder_apply(z_obs, z_prev) - axis_prev


def to_obs(pred, z_prev):
    prev_norm = z_prev.norm(dim=-1, keepdim=True)
    axis_prev = torch.zeros_like(pred)
    axis_prev[..., :1] = prev_norm
    return _householder_apply(pred + axis_prev, z_prev)
