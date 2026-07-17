"""target chunk: predict a partial residual z_obs - 0.65*z_prev.

This keeps more absolute observation information than a full delta while subtracting much of the
stable carryover from the previous shell observation. The fixed affine transform is exactly
invertible, and may give the predictor a lower-variance target without making the contrastive
space depend entirely on small residual changes.
"""

import torch

NAME = "partial_residual_065"
DESCRIPTION = "Predict z_obs - 0.65*z_prev; reconstruct prediction + 0.65*z_prev."

_ALPHA = 0.65


def make_target(z_obs, z_prev):
    return z_obs - (_ALPHA * z_prev)


def to_obs(pred, z_prev):
    return pred + (_ALPHA * z_prev)
