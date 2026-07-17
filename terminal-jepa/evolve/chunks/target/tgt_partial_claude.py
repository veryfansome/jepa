"""target chunk: partial residual z_obs - alpha*z_prev, alpha=0.75 (between identity and delta).
Removes most of the predictable shared previous-obs component without the over-subtraction of full
delta; alpha ~ regression coeff of z_obs on z_prev. Invertible: reconstruct alpha*z_prev + pred."""
import torch
NAME = "partial_residual_a075"
DESCRIPTION = "Predict z_obs - 0.75*z_prev; reconstruct 0.75*z_prev + pred."
_ALPHA = 0.75
def make_target(z_obs, z_prev):
    return z_obs - _ALPHA * z_prev
def to_obs(pred, z_prev):
    return _ALPHA * z_prev + pred
