"""LEARNED target space (the epistasis retry of the thrice-recorded-negative 'target_space'
idea, now recombined with the e5 encoder + contrastive objectives — a context that did not exist
when it was last tried): predict in a learned per-dimension re-weighting of the obs space.

  g = D * softmax(theta)          # positive gates, mean EXACTLY 1 by construction; theta=0 at
                                  # init -> g == 1 -> exact identity at initialization
  make_target(z_obs, z_prev) = g * z_obs        # trained jointly with the predictor
  to_obs(pred, z_prev)       = pred / g         # exact inverse; eval stays in the FIXED obs space
  reg() = REG_LAM * KL(softmax(theta) || uniform)

The training loss sees the gated space, so gradient can down-weight dimensions whose variation is
unpredictable noise and emphasize dimensions that carry predictable structure. Honesty: the
retrieval eval always ranks in the ORIGINAL obs space via the exact inverse, so shrinking a
dimension to dodge prediction error re-amplifies that error at eval — a collapsed gate cannot
win. The KL regularizer only stabilizes against extreme gate concentration (softmax winner-take-
all); at lam=0.01 it is a weak pull toward identity.

Contract (learned extension): LEARNED = True; make(D) -> nn.Module exposing
make_target(z_obs, z_prev), to_obs(pred, z_prev), reg(). Params are registered on the net by the
harness and trained by the genome's optimizer.
"""

import math

import torch
import torch.nn as nn

NAME = "tgt_space_diag_gate"
DESCRIPTION = ("Learned mean-one per-dim target gates g = D*softmax(theta) (identity at init), "
               "exact inverse at eval, weak KL(uniform) stabilizer — the target_space retry.")

LEARNED = True
REG_LAM = 0.01


class DiagGateTarget(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.theta = nn.Parameter(torch.zeros(dim))

    def _g(self):
        return self.dim * torch.softmax(self.theta, dim=0)

    def make_target(self, z_obs, z_prev):
        return self._g() * z_obs

    def to_obs(self, pred, z_prev):
        return pred / self._g().clamp(min=1e-3)

    def reg(self):
        p = torch.softmax(self.theta, dim=0)
        kl = (p * (p.clamp(min=1e-12).log() + math.log(self.dim))).sum()
        return REG_LAM * kl


def make(dim):
    return DiagGateTarget(dim)
