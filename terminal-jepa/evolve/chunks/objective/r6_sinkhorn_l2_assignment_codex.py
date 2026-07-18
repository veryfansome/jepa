"""
Doubly-stochastic L2 assignment loss.

Instead of only asking each prediction to pick its own target independently, this loss
turns the batchwise negative squared-L2 matrix into a soft one-to-one matching with
Sinkhorn normalization, then maximizes the diagonal assignment probability. This may
raise retrieval margin by penalizing many predictions collapsing onto the same nearby
target while staying exactly aligned with squared-L2 retrieval.
"""

import torch
import torch.nn.functional as F

NAME = "sinkhorn_l2_assignment"
DESCRIPTION = "Entropic batchwise one-to-one assignment over negative squared-L2 distances, plus a small alignment anchor."


def loss(pred, tgt):
    n = pred.shape[0]
    if n <= 1:
        return ((pred - tgt) ** 2).mean()

    pred_f = pred.float()
    tgt_f = tgt.float()

    p2 = (pred_f * pred_f).sum(dim=1, keepdim=True)
    t2 = (tgt_f * tgt_f).sum(dim=1, keepdim=True).t()
    d2 = (p2 + t2 - 2.0 * pred_f @ tgt_f.t()).clamp_min(0.0)

    diag_d2 = d2.diag()
    off = d2.detach()
    neg_scale = off[~torch.eye(n, dtype=torch.bool, device=d2.device)].median()
    tau = neg_scale.clamp_min(1.0).sqrt() * 0.22

    log_p = -d2 / tau.clamp_min(1e-4)

    # Differentiable Sinkhorn in log-space: rows are predictions, columns are targets.
    for _ in range(5):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)

    assign = -log_p.diag().mean()

    # A light pointwise anchor keeps absolute coordinates calibrated for L2 retrieval.
    huber = F.smooth_l1_loss(pred_f, tgt_f, beta=1.0)

    # Penalize diagonal distances that fail to beat the closest nonmatching target.
    masked = d2 + torch.eye(n, device=d2.device, dtype=d2.dtype) * 1e6
    nearest_neg = masked.min(dim=1).values.detach()
    margin = F.softplus((diag_d2 - nearest_neg + 0.5) / tau.clamp_min(1e-4)).mean()

    return assign + 0.08 * huber + 0.25 * margin
