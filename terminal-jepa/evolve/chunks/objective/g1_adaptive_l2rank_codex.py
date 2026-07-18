"""Adaptive L2 rank objective: fit the target embedding while enforcing that each
prediction is closer to its own target than to in-batch foils under squared L2.
Margins shrink for semantically near target foils and grow for far foils, which
matches retrieval ranking without over-repelling genuinely similar observations.
"""

import torch
import torch.nn.functional as F

NAME = "adaptive_l2_rank"
DESCRIPTION = "MSE plus adaptive in-batch squared-L2 ranking margins against target foils."


def loss(pred, tgt):
    n = pred.shape[0]
    base = F.mse_loss(pred, tgt)

    if n < 2:
        return base

    d = torch.cdist(pred.float(), tgt.float(), p=2).pow(2)
    pos = d.diag().unsqueeze(1)

    with torch.no_grad():
        td = torch.cdist(tgt.float(), tgt.float(), p=2).pow(2)
        scale = td[~torch.eye(n, dtype=torch.bool, device=tgt.device)].median().clamp_min(1e-6)
        margin = 0.15 + 0.35 * (td / scale).clamp(max=2.0)
        mask = ~torch.eye(n, dtype=torch.bool, device=tgt.device)

    violations = (pos - d + margin) / 24.0
    rank = F.softplus(violations[mask]).mean()

    logits = -d / 48.0
    nce = F.cross_entropy(logits, torch.arange(n, device=pred.device))

    return base + 0.25 * rank + 0.10 * nce
