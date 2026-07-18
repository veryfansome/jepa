"""objective chunk: hard-shell L2 ranking with target-geometry calibration.

Idea: optimize the retrieval rule without InfoNCE by making each prediction beat only the
most dangerous in-batch target foils under squared L2. The margin is calibrated from the
true target-target distance, so near-duplicate observations are not forced unrealistically
far apart while genuinely different observations get a stronger separation. A small
Barlow-style cross-correlation term keeps dimensions aligned/decorrelated, which may help
the frozen embedding space preserve retrieval-relevant geometry beyond pointwise MSE.
"""

import torch
import torch.nn.functional as F

NAME = "hard_shell_barlow_l2"
DESCRIPTION = (
    "Smooth hard-negative squared-L2 ranking with target-distance-calibrated margins, "
    "plus MSE alignment and a lightweight Barlow-style geometry term."
)


def _off_diagonal(x):
    n, m = x.shape
    if n != m:
        return x.new_empty(0)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def loss(pred, tgt):
    n, d = pred.shape
    diff = pred - tgt

    align = F.smooth_l1_loss(pred, tgt, beta=0.75)
    mse = diff.pow(2).mean()

    if n < 2:
        return align + 0.15 * mse

    cross_d2 = torch.cdist(pred.float(), tgt.float(), p=2).pow(2)
    pos = cross_d2.diag()

    with torch.no_grad():
        tgt_d2 = torch.cdist(tgt.float(), tgt.float(), p=2).pow(2)
        eye = torch.eye(n, dtype=torch.bool, device=tgt.device)
        scale = tgt_d2[~eye].median().clamp_min(1.0)
        margin = (0.05 + 0.25 * (tgt_d2 / scale).clamp(max=4.0)).to(cross_d2.dtype)

    eye = torch.eye(n, dtype=torch.bool, device=pred.device)
    violations = (pos[:, None] - cross_d2 + margin).masked_fill(eye, -1.0e6)

    k = min(8, n - 1)
    hard = torch.topk(violations, k=k, dim=1).values
    rank = F.softplus(hard / 8.0).mean() * 8.0

    pred_z = (pred - pred.mean(dim=0)) / pred.std(dim=0, unbiased=False).clamp_min(1.0e-4)
    tgt_z = (tgt - tgt.mean(dim=0)) / tgt.std(dim=0, unbiased=False).clamp_min(1.0e-4)
    c = (pred_z.T @ tgt_z) / float(n)
    on_diag = (torch.diagonal(c) - 1.0).pow(2).mean()
    off_diag = _off_diagonal(c).pow(2).mean()
    barlow = on_diag + 0.01 * off_diag

    return align + 0.10 * mse + 0.35 * rank + 0.03 * barlow
