"""
Objective idea: local-geometry calibrated L2 retrieval.

Instead of only pulling each prediction to its own target, this loss also makes the
batchwise prediction distance graph match the target distance graph. That should help
retrieval margin because nearest-neighbor ranking in squared-L2 depends on the local
geometry of the embedding cloud, not just independent pointwise error; preserving that
geometry can reduce hub-like predictions and keep hard same-verb foils separated.
"""

import torch
import torch.nn.functional as F

NAME = "r7_local_geometry_l2"
DESCRIPTION = "MSE plus soft L2 ranking and batchwise local-distance-graph calibration."


def loss(pred, tgt):
    n, d = pred.shape
    diff = pred - tgt

    mse = diff.pow(2).mean()
    if n < 2:
        return mse

    # Retrieval-shaped logits: smaller squared L2 to a target should win.
    dist_pt = torch.cdist(pred.float(), tgt.float(), p=2).pow(2)
    pos = dist_pt.diag()

    tau = dist_pt.detach().median().clamp_min(1.0) * 0.07
    logits = -dist_pt / tau
    labels = torch.arange(n, device=pred.device)
    nce = F.cross_entropy(logits, labels)

    # Pairwise hinge focuses directly on "true target closer than batch foils".
    tgt_dist = torch.cdist(tgt.float(), tgt.float(), p=2).pow(2).detach()
    eye = torch.eye(n, device=pred.device, dtype=torch.bool)
    nonzero = tgt_dist[~eye]
    margin = nonzero.median().clamp_min(1.0) * 0.04

    rank_terms = F.softplus((pos[:, None] - dist_pt + margin) / tau)
    rank_terms = rank_terms.masked_fill(eye, 0.0)
    rank = rank_terms.sum() / (n * (n - 1))

    # Local geometry calibration: predictions should preserve target neighborhood
    # distances. Weight nearby target pairs more because retrieval errors are local.
    dist_pp = torch.cdist(pred.float(), pred.float(), p=2).pow(2)
    scale = nonzero.median().clamp_min(1.0)
    local_w = torch.exp(-(tgt_dist / scale)).masked_fill(eye, 0.0).detach()

    pp_norm = dist_pp / dist_pp.detach()[~eye].median().clamp_min(1.0)
    tt_norm = tgt_dist / scale
    geom = (local_w * F.smooth_l1_loss(pp_norm, tt_norm, reduction="none")).sum()
    geom = geom / local_w.sum().clamp_min(1.0)

    return mse + 0.35 * nce + 0.20 * rank + 0.12 * geom
