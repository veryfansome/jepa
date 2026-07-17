"""Recombination: Adaptive-margin symmetric L2-InfoNCE (AM-L2-InfoNCE).

Fuses the two leaders into ONE contrastive core instead of a sum of separate terms.
From the winner (g1_l2infonce_claude, margin 0.47): a SYMMETRIC InfoNCE whose logits are
negative squared-L2 distances between predictions and all in-batch targets — the exact geometry
the retrieval metric scores by, so the softmax is a differentiable surrogate for the eval. From
the ranker (g1_adaptive_l2rank_codex, margin 0.28): the insight that in-batch foils are NOT all
equally repellable — some are genuinely semantically close (small target-target distance) and
hard-repelling them corrupts the geometry we want. The recombination bakes a distance-aware
additive margin DIRECTLY into the InfoNCE logits: logit[i,j] = -(dist2[i,j] + (i!=j)*m_ij)/T,
with m_ij small for near foils and large for far foils (m_ij from target-target distance under
no_grad). Adaptive temperature tracks the batch distance scale (floored by sqrt(d)). Anti-collapse:
a constant prediction gives per-row-constant logits (margins depend only on tgt-tgt distances) ->
uniform softmax -> InfoNCE pinned at log(n); a small MSE anchor fixes absolute placement.
"""

import torch
import torch.nn.functional as F

NAME = "adaptive_margin_l2_infonce"
DESCRIPTION = (
    "Symmetric L2-distance InfoNCE (the winner's retrieval-aligned engine) with a distance-aware "
    "additive margin baked into the logits: foils semantically near the true target get a small "
    "margin (not over-repelled), far foils a large one. Adaptive temperature; small MSE anchor."
)


def loss(pred, tgt):
    n, d = pred.shape

    mse_anchor = ((pred - tgt) ** 2).mean()

    if n < 2:
        return mse_anchor

    pred_sq = (pred * pred).sum(dim=1, keepdim=True)          # [n, 1]
    tgt_sq = (tgt * tgt).sum(dim=1, keepdim=True)             # [n, 1]
    cross = pred @ tgt.t()                                    # [n, n]
    dist2 = (pred_sq + tgt_sq.t() - 2.0 * cross).clamp_min(0.0)  # [n, n]

    eye = torch.eye(n, dtype=torch.bool, device=pred.device)
    off = ~eye

    with torch.no_grad():
        td = tgt_sq + tgt_sq.t() - 2.0 * (tgt @ tgt.t())     # ||tgt_i - tgt_j||^2
        td = td.clamp_min(0.0)
        off_td = td[off]
        scale = off_td.median().clamp_min(1e-6)

        m_min, m_max = 0.15 * scale, 0.85 * scale
        margin = m_min + (m_max - m_min) * (td / scale).clamp(0.0, 1.0)
        margin = margin * off.to(margin.dtype)

        temperature = torch.clamp(scale, min=float(d) ** 0.5)

    logits_row = -(dist2 + margin) / temperature             # pred_i ranks tgt_j
    logits_col = -(dist2.t() + margin.t()) / temperature     # tgt_i ranks pred_j
    labels = torch.arange(n, device=pred.device)

    loss_row = F.cross_entropy(logits_row, labels)
    loss_col = F.cross_entropy(logits_col, labels)
    contrastive = 0.5 * (loss_row + loss_col)

    return contrastive + 0.05 * mse_anchor
