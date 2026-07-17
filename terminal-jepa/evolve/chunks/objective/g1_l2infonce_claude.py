"""Idea: Retrieval-aligned InfoNCE on L2 (squared-distance) similarity — directly optimize
the metric fitness is computed from.

WHY this might raise the held-out retrieval margin:
The downstream metric ranks the true next-observation against same-verb foils by *squared L2
distance* and reports top-1. MSE only pushes each prediction toward its own target in isolation;
it never tells the predictor to make the true target CLOSER than the *other* targets in the space.
That relative-ranking pressure is exactly what a contrastive loss supplies. So instead of a
cosine/dot InfoNCE, I use negative squared-Euclidean distance as the logit — the SAME geometry
the retrieval metric scores by — turning the training objective into a differentiable surrogate
for "rank the correct next-obs first by L2." Every other tgt in the batch is an in-batch negative
(a plausible same-space foil), which mirrors the foil ranking at eval time.

Anti-collapse: a constant prediction makes all rows identical, so every example's logit vector is
constant across columns -> the softmax is uniform -> InfoNCE stays at log(n), never minimized. A
small MSE anchor is added so predictions live at the correct absolute location (targets are
standardized, so absolute placement matters for the L2 metric), but the MSE weight is tiny so the
ranking term dominates. The distance logits are scaled by a temperature to keep
gradients well-conditioned across the 768-d standardized space.
"""

import torch
import torch.nn.functional as F

NAME = "l2_infonce_retrieval"
DESCRIPTION = (
    "InfoNCE whose logits are negative squared-L2 distances between predictions and all "
    "in-batch targets (matching the retrieval metric's geometry), plus a small MSE anchor "
    "for absolute placement. Symmetric over the pred->tgt and tgt->pred directions."
)


def loss(pred, tgt):
    n, d = pred.shape

    # Small MSE anchor: keeps predictions at the correct absolute location in the
    # standardized target space (the L2 retrieval metric is not scale/shift invariant).
    mse_anchor = ((pred - tgt) ** 2).mean()

    # Degenerate batch (no negatives available): fall back to the anchor only.
    if n < 2:
        return mse_anchor

    # Pairwise squared-Euclidean distances: dist2[i, j] = || pred_i - tgt_j ||^2.
    # Computed stably via the (a-b)^2 = a^2 - 2ab + b^2 expansion.
    pred_sq = (pred * pred).sum(dim=1, keepdim=True)          # [n, 1]
    tgt_sq = (tgt * tgt).sum(dim=1, keepdim=True)             # [n, 1]
    cross = pred @ tgt.t()                                    # [n, n]
    dist2 = pred_sq + tgt_sq.t() - 2.0 * cross               # [n, n]
    dist2 = dist2.clamp_min(0.0)                              # guard tiny negatives

    # Temperature that normalizes for dimensionality so logits are O(1) regardless of d.
    temperature = float(d) ** 0.5

    logits = -dist2 / temperature                            # closer target => higher logit
    targets = torch.arange(n, device=pred.device)

    # Symmetric InfoNCE: rank correct target among target-foils (rows) AND correct pred
    # among pred-foils (cols). Both directions reinforce L2-separated, retrievable geometry.
    loss_row = F.cross_entropy(logits, targets)              # pred_i picks tgt_i
    loss_col = F.cross_entropy(logits.t(), targets)          # tgt_i picks pred_i
    contrastive = 0.5 * (loss_row + loss_col)

    # Ranking term dominates; MSE only anchors absolute location.
    return contrastive + 0.05 * mse_anchor
