"""Recombination: same-verb-weighted L2-InfoNCE (fix of the AM-L2-InfoNCE failure).

DIAGNOSIS OF WHY AM-L2-INFONCE HURT (g2_am_l2infonce_claude, 0.39 -> 0.18):
Its `temperature = clamp(scale, min=sqrt(d))` where `scale = median off-diagonal ||tgt_i-tgt_j||^2`.
Targets are per-dim standardized (mean 0, std 1), so E||tgt_i-tgt_j||^2 ~ 2*768 ~ 1536, and the
median is ~1532 (verified numerically). That is ~55x LARGER than the winner's fixed sqrt(768)=27.7.
Dividing the L2 logits by ~1532 instead of ~28 flattens -dist2/T toward 0 -> the InfoNCE softmax
goes near-uniform -> cross-entropy sits near log(n) -> the contrastive gradient vanishes -> only the
0.05*MSE anchor survives, collapsing the objective back to MSE behaviour (exactly the 0.18 it scored,
== the MSE baseline). The distance-aware additive margin was a second, smaller mistake: it grows WITH
the distance it is added to and is dominated by the same over-large T, and it pushes FAR (easy) foils
further down rather than sharpening the near ones the metric cares about.

WHAT THE WINNER GETS RIGHT (must be preserved): a FIXED, small temperature (sqrt(d)) that keeps
-dist2/T at an O(1)-to-O(10) scale, a symmetric row+col InfoNCE on negative-squared-L2 logits (the
metric's exact geometry, per `_rank_stats`: candidates scored by squared L2 to pred, top-1 strict),
and a tiny MSE anchor for absolute placement (the standardized-L2 metric is not shift-invariant).

THE REAL HEADROOM (from reading the metric): fitness ranks the true obs only against SAME-VERB foils
(`_foils_sameverb`) -- in-space-NEAR competitors. But the winner's in-batch negatives are dominated
by different-verb, unrelated observations that are trivially far, so with T=sqrt(d) the in-batch CE
saturates to ~0 (verified: dead gradient once the predictor is decent). The objective stops applying
pressure exactly on the near-in-space negatives that decide the metric.

THE FIX (two disciplined changes, everything else kept intact):
(1) Temperature = 4*sqrt(d): un-saturates the softmax (CE moves from ~0 into a productive O(0.1-1)
    range at the standardized scale) while staying 14x SMALLER than the ~55x that killed AM -- the
    knob AM got catastrophically wrong, corrected in the safe direction.
(2) Same-verb-foil mimic: weight each in-batch negative j (for anchor i) by exp(-||tgt_i-tgt_j||^2/med)
    -- negatives whose TARGET is L2-near the true target (the in-batch analog of same-verb foils) keep
    full weight; unrelated far targets are softly discounted. Detached, so it only reshapes WHICH
    negatives the softmax competes against; it is a gradient-free reweighting, never a leakage path.
    Unlike AM's additive margin it does NOT over-repel far foils and does NOT touch the logit scale.

Anti-collapse: a constant prediction makes each dist2 row constant across columns -> per-row logits
constant -> the per-row softmax is fixed by the (target-only) weights, independent of pred -> the
contrastive term is stuck high (~3.3, verified) and the MSE anchor is not minimized by a constant
either (tgt varies). Pure function of (pred, tgt); no state, no file, NaN-safe under large values.
"""

import torch
import torch.nn.functional as F

NAME = "sameverb_weighted_l2_infonce"
DESCRIPTION = (
    "Winner's symmetric negative-squared-L2 InfoNCE, kept intact, with two disciplined recombination "
    "changes: (1) a moderate 4x-sqrt(d) temperature that un-saturates the in-batch softmax at the "
    "standardized-768-d scale (vs the ~55x adaptive temperature that collapsed AM-L2-InfoNCE to MSE), "
    "and (2) a same-verb-foil-mimicking hardness weighting that concentrates the revived contrastive "
    "gradient on in-batch negatives whose TARGET is L2-near the true target. Small MSE anchor unchanged."
)


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor (winner's; unchanged weight). The standardized-L2 metric is not
    # shift-invariant, so predictions must sit at the correct location.
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # Pairwise squared-Euclidean distances dist2[i,j] = ||pred_i - tgt_j||^2 -- the metric's geometry.
    pred_sq = (pred * pred).sum(dim=1, keepdim=True)          # [n,1]
    tgt_sq = (tgt * tgt).sum(dim=1, keepdim=True)             # [n,1]
    dist2 = (pred_sq + tgt_sq.t() - 2.0 * (pred @ tgt.t())).clamp_min(0.0)  # [n,n]

    # Temperature = 4*sqrt(d). sqrt(d) alone (the winner's) saturates the in-batch softmax at the
    # standardized scale (E||tgt_i-tgt_j||^2 ~ 2d ~ 1536), leaving no late-training gradient; 4x
    # holds CE in a productive range. It is ~14x SMALLER than the adaptive temperature
    # (~median tgt-tgt dist ~ 55*sqrt(d)) that collapsed AM-L2-InfoNCE back to MSE.
    T = 4.0 * (float(d) ** 0.5)
    logits = -dist2 / T                                       # closer target => higher logit
    eye = torch.eye(n, dtype=torch.bool, device=pred.device)

    # Same-verb-foil mimic: the eval ranks the true obs only against SAME-VERB (in-space NEAR) foils.
    # The in-batch analog is negatives whose TARGET is L2-near the true target. Additive log-weight
    # exp(-||tgt_i - tgt_j||^2 / median): near foils keep full weight, unrelated targets are softly
    # discounted. Detached -> reshapes WHICH negatives the softmax competes against, not a grad path.
    with torch.no_grad():
        td = (tgt_sq + tgt_sq.t() - 2.0 * (tgt @ tgt.t())).clamp_min(0.0)   # ||tgt_i - tgt_j||^2
        med = td[~eye].median().clamp_min(1e-6)
        log_w = (-td / med).masked_fill(eye, 0.0)            # additive log-weight on negative logits

    idx = torch.arange(n, device=pred.device)

    def _dir(lg, lw):
        # Weighted InfoNCE: -log( exp(pos_i) / (exp(pos_i) + sum_{j!=i} w_ij * exp(lg_ij)) ).
        pos = lg[idx, idx]                                    # [n]
        neg = lg.masked_fill(eye, float("-inf")) + lw        # [n,n], diagonal -> -inf
        denom = torch.logsumexp(torch.cat([pos[:, None], neg], dim=1), dim=1)
        return (denom - pos).mean()

    # Symmetric: rank true target among target-foils (rows) AND true pred among pred-foils (cols).
    contrastive = 0.5 * (_dir(logits, log_w) + _dir(logits.t(), log_w.t()))

    return contrastive + 0.05 * mse_anchor

