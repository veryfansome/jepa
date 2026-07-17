"""objective chunk: hubness-aware (CSLS local-scaling) hard negatives on the metric-exact
focal-listwise top-1 surrogate.

MECHANISM SHARED BY THE ARCHIVE WINNERS, AND WHAT THEY MISS
-----------------------------------------------------------
l2_infonce / sameverb_weighted_l2_infonce / l2_top1_focal_listwise all: (a) use NEGATIVE
per-dim squared-L2 as the contrastive logit (the exact geometry `_rank_stats` scores by --
((true-pred)**2).mean(-1), strict top-1), (b) use in-batch targets as the only same-space
negatives, (c) add a small MSE anchor for absolute placement (the L2 metric is shift-sensitive),
and (d) use a fixed temperature. The incumbent (l2_top1_focal_listwise, proxy 0.428) further
keeps ROW-DIRECTION ONLY (the metric ranks pred_i vs TRUE targets, never tgt_i vs predictions)
and a detached focal (1-p_true)^gamma reweighting that concentrates gradient on examples whose
true target is not yet closest.

What NONE of them model: the eval draws its 63 same-verb foils from the POOL OF TRUE
observations (`content_retrieval` -> `_foils_sameverb`). In 768-d e5 space that pool is
HUB-structured (a well-documented high-dim retrieval failure, Schnitzer et al. JMLR 2012;
arXiv 2606.28330): a few centrally-located same-verb observations are near-neighbors to many
predictions and repeatedly out-rank the TRUE target, flipping top-1. sameverb_weighted weights
a negative by its proximity to the ANCHOR (a same-verb proxy) -- orthogonal to a candidate's own
LOCAL centrality. So the exact points that most often steal top-1 (the hubs) get no extra
training pressure.

THE NEW MECHANISM (CSLS local-scaling hard negatives)
-----------------------------------------------------
Everything in l2_top1_focal_listwise is preserved verbatim (per-dim-mean L2 logits, T=0.25,
focal gamma=1, MSE anchor 0.05). Added: a hubness boost on the NEGATIVE logits.
  * For each candidate target j, estimate its LOCAL DENSITY = mean per-dim squared-L2 distance
    to its k nearest OTHER targets (its neighborhood). Same-verb observations are L2-near in e5
    space, so a candidate's nearest neighbors are effectively its same-verb cluster -> local
    density is a same-verb-centrality (hubness) estimate, exactly CSLS/local-scaling's r(y).
  * hub_j = standardize(-local_density_j): high for central hubs, low for peripheral points.
  * Add BETA*hub_j (detached) to every NEGATIVE logit column j (never the positive/diagonal),
    making central hub candidates HARDER negatives. The predictor must place pred_i closer to
    tgt_i than to the boosted hubs -> it learns the within-verb DISTINCTIVE directions rather
    than regressing to the same-verb centroid (VICReg's conditional-mean-collapse failure,
    arXiv 2105.04906). Beating hubs under this strictly harder training criterion means at plain
    -L2 eval those hubs less often out-rank the true target -> higher content top-1 -> margin.

Why this is NOT the am_l2infonce failure: the boost is PER-CANDIDATE (a property of j), not a
per-pair additive margin that grows with distance; it only lifts central (near) negatives and
leaves far/easy foils untouched; and it does NOT touch the temperature (am's fatal ~55x scale
error). It re-shapes WHICH negatives compete, like sameverb_weighted, but on a genuinely new axis
(candidate hubness vs anchor-proximity).

Anti-collapse: a constant prediction c makes each row's data-logit vector -||c-tgt_j||^2 IDENTICAL
across rows i but the label for row i is column i, so the mean focal-NLL is a cross-entropy of a
uniform-over-i target against a FIXED column distribution, which is >= log(n) by Jensen and cannot
be driven down (equality needs all ||c-tgt_j||^2 equal, impossible for constant c vs varying tgt).
The hub boost is detached and finite so it cannot create a collapse escape, and MSE(const, varying
tgt) is strictly positive. Pure function of (pred, tgt); no state/file/network; no in-place edits
of inputs; NaN-safe (clamp_min on distances, eps in the standardizer, finite k). MPS-fast: two
[n,n] matmuls + one topk, same order as the incumbent.
"""

import torch
import torch.nn.functional as F

NAME = "csls_hub_listwise"
DESCRIPTION = (
    "Metric-exact row-only focal-listwise loss on per-dim-mean squared-L2 logits (the incumbent "
    "l2_top1_focal_listwise, preserved), plus a CSLS/local-scaling hubness boost: each in-batch "
    "candidate's negative logit is raised in proportion to its local centrality (mean distance to "
    "its k nearest targets, a same-verb-hub estimate), forcing predictions to separate from the "
    "same-verb hubs that flip held-out top-1. Detached boost; small MSE anchor unchanged."
)

_TEMP = 0.25     # on dist2/d; gap d_true(~0.1-1) vs same-verb foil(~2) is O(1) in standardized space
_GAMMA = 1.0     # focal focusing on not-yet-#1 examples (the ones that decide top-1)
_ANCHOR = 0.05   # absolute-placement anchor (metric is shift-sensitive); also anti-collapse
_BETA = 0.5      # strength of the hubness hard-negative boost (added to standardized hub scores)
_KFRAC = 0.10    # neighborhood size for local density = this fraction of the batch (a verb-cluster proxy)
_KMAX = 64       # cap on k for cost/robustness


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee (positive for constant pred vs varying tgt).
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # Pairwise squared-Euclidean pred_i vs tgt_j, PER-DIM MEAN (/d) to match the metric's .mean(-1).
    pred_sq = (pred * pred).sum(dim=1, keepdim=True)          # [n,1]
    tgt_sq = (tgt * tgt).sum(dim=1, keepdim=True)             # [n,1]
    dist2 = pred_sq + tgt_sq.t() - 2.0 * (pred @ tgt.t())     # [n,n] = ||pred_i - tgt_j||^2
    dist2 = dist2.clamp_min(0.0) / float(d)                   # per-dim-mean squared L2 (metric geometry)

    logits = -dist2 / _TEMP                                   # closer true target => higher logit (data term)
    labels = torch.arange(n, device=pred.device)

    # ---- CSLS local-scaling hubness estimate over the TRUE-target pool (detached, gradient-free) ----
    with torch.no_grad():
        # target-target per-dim-mean squared-L2; self set to +inf so it is never a "neighbor".
        tt = (tgt_sq + tgt_sq.t() - 2.0 * (tgt @ tgt.t())).clamp_min(0.0) / float(d)   # [n,n]
        tt.fill_diagonal_(float("inf"))
        k = max(1, min(_KMAX, int(_KFRAC * n), n - 1))
        near, _ = tt.topk(k, dim=1, largest=False)           # k smallest distances per candidate
        local_density = near.mean(dim=1)                     # small => central hub, large => peripheral
        hub = -local_density                                 # high for hubs
        hub = (hub - hub.mean()) / (hub.std() + 1e-6)        # standardize to O(1)
        boost = _BETA * hub.unsqueeze(0).repeat(n, 1)        # [n,n]; column j gets hub_j
        boost.fill_diagonal_(0.0)                            # never boost the positive (diagonal)

    logits = logits + boost                                  # central candidates become harder negatives

    logp = F.log_softmax(logits, dim=1)                      # [n,n]
    nll = -logp.gather(1, labels[:, None]).squeeze(1)        # [n] = -log P(rank tgt_i first)

    # Detached focal weight: focus on examples whose true target is NOT yet closest (top-1 at risk).
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)                # softmax prob on the true target
        weight = (1.0 - p_true).pow(_GAMMA)                  # ~1 when a foil out-ranks true; ~0 when solved

    listwise = (weight * nll).mean()

    return listwise + _ANCHOR * mse_anchor

