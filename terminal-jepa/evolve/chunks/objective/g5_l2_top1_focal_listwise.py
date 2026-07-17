"""objective chunk: metric-exact listwise TOP-1 surrogate on per-dim-mean squared-L2.

Reading of the EXACT metric (realenv/seq_worldmodel.py):
  _rank_stats scores every candidate by  ((true[cand] - pred)**2).mean(-1)  and counts
  closer = (d_foil < d_true).sum(1); top-1 = (closer==0). The candidate set for query i is
  {true_i} UNION {true_j : j in same-verb foils}. So the metric ranks pred_i against the
  TRUE TARGET embeddings, by PER-DIM-MEAN squared L2 (note the .mean(-1), not a sum), strictly.
  content_retrieval restricts to ls/cat first, then _foils_sameverb draws foils from the SAME
  verb inside that subset -> the hard negatives are other ls/cat observations.

This loss is a listwise top-1 surrogate that matches that structure exactly:
  * Logits are NEGATIVE per-dim-mean squared-L2 from pred_i to every in-batch target tgt_j
    (dist2/d) -- the SAME geometry AND the SAME per-dim normalization the metric scores by.
    In-batch targets are the only same-space foils available, mirroring true[foil_idx].
  * ROW-DIRECTION ONLY. The metric evaluates pred_i vs candidate TRUE targets; it never ranks
    tgt_i against other PREDICTIONS. The winner's symmetric (col) term is an off-metric prior
    that dilutes gradient away from the eval structure, so it is dropped.
  * FOCAL top-1 reweighting: the metric only flips when a foil out-ranks true, i.e. when the
    true target is NOT already closest. We weight each example by (1 - p_true)^gamma (detached),
    concentrating gradient on examples at risk of a top-1 flip -- a top-1 surrogate rather than
    InfoNCE's uniform log-partition, which spends equal gradient on already-solved rows.
  * Temperature 0.25 on dist2/d is justified from the standardized 768-d scale: a random
    same-space same-verb foil sits at per-dim-mean sqL2 ~= 2.0, while a good prediction sits at
    ~0.1-1.0, so the decisive gap d_foil - d_true is O(1); dividing by 0.25 yields sharp,
    well-separated logits (NOT the near-uniform softmax that sank the adaptive-margin recombiner,
    which divided distances by an O(d) sum-space scale and collapsed to its MSE anchor -> 0.18).
  * Small MSE anchor (0.05): targets are standardized and the L2 metric is NOT shift-invariant,
    so absolute placement matters. Anti-collapse: a constant pred makes every row's logit vector
    constant across columns -> uniform softmax -> the focal-NLL term is pinned at log(n) (never
    minimized), and MSE(const, varying tgt) is strictly positive -- collapse cannot win.
"""

import torch
import torch.nn.functional as F

NAME = "l2_top1_focal_listwise"
DESCRIPTION = (
    "Row-only (metric-direction) InfoNCE-style listwise loss on per-dim-mean squared-L2 logits, "
    "with a detached focal (1-p_true)^gamma reweighting that concentrates gradient on examples at "
    "risk of a same-verb top-1 flip, plus a small MSE anchor for absolute placement."
)

_TEMP = 0.25     # on dist2/d; gap between d_true(~0.1-1) and same-verb foil(~2) is O(1) in this space
_GAMMA = 1.0     # focal focusing on not-yet-#1 examples (the ones that decide top-1)
_ANCHOR = 0.05   # absolute-placement anchor (metric is shift-sensitive); also anti-collapse


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee (positive for constant pred vs varying tgt).
    mse_anchor = ((pred - tgt) ** 2).mean()

    if n < 2:
        return mse_anchor

    # Pairwise squared-Euclidean, then PER-DIM MEAN (/d) to match the metric's .mean(-1) exactly.
    pred_sq = (pred * pred).sum(dim=1, keepdim=True)          # [n,1]
    tgt_sq = (tgt * tgt).sum(dim=1, keepdim=True)             # [n,1]
    dist2 = pred_sq + tgt_sq.t() - 2.0 * (pred @ tgt.t())     # [n,n] = ||pred_i - tgt_j||^2
    dist2 = dist2.clamp_min(0.0) / float(d)                   # per-dim-mean squared L2 (metric geometry)

    # Metric direction only: pred_i ranked against candidate TRUE targets by (negative) L2.
    logits = -dist2 / _TEMP                                   # closer true target => higher logit
    labels = torch.arange(n, device=pred.device)

    logp = F.log_softmax(logits, dim=1)                       # [n,n]
    nll = -logp.gather(1, labels[:, None]).squeeze(1)         # [n] = -log P(rank tgt_i first)

    # Detached focal weight: focus on examples whose true target is NOT yet closest (top-1 at risk).
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)                 # softmax prob mass on the true target
        weight = (1.0 - p_true).pow(_GAMMA)                   # ~1 when a foil out-ranks true; ~0 when solved

    listwise = (weight * nll).mean()

    return listwise + _ANCHOR * mse_anchor
