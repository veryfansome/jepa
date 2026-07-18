"""Retrieval-aligned NCA + relative-margin hinge in the EXACT squared-L2 geometry the
metric scores by — a listwise soft-top-1 objective, NOT InfoNCE.

=== What the metric actually is (read from realenv/seq_worldmodel.py) ===
`_rank_stats` scores each candidate (the true next-obs + 63 foils) by
    d = ((candidate - pred) ** 2).mean(-1)
and counts foils with `d_foil < d_true` STRICTLY; top-1 = fraction with zero closer foils.
`_foils_sameverb` draws every foil from SAME-verb *real target* embeddings, and
`content_retrieval` restricts to ls/cat. So the decision the model is judged on is purely:

    is pred_i closer to its OWN target tgt_i than to any OTHER real same-verb target tgt_j?

i.e. pred_i must land in tgt_i's Voronoi cell among the real targets under squared L2. The
in-batch targets ARE the same-space foils (mostly different verbs, but the same geometry the
eval uses). Two consequences drive this design:

  (1) The metric compares distances from pred to *different real targets* (tgt_j), never
      pred-to-pred. So the useful contrast is pred_i vs. {tgt_j}, and the NEAR real targets
      are the only ones that can produce a rank error. Down-weighting near foils (what the
      failed adaptive-margin recombiner and g1_adaptive_l2rank did) removes pressure from
      exactly the foils the metric penalizes — that is why they scored 0.18 / 0.28 vs the
      winner's 0.47. This loss does the opposite: it concentrates gradient on near foils.

  (2) The comparison is STRICT (`d_foil < d_true` beats the true; ties do not). Being ranked
      #1 by an infinitesimal margin is fragile at eval where foils are resampled. So even
      after pred_i is nominally closest, we keep pushing every near foil a relative margin
      farther than the true — a hinge that only goes quiet once there is real slack.

=== Why NCA over InfoNCE, and the temperature (the real bug in the champion) ===
Measured on standardized 768-d targets: a decent predictor gives positive squared-L2
~3.7e2 and off-diagonal ~2.1e3. The champion divides these by temperature sqrt(d)=27.7,
so logits differ by thousands and the softmax is a HARDMAX (measured entropy 0.0): it sees
only the single nearest point and its gradient saturates to ~0 the instant that one point is
separated. It is effectively a saturating single-nearest-negative loss with a mis-scaled
temperature. Here the temperature is SELF-CALIBRATED to the batch's own positive-distance
scale T = mean_i ||pred_i - tgt_i||^2 (the metric's d_true scale). At that temperature the
softmax stays in an informative regime (entropy ~2.5), so the loss keeps applying graded
pressure across ALL near foils that decide top-1, not just the single closest — this is the
listwise Neighbourhood-Components-Analysis objective: -log of the soft-probability that pred_i
retrieves tgt_i first among the real targets, which is a smooth surrogate for the exact top-1
event `no foil closer than true`.

=== Anti-collapse / safety ===
A constant prediction makes every d(pred, tgt_j) equal across j -> uniform softmax ->
cross-entropy pinned at log(n) (verified: const loss 45.8 vs perfect 0.0), and the hinge
becomes (pos - dist + margin) constant across j with pos=dist so it reduces to softplus(margin)>0,
also un-minimized -> constant predictions never minimize the loss. A tiny MSE anchor fixes
absolute placement (the L2 metric is not shift-invariant) without dominating. All masking of
the diagonal is MULTIPLICATIVE (off-mask), never additive -inf, because -inf through softplus
yields finite forward but NaN backward (verified) — the hard NaN filter would kill it.
"""

import torch
import torch.nn.functional as F

NAME = "l2_nca_hinge_retrieval"
DESCRIPTION = (
    "Listwise NCA (soft-top-1) in the retrieval metric's squared-L2 geometry with a "
    "self-calibrated temperature tied to the positive-distance scale, plus a relative-margin "
    "hinge that concentrates gradient on the NEAR real-target foils that decide strict top-1, "
    "plus a tiny MSE anchor. A different mechanism from InfoNCE: no pred-pred contrast, "
    "temperature from the batch's own d_true scale rather than a fixed sqrt(d)."
)


def loss(pred, tgt):
    n, d = pred.shape

    # Tiny anchor: the squared-L2 retrieval metric is not shift/scale invariant, so pin the
    # absolute location. Kept small so the ranking terms dominate.
    mse = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse

    # Pairwise squared-Euclidean distances dist2[i, j] = || pred_i - tgt_j ||^2, the EXACT
    # quantity `_rank_stats` compares (up to the /d mean, absorbed by the temperature).
    pred_sq = (pred * pred).sum(dim=1, keepdim=True)          # [n, 1]
    tgt_sq = (tgt * tgt).sum(dim=1, keepdim=True)             # [n, 1]
    dist2 = (pred_sq + tgt_sq.t() - 2.0 * (pred @ tgt.t())).clamp_min(0.0)  # [n, n]

    eye = torch.eye(n, dtype=torch.bool, device=pred.device)
    off = (~eye).to(pred.dtype)
    pos = dist2.diagonal().unsqueeze(1)                       # d_true per row, [n, 1]

    # Self-calibrated temperature = mean positive squared-distance (the metric's d_true scale).
    # Keeps the softmax informative instead of a hardmax; detached so it only sets scale.
    with torch.no_grad():
        T = dist2.diagonal().mean().clamp_min(1e-6)

    # (1) Listwise NCA / soft-top-1: -log P(pred_i retrieves tgt_i first among real targets).
    #     Symmetric so tgt_i must also retrieve pred_i, reinforcing separable L2 geometry.
    logits = -dist2 / T
    labels = torch.arange(n, device=pred.device)
    nca = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    # (2) Relative-margin hinge on the STRICT top-1 event: each near foil must sit at least a
    #     margin (scaled by T) farther than the true. Softplus stays active until real slack
    #     exists, and (unlike the failed adaptive-margin) applies the SAME margin to every foil
    #     so the near ones — the only ones that cause rank errors — get the most gradient.
    #     Off-mask is multiplicative (never additive -inf) to keep the backward NaN-free.
    margin = 0.25 * T
    hinge = (F.softplus((pos - dist2 + margin) / T * 4.0) * off).sum(dim=1).mean()

    return nca + 0.5 * hinge + 0.02 * mse

