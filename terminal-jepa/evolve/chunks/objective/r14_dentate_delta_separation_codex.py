"""objective chunk: dentate delta separation.

The cat bottleneck looks like common-template interference: many same-path file bodies
share most content, while the held-out retrieval win depends on preserving small
system-specific deltas. Inspired by hippocampal dentate-gyrus pattern separation and
center-surround common-mode rejection, this loss adds a local differential constraint:
for close-but-distinct target pairs, prediction differences should preserve target
differences. A model that predicts only the shared file template for two systems gets
penalized even if its absolute L2 error is modest.

The pair term is scale-free, target-neighborhood-gated, and duplicate-safe:
  log1p(||(pred_i-tgt_i) - (pred_j-tgt_j)||^2 / ||tgt_i-tgt_j||^2)
on detached close target pairs. It complements, rather than reweights, the usual
retrieval softmax.
"""

import torch
import torch.nn.functional as F

NAME = "r14_dentate_delta_separation_codex"
DESCRIPTION = (
    "Precision-weighted focal L2 listwise retrieval loss plus a dentate-style local "
    "delta-separation term: for close-but-distinct target pairs, preserve the signed "
    "target displacement between predictions by penalizing relative pairwise error "
    "differences. This common-mode-rejected constraint targets cat system-variant file "
    "bodies where the shared template is easy but the small cross-system delta decides "
    "same-verb L2 retrieval."
)

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05

_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_DELTA = 0.04
_LAM_FRAC = 0.35
_PAIR_WEIGHT = 0.35
_PAIR_TAU = 0.50
_PAIR_FOCAL = 0.5


def loss(pred, tgt):
    n, d = pred.shape

    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        sw = w.sqrt().unsqueeze(0)

    pw = pred * sw
    tw = tgt * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)

    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(-dist2 / _TEMP, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)

    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)

    listwise = (focal * nll).mean()

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)

        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)

        near = torch.exp(-tt / lam)
        nondup = 1.0 - torch.exp(-tt / _DELTA)
        pair_w = (near * nondup).masked_fill(eye, 0.0)

        row_mean = pair_w.sum(dim=1, keepdim=True) / float(n - 1)
        pair_w = pair_w / row_mean.clamp_min(1e-6)
        pair_w = pair_w.masked_fill(eye, 0.0).clamp_max(8.0)

        sep = tt.clamp_min(_DELTA)

    ew = (pred - tgt) * sw
    ew_sq = (ew * ew).sum(dim=1, keepdim=True)
    ed2 = (ew_sq + ew_sq.t() - 2.0 * (ew @ ew.t())).clamp_min(0.0) / float(d)

    rel = ed2 / sep
    pair_terms = _PAIR_TAU * torch.log1p(rel / _PAIR_TAU)

    with torch.no_grad():
        pair_focus = (rel / (rel + _PAIR_FOCAL)).clamp(0.0, 1.0)

    pair = (pair_w * pair_focus * pair_terms).sum() / pair_w.sum().clamp_min(1.0)

    return listwise + _ANCHOR * mse_anchor + _PAIR_WEIGHT * pair
