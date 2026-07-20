"""R10 objective: precision ranking plus cross-example and radial calibration.

The incumbent row softmax keeps the right free-energy precision L2 ranking geometry, but each
row's own ||pred||^2 cancels out, allowing norm inflation. This objective preserves that ranking
core, then makes absolute placement pay in two norm-sensitive ways: a Cross-Example-Softmax-style
global hard-negative threshold in the unweighted eval space, and a detached-target log radial
penalty that pulls ||pred||^2 toward ||tgt||^2. Calibration terms are gated by the row retrieval
probability, so they become strongest once ranking is already working instead of replacing it.
"""

import torch
import torch.nn.functional as F

NAME = "r10_cross_example_radial_codex"
DESCRIPTION = (
    "R10 calibrated contrastive objective: incumbent free-energy precision row-focal L2 "
    "listwise loss for ranking, plus a global cross-example hard-negative softmax in the "
    "unweighted eval L2 space and a log radial shell penalty matching ||pred||^2 to ||tgt||^2. "
    "The calibration terms are detached-gated by current row retrieval confidence to preserve "
    "the champion ranking signal while breaking the row-softmax norm invariance."
)

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05

_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_CAL_TEMP = 0.25
_CAL_WEIGHT = 0.08
_NORM_WEIGHT = 0.16
_NORM_BETA = 0.25
_GATE_FLOOR = 0.20
_DUP_SEP = 0.03
_HARD_K = 128


def loss(pred, tgt):
    n, d = pred.shape

    mse_anchor = ((pred - tgt) ** 2).mean()

    pred_norm2 = (pred * pred).sum(dim=1)
    tgt_norm2 = (tgt * tgt).sum(dim=1).detach()
    log_ratio = torch.log(pred_norm2.clamp_min(1e-8)) - torch.log(tgt_norm2.clamp_min(1e-8))
    norm_rows = F.smooth_l1_loss(
        log_ratio, torch.zeros_like(log_ratio), beta=_NORM_BETA, reduction="none"
    )

    if n < 2:
        return mse_anchor + _NORM_WEIGHT * norm_rows.mean()

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
    dist_prec = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)

    logits = -dist_prec / _TEMP
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)

    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
        cal_gate = _GATE_FLOOR + (1.0 - _GATE_FLOOR) * p_true.sqrt()

    listwise = (focal * nll).mean()

    ps = (pred * pred).sum(dim=1, keepdim=True)
    ts = (tgt * tgt).sum(dim=1, keepdim=True)
    dist_cal = (ps + ts.t() - 2.0 * (pred @ tgt.t())).clamp_min(0.0) / float(d)
    logits_cal = -dist_cal / _CAL_TEMP
    eye = torch.eye(n, dtype=torch.bool, device=pred.device)

    with torch.no_grad():
        tt_cal = (ts + ts.t() - 2.0 * (tgt @ tgt.t())).clamp_min(0.0) / float(d)
        valid_neg = (~eye) & (tt_cal >= _DUP_SEP)

    off_logits = logits_cal[valid_neg]
    if off_logits.numel() > 0:
        k = min(_HARD_K, off_logits.numel())
        hard = off_logits.topk(k, largest=True).values
        hard_ref = torch.logsumexp(hard, dim=0) - hard.new_tensor(float(k)).log()
        xsoft_rows = F.softplus(hard_ref - logits_cal.diagonal())
        xsoft = (cal_gate * xsoft_rows).mean()
    else:
        xsoft = pred.new_zeros(())

    norm_penalty = (cal_gate * norm_rows).mean()

    return (
        listwise
        + _ANCHOR * mse_anchor
        + _CAL_WEIGHT * xsoft
        + _NORM_WEIGHT * norm_penalty
    )
