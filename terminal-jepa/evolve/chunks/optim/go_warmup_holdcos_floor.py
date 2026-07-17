"""AdamW + short warmup, hold near-peak, then cosine decay to a nonzero floor.

Schedule shape (fraction of total steps):
  [0, w)        linear warmup 0 -> peak         (w = 4% of steps, min 20)
  [w, w+h)      HOLD at peak                     (h = 30% of steps)
  [w+h, steps]  cosine peak -> floor            (floor = peak/20, never 0)

Rationale for a ~2M-param pre-LN transformer on ~3200 seqs (many epochs), fixed step budget:
- Short warmup: pre-LN + grad-clip(1.0) tolerate a higher peak LR (5e-4 > baseline 3e-4);
  warmup just avoids the first-step attention/embedding blow-up.
- Hold: "more training helps a lot" here means we are convergence-limited, so keep the
  effective LR near peak through the middle of the run instead of cosine's immediate decay.
- Cosine-to-FLOOR (not 0): plain cosine spends its last ~10% of steps at ~0 LR (dead steps).
  A small floor keeps the contrastive L2-InfoNCE objective refining the minimum through the
  tail, converting otherwise-wasted budget into a sharper, better-generalizing solution.
- beta2=0.95 (vs 0.999): short/noisy contrastive run adapts faster; standard for small-data,
  short-horizon transformer training. Higher wd (5e-4) regularizes the over-epoched small net.

Contract: make(params, steps, **kw) -> (optimizer, LambdaLR). Scheduler .step()'d once per
iteration by the harness. Pure/self-contained; only torch.optim + lr_scheduler; no file/state.
"""
import math
import torch

NAME = "warmup_holdcos_floor"
DESCRIPTION = ("AdamW(lr 5e-4, wd 5e-4, betas (0.9,0.95)); 4% warmup, 30% hold at peak, "
               "then cosine decay to floor=peak/20 (never 0).")


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta2=0.95):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    warm = max(20, int(warmup_frac * steps))
    hold = int(hold_frac * steps)
    decay_start = warm + hold
    decay_len = max(1, steps - decay_start)

    def lr_lambda(step):  # returns multiplier in [floor_ratio, 1.0]
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / decay_len          # 0 -> 1 over the decay phase
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))  # 1 -> 0
        return floor_ratio + (1.0 - floor_ratio) * cos       # 1 -> floor_ratio

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
