"""OPTIM chunk: AdamW tuned for a small pre-LN transformer on frozen features under a FIXED,
convergence-limited step budget (proxy=1000, full=4000), with 6% linear warmup then cosine
decay to a nonzero LR floor.

Design (justified from the real setup, not defaults):
- Regime is UNDER-optimized, not over-regularized: recorded proxy->full jump (InfoNCE 0.39->0.47)
  says more optimization keeps helping at 4000 steps. So the optimizer should (a) reach the good
  basin faster and (b) NOT waste the final steps at lr~=0 -> cosine to a 0.1*peak FLOOR, not 0.
- beta2=0.98 (transformer-standard), not 0.999: at 0.999 the 2nd-moment memory (~1000 steps) is
  as long as the whole proxy run, so the preconditioner barely converges; 0.98 (~50-step memory)
  tracks the contrastive loss's curvature and adapts within budget. beta1=0.9, eps=1e-8 (features
  are standardized, grads are unit-scale after clip_grad_norm_(1.0)).
- peak lr 4e-4 (slightly above the 3e-4 constant baseline): the harness already clips grad-norm
  to 1.0 every step and the 6% warmup ramps in safely, so a modestly higher peak converts the
  fixed budget into faster convergence without divergence (NaN is hard-filtered anyway).
- Decoupled weight decay done RIGHT from the raw param iterator: wd=5e-2 on >=2-D weights
  (matmuls/embeddings) for generalization to unseen images, wd=0 on 1-D params (LayerNorm gains
  + biases) -- decaying norm gains would suppress signal in a pre-LN net. The contract passes a
  flat iterator, so we split by tensor ndim (the standard no-decay-on-norm/bias rule).

Contract: make(params, steps) -> (optimizer, scheduler). scheduler.step() is called once per
iteration by the harness. Pure/self-contained; torch.optim / lr_scheduler only.
"""
import math
import torch

NAME = "warmup_cosine_floor_beta98_decoupled"
DESCRIPTION = ("AdamW betas(0.9,0.98) eps1e-8, decoupled wd 5e-2 on >=2-D weights / 0 on "
               "norms+biases, peak lr 4e-4, 6% linear warmup then cosine decay to 0.1*peak floor.")


def make(params, steps, lr=4e-4, wd=5e-2, warmup_frac=0.06, floor_frac=0.1,
         betas=(0.9, 0.98), eps=1e-8):
    # Materialize once (net.parameters() is a generator) and split into decay / no-decay groups
    # by tensor rank: >=2-D = matmul/embedding weights (decayed); 1-D = LayerNorm gains + biases
    # (NOT decayed). Standard no-decay-on-norm/bias rule, achievable from the flat iterator.
    plist = [p for p in params if p.requires_grad]
    decay, no_decay = [], []
    for p in plist:
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": wd})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:  # degenerate safety
        groups = [{"params": plist, "weight_decay": wd}]

    opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)

    warm = max(1, int(warmup_frac * steps))

    def lr_lambda(step):
        # step is 0-indexed at first scheduler.step() call (called AFTER the first opt.step()).
        if step < warm:
            return (step + 1) / warm                      # linear warmup to 1.0
        p = (step - warm) / max(1, steps - warm)          # 0 -> 1 over the decay phase
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return floor_frac + (1.0 - floor_frac) * cos       # cosine from 1.0 down to floor_frac

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched
