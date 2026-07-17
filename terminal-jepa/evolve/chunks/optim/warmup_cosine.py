"""AdamW + linear warmup (first 5% of steps) then cosine decay to 0."""
import math, torch
NAME = "warmup_cosine"
DESCRIPTION = "AdamW lr 3e-4 wd 1e-4, 5% linear warmup then cosine decay to 0."
def make(params, steps, lr=3e-4, wd=1e-4, warmup_frac=0.05):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    warm = max(1, int(warmup_frac * steps))
    def lr_lambda(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
