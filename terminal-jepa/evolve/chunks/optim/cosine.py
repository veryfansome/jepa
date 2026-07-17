"""AdamW + cosine LR annealing to 0 over the training run."""
import torch
NAME = "cosine"
DESCRIPTION = "AdamW lr 3e-4 wd 1e-4 with cosine annealing to 0 over all steps."
def make(params, steps, lr=3e-4, wd=1e-4):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
