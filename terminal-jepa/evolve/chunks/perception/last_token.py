"""perception variant: ModernBERT-base, full render, last_token pooling (findings 14/22 — the readout
is the graded lever: pooled content decodes 44.4% vs structured 100% on this encoder)."""
import torch
from evolve.chunks.perception.baseline import MODEL, render_obs, render_cmd
def pool(h, mask):  # last non-pad token state (bidirectional -> full-context summary)
    idx = mask.sum(1).long() - 1
    return h[torch.arange(h.shape[0], device=h.device), idx.clamp(min=0)]
