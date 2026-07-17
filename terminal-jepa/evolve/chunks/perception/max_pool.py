"""perception variant: ModernBERT-base, full render, max_pool pooling (findings 14/22 — the readout
is the graded lever: pooled content decodes 44.4% vs structured 100% on this encoder)."""
import torch
from evolve.chunks.perception.baseline import MODEL, render_obs, render_cmd
def pool(h, mask):  # max over non-pad token states
    return h.masked_fill(mask.unsqueeze(-1) == 0, float("-inf")).max(1).values
