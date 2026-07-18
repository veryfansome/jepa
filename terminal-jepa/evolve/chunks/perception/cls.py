"""perception variant: ModernBERT-base, full render, cls pooling (findings 14/22 — the readout
is the graded lever: pooled content decodes 44.4% vs structured 100% on this encoder)."""
import torch
from evolve.chunks.perception.baseline import MODEL, render_obs, render_cmd
def pool(h, mask):  # first (CLS-position) token state
    return h[:, 0]
