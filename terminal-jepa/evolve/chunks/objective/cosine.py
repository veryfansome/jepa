"""objective chunk: cosine-distance loss (directional target, magnitude-free)."""

import torch.nn.functional as F

NAME = "cosine"
DESCRIPTION = "1 - cosine similarity to the target embedding (direction only, ignores norm)."


def loss(pred, tgt):
    return (1.0 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
