"""objective chunk: MSE + a cosine-alignment term.

Seed hypothesis: the retrieval metric ranks by MSE distance but also tracks cosine; adding a
directional term may sharpen the embedding direction without hurting the magnitude fit."""

import torch.nn.functional as F

NAME = "mse_cosine"
DESCRIPTION = "MSE + lambda * (1 - cosine similarity); lambda=0.5."
LAMBDA = 0.5


def loss(pred, tgt):
    return ((pred - tgt) ** 2).mean() + LAMBDA * (1.0 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
