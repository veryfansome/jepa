"""objective chunk: plain MSE — the R4 baseline loss.

Contract for any objective impl: expose `loss(pred, tgt) -> scalar tensor`, where
  pred: [n, D] the world model's predicted next-observation embeddings at command positions
  tgt : [n, D] the true (standardized) next-observation embeddings
Both are already the flattened cmd-position tensors for the training batch, so batch-level
objectives (contrastive/InfoNCE, variance regularizers) can be formed from them directly.
The loss must be a scalar torch tensor with grad. Keep it anti-collapse-safe: a constant
prediction should NOT minimize it (plain MSE is fine because tgt varies)."""

NAME = "mse"
DESCRIPTION = "R4 baseline: mean squared error to the standardized target embedding."


def loss(pred, tgt):
    return ((pred - tgt) ** 2).mean()
