"""perception TIER-2 encoder swap: BAAI/bge-base-en-v1.5, a top retrieval-tuned 768-d encoder.
bge's standard readout is the [CLS] token."""
from evolve.chunks.perception.baseline import render_obs, render_cmd
MODEL = "BAAI/bge-base-en-v1.5"
def pool(h, mask):
    return h[:, 0]  # CLS token (bge standard)
