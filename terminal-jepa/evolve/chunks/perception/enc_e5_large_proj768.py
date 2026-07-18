"""perception Tier-2: intfloat/e5-large-v2 (1024-d, the larger retrieval-tuned e5) adapted to the
hard-wired D=768 via a FIXED, deterministic, data-independent orthonormal projection: P is the
Q-factor of a seeded standard-gaussian [1024, 768] (torch.Generator seed 0, CPU), computed once at
module import. Orthonormal columns make the map a near-isometry on the 768-d image (Johnson-
Lindenstrauss style), preserving the L2 geometry the eval ranks by; no learned params, nothing
fitted on our data. Render/prefix identical to enc_e5_base ('passage: ' both sides)."""

import torch

from evolve.chunks.perception.enc_e5_base import render_obs, render_cmd  # noqa: F401 (same render)

MODEL = "intfloat/e5-large-v2"

_g = torch.Generator().manual_seed(0)
_P, _ = torch.linalg.qr(torch.randn(1024, 768, generator=_g, dtype=torch.float64))
_P = _P.float()  # [1024, 768], orthonormal columns


def pool(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    mean = (h * m).sum(1) / m.sum(1).clamp(min=1)          # [B, 1024] e5 mean-pool
    return mean @ _P.to(mean.device, mean.dtype)           # [B, 768] fixed orthonormal projection
