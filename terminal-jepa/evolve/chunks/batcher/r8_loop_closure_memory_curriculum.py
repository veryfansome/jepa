"""batcher chunk: loop-closure-weighted sysblock hard-negative curriculum.

Keep the champion's proven batch STRUCTURE (uniform/easy part + image-blocked hard part,
hard fraction ramping 0 -> hard_frac_max over ramp_frac of training — the composition whose
same-system negative density matches the eval's same-verb same-system foils), and add ONE
new, annealed lever aimed at the CURRENT champion arch: oversample sequences with high
LOOP-CLOSURE density early in training.

WHY (mechanism, designed for r7_path_delta_fastweights): the arch writes each completed
(cmd, obs) pair into per-sequence delta-rule associative memories and reads them back at
later command positions. Those read/write/decay gates receive useful gradient ONLY from
sequences whose later commands actually resemble earlier ones — i.e. sequences with
"loop closures" in the SLAM sense (the trajectory re-queries previously visited keys).
On a sequence with no internal recurrence the memory read is unsupported noise and the
gradient can only teach the gates to shut. Neural evidence points the same way: networks
learn WHEN to retrieve from episodic memory only when the training distribution contains
recurring items (Lu, Hasson & Norman, eLife 2022). We therefore score each train sequence
once, read-only, by

    rev_i = mean over steps t>=1 of max_{t'<t} cos(z_cmd[t], z_cmd[t'])   (min-max
    normalized over fit to [0,1])

— i.e. how much support later memory queries have in that sequence's own past, measured in
the very (frozen cmd-embedding) space the arch's content_read/content_write keys project
from — and draw BOTH batch parts multinomially with weight exp(beta * rev_i), where beta
anneals beta0 -> 0 over the first warm_frac of training. Early on (while the hard fraction
is still ramping and the loss is easy) the fastweight gates train on memory-informative,
closure-rich trajectories; by mid-training beta = 0 and the sampling distribution is exactly
the incumbent's (uniform within each part), so the eval-matched hard-negative geometry at
steady state is untouched. This complements the target-space memory instead of duplicating
it: no other chunk controls WHICH trajectories the memory machinery learns under.

Anti-collapse: unchanged from the objective — this module never touches predictions or
targets, only which sequence indices are drawn; a constant prediction still yields uniform
softmax rows under the contrastive objectives regardless of composition.

Degenerate-data safety: missing/short z_cmd or constant closure scores -> weights collapse
to uniform; < 2 distinct images (or hard_frac_max <= 0) -> no image blocking; both
degenerate -> exact uniform sampling. All randomness flows through one private,
seed-derived torch.Generator (deterministic per seed; global RNG untouched; fit never
mutated). Every returned batch is exactly bs indices in [0, len(fit)).
"""

import torch

NAME = "r8_loop_closure_memory_curriculum"
DESCRIPTION = (
    "Sysblock hard-negative curriculum (champion structure/defaults) whose draws are, early "
    "in training, exponentially tilted toward sequences with high loop-closure density — "
    "later commands cosine-similar to earlier ones in the frozen cmd space — so the "
    "fastweight arch's delta-rule read/write gates train on memory-informative trajectories; "
    "the tilt anneals to zero so the steady-state negative geometry is exactly the incumbent's."
)


def _loop_closure_scores(fit):
    """Per-sequence loop-closure density, min-max normalized to [0,1] over fit.

    rev_i = mean over steps t>=1 of max_{t'<t} cos(z_cmd[t], z_cmd[t']): the average support
    a later memory query has in the sequence's own past. Deterministic, read-only, cheap
    (n_seq small matmuls at make_batcher time)."""
    n = len(fit)
    rev = torch.zeros(n)
    for i, s in enumerate(fit):
        zc = s.get("z_cmd", None)
        if zc is None or not torch.is_tensor(zc) or zc.dim() != 2 or zc.shape[0] < 2:
            continue
        z = zc.detach().float()
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sim = z @ z.T                                     # [T,T] cosine, in [-1,1]
        T = sim.shape[0]
        past = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-1)
        sim = sim.masked_fill(~past, float("-inf"))
        best = sim[1:].max(dim=1).values                  # [T-1] max sim to any earlier cmd
        rev[i] = best.mean().nan_to_num(0.0)
    lo, hi = float(rev.min()), float(rev.max())
    if hi - lo < 1e-8:
        return torch.zeros(n)                             # constant -> uniform weighting
    return (rev - lo) / (hi - lo)


def make_batcher(fit, bs, seed, n_block_images=2, hard_frac_max=0.75, ramp_frac=0.30,
                 beta0=1.5, warm_frac=0.5):
    """Loop-closure-weighted sysblock hard-negative curriculum.

    n_block_images/hard_frac_max/ramp_frac : the champion's proven composition knobs
        (defaults = block2/hard75/ramp30, the incumbent's winning params).
    beta0     : initial tilt strength; draw weight = exp(beta * rev_i) with rev in [0,1],
                so the max/min oversampling ratio starts at e^beta0 (~4.5 at 1.5) — a bias,
                never a hard filter.
    warm_frac : beta anneals linearly beta0 -> 0 over the first warm_frac of training;
                afterwards sampling is uniform-within-part (incumbent distribution).
    """
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    rev = _loop_closure_scores(fit)                       # [n] in [0,1]; zeros if degenerate
    weighted = bool((rev > 0).any()) and beta0 > 0.0

    # Image groups for the hard (system-blocked) part (read-only metadata; sorted order).
    groups = {}
    for i, s in enumerate(fit):
        groups.setdefault(s.get("image", "?"), []).append(i)
    pools = [torch.tensor(groups[k], dtype=torch.long) for k in sorted(groups)]
    sizes = torch.tensor([float(p.numel()) for p in pools])
    blocking = len(pools) >= 2 and hard_frac_max > 0.0

    if not blocking and not weighted:
        # Fully degenerate: exact uniform fallback.
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    k_imgs = max(1, min(int(n_block_images), len(pools)))

    def _draw(k, pool, beta):
        """k indices from `pool` (None = all of fit), tilted by exp(beta * rev)."""
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        if not weighted or beta <= 1e-6:
            if pool is None:
                return torch.randint(0, n, (k,), generator=g)
            return pool[torch.randint(0, pool.numel(), (k,), generator=g)]
        w = torch.exp(beta * (rev if pool is None else rev[pool]))
        idx = torch.multinomial(w, k, replacement=True, generator=g)
        return idx if pool is None else pool[idx]

    def next_batch(step, total_steps):
        # Hard-fraction curriculum: 0 -> hard_frac_max over the first ramp_frac of training,
        # then holds (identical schedule to the incumbent).
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = (hard_frac_max * min(1.0, step / ramp_steps)) if blocking else 0.0
        n_hard = max(0, min(bs, int(round(bs * frac))))

        # Loop-closure tilt: beta0 -> 0 over the first warm_frac of training, then exactly 0
        # (steady-state distribution == incumbent's uniform-within-part draws).
        warm_steps = max(1, int(warm_frac * max(1, total_steps)))
        beta = beta0 * max(0.0, 1.0 - (step - 1) / warm_steps)

        parts = [_draw(bs - n_hard, None, beta)]          # easy part: tilted-uniform over fit
        if n_hard > 0:
            # Pick a few images (size-weighted, without replacement) exactly as the
            # incumbent does, then tilted draw within their union.
            chosen = torch.multinomial(sizes, k_imgs, replacement=False, generator=g)
            pool = torch.cat([pools[int(c)] for c in chosen])
            parts.append(_draw(n_hard, pool, beta))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()  # shuffle hard/easy mix

    return next_batch
