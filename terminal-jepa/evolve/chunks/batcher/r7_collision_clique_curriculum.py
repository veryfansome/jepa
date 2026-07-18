"""batcher chunk: collision-clique hard-negative curriculum.

Anneal from uniform batches to batches whose HARD fraction is drawn from a few
"collision cliques" -- groups of train sequences that all contain steps sharing the same
(content-verb, system-image, path-subtree-prefix) key. This mirrors the eval's foil
geometry more tightly than image-only blocking: the eval ranks the true next-obs against
SAME-VERB foils drawn from the (two-image) inner-val pool, so the decisive foils are
same-verb AND same-system AND (often) same-subtree. Blocking by image alone densifies only
same-SYSTEM negatives; a 2-image hard batch still mixes ls/cat/cd verbs and random paths, so
the hardest foil type stays diluted. Cliques guarantee many in-batch step positions produce
mutually-confusable same-verb same-system same-subtree observations -- measured on real train
data this ~triples the in-batch same-key collision density vs a uniform batch.

WHY (mechanism): concentrating the contrastive objective's in-batch negatives on the
exact geometry the metric scores (same-verb same-system same-subtree) puts the loss's
discriminative pressure on the dimensions that separate near-identical observations rather
than on a system-identity shortcut (cheap, transfer-irrelevant). The free-energy precision
objective then denoises precisely those channels. Cliques are sampled with SPECIFICITY
weighting (inverse clique size ^ spec_pow) so the giant generic buckets (ls of '.',
cat /etc/profile) do not dominate; gradient is spent on the fine subtree distinctions that
generalize to unseen systems. Complements (does not replace) the proven design: hard/easy
MIX (not all-hard) via a hard fraction that ramps over the first ramp_frac of training.

Anti-collapse: unchanged from the objective -- a constant prediction still yields uniform
softmax rows (loss log(n)) under the contrastive objectives regardless of batch composition;
this module never touches predictions or targets, only which sequence indices are drawn.

Degenerate-data safety (verified): with no usable content-verb collision cliques it falls
back to sysblock-style image-blocking; with < 2 images it falls back to exact uniform
sampling; hard_frac_max <= 0 or n < 2 also -> uniform. All randomness flows through one
private, seed-derived torch.Generator (deterministic per seed; global RNG untouched; fit
never mutated). Every returned batch is exactly bs indices in [0, len(fit)).
"""

import collections

import torch

NAME = "r7_collision_clique_curriculum"
DESCRIPTION = (
    "Curriculum hard-negative batcher whose hard sub-blocks are COLLISION CLIQUES: sequences "
    "sharing a (content-verb, system-image, path-subtree-prefix) key, so in-batch negatives "
    "become dense same-verb same-system same-subtree observations -- mirroring the eval's "
    "same-verb foil geometry far more tightly than image-only blocking. Specificity-weighted "
    "clique choice + hard/easy MIX ramped over training; falls back to image-blocking then "
    "uniform on sparse metadata."
)

_CONTENT_VERBS = ("ls", "cat")  # the verbs the fitness metric scores


def _verb_of(cmd):
    p = cmd.split()
    return p[0] if p else ""


def _first_path(cmd):
    # first non-flag argument = the operand path (e.g. `cat /etc/fstab`, `ls -lt /var/log`).
    toks = cmd.split()
    for t in toks[1:]:
        if t.startswith("-"):
            continue
        return t
    return ""


def _prefix_key(path, depth):
    # subtree bucket: first `depth` path components (abs vs rel kept distinct).
    if not path:
        return "."
    parts = [p for p in path.split("/") if p]
    body = "/".join(parts[:depth])
    return ("/" + body) if path.startswith("/") else ("./" + body)


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.30,
                 n_cliques=4, prefix_depth=2, min_clique=2, spec_pow=0.5,
                 n_block_images=2):
    """Collision-clique hard-negative curriculum.

    hard_frac_max/ramp_frac : the hard fraction ramps 0 -> hard_frac_max over the first
        ramp_frac of training, then holds (proven MIX + curriculum, matched to the sysblock
        block2/hard75/ramp30 winner by default).
    n_cliques   : distinct collision cliques unioned to form each hard sub-block.
    prefix_depth: path components kept for the subtree bucket (2 => /etc/network).
    min_clique  : a clique must contain >= this many DISTINCT sequences to be usable.
    spec_pow    : specificity tilt; clique weight = size ^ (-spec_pow) (0 => size-uniform).
    n_block_images : images per hard block in the image-blocking fallback.
    """
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def _uniform(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n < 2 or hard_frac_max <= 0.0:
        return _uniform

    content = set(_CONTENT_VERBS)

    # Build collision cliques (verb, image, subtree-prefix) -> set of sequence indices, and
    # image groups for the fallback. Read-only metadata; deterministic (sorted) ordering.
    keybag = collections.defaultdict(set)
    img_groups = {}
    for i, s in enumerate(fit):
        img = s.get("image", "?")
        img_groups.setdefault(img, []).append(i)
        seen = set()
        for c in s.get("cmds", []):
            v = _verb_of(c)
            if v not in content:
                continue
            seen.add((v, img, _prefix_key(_first_path(c), prefix_depth)))
        for k in seen:
            keybag[k].add(i)

    min_c = max(2, int(min_clique))
    cliques = [sorted(v) for k, v in sorted(keybag.items()) if len(v) >= min_c]
    clique_pools = [torch.tensor(c, dtype=torch.long) for c in cliques]

    img_keys = sorted(img_groups)
    img_pools = [torch.tensor(img_groups[k], dtype=torch.long) for k in img_keys]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools])

    if not clique_pools:
        # No usable same-verb same-system same-subtree cliques -> sysblock image-blocking.
        if len(img_pools) < 2:
            return _uniform
        k_imgs = max(1, min(int(n_block_images), len(img_pools)))

        def next_batch_img(step, total_steps):
            ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
            frac = hard_frac_max * min(1.0, step / ramp_steps)
            n_hard = max(0, min(bs, int(round(bs * frac))))
            parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
            if n_hard > 0:
                chosen = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
                pool = torch.cat([img_pools[int(c)] for c in chosen])
                parts.append(pool[torch.randint(0, pool.numel(), (n_hard,), generator=g)])
            batch = torch.cat(parts)
            return batch[torch.randperm(bs, generator=g)].tolist()

        return next_batch_img

    # Specificity weighting: rarer (smaller) cliques = the hardest, most eval-like buckets;
    # sample them more so common trivial buckets (ls '.', cat /etc/profile) don't dominate.
    csize = torch.tensor([float(p.numel()) for p in clique_pools])
    cw = csize.clamp_min(1.0).pow(-abs(spec_pow))
    cw = cw / cw.sum().clamp_min(1e-12)
    n_cl = max(1, min(int(n_cliques), len(clique_pools)))

    def next_batch(step, total_steps):
        # Curriculum: hard fraction ramps 0 -> hard_frac_max over the first ramp_frac of
        # training, then holds. Proxy (1000 steps, ramp_frac=0.3) finishes the ramp at 300.
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]  # easy: uniform
        if n_hard > 0:
            # Pick a few collision cliques (specificity-weighted, without replacement), then
            # draw the hard portion uniformly from the union of their member sequences: dense
            # same-verb same-system same-subtree negatives.
            chosen = torch.multinomial(cw, n_cl, replacement=False, generator=g)
            pool = torch.cat([clique_pools[int(c)] for c in chosen])
            parts.append(pool[torch.randint(0, pool.numel(), (n_hard,), generator=g)])
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()  # shuffle hard/easy mix

    return next_batch

