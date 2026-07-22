"""batcher chunk: weak-verb cross-system clique curriculum (coarse-to-fine negatives).

Keep the champion sysblock curriculum EXACTLY as the outer schedule (uniform -> partially
image-blocked, hard fraction ramping over the first ramp_frac of training), then add a SECOND,
later ramp INSIDE the hard block: an increasing share of the hard block is composed of
SAME-PATH CROSS-SYSTEM CLIQUES anchored on the weak content verbs (cat/head/tail) — 3-4
sequences that all read the SAME file path on DIFFERENT Docker images, always including the
batch's block image when possible. The rest of the hard block stays image-blocked, but its
within-image fill is re-weighted toward sequences RICH in weak-verb steps.

WHY (mechanism): the standing diagnosis says cat's remaining margin lives in SYSTEM-VARIANT
file bodies — same path, different content per system — and the champion objective
(antiretrieval ring) takes its negatives from the batch, so batch composition IS the negative
distribution. Three failure-mode fixes over the twice-failed r12 minimal-pair batcher:
  1. CLIQUES, NOT PAIRS — a 2-member pair contributes 2 colliding observations among the
     ~bs*seq_len obs in the pool (sparse; the softmax barely moves). A clique puts 3-4
     system-variants of the SAME path in one batch, several cliques per batch, up to half the
     hard block late in training — collision as a BULK property, which is exactly how the
     winning sysblock batcher works at the system level.
  2. ONE FOIL FAMILY — all cross-system budget goes to the single family holding the stalled
     margin (system-variant file bodies), instead of splitting 4 ways across ht/grep/find
     packet types aimed at axes the fitness does not score.
  3. COARSE-TO-FINE SCHEDULING — Reverse Hierarchy Theory (Ahissar & Hochstein, TiCS 2004):
     perceptual learning consolidates coarse discriminations before fine ones transfer. Here
     system-level discrimination (the proven sysblock ramp) trains FIRST and is untouched;
     only after it completes does the fine content-level discrimination (same path, which
     system's body?) phase in — the hard-negative-hardness curriculum of Kalantidis et al.
     (NeurIPS 2020) realized purely by composition, with zero new parameters.
Block-image-inclusive cliques mean the block image's cat observation simultaneously has dense
same-system negatives (rest of the hard block) AND its own cross-system same-path variants
(clique partners) — the union of the eval's foil geometry, not a trade of one for the other.
The verb-richness fill weight (1 + #weak-verb steps)^rich_pow counters interference from ls
dominance in the step distribution (8580 ls vs 5910 cat steps in inner): class-balanced
sampling (Cui et al., CVPR 2019) applied at the sequence level.

Anti-collapse: unchanged from the objective — this module never touches predictions or
targets, only which sequence indices are drawn; a constant prediction still yields uniform
softmax rows (loss log(n)) under the contrastive objectives regardless of composition.

Degenerate-data safety: no cross-image path anchors -> the clique share stays 0 and the
behavior degrades to sysblock-with-richness-weighting; < 2 distinct images or
hard_frac_max <= 0 or n < 2 -> exact uniform sampling. All randomness flows through one
private, seed-derived torch.Generator (deterministic per seed; global RNG untouched; fit
never mutated). Every returned batch is exactly bs indices in [0, len(fit)).
"""

import torch

NAME = "r16_weakverb_xsys_clique_curriculum"
DESCRIPTION = (
    "Two-stage coarse-to-fine negative curriculum: the proven sysblock hard-fraction ramp "
    "runs first, then a growing share of the hard block becomes same-path cross-system "
    "cliques anchored on weak verbs (cat/head/tail) — 3-4 sequences reading the SAME file on "
    "DIFFERENT images, block-image-inclusive — making the antiretrieval ring's in-batch "
    "negatives exactly the system-variant file bodies where cat's margin is stalled; the "
    "image-blocked remainder is re-weighted toward weak-verb-rich sequences to counter ls "
    "step dominance."
)

_ANCHOR_VERBS = ("cat", "head", "tail")  # the file-body verbs; cat stalled, tail weakest


def _op_path(toks):
    # first non-flag argument = the operand path ("cat /etc/os-release", "head -n 5 p").
    for t in toks[1:]:
        if not t.startswith("-"):
            return t
    return ""


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.30,
                 xsys_frac_max=0.5, xsys_start=0.30, xsys_end=0.80,
                 packet_max=4, rich_pow=1.0):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    # ---- image pools (the sysblock backbone; sorted for determinism) ----
    groups = {}
    for i, s in enumerate(fit):
        groups.setdefault(s.get("image", "?"), []).append(i)
    img_names = sorted(groups)
    pools = {k: torch.tensor(groups[k], dtype=torch.long) for k in img_names}
    sizes = torch.tensor([float(pools[k].numel()) for k in img_names])

    if n < 2 or len(img_names) < 2 or hard_frac_max <= 0.0:
        # Degenerate: nothing to block on — exact uniform fallback.
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    # ---- mine weak-verb path anchors + per-sequence weak-verb richness (read-only) ----
    # anchor = a file path read (cat/head/tail) on >= 2 DISTINCT images; its clique members
    # are one sequence per image. n_weak[i] = weak-verb step count of sequence i.
    n_weak = [0] * n
    raw = {}
    for i, s in enumerate(fit):
        img = s.get("image", "?")
        for cmd in (s.get("cmds") or []):
            toks = cmd.split()
            if len(toks) < 2 or toks[0] not in _ANCHOR_VERBS:
                continue
            p = _op_path(toks)
            if not p:
                continue
            n_weak[i] += 1
            raw.setdefault(p, {}).setdefault(img, set()).add(i)

    anchors = []  # [(path, {img -> LongTensor of member seq indices})], sorted-deterministic
    for p in sorted(raw):
        by_img = raw[p]
        if len(by_img) >= 2:
            anchors.append((p, {im: torch.tensor(sorted(by_img[im]), dtype=torch.long)
                                for im in sorted(by_img)}))
    a_by_img = {im: [] for im in img_names}   # img -> anchor ids that include this image
    for ai, (_, by_img) in enumerate(anchors):
        for im in by_img:
            a_by_img[im].append(ai)
    # anchor weight = image multiplicity (more system variants = richer clique)
    a_w = torch.tensor([float(len(by_img)) for _, by_img in anchors]) if anchors else None
    a_w_img = {im: torch.tensor([float(len(anchors[ai][1])) for ai in a_by_img[im]])
               for im in img_names if a_by_img[im]}

    # within-image fill weights: oversample weak-verb-rich sequences inside the hard block
    rich = {im: (1.0 + torch.tensor([float(n_weak[int(i)]) for i in pools[im]]))
            ** float(rich_pow) for im in img_names}

    def draw_clique(img, budget):
        # One anchor (prefer anchors that include the block image, multiplicity-weighted),
        # then one member sequence per DISTINCT image — block image first, remaining images
        # in shuffled order — up to min(packet_max, budget) members.
        if img in a_w_img:
            lst = a_by_img[img]
            ai = lst[int(torch.multinomial(a_w_img[img], 1, generator=g))]
        else:
            ai = int(torch.multinomial(a_w, 1, generator=g))
        _, by_img = anchors[ai]
        others = [im for im in by_img if im != img]
        order = ([img] if img in by_img else []) \
            + [others[j] for j in torch.randperm(len(others), generator=g).tolist()]
        out = []
        for im in order[:max(1, min(int(packet_max), budget))]:
            mem = by_img[im]
            out.append(int(mem[int(torch.randint(0, mem.numel(), (1,), generator=g))]))
        return out

    def next_batch(step, total_steps):
        T = max(1, total_steps)
        # Outer curriculum: hard fraction ramps 0 -> hard_frac_max over the first ramp_frac
        # of training, then holds (identical schedule to the incumbent sysblock champion).
        ramp_steps = max(1, int(ramp_frac * T))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))
        # Inner (coarse-to-fine) curriculum: cross-system clique share of the hard block
        # ramps 0 -> xsys_frac_max between xsys_start*T and xsys_end*T, then holds.
        t = step / T
        if not anchors or t <= xsys_start:
            xw = 0.0
        elif t >= xsys_end:
            xw = 1.0
        else:
            xw = (t - xsys_start) / max(1e-9, xsys_end - xsys_start)
        n_x = min(n_hard, int(round(n_hard * xsys_frac_max * xw)))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            ci = int(torch.multinomial(sizes, 1, generator=g))  # size-weighted block image
            img = img_names[ci]
            hard = []
            while len(hard) < n_x:                       # same-path cross-system cliques
                hard.extend(draw_clique(img, n_x - len(hard)))
            n_blk = n_hard - len(hard)                   # image-blocked, richness-weighted
            if n_blk > 0:
                sel = torch.multinomial(rich[img], n_blk, replacement=True, generator=g)
                hard.extend(pools[img][sel].tolist())
            parts.append(torch.tensor(hard[:n_hard], dtype=torch.long))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()  # shuffle hard/easy mix

    return next_batch
