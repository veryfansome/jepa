"""batcher chunk: system-variant anchor-cohort packing (interleaved cross-system same-path).

Keep the champion sysblock backbone EXACTLY (hard/easy mix, single-block-image hard draw,
hard fraction ramping 0 -> hard_frac_max over ramp_frac of training), but convert part of
the hard block into SYSTEM-VARIANT COHORTS: pick a FEW anchor paths (absolute operand paths
of the scored content verbs) that are visited on MANY Docker images, and fill the cohort
round-robin across the hosting images — at most one member sequence per (anchor, image) per
cycle. The flattened loss pools every cmd position of the batch as mutual negatives, so a
cohort makes the SAME path's observation on k different systems co-occur as k mutually
confusable target rows — precisely the system-variant file bodies (alpine's vs debian's
/etc/os-release, ls of the same /etc across distros) where the stalled cat margin lives,
because retrieve-by-cmd's candidate is exactly another system's version of the file.

WHY THIS SERVES THE CURRENT CHAMPION OBJECTIVE (antiretrieval ring): the ring band-passes
in-batch negatives by target-target distance — up-weights close-but-distinct pairs, gates
out near-identical ones as false negatives. Cross-system versions of one path are the
canonical close-but-distinct pair (same file schema, system-specific content), so cohort
rows land in the ring's pass-band and receive the repulsion hinge; same-system re-reads of
the path are near-identical (same right answer) and the dupmask removes them — packing and
objective are complementary by construction. Round-robin one-per-image maximizes pass-band
mass and minimizes gated duplicate mass per drawn index.

WHY THE FAILED R12 minimalpair BATCHER DOESN'T REFUTE THIS: its cross-image cat echoes were
(a) only ~15% of the hard block, diluted among ht/grep/find packet types aimed at verbs the
fitness does not score, and (b) drawn as isolated PAIRS — one cross-system negative per
anchor path, a vanishing fraction of the ~bs*seq_len flattened rows after the ring's row
normalization. Here the WHOLE variant budget concentrates on n_anchors (default 2) paths as
many-image cohorts, so each anchor contributes a clique of mutually-variant rows, not one
pair — the same blocked->cohort concentration that made sysblock beat uniform, applied one
level deeper (path-concentrated, system-diverse: the exact dual of sysblock's
system-concentrated, path-diverse hard block, which is retained for the other half).

CROSS-DOMAIN MECHANISM (category learning): the discriminative-contrast hypothesis
(Kornell & Bjork 2008; Birnbaum et al. 2013) — INTERLEAVING exemplars of confusable
categories beats blocking because temporal juxtaposition exposes between-category
differences. For an in-batch contrastive loss "juxtaposition" is literal (negatives = the
batch), and the confusable categories are the per-system variants of one path; blocking by
system (the incumbent alone) never juxtaposes them. High inter-category confusability is
the regime where interleaving wins — matching the diagnosis that cat's residual margin
lives in system-variant bodies.

Anti-collapse: unchanged from the objective — this module never touches predictions or
targets; a constant prediction still yields uniform softmax rows under the contrastive
objectives regardless of composition.

Degenerate-data safety: no multi-image anchor paths -> the variant budget is 0 and the
module reproduces the sysblock champion's draw stream exactly (same generator calls in the
same order); < 2 distinct images or hard_frac_max <= 0 -> exact uniform sampling. All
randomness flows through one private, seed-derived torch.Generator (deterministic per
seed; global RNG untouched; fit never mutated). Every batch is exactly bs indices in
[0, len(fit)).
"""

import torch

NAME = "r16_sysvariant_anchor_cohorts"
DESCRIPTION = (
    "Sysblock-shaped curriculum whose hard block is split between the champion's "
    "single-image system block and SYSTEM-VARIANT ANCHOR COHORTS: a few absolute "
    "content-verb paths hosted by many images, filled round-robin one-sequence-per-image, "
    "so the same path's per-system observations co-occur as dense in-batch negatives — "
    "exactly the close-but-distinct pairs the antiretrieval ring objective up-weights and "
    "the system-variant foils behind the stalled cat margin."
)

_DEF_VERBS = ("cat", "ls")


def _op_path(toks):
    # first non-flag argument = the operand path (v2 cat has no flags, but stay safe)
    for t in toks[1:]:
        if not t.startswith("-"):
            return t
    return ""


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.30,
                 n_block_images=1, variant_frac=0.5, n_anchors=2,
                 min_images=2, anchor_verbs=_DEF_VERBS):
    """variant_frac of the hard block = anchor cohorts; the rest = exact sysblock draw.

    n_anchors    : distinct anchor paths per batch (few => dense cliques, not thin pairs).
    min_images   : an anchor path must be visited on >= this many distinct images.
    anchor_verbs : verbs whose absolute operand paths define anchors (the scored content
                   verbs; cat is the diagnosed stalled lever, ls shares the geometry).
    """
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    # ---- image pools (the sysblock backbone; sorted for determinism) ----
    groups = {}
    for i, s in enumerate(fit):
        groups.setdefault(s.get("image", "?"), []).append(i)
    pools = [torch.tensor(groups[k], dtype=torch.long) for k in sorted(groups)]
    sizes = torch.tensor([float(p.numel()) for p in pools])

    def _uniform(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n < 2 or len(pools) < 2 or hard_frac_max <= 0.0:
        return _uniform

    k_imgs = max(1, min(int(n_block_images), len(pools)))

    # ---- mine anchors: absolute content-verb operand path -> hosting image -> members ----
    verbs = set(anchor_verbs)
    raw = {}
    for i, s in enumerate(fit):
        img = s.get("image", "?")
        for cmd in (s.get("cmds") or []):
            toks = cmd.split()
            if len(toks) < 2 or toks[0] not in verbs:
                continue
            p = _op_path(toks)
            if p.startswith("/"):  # absolute only: relative paths are cwd-ambiguous keys
                raw.setdefault(p, {}).setdefault(img, set()).add(i)

    anchors, weights = [], []
    need = max(2, int(min_images))
    for p in sorted(raw):
        by_img = raw[p]
        if len(by_img) < need:
            continue
        anchors.append([torch.tensor(sorted(by_img[k]), dtype=torch.long)
                        for k in sorted(by_img)])
        weights.append(float(len(by_img) - 1))  # cross-system richness of the anchor
    aw = torch.tensor(weights) if anchors else None

    def draw_cohort(budget):
        # A few anchor paths, each filled ROUND-ROBIN over its hosting images (at most one
        # member per image per cycle): maximal cross-system variant density, minimal
        # same-image duplicate mass (which the ring objective would gate out anyway).
        k = max(1, min(int(n_anchors), len(anchors)))
        chosen = torch.multinomial(aw, k, replacement=False, generator=g).tolist()
        out = []
        for j, ai in enumerate(chosen):
            share = budget // k + (1 if j < budget % k else 0)
            imgs = anchors[ai]
            order = torch.randperm(len(imgs), generator=g).tolist()
            c = 0
            while share > 0:
                pool = imgs[order[c % len(imgs)]]
                out.append(int(pool[int(torch.randint(0, pool.numel(), (1,),
                                                      generator=g))]))
                share -= 1
                c += 1
        return out

    def next_batch(step, total_steps):
        # Curriculum: hard fraction ramps 0 -> hard_frac_max over the first ramp_frac of
        # training, then holds (identical schedule to the incumbent sysblock champion).
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            hard = []
            if anchors:
                n_var = max(0, min(n_hard, int(round(n_hard * variant_frac))))
                if n_var > 0:
                    hard.extend(draw_cohort(n_var))
            rest = n_hard - len(hard)
            if rest > 0:
                # exact sysblock draw for the remainder of the hard block
                chosen = torch.multinomial(sizes, k_imgs, replacement=False, generator=g)
                pool = torch.cat([pools[int(c)] for c in chosen])
                hard.extend(pool[torch.randint(0, pool.numel(), (rest,),
                                               generator=g)].tolist())
            parts.append(torch.tensor(hard, dtype=torch.long))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()  # shuffle hard/easy mix

    return next_batch
