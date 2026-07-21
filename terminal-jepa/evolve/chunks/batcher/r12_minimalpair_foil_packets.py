"""batcher chunk: v2 minimal-pair foil-packet curriculum.

Keep the champion's proven batch STRUCTURE (uniform/easy part + a hard part drawn from ONE
block image, hard fraction ramping 0 -> hard_frac_max over ramp_frac of training), but
compose most of the hard block from MINIMAL PAIRS mined from the v2 collection policy's own
hard-negative machinery — collision structure that did not exist when the incumbent sysblock
batcher was designed (v1: 4 verbs, no head/tail/grep/find):

  * HT packets   — two sequences that ran head/tail on the SAME file of the SAME image with
                   DIFFERENT commands (head-vs-tail, or different global-K): observations
                   share file identity but differ in extent/end — exactly the same-verb foil
                   family where the wm is weakest (tail .717, margin +.347).
  * GREP packets — two sequences that grepped the SAME file with different (token, outcome):
                   polar keys (one hit + one intended-miss, recovered from the cached ok
                   flags) are preferred — the pair differs ONLY in whether the token occurs,
                   the purest possible negative for in-context which-lines-match reasoning
                   (grep margin +.326 vs a strong .513 within-traj lookup).
  * FIND packets — two sequences that ran find on the SAME root with different modifiers
                   (-maxdepth/-type/-name): same-subtree different-depth listings.
  * CAT echoes   — one sequence per DIFFERENT image cat-ing the SAME path (v2 config/lexicon
                   files repeat across distros; rbc .530 is the binding baseline): making
                   alpine's and debian's /etc/os-release mutual in-batch negatives puts the
                   contrastive pressure exactly on the SYSTEM-SPECIFIC content that the
                   cross-image corpus lookup cannot supply (cat margin +.293).

WHY (mechanism): the contrastive objectives rank each true next-obs against the other
observations IN THE BATCH; the fitness ranks it against same-verb foils inside a two-image
pool. Image-blocking (the incumbent) matches the foil geometry only at the SYSTEM level;
under v2 the frozen prereg foils are far harder — same file different K, hit vs miss, same
subtree different depth. A uniform or image-blocked batch realizes such collisions only by
chance; each packet GUARANTEES one, so every gradient step spends discriminative pressure on
the axes the metric actually scores. Packet members are drawn from DISTINCT command
signatures, so a pair is always a near-miss and never a duplicated observation (a duplicate
target is a false negative that corrupts the contrastive loss). Type budget is aimed at the
measured v2 margin ordering (tail < grep < cat weakest); the easy remainder and the ramp
keep the proven hard/easy MIX and early-training stability.

Anti-collapse: unchanged from the objective — this module never touches predictions or
targets, only which sequence indices are drawn; a constant prediction still yields uniform
softmax rows (loss log(n)) under the contrastive objectives regardless of composition.

Degenerate-data safety: no minable v2 keys -> the hard block degrades to exact sysblock
image-blocking; < 2 distinct images or hard_frac_max <= 0 -> exact uniform sampling.
Missing "ok" defaults to all-True (v1 caches). All randomness flows through one private,
seed-derived torch.Generator (deterministic per seed; global RNG untouched; fit never
mutated). Every returned batch is exactly bs indices in [0, len(fit)).
"""

import torch

NAME = "r12_minimalpair_foil_packets"
DESCRIPTION = (
    "Sysblock-shaped curriculum whose hard block is built from GUARANTEED minimal pairs "
    "mined from the v2 collision structure: same-file head-vs-tail/different-K packets, "
    "same-file grep hit-vs-miss packets (ok-flag polar pairs), same-root different-depth "
    "find packets, and cross-image same-path cat echoes — matching the frozen v2 foil "
    "geometry pair-by-pair instead of only at the system level."
)

_HT = ("head", "tail")


def _op_path(toks):
    # first non-flag argument = the operand path (v2 cat has no flags, but stay safe)
    for t in toks[1:]:
        if not t.startswith("-"):
            return t
    return ""


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.30,
                 packet_frac=0.60, cat_echo_frac=0.15, polar_pref=0.7,
                 w_ht=0.45, w_grep=0.35, w_find=0.20):
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

    # ---- mine v2 collision keys (one read-only pass over cmds/ok) ----
    # key -> {command signature -> [seq idx, ...]}. A key is USABLE iff it has >= 2 distinct
    # signatures: a drawn pair then differs in command/outcome, never a duplicate obs.
    ht_raw, grep_raw, find_raw, cat_raw = {}, {}, {}, {}
    for i, s in enumerate(fit):
        img = s.get("image", "?")
        cmds = s.get("cmds") or []
        oks = s.get("ok") or [True] * len(cmds)
        for t, cmd in enumerate(cmds):
            toks = cmd.split()
            if len(toks) < 2:
                continue
            v = toks[0]
            if v in _HT:                      # "head -n K path" -> key (img, path)
                ht_raw.setdefault((img, toks[-1]), {}).setdefault(cmd, []).append(i)
            elif v == "grep" and len(toks) >= 3:   # "grep -F -m 8 'tok' path"
                hit = bool(oks[t]) if t < len(oks) else True
                grep_raw.setdefault((img, toks[-1]), {}) \
                        .setdefault((cmd, hit), []).append(i)
            elif v == "find":                 # "find root -maxdepth d ... -name g"
                find_raw.setdefault((img, toks[1]), {}).setdefault(cmd, []).append(i)
            elif v == "cat":                  # cross-image echo: key = path, sig = image
                p = _op_path(toks)
                if p:
                    cat_raw.setdefault(p, {}).setdefault(img, []).append(i)

    def finalize(raw):
        # img -> list of usable keys; each key = list of member tensors (one per signature)
        out = {}
        for key in sorted(raw):
            sig_map = raw[key]
            if len(sig_map) < 2:
                continue
            sigs = [torch.tensor(sig_map[sg], dtype=torch.long)
                    for sg in sorted(sig_map, key=repr)]
            out.setdefault(key[0], []).append(sigs)
        return out

    ht_keys, find_keys = finalize(ht_raw), finalize(find_raw)
    grep_keys, grep_polar = {}, {}
    for key in sorted(grep_raw):
        sig_map = grep_raw[key]
        if len(sig_map) < 2:
            continue
        sigs = [torch.tensor(sig_map[sg], dtype=torch.long)
                for sg in sorted(sig_map, key=repr)]
        grep_keys.setdefault(key[0], []).append(sigs)
        if len({hit for (_, hit) in sig_map}) == 2:      # both hit and miss present
            grep_polar.setdefault(key[0], []).append(sigs)
    cat_echo = [[torch.tensor(sig_map[im], dtype=torch.long)
                 for im in sorted(sig_map)]
                for p, sig_map in ((p, cat_raw[p]) for p in sorted(cat_raw))
                if len(sig_map) >= 2]
    type_w = {"ht": float(w_ht), "grep": float(w_grep), "find": float(w_find)}

    def draw_pair(keys):
        # one key uniformly; two DISTINCT signatures; one member from each.
        key = keys[int(torch.randint(0, len(keys), (1,), generator=g))]
        a, b = torch.randperm(len(key), generator=g)[:2].tolist()
        out = []
        for sig in (key[a], key[b]):
            out.append(int(sig[int(torch.randint(0, sig.numel(), (1,), generator=g))]))
        return out

    def next_batch(step, total_steps):
        # Curriculum: hard fraction ramps 0 -> hard_frac_max over the first ramp_frac of
        # training, then holds (identical schedule to the incumbent sysblock champion).
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            ci = int(torch.multinomial(sizes, 1, generator=g))   # size-weighted block image
            img = img_names[ci]
            hard = []

            # -- minimal-pair packets inside the block image --
            budget = int(round(n_hard * packet_frac))
            avail = [(t, k) for t, k in (("ht", ht_keys.get(img)),
                                         ("grep", grep_keys.get(img)),
                                         ("find", find_keys.get(img))) if k]
            while len(hard) + 2 <= budget and avail:
                w = torch.tensor([type_w[t] for t, _ in avail])
                t, keys = avail[int(torch.multinomial(w, 1, generator=g))]
                if t == "grep":
                    pol = grep_polar.get(img)
                    if pol and float(torch.rand(1, generator=g)) < polar_pref:
                        keys = pol                    # prefer hit-vs-miss polar pairs
                hard.extend(draw_pair(keys))

            # -- cross-image cat echoes (same path, two different systems) --
            budget = len(hard) + int(round(n_hard * cat_echo_frac))
            while len(hard) + 2 <= min(budget, n_hard) and cat_echo:
                hard.extend(draw_pair(cat_echo))

            # -- fill the remainder from the block image (exact sysblock behavior) --
            rest = n_hard - len(hard)
            if rest > 0:
                pool = pools[img]
                hard.extend(pool[torch.randint(0, pool.numel(), (rest,),
                                               generator=g)].tolist())
            parts.append(torch.tensor(hard[:n_hard], dtype=torch.long))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()   # shuffle hard/easy mix

    return next_batch
