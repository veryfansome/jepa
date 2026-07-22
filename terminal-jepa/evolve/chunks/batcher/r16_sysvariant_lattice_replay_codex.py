"""batcher chunk: system-variant lattice replay.

Build hard batches as balanced contrast assemblies: same normalized path and command view,
several Docker images, one sequence per image. The champion objective's negatives are the
other observations in the batch; this makes same-path cross-system file bodies co-occur as
retrieval-confusable negatives instead of appearing only by chance. Keys are selected by a
train-only utility score: high between-system latent separation, low within-system volatility
(to avoid hostname/container noise), close-but-distinct ring mass, and mild path specificity.

Why it may raise the margin: retrieve-by-command is strongest on cat because the command text
points to corpus-common paths whose body changes by system. This batcher supplies those
system-variant foils directly to the in-batch antiretrieval ring, while preserving the proven
hard/easy curriculum and a sysblock fallback for sparse or early training.
"""

import collections
import math
import shlex

import torch

NAME = "r16_sysvariant_lattice_replay_codex"
DESCRIPTION = (
    "System-variant contrast-lattice batcher: anneal from uniform to hard batches packed with "
    "same-path same-view content commands across several Docker images, using train-only latent "
    "between/within statistics to prefer stable system-specific files and avoid volatile false "
    "negatives; residual hard budget remains image-cohort blocked."
)

_CONTENT_VERBS = ("cat", "ls", "head", "tail", "find", "grep")
_VERB_WEIGHT = {
    "cat": 1.65,
    "ls": 1.10,
    "tail": 0.90,
    "head": 0.80,
    "grep": 0.65,
    "find": 0.55,
}


def _split(cmd):
    try:
        return shlex.split(str(cmd))
    except Exception:
        return str(cmd).split()


def _clean_path(path):
    return (path or "").strip().strip("'\"")


def _norm_path(path):
    path = _clean_path(path)
    if not path:
        return "."
    if path.startswith("~"):
        return path
    absolute = path.startswith("/")
    parts = []
    for p in path.split("/"):
        if not p or p == ".":
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    body = "/".join(parts)
    if absolute:
        return "/" + body if body else "/"
    return body or "."


def _join_path(cwd, path):
    path = _clean_path(path)
    if not path or path in (".", "./"):
        return _norm_path(cwd or "/")
    if path.startswith("/") or path.startswith("~"):
        return _norm_path(path)
    base = _norm_path(cwd or "/")
    return _norm_path(("/" + path) if base == "/" else (base.rstrip("/") + "/" + path))


def _operands(toks):
    out = []
    skip = False
    takes_value = {
        "-n", "--lines", "-m", "--max-count", "-A", "-B", "-C",
        "-e", "-f", "--exclude", "--include", "-name", "-type", "-maxdepth",
        "-mindepth", "-path",
    }
    for t in toks[1:]:
        if skip:
            skip = False
            continue
        if t in takes_value:
            skip = True
            continue
        if t.startswith("-"):
            continue
        out.append(t)
    return out


def _flagsig(toks, path_pos=None):
    vals = []
    skip = False
    takes_value = {"-n", "--lines", "-m", "--max-count", "-A", "-B", "-C", "-maxdepth", "-mindepth"}
    for i, t in enumerate(toks[1:], start=1):
        if path_pos is not None and i == path_pos:
            continue
        if skip:
            vals.append("<v>")
            skip = False
            continue
        if t in takes_value:
            vals.append(t)
            skip = True
        elif t.startswith("-"):
            vals.append(t)
    return " ".join(vals) or "<plain>"


def _content_key(cmd, cwd, ok):
    toks = _split(cmd)
    if not toks:
        return None, cwd

    v = toks[0]
    if v == "cd":
        args = _operands(toks)
        return None, _join_path(cwd, args[0] if args else "/")

    if v not in _CONTENT_VERBS or not bool(ok):
        return None, cwd

    if v == "cat":
        args = _operands(toks)
        if not args:
            return None, cwd
        return (v, _join_path(cwd, args[-1]), "cat"), cwd

    if v == "ls":
        args = _operands(toks)
        p = args[-1] if args else "."
        return (v, _join_path(cwd, p), _flagsig(toks)), cwd

    if v in ("head", "tail"):
        args = _operands(toks)
        if not args:
            return None, cwd
        return (v, _join_path(cwd, args[-1]), _flagsig(toks)), cwd

    if v == "grep":
        if len(toks) < 3:
            return None, cwd
        path = _join_path(cwd, toks[-1])
        sig = " ".join(toks[1:-1]) or "<grep>"
        return (v, path, sig), cwd

    if v == "find":
        args = _operands(toks)
        root = args[0] if args else "."
        sig = " ".join(toks[2:]) if len(toks) > 2 else "<find>"
        return (v, _join_path(cwd, root), sig), cwd

    return None, cwd


def _specificity(key):
    verb, path, _sig = key
    if path in ("", ".", "/"):
        return 0.20 if verb == "ls" else 0.35
    depth = len([p for p in path.split("/") if p])
    base = 0.55 + 0.12 * min(depth, 6)
    if verb == "cat":
        base += 0.25
    elif verb in ("head", "tail"):
        base += 0.10
    return max(0.25, min(1.35, base))


def _smooth01(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def make_batcher(
    fit,
    bs,
    seed,
    hard_frac_max=0.75,
    ramp_frac=0.30,
    variant_frac_max=0.78,
    cohort_images=4,
    assembly_width=4,
    min_systems=3,
    n_block_images=1,
    cooldown=64,
    ring_lam_frac=0.55,
    dup_delta_frac=0.045,
    min_reliability=0.18,
    posting_pow=0.08,
):
    n = len(fit)
    bs = int(bs)
    g = torch.Generator().manual_seed(int(seed))

    def uniform(_step, _total_steps):
        if bs <= 0:
            return []
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if bs <= 0:
        return uniform
    if n < 2 or hard_frac_max <= 0.0:
        return uniform

    img_groups = collections.defaultdict(list)
    img_of = []
    raw = collections.defaultdict(dict)

    for i, s in enumerate(fit):
        img = s.get("image", "?")
        img_of.append(img)
        img_groups[img].append(i)
        cmds = s.get("cmds") or ()
        oks = s.get("ok") or [True] * len(cmds)
        zobs = s.get("z_obs", None)
        cwd = "/"
        for t, cmd in enumerate(cmds):
            ok = oks[t] if t < len(oks) else True
            key, cwd = _content_key(cmd, cwd, ok)
            if key is None or not torch.is_tensor(zobs) or t >= zobs.shape[0]:
                continue
            z = torch.nan_to_num(zobs[t].detach().float().cpu(), nan=0.0, posinf=1e4, neginf=-1e4)
            e = raw[key].setdefault(img, {"idx": [], "n": 0, "sum": None, "msq": 0.0})
            if e["sum"] is None:
                e["sum"] = torch.zeros_like(z)
            e["idx"].append(i)
            e["n"] += 1
            e["sum"].add_(z)
            e["msq"] += float((z * z).mean().item())

    img_names = sorted(img_groups)
    img_pools = [torch.tensor(img_groups[k], dtype=torch.long) for k in img_names]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools])
    img_id = {k: i for i, k in enumerate(img_names)}
    all_idx = torch.arange(n, dtype=torch.long)

    temps, all_off = [], []
    min_systems = max(2, int(min_systems))
    for key in sorted(raw):
        entries = []
        for img in sorted(raw[key]):
            e = raw[key][img]
            if e["n"] <= 0 or not e["idx"]:
                continue
            center = e["sum"] / float(e["n"])
            intra = max(0.0, e["msq"] / float(e["n"]) - float((center * center).mean().item()))
            entries.append((img, center, intra, torch.tensor(e["idx"], dtype=torch.long)))
        if len(entries) < min_systems:
            continue
        centers = torch.stack([e[1] for e in entries])
        dist = ((centers[:, None, :] - centers[None, :, :]) ** 2).mean(-1).clamp_min(0.0)
        eye = torch.eye(dist.shape[0], dtype=torch.bool)
        off = dist[~eye]
        if off.numel() == 0 or float(off.mean().item()) <= 1e-12:
            continue
        temps.append((key, entries, dist, off))
        all_off.append(off)

    if not temps or len(img_pools) < 2:
        if len(img_pools) < 2:
            return uniform
        k_imgs = max(1, min(int(n_block_images), len(img_pools)))

        def next_batch_sysblock(step, total_steps):
            ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
            frac = float(hard_frac_max) * min(1.0, float(step) / float(ramp_steps))
            n_hard = max(0, min(bs, int(round(bs * frac))))
            parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
            if n_hard > 0:
                chosen = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
                pool = torch.cat([img_pools[int(c)] for c in chosen])
                parts.append(pool[torch.randint(0, pool.numel(), (n_hard,), generator=g)])
            batch = torch.cat(parts)
            return batch[torch.randperm(bs, generator=g)].tolist()

        return next_batch_sysblock

    scale = float(torch.cat(all_off).median().item())
    scale = max(scale, 1e-6)
    lam = max(1e-6, float(ring_lam_frac) * scale)
    dup_delta = max(1e-6, float(dup_delta_frac) * scale)

    assemblies = []
    weights = []
    for key, entries, dist, off in temps:
        ring_full = torch.exp(-dist / lam) * (1.0 - torch.exp(-dist / dup_delta))
        ring_full = ring_full.masked_fill(torch.eye(dist.shape[0], dtype=torch.bool), 0.0)
        ring = float(ring_full.sum().item() / max(1, dist.shape[0] * (dist.shape[0] - 1)))
        between = float(off.mean().item())
        within = sum(float(e[2]) for e in entries) / max(1, len(entries))
        reliability = between / (between + within + 1e-8)
        if reliability < float(min_reliability):
            continue
        total_post = sum(int(e[3].numel()) for e in entries)
        score = (
            _VERB_WEIGHT.get(key[0], 0.5)
            * (len(entries) ** 0.55)
            * _specificity(key)
            * (ring + 0.03)
            * reliability
            * (max(1, total_post) ** (-abs(float(posting_pow))))
        )
        if not math.isfinite(score) or score <= 0.0:
            continue
        img_w = ring_full.sum(1).clamp_min(0.0) + 0.05
        assemblies.append({
            "key": key,
            "pools": [e[3] for e in entries],
            "img_ids": [img_id[e[0]] for e in entries],
            "img_w": img_w.float(),
        })
        weights.append(score)

    if not assemblies:
        return uniform

    key_w = torch.tensor(weights, dtype=torch.float).clamp_min(1e-6)
    key_w = key_w / key_w.sum().clamp_min(1e-12)
    hard_frac_max = max(0.0, min(1.0, float(hard_frac_max)))
    variant_frac_max = max(0.0, min(1.0, float(variant_frac_max)))
    assembly_width = max(2, int(assembly_width))
    cohort_images = max(2, min(int(cohort_images), len(img_pools)))
    n_block_images = max(1, min(int(n_block_images), len(img_pools)))
    recent = []

    def append_one(pool, out, selected):
        if pool.numel() == 0:
            return False
        for _ in range(8):
            j = int(pool[int(torch.randint(0, pool.numel(), (1,), generator=g))])
            if n < bs or j not in selected:
                out.append(j)
                selected.add(j)
                return True
        vals = [int(x) for x in pool.tolist() if n < bs or int(x) not in selected]
        if not vals:
            return False
        j = vals[int(torch.randint(0, len(vals), (1,), generator=g))]
        out.append(j)
        selected.add(j)
        return True

    def append_many(pool, count, out, selected):
        for _ in range(max(0, int(count))):
            if not append_one(pool, out, selected):
                break

    def choose_cohort():
        if len(img_pools) <= cohort_images:
            return list(range(len(img_pools)))
        return torch.multinomial(img_sizes, cohort_images, replacement=False, generator=g).tolist()

    def sample_keys(candidates, count):
        if not candidates or count <= 0:
            return []
        banned = set(recent[-max(0, int(cooldown)):])
        fresh = [k for k in candidates if k not in banned]
        if len(fresh) >= min(len(candidates), max(1, count // 2)):
            candidates = fresh
        cand = torch.tensor(candidates, dtype=torch.long)
        w = key_w[cand].clone().clamp_min(1e-6)
        repl = cand.numel() < count
        pos = torch.multinomial(w, count, replacement=repl, generator=g)
        return cand[pos].tolist()

    def draw_assembly(ai, cohort_set, remain, out, selected):
        a = assemblies[int(ai)]
        locs = [j for j, gid in enumerate(a["img_ids"]) if gid in cohort_set]
        if len(locs) < 2:
            locs = list(range(len(a["pools"])))
        width = min(max(1, int(remain)), assembly_width, len(locs))
        cand = torch.tensor(locs, dtype=torch.long)
        w = a["img_w"][cand].clone().clamp_min(1e-6)
        chosen = cand[torch.multinomial(w, width, replacement=False, generator=g)].tolist()
        before = len(out)
        for p in chosen:
            append_one(a["pools"][int(p)], out, selected)
        if len(out) > before:
            recent.append(int(ai))
            del recent[:-4 * max(1, int(cooldown))]
        return len(out) - before

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = _smooth01(float(step) / float(ramp_steps))
        n_hard = max(0, min(bs, int(round(bs * hard_frac_max * ramp))))
        n_variant = max(0, min(n_hard, int(round(n_hard * variant_frac_max * ramp))))

        out = []
        selected = set()
        append_many(all_idx, bs - n_hard, out, selected)

        cohort = choose_cohort()
        cohort_set = set(int(x) for x in cohort)
        eligible = [
            i for i, a in enumerate(assemblies)
            if sum(1 for gid in a["img_ids"] if gid in cohort_set) >= 2
        ]
        if not eligible:
            eligible = list(range(len(assemblies)))

        n_keys = max(1, int(math.ceil(n_variant / float(max(2, assembly_width)))))
        for ai in sample_keys(eligible, n_keys):
            if len(out) >= bs - n_hard + n_variant:
                break
            draw_assembly(ai, cohort_set, bs - n_hard + n_variant - len(out), out, selected)

        cohort_pool = torch.cat([img_pools[int(i)] for i in cohort])
        append_many(cohort_pool, bs - len(out), out, selected)

        if len(out) < bs:
            chosen = torch.multinomial(img_sizes, n_block_images, replacement=False, generator=g)
            block_pool = torch.cat([img_pools[int(c)] for c in chosen])
            append_many(block_pool, bs - len(out), out, selected)

        if len(out) < bs:
            out.extend(torch.randint(0, n, (bs - len(out),), generator=g).tolist())

        batch = torch.tensor(out[:bs], dtype=torch.long)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
