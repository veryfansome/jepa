"""batcher chunk: similarity-weighted path replay curriculum.

Inspired by complementary-learning-systems / similarity-weighted interleaved learning:
when a high-interference memory is replayed, interleave nearby memories instead of the whole
buffer, but keep broad coverage so old structure is not overwritten.

Here a "nearby" shell episode is concrete, not metaphorical. For each sequence i we infer a
rough cwd from its own `cd` commands, then form route keys K_i from content operations
(ls/cat) at their effective path. Neighbor score is

    q(i,j) = sum_{k in K_i cap K_j} |posting(k)|^-1/2

with same-image neighbors used for local hard negatives and cross-image neighbors used as
schema bridges. Anchor priority is higher for sequences where content commands are
context-dependent (e.g. `ls -1` after `cd`) or repeat the same command text at different
effective paths. That should complement the R7 path-delta fastweight arch: batches now
exercise the target-space memory on ambiguous command/content reads whose answer depends on
the latent path state, while retaining the proven image-block hard/easy curriculum.
"""

import collections
import math

import torch

NAME = "r8_swil_path_replay_curriculum"
DESCRIPTION = (
    "Similarity-weighted interleaved path replay batcher: keep the winning image-blocked "
    "hard/easy curriculum, but replace part of the hard block with route-neighbor packets "
    "around high-priority context-dependent episodes. Neighbors are sequences sharing inferred "
    "cwd/path content keys, plus a small cross-image schema bridge, targeting the R7 fastweight "
    "path memory rather than only densifying same-system negatives."
)

_CONTENT_VERBS = ("ls", "cat")


def _verb_of(cmd):
    toks = cmd.split()
    return toks[0] if toks else ""


def _first_path(cmd):
    toks = cmd.split()
    for t in toks[1:]:
        if t.startswith("-"):
            continue
        return _clean_path(t)
    return ""


def _clean_path(path):
    return (path or "").strip().strip("'\"")


def _norm_path(path):
    path = _clean_path(path)
    if not path:
        return "."
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
    if path.startswith("/"):
        return _norm_path(path)
    base = _norm_path(cwd or "/")
    joined = "/" + path if base == "/" else base.rstrip("/") + "/" + path
    return _norm_path(joined)


def _prefix_key(path, depth):
    path = _norm_path(path)
    depth = max(1, int(depth))
    if path == "/":
        return "/"
    if path == ".":
        return "."
    absolute = path.startswith("/")
    parts = [p for p in path.split("/") if p]
    body = "/".join(parts[:depth])
    return "/" + body if absolute else "./" + body


def _contextual_arg(path):
    path = _clean_path(path)
    return (not path) or path in (".", "./") or (
        not path.startswith("/") and not path.startswith("~")
    )


def _cmd_shape(cmd):
    toks = cmd.split()
    if not toks:
        return ""
    v = toks[0]
    flags = [t for t in toks[1:] if t.startswith("-")]
    path = _first_path(cmd)
    if not path:
        tag = "<cwd>"
    elif path.startswith("/"):
        tag = "<abs>"
    elif path.startswith("~"):
        tag = "<home>"
    else:
        tag = "<rel>"
    return " ".join([v] + flags + [tag])


def _seq_route_and_priority(cmds, prefix_depth):
    keys = set()
    cwd = "/"
    saw_cd = False
    content_n = 0
    contextual_n = 0
    after_cd_n = 0
    late_mass = 0.0
    text_to_effective = collections.defaultdict(set)
    effective_counts = collections.Counter()
    L = max(1, len(cmds))

    for t, cmd in enumerate(cmds):
        v = _verb_of(cmd)
        path = _first_path(cmd)

        if v == "cd":
            cwd = _join_path(cwd, path)
            pk = _prefix_key(cwd, prefix_depth)
            if pk not in (".", "/"):
                keys.add(("cd", pk))
                keys.add(("route", pk))
            saw_cd = True
            continue

        if v not in _CONTENT_VERBS:
            continue

        eff = _join_path(cwd, path)
        pk = _prefix_key(eff, prefix_depth)
        if pk not in (".", "/"):
            keys.add(("op", v, pk))
            keys.add(("route", pk))
            if _contextual_arg(path):
                keys.add(("ctxop", v, pk))

        if _contextual_arg(path):
            contextual_n += 1
            shape = _cmd_shape(cmd)
            if shape:
                keys.add(("shape", shape))

        if saw_cd:
            after_cd_n += 1
        content_n += 1
        late_mass += float(t + 1) / float(L)

        fine = _prefix_key(eff, prefix_depth + 1)
        text_to_effective[cmd.strip()].add(fine)
        effective_counts[(v, fine)] += 1

    ambiguity = sum(max(0, len(v) - 1) for v in text_to_effective.values())
    repeats = sum(max(0, c - 1) for c in effective_counts.values())
    if content_n == 0:
        return keys, 1.0

    priority = (
        1.0
        + 0.70 * ambiguity
        + 0.35 * contextual_n
        + 0.16 * after_cd_n
        + 0.20 * repeats
        + 0.06 * late_mass
    )
    return keys, priority


def _ranked_pool(counts, limit):
    if not counts or limit <= 0:
        return torch.empty(0, dtype=torch.long)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, int(limit))]
    return torch.tensor([j for j, _ in ranked], dtype=torch.long)


def make_batcher(
    fit,
    bs,
    seed,
    n_block_images=2,
    hard_frac_max=0.75,
    ramp_frac=0.30,
    packet_frac_max=0.60,
    packet_size=4,
    bridge_frac=0.30,
    prefix_depth=2,
    nn_k=24,
    priority_pow=0.65,
    max_same_posting=384,
    max_global_posting=768,
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

    groups = {}
    images = []
    for i, s in enumerate(fit):
        img = s.get("image", "?")
        images.append(img)
        groups.setdefault(img, []).append(i)

    img_keys = sorted(groups)
    img_pools = [torch.tensor(groups[k], dtype=torch.long) for k in img_keys]
    if len(img_pools) < 2:
        return uniform

    all_idx = torch.arange(n, dtype=torch.long)
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools])
    k_imgs = max(1, min(int(n_block_images), len(img_pools)))

    route_keys = []
    raw_priority = []
    for s in fit:
        keys, p = _seq_route_and_priority(s.get("cmds", ()), prefix_depth)
        route_keys.append(keys)
        raw_priority.append(max(1e-3, float(p)))

    priority = torch.tensor(raw_priority, dtype=torch.float)
    priority = priority.pow(max(0.0, float(priority_pow)))
    priority = priority / priority.mean().clamp_min(1e-12)

    same_postings = collections.defaultdict(list)
    global_postings = collections.defaultdict(list)
    for i, keys in enumerate(route_keys):
        for k in sorted(keys):
            same_postings[(images[i], k)].append(i)
            global_postings[k].append(i)

    same_neighbors = []
    bridge_neighbors = []
    for i, keys in enumerate(route_keys):
        same_counts = collections.defaultdict(float)
        bridge_counts = collections.defaultdict(float)
        img = images[i]

        for k in sorted(keys):
            sp = same_postings[(img, k)]
            if len(sp) <= max_same_posting:
                w = 1.0 / math.sqrt(max(1, len(sp)))
                for j in sp:
                    if j != i:
                        same_counts[j] += w

            gp = global_postings[k]
            if len(gp) <= max_global_posting:
                w = 1.0 / math.sqrt(max(1, len(gp)))
                for j in gp:
                    if images[j] != img:
                        bridge_counts[j] += w

        same_neighbors.append(_ranked_pool(same_counts, nn_k))
        bridge_neighbors.append(_ranked_pool(bridge_counts, nn_k))

    hard_frac_max = max(0.0, min(1.0, float(hard_frac_max)))
    packet_frac_max = max(0.0, min(1.0, float(packet_frac_max)))
    bridge_frac = max(0.0, min(1.0, float(bridge_frac)))
    packet_size = max(2, int(packet_size))

    def choose_image_pool():
        chosen = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
        return torch.cat([img_pools[int(c)] for c in chosen])

    def draw_unique(pool, count, out, selected):
        count = int(count)
        if count <= 0 or pool.numel() == 0:
            return
        vals = []
        for x in pool.tolist():
            j = int(x)
            if n < bs or j not in selected:
                vals.append(j)
        if not vals:
            return
        k = min(count, len(vals))
        order = torch.randperm(len(vals), generator=g)[:k].tolist()
        for pos in order:
            j = vals[int(pos)]
            out.append(j)
            selected.add(j)

    def weighted_one(pool, selected):
        if pool.numel() == 0:
            return None
        vals = []
        for x in pool.tolist():
            j = int(x)
            if n < bs or j not in selected:
                vals.append(j)
        if not vals:
            return None
        idx = torch.tensor(vals, dtype=torch.long)
        w = priority[idx].clamp_min(1e-6)
        pos = int(torch.multinomial(w, 1, replacement=True, generator=g).item())
        return int(idx[pos])

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, float(step) / float(ramp_steps))
        n_hard = max(0, min(bs, int(round(bs * hard_frac_max * ramp))))

        out = []
        selected = set()
        chosen_pool = choose_image_pool() if n_hard > 0 else all_idx

        packet_frac = packet_frac_max * ramp
        n_packet = max(0, min(n_hard, int(round(n_hard * packet_frac))))
        n_image = n_hard - n_packet

        draw_unique(chosen_pool, n_image, out, selected)
        hard_target = n_hard

        while len(out) < hard_target:
            before = len(out)
            anchor = weighted_one(chosen_pool, selected)
            if anchor is None:
                anchor = weighted_one(all_idx, selected)
            if anchor is None:
                break

            out.append(anchor)
            selected.add(anchor)

            remain = min(packet_size - 1, hard_target - len(out))
            n_bridge = int(round(remain * bridge_frac))
            n_same = remain - n_bridge

            same_pool = same_neighbors[anchor]
            if same_pool.numel() == 0:
                same_pool = chosen_pool
            bridge_pool = bridge_neighbors[anchor]
            if bridge_pool.numel() == 0:
                bridge_pool = all_idx

            draw_unique(same_pool, n_same, out, selected)
            draw_unique(bridge_pool, n_bridge, out, selected)

            if len(out) == before:
                break

        draw_unique(chosen_pool, hard_target - len(out), out, selected)
        draw_unique(all_idx, bs - len(out), out, selected)

        if len(out) < bs:
            out.extend(torch.randint(0, n, (bs - len(out),), generator=g).tolist())

        batch = torch.tensor(out[:bs], dtype=torch.long)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
