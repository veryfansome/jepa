"""perception chunk: e5-base with a multiscale tree-coordinate render for cd observations.

Inspired by entorhinal grid codes whose modules use geometric scale ratios
(https://arxiv.org/abs/1304.0031) and cognitive-map/successor representations
(https://arxiv.org/abs/2202.11190), a directory is rendered as a path coordinate: exact
cwd, parent, leaf, root-to-leaf prefixes, and Fibonacci-scale suffixes. Text overlap then
tracks tree distance, so nearby cd states should embed closer than distant subtrees. ls,
cat, uname, and command renders stay byte-identical to enc_e5_base; only cd state tokens
change, limiting content-verb baseline lift while giving planning a less-flat state field.
"""

import posixpath

NAME = "r11_gridpath_cdstate_codex"
DESCRIPTION = (
    "Path-structured cd perception: keep e5-base and the champion command/content renders, "
    "but replace successful cd observations with a compact multiscale tree coordinate "
    "(cwd, parent, leaf, ancestor prefixes, Fibonacci suffix scales) so cd states carry "
    "graded filesystem distance instead of near-identical empty-output strings."
)

MODEL = "intfloat/e5-base-v2"
OBS_CAP = 1600
PATH_CAP = 1200
CD_OUT_CAP = 360

_SCALES = (1, 2, 3, 5, 8, 13)
_MAX_PREFIXES = 14


def _verb(cmd):
    parts = (cmd or "").split()
    return parts[0] if parts else ""


def _norm_path(path):
    p = str(path or "/")
    if not p.startswith("/"):
        p = "/" + p
    p = posixpath.normpath(p)
    if p in ("", "."):
        return "/"
    if not p.startswith("/"):
        p = "/" + p
    return p


def _parts(path):
    return [p for p in path.split("/") if p and p != "."]


def _prefix(parts, n):
    if n <= 0:
        return "/"
    return "/" + "/".join(parts[:n])


def _keep_ends(items, max_items):
    if len(items) <= max_items:
        return items
    head = max(3, max_items // 3)
    tail = max_items - head
    return items[:head] + items[-tail:]


def _dedup(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _cap_middle(text, cap):
    if len(text) <= cap:
        return text
    head = (2 * cap) // 3
    tail = cap - head
    omitted = len(text) - head - tail
    return text[:head].rstrip() + f" ...[{omitted} path chars]... " + text[-tail:].lstrip()


def _clip_output(out, cap):
    if len(out) <= cap:
        return out
    return out[:cap] + f"\n...[{len(out) - cap} more chars]"


def _depth_ring(n):
    if n <= 1:
        return "r1"
    if n <= 2:
        return "r2"
    if n <= 4:
        return "r4"
    if n <= 7:
        return "r7"
    if n <= 12:
        return "r12"
    return "r13p"


def _path_code(cwd):
    cwd = _norm_path(cwd)
    parts = _parts(cwd)
    depth = len(parts)
    parent = _prefix(parts, depth - 1)
    leaf = parts[-1] if parts else "/"

    prefixes = [_prefix(parts, i) for i in range(depth + 1)]
    prefixes = _keep_ends(prefixes, _MAX_PREFIXES)

    suffixes = []
    for scale in _SCALES:
        if depth >= scale:
            suffixes.append("/".join(parts[-scale:]))

    atoms = [
        f"d{depth}",
        _depth_ring(depth),
        cwd,
        f"parent:{parent}",
        f"leaf:{leaf}",
    ]
    atoms.extend(prefixes)
    atoms.extend(suffixes)
    return _cap_middle(" ".join(_dedup(atoms)), PATH_CAP)


def _render_cd_obs(step, out):
    exit_ = step.get("exit", 0)
    ok = exit_ == 0 or exit_ == "0"
    status = "cd_ok" if ok else "cd_fail"
    text = f"passage: {status} {_path_code(step.get('cwd', '/'))}"
    if out:
        text += "\n" + _clip_output(out, CD_OUT_CAP)
    return text


def render_obs(step):
    out = step.get("output", "") or ""
    if _verb(step.get("cmd", "")) == "cd":
        return _render_cd_obs(step, out)

    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"


def render_cmd(step):
    return "passage: " + step["cmd"]


def pool(h, mask):
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)
