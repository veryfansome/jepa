"""perception recipe (R8) for the PATH-KEYED multi-vector stream: e5-base-v2 (the champion
encoder) with up to K=4 LOCATION-KEYED segments per observation, alongside the standard e5
single-vector recipe (so the single z_obs/z_cmd — the FIXED target/eval space — are IDENTICAL to
enc_e5_base / the data/dockerfs-e5 root; mv_encode copies them verbatim).

THE KEY IDEA (from the retired synthetic phase, where a path-keyed readout decoded held-out state
100% vs 44% for mean-pooling): every segment's text BEGINS with the ABSOLUTE PATH(S) it describes,
in the same lexical form a future command uses. Commands are encoded as "passage: cat
/etc/os-release"; a segment "passage: /etc/os-release\n<content>" therefore lands NEAR that command
in the frozen retrieval-tuned e5 space — the encoder itself supplies the content-to-location
binding, so the key space of stored segments coincides with the query space of commands. The two
earlier multi-vector renders lacked exactly this: mv_obs_k4 keyed segments by line-INDEX,
r7_role_multivec by semantic ROLE — neither key could be addressed by a command.

Segments (>=1, <=K; empty roles are simply absent -> masked in the stream):
  ls  : listed entries rewritten as sorted ABSOLUTE paths (dir resolved from cmd arg + cwd;
        metadata columns stripped; dirs get a trailing '/', symlinks keep '->target'), split into
        up to K contiguous alphabetical buckets — each bucket a sharp path-keyed vector instead of
        one mean-pool over 60 metadata-heavy lines. `-R` section headers re-key following entries
        to their subdirectory.
  cat : the file's content in up to K contiguous chunks, EACH prefixed by the file's absolute
        path (a short file = ONE vector binding path -> content).
  cd  : a single segment naming the new cwd.
  other (uname/...): a single "cmd | output" segment.
"""

import math
import posixpath
import re

from evolve.chunks.perception.baseline import OBS_CAP
from evolve.chunks.perception.enc_e5_base import MODEL, render_obs, render_cmd, pool  # noqa: F401 (single-vector recipe unchanged)

K = 4
SEG_CAP = 700  # chars per segment text (after the "passage: " prefix)

# Long-format ls line: pull the TYPE char and the NAME, discard permission/link/owner/group/
# size/date nuisance columns (same convention as r6_foveal_ls_names / r7_role_multivec).
_LS_LINE = re.compile(
    r'^([bcdlpsD\-])[rwxsStTlL.\-]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+'
    r'(?:\d+,\s+\d+|\S+)\s+\S+\s+\S+\s+\S+\s+(.+?)\s*$')
_TOTAL = re.compile(r'^total\s+\S+\s*$')  # `total 56` and human-readable `total 4.0K`
_SECTION = re.compile(r'^(\S.*):\s*$')   # `ls -R` section header like `./etc:` or `/etc:`
_NUM = re.compile(r'^\d+$')              # inode / size tokens in short formats (`ls -i`, `ls -s`)


def _resolve(path, cwd):
    if not path.startswith("/"):
        path = posixpath.join(cwd or "/", path)
    return posixpath.normpath(path)


def _arg_of(cmd):
    """First non-flag argument of the command, or None."""
    for p in (cmd or "").split()[1:]:
        if not p.startswith("-"):
            return p
    return None


def _ls_paths(out, base_dir):
    """Absolute-path entries from an ls output (any format), '.'/'..' dropped."""
    paths = []
    cur = base_dir
    for ln in out.split("\n"):
        s = ln.rstrip()
        if not s or _TOTAL.match(s):
            continue
        msec = _SECTION.match(s)
        if msec:
            cur = _resolve(msec.group(1), base_dir)   # -R: re-key following entries
            continue
        m = _LS_LINE.match(s)
        if m:
            typ, name = m.group(1), m.group(2).strip()
            link = ""
            if " -> " in name:
                name, _, tgt = name.partition(" -> ")
                link = "->" + tgt.strip()
            if name in (".", ".."):
                continue
            p = posixpath.join(cur, name)
            paths.append(p + "/" if typ == "d" else p + link)
        else:
            # short format (-1 / -i / columns): tokens are names; drop numeric inode/size tokens
            for tok in s.split():
                if _NUM.match(tok) or tok in (".", ".."):
                    continue
                paths.append(posixpath.join(cur, tok))
    return paths


def _chunks(items, n):
    """Split a list into n contiguous chunks of near-equal length (no empty chunks)."""
    per = math.ceil(len(items) / n)
    return [items[i * per:(i + 1) * per] for i in range(n) if items[i * per:(i + 1) * per]]


def render_obs_multi(step):
    cmd = step.get("cmd", "") or ""
    verb = cmd.split()[0] if cmd.split() else ""
    cwd = step.get("cwd", "/") or "/"
    exit_ = step.get("exit", 0)
    err = f" exit={exit_}" if exit_ else ""
    out = (step.get("output", "") or "")[:OBS_CAP]

    if verb == "ls":
        target = _resolve(_arg_of(cmd) or cwd, cwd)
        paths = sorted(set(_ls_paths(out, target)))
        if not paths:
            return [f"passage: {target}/ (empty){err}"]
        nseg = min(K, len(paths))
        return ["passage: " + "\n".join(c)[:SEG_CAP] for c in _chunks(paths, nseg)]

    if verb == "cat":
        target = _resolve(_arg_of(cmd) or cwd, cwd)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if not lines:
            return [f"passage: {target} (empty){err}"]
        body = "\n".join(lines)
        nseg = min(K, max(1, math.ceil(len(body) / SEG_CAP)), len(lines))
        return [("passage: " + target + err + "\n" + "\n".join(c))[: SEG_CAP + 96]
                for c in _chunks(lines, nseg)]

    if verb == "cd":
        return [f"passage: {cwd}/{err}"]

    one = " ".join(out.split())[:SEG_CAP]
    return [f"passage: {cmd} | {one}{err}"]
