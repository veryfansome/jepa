"""Factored actions (verb, arg1, arg2), validity semantics, transition typing, and the
typed samplers used by datagen (terminal-jepa.md §3)."""

import random

from . import vocab

VERBS = ["cd", "ls", "cat", "mkdir", "touch", "rm", "cp", "mv", "write"]

# Invalid-action failure taxonomy; datagen reports coverage per type (terminal-jepa.md §3).
FAILURE_TYPES = [
    "malformed-arg",
    "nonexistent-path",
    "wrong-type",
    "already-exists",
    "missing-parent",
    "not-empty",
    "busy",
    "same-src-dst",
]

STATE_CHANGING = "state-changing"
VALID_NO_OP = "valid-no-op"
INVALID = "invalid"


class Result:
    def __init__(self, state, stdout, ttype, failure=None):
        self.state = state
        self.stdout = stdout
        self.ttype = ttype
        self.failure = failure

    @property
    def valid(self):
        return self.ttype != INVALID


def _err(state, verb, msg, failure):
    return Result(state, f"{verb}: error: {msg}", INVALID, failure)


def _parse_path(s):
    p = vocab.str_to_path(s)
    if p != () and p not in vocab.DIR_PATH_INDEX and p not in vocab.FILE_PATH_INDEX:
        raise ValueError(f"path outside vocabulary: {s!r}")
    return p


def apply(state, action):
    """Pure transition function: returns Result with a fresh state (input untouched).
    Invalid actions never change state; ttype is derived from the actual state delta."""
    verb, a1, a2 = action
    if verb not in VERBS:
        return _err(state, str(verb), f"unknown verb {verb!r}", "malformed-arg")
    try:
        return _APPLY[verb](state, a1, a2)
    except ValueError as e:
        return _err(state, verb, str(e), "malformed-arg")


def _finish(old, new, stdout=""):
    ttype = VALID_NO_OP if new == old else STATE_CHANGING
    return Result(new, stdout, ttype)


def _apply_cd(state, a1, a2):
    d = state.cwd if a1 == "." else _parse_path(a1)
    if d in state.files:
        return _err(state, "cd", f"not a directory: {a1}", "wrong-type")
    if d != () and d not in state.dirs:
        return _err(state, "cd", f"no such directory: {a1}", "nonexistent-path")
    new = state.copy()
    new.cwd = d
    return _finish(state, new)


def _apply_ls(state, a1, a2):
    d = state.cwd if a1 == "." else _parse_path(a1)
    if d in state.files:
        return _err(state, "ls", f"not a directory: {a1}", "wrong-type")
    if d != () and d not in state.dirs:
        return _err(state, "ls", f"no such directory: {a1}", "nonexistent-path")
    entries = sorted(
        [p[-1] + "/" for p in state.dirs if p[:-1] == d]
        + [p[-1] for p in state.files if p[:-1] == d]
    )
    return _finish(state, state.copy(), "\n".join(entries))


def _apply_cat(state, a1, a2):
    f = _parse_path(a1)
    if f in state.dirs or f == ():
        return _err(state, "cat", f"is a directory: {a1}", "wrong-type")
    if f not in state.files:
        return _err(state, "cat", f"no such file: {a1}", "nonexistent-path")
    return _finish(state, state.copy(), vocab.content_to_str(state.files[f]))


def _apply_mkdir(state, a1, a2):
    d = _parse_path(a1)
    if d not in vocab.DIR_PATH_INDEX:
        return _err(state, "mkdir", f"not a directory path: {a1}", "wrong-type")
    if d in state.dirs:
        return _err(state, "mkdir", f"already exists: {a1}", "already-exists")
    if len(d) > 1 and d[:-1] not in state.dirs:
        return _err(state, "mkdir", f"missing parent: {a1}", "missing-parent")
    new = state.copy()
    new.dirs.add(d)
    return _finish(state, new)


def _apply_touch(state, a1, a2):
    f = _parse_path(a1)
    if f not in vocab.FILE_PATH_INDEX:
        return _err(state, "touch", f"not a file path: {a1}", "wrong-type")
    if f in state.files:
        return _err(state, "touch", f"already exists: {a1}", "already-exists")
    if f[:-1] != () and f[:-1] not in state.dirs:
        return _err(state, "touch", f"missing parent: {a1}", "missing-parent")
    new = state.copy()
    new.files[f] = 0
    return _finish(state, new)


def _apply_rm(state, a1, a2):
    p = _parse_path(a1)
    if p in state.files:
        new = state.copy()
        del new.files[p]
        return _finish(state, new)
    if p in state.dirs:
        if state.cwd[: len(p)] == p:
            return _err(state, "rm", f"directory busy (cwd): {a1}", "busy")
        occupied = any(q[:-1] == p for q in state.dirs) or any(
            q[:-1] == p for q in state.files
        )
        if occupied:
            return _err(state, "rm", f"directory not empty: {a1}", "not-empty")
        new = state.copy()
        new.dirs.discard(p)
        return _finish(state, new)
    return _err(state, "rm", f"no such path: {a1}", "nonexistent-path")


def _copy_like(verb, move):
    def op(state, a1, a2):
        s, d = _parse_path(a1), _parse_path(a2)
        if s in state.dirs or s == ():
            return _err(state, verb, f"source is a directory: {a1}", "wrong-type")
        if s not in state.files:
            return _err(state, verb, f"no such file: {a1}", "nonexistent-path")
        if d not in vocab.FILE_PATH_INDEX:
            return _err(state, verb, f"destination not a file path: {a2}", "wrong-type")
        if d == s:
            return _err(state, verb, "source equals destination", "same-src-dst")
        if d[:-1] != () and d[:-1] not in state.dirs:
            return _err(state, verb, f"missing parent: {a2}", "missing-parent")
        new = state.copy()
        new.files[d] = new.files[s]
        if move:
            del new.files[s]
        return _finish(state, new)

    return op


def _apply_write(state, a1, a2):
    f = _parse_path(a1)
    if f not in vocab.FILE_PATH_INDEX:
        return _err(state, "write", f"not a file path: {a1}", "wrong-type")
    k = vocab.str_to_content(a2)
    if f[:-1] != () and f[:-1] not in state.dirs:
        return _err(state, "write", f"missing parent: {a1}", "missing-parent")
    new = state.copy()
    new.files[f] = k
    return _finish(state, new)


_APPLY = {
    "cd": _apply_cd,
    "ls": _apply_ls,
    "cat": _apply_cat,
    "mkdir": _apply_mkdir,
    "touch": _apply_touch,
    "rm": _apply_rm,
    "cp": _copy_like("cp", move=False),
    "mv": _copy_like("mv", move=True),
    "write": _apply_write,
}


# -- typed samplers ----------------------------------------------------------------
# Sampling is two-stage (uniform over verbs with >=1 valid instantiation, then uniform
# over that verb's valid args) so high-arity verbs (cp/write) don't dominate the mix.


def _ps(path):
    return vocab.path_to_str(path)


def _valid_args(state, verb):
    dirs_all = [()] + sorted(state.dirs)
    files = sorted(state.files)
    if verb in ("cd", "ls"):
        return [(_ps(d), "") for d in dirs_all]
    if verb == "cat":
        return [(_ps(f), "") for f in files]
    if verb == "mkdir":
        return [
            (_ps(d), "")
            for d in vocab.DIR_PATHS
            if d not in state.dirs and (len(d) == 1 or d[:-1] in state.dirs)
        ]
    if verb == "touch":
        return [
            (_ps(f), "")
            for f in vocab.FILE_PATHS
            if f not in state.files and (f[:-1] == () or f[:-1] in state.dirs)
        ]
    if verb == "rm":
        removable_dirs = [
            d
            for d in state.dirs
            if state.cwd[: len(d)] != d
            and not any(q[:-1] == d for q in state.dirs)
            and not any(q[:-1] == d for q in state.files)
        ]
        return [(_ps(p), "") for p in files + sorted(removable_dirs)]
    if verb in ("cp", "mv"):
        dsts = [
            f
            for f in vocab.FILE_PATHS
            if f[:-1] == () or f[:-1] in state.dirs
        ]
        return [
            (_ps(s), _ps(d)) for s in files for d in dsts if d != s
        ]
    if verb == "write":
        targets = [
            f for f in vocab.FILE_PATHS if f[:-1] == () or f[:-1] in state.dirs
        ]
        return [(_ps(f), f"c{k}") for f in targets for k in range(vocab.N_CONTENT)]
    raise ValueError(verb)


def sample_valid(state, rng):
    verbs = [v for v in VERBS if _valid_args(state, v)]
    verb = rng.choice(verbs)
    a1, a2 = rng.choice(_valid_args(state, verb))
    return (verb, a1, a2)


def _invalid_candidates(state, rng):
    """One concrete candidate per achievable failure type; caller picks among them."""
    existing_dir = _ps(rng.choice(sorted(state.dirs))) if state.dirs else None
    existing_file = _ps(rng.choice(sorted(state.files))) if state.files else None
    absent_files = [f for f in vocab.FILE_PATHS if f not in state.files]
    absent_file = _ps(rng.choice(absent_files)) if absent_files else None
    orphan_files = [
        f for f in vocab.FILE_PATHS if len(f) > 1 and f[:-1] not in state.dirs
    ]
    orphan_file = _ps(rng.choice(orphan_files)) if orphan_files else None
    # Exclude cwd ancestors: apply() answers "busy" before "not-empty" for those.
    nonempty_dirs = [
        d
        for d in state.dirs
        if state.cwd[: len(d)] != d
        and (any(q[:-1] == d for q in state.dirs) or any(q[:-1] == d for q in state.files))
    ]

    out = []
    if absent_file:
        out.append((("cat", absent_file, ""), "nonexistent-path"))
    if existing_dir:
        out.append((("cat", existing_dir, ""), "wrong-type"))
    if existing_dir:
        out.append((("mkdir", existing_dir, ""), "already-exists"))
    if existing_file:
        out.append((("touch", existing_file, ""), "already-exists"))
    if orphan_file:
        out.append((("touch", orphan_file, ""), "missing-parent"))
        out.append((("write", orphan_file, "c0"), "missing-parent"))
    if nonempty_dirs:
        out.append((("rm", _ps(rng.choice(sorted(nonempty_dirs))), ""), "not-empty"))
    if state.cwd != ():
        out.append((("rm", _ps(state.cwd), ""), "busy"))
    if existing_file:
        out.append((("cp", existing_file, existing_file), "same-src-dst"))
    return out


def sample_invalid(state, rng):
    cands = _invalid_candidates(state, rng)
    action, failure = rng.choice(cands)
    return action, failure
