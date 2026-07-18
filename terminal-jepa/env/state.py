"""Filesystem state, invariants, ground-truth probe features, predicates, and the
validated state constructor used for goal exemplars (terminal-jepa.md §5)."""

import hashlib
import json

from . import vocab


class FsState:
    """dirs: set of dir-path tuples; files: dict file-path tuple -> content class; cwd: tuple."""

    def __init__(self, dirs=(), files=None, cwd=()):
        self.dirs = set(dirs)
        self.files = dict(files or {})
        self.cwd = tuple(cwd)

    def copy(self):
        return FsState(self.dirs, self.files, self.cwd)

    def __eq__(self, other):
        return (
            isinstance(other, FsState)
            and self.dirs == other.dirs
            and self.files == other.files
            and self.cwd == other.cwd
        )

    def __hash__(self):
        return hash(self.canonical_json())

    def __repr__(self):
        return f"FsState(dirs={sorted(self.dirs)}, files={sorted(self.files.items())}, cwd={self.cwd})"

    # -- invariants ------------------------------------------------------------

    def invariant_violations(self):
        v = []
        for d in self.dirs:
            if d not in vocab.DIR_PATH_INDEX:
                v.append(f"dir not in vocab: {d}")
            elif len(d) > 1 and d[:-1] not in self.dirs:
                v.append(f"dir parent missing: {d}")
        for f, k in self.files.items():
            if f not in vocab.FILE_PATH_INDEX:
                v.append(f"file not in vocab: {f}")
            elif f[:-1] != () and f[:-1] not in self.dirs:
                v.append(f"file parent missing: {f}")
            if not (0 <= k < vocab.N_CONTENT):
                v.append(f"bad content class: {f} -> {k}")
        if self.cwd != () and self.cwd not in self.dirs:
            v.append(f"cwd missing: {self.cwd}")
        return v

    def check_invariants(self):
        v = self.invariant_violations()
        if v:
            raise ValueError("invariant violations: " + "; ".join(v))
        return self

    # -- serialization ----------------------------------------------------------

    def to_json(self):
        return {
            "dirs": sorted("/".join(d) for d in self.dirs),
            "files": {"/".join(f): k for f, k in sorted(self.files.items())},
            "cwd": "/".join(self.cwd),
        }

    @classmethod
    def from_json(cls, obj):
        split = lambda s: tuple(s.split("/")) if s else ()
        return cls(
            dirs={split(d) for d in obj["dirs"]},
            files={split(f): k for f, k in obj["files"].items()},
            cwd=split(obj["cwd"]),
        )

    def canonical_json(self):
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))

    def state_id(self):
        return hashlib.sha1(self.canonical_json().encode()).hexdigest()[:16]

    # -- ground-truth probe features (terminal-jepa.md §3) ----------------------

    def features(self):
        return {
            "cwd_index": vocab.CWD_INDEX[self.cwd],
            "dir_exists": [1 if p in self.dirs else 0 for p in vocab.DIR_PATHS],
            "file_exists": [1 if p in self.files else 0 for p in vocab.FILE_PATHS],
            "file_class": [self.files.get(p, -1) for p in vocab.FILE_PATHS],
        }


# -- predicates -----------------------------------------------------------------


class Predicate:
    kind = None

    def check(self, state):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    @staticmethod
    def from_json(obj):
        kind = obj["kind"]
        if kind == "file_exists_with_class":
            return FileExistsWithClass(vocab.str_to_path(obj["path"]), obj["cls"])
        if kind == "file_exists":
            return FileExists(vocab.str_to_path(obj["path"]))
        if kind == "file_absent":
            return FileAbsent(vocab.str_to_path(obj["path"]))
        if kind == "cwd_is":
            return CwdIs(vocab.str_to_path(obj["path"]))
        raise ValueError(f"unknown predicate kind: {kind}")

    def __eq__(self, other):
        return type(self) is type(other) and self.to_json() == other.to_json()

    def __hash__(self):
        return hash(json.dumps(self.to_json(), sort_keys=True))

    def __repr__(self):
        return json.dumps(self.to_json())


class FileExistsWithClass(Predicate):
    kind = "file_exists_with_class"

    def __init__(self, path, cls):
        assert path in vocab.FILE_PATH_INDEX and 0 <= cls < vocab.N_CONTENT
        self.path, self.cls = tuple(path), cls

    def check(self, state):
        return state.files.get(self.path) == self.cls

    def to_json(self):
        return {"kind": self.kind, "path": vocab.path_to_str(self.path), "cls": self.cls}


class FileExists(Predicate):
    kind = "file_exists"

    def __init__(self, path):
        assert path in vocab.FILE_PATH_INDEX
        self.path = tuple(path)

    def check(self, state):
        return self.path in state.files

    def to_json(self):
        return {"kind": self.kind, "path": vocab.path_to_str(self.path)}


class FileAbsent(Predicate):
    kind = "file_absent"

    def __init__(self, path):
        assert path in vocab.FILE_PATH_INDEX
        self.path = tuple(path)

    def check(self, state):
        return self.path not in state.files

    def to_json(self):
        return {"kind": self.kind, "path": vocab.path_to_str(self.path)}


class CwdIs(Predicate):
    kind = "cwd_is"

    def __init__(self, path):
        assert path == () or path in vocab.DIR_PATH_INDEX
        self.path = tuple(path)

    def check(self, state):
        return state.cwd == self.path

    def to_json(self):
        return {"kind": self.kind, "path": vocab.path_to_str(self.path)}


# -- validated state constructor --------------------------------------------------


def _ensure_dirs(state, path):
    for i in range(1, len(path) + 1):
        state.dirs.add(path[:i])


def make_satisfying(state, pred, rng=None, n_variants=1):
    """Goal exemplars: minimal predicate-satisfying edit of `state`, built only through
    validated mutations so exemplars can never be off-manifold (terminal-jepa.md §5).
    Variant 0 is the minimal edit; further variants also move cwd, covering plans whose
    side effects include an intermediate `cd`. Every result is invariant-checked and
    predicate-checked before it is returned."""
    base = state.copy()
    if isinstance(pred, FileExistsWithClass):
        _ensure_dirs(base, pred.path[:-1])
        base.files[pred.path] = pred.cls
    elif isinstance(pred, FileExists):
        _ensure_dirs(base, pred.path[:-1])
        base.files.setdefault(pred.path, 0)
    elif isinstance(pred, FileAbsent):
        base.files.pop(pred.path, None)
    elif isinstance(pred, CwdIs):
        _ensure_dirs(base, pred.path)
        base.cwd = pred.path
    else:
        raise ValueError(f"unsupported predicate: {pred!r}")

    variants = [base]
    if n_variants > 1 and rng is not None:
        cwd_options = [c for c in [()] + sorted(base.dirs) if c != base.cwd]
        rng.shuffle(cwd_options)
        for c in cwd_options:
            if len(variants) >= n_variants:
                break
            v = base.copy()
            v.cwd = c
            # cwd variants can break cwd-pinning predicates (CwdIs); keep only
            # variants that still satisfy the predicate.
            if pred.check(v):
                variants.append(v)

    for v in variants:
        v.check_invariants()
        if not pred.check(v):
            raise AssertionError(f"exemplar does not satisfy {pred!r}")
    return variants
