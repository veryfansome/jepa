"""Shared enumerations. Every layout, probe label space, and action argument draws from
these fixed vocabularies (terminal-jepa.md §3: shared enumeration across layouts/splits)."""

DIR_NAMES = ["a", "b", "c", "d", "e", "f"]
FILE_NAMES = ["notes.txt", "data.csv", "run.sh", "log.txt", "cfg.ini", "readme.md"]
N_CONTENT = 8
MAX_DIR_DEPTH = 2  # dirs at /x or /x/y; files therefore at depth <= 3 components

# All syntactically possible dir paths (tuples of names), depth 1..MAX_DIR_DEPTH.
DIR_PATHS = [(n,) for n in DIR_NAMES] + [
    (a, b) for a in DIR_NAMES for b in DIR_NAMES
]
DIR_PATH_INDEX = {p: i for i, p in enumerate(DIR_PATHS)}

# All syntactically possible file paths: parent is root or any dir path.
FILE_PARENTS = [()] + DIR_PATHS
FILE_PATHS = [par + (f,) for par in FILE_PARENTS for f in FILE_NAMES]
FILE_PATH_INDEX = {p: i for i, p in enumerate(FILE_PATHS)}

# cwd label space: root + every dir path.
CWD_PATHS = [()] + DIR_PATHS
CWD_INDEX = {p: i for i, p in enumerate(CWD_PATHS)}

CONTENT_TOKENS = [f"content-{k}" for k in range(N_CONTENT)]

# Banner vocabulary for the slow-feature distractor. Sampled i.i.d. per trajectory,
# independent of layout, policy, and split (terminal-jepa.md §3).
_BANNER_WORDS = ["quartz", "onyx", "maple", "cobalt", "ember", "juniper", "slate", "fjord"]
BANNERS = [
    f"### {w} terminal build {i:03d} ###" for i, w in
    ((i, _BANNER_WORDS[i % len(_BANNER_WORDS)]) for i in range(64))
]


def path_to_str(path):
    """() -> "/", ("a","b") -> "/a/b"."""
    return "/" + "/".join(path)


def str_to_path(s):
    """Inverse of path_to_str. Raises ValueError on anything not shaped like a path."""
    if not isinstance(s, str) or not s.startswith("/"):
        raise ValueError(f"not a path: {s!r}")
    if s == "/":
        return ()
    parts = tuple(s[1:].split("/"))
    if any(not p for p in parts):
        raise ValueError(f"not a path: {s!r}")
    return parts


def content_to_str(k):
    return CONTENT_TOKENS[k]


def str_to_content(s):
    """"c3" or "content-3" -> 3. Raises ValueError on anything else."""
    if isinstance(s, str):
        body = None
        if s.startswith("content-"):
            body = s[len("content-"):]
        elif s.startswith("c"):
            body = s[1:]
        if body is not None and body.isdigit():
            k = int(body)
            if 0 <= k < N_CONTENT:
                return k
    raise ValueError(f"not a content class: {s!r}")
