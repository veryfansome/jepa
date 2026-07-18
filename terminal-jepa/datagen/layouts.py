"""Layout generation and the train/val splits (terminal-jepa.md §3): held-out layouts are
unseen arrangements of the shared vocabulary (compositional recombination); held-out goal
predicates are a hash-based slice of the predicate universe."""

import hashlib
import json
import random

from env import vocab
from env.state import CwdIs, FileAbsent, FileExists, FileExistsWithClass, FsState

MAX_LAYOUT_DIRS = 12
MAX_LAYOUT_FILES = 20


def gen_layout(rng):
    n_dirs = rng.randint(3, MAX_LAYOUT_DIRS)
    dirs = set()
    while len(dirs) < n_dirs:
        parents = [()] + [d for d in dirs if len(d) < vocab.MAX_DIR_DEPTH]
        cand = rng.choice(parents) + (rng.choice(vocab.DIR_NAMES),)
        dirs.add(cand)

    n_files = rng.randint(4, MAX_LAYOUT_FILES)
    files = {}
    parents = [()] + sorted(dirs)
    attempts = 0
    while len(files) < n_files and attempts < 200:
        attempts += 1
        f = rng.choice(parents) + (rng.choice(vocab.FILE_NAMES),)
        if f not in files:
            files[f] = rng.randrange(vocab.N_CONTENT)

    cwd = rng.choice([()] + sorted(dirs))
    return FsState(dirs, files, cwd).check_invariants()


def layout_key(state):
    """Layout identity = tree shape + content assignment ONLY. cwd is initial state, not
    layout (terminal-jepa.md §3) — including it would let the same arrangement appear in
    both splits as different "layouts"."""
    obj = {
        "dirs": sorted("/".join(d) for d in state.dirs),
        "files": {"/".join(f): k for f, k in sorted(state.files.items())},
    }
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def make_layout_split(n_layouts, val_frac, seed):
    """Deduped by cwd-free layout key, split by hash of key so membership is stable
    across runs."""
    rng = random.Random(f"layouts:{seed}")
    layouts = {}
    while len(layouts) < n_layouts:
        st = gen_layout(rng)
        layouts.setdefault(layout_key(st), st)
    train, val = [], []
    threshold = int(val_frac * 1000)
    for lid in sorted(layouts):
        h = int(hashlib.sha1(f"layout-split:{lid}".encode()).hexdigest(), 16) % 1000
        (val if h < threshold else train).append((lid, layouts[lid]))
    return train, val


def predicate_universe():
    preds = []
    for p in vocab.FILE_PATHS:
        preds.append(FileAbsent(p))
        preds.append(FileExists(p))
        for k in range(vocab.N_CONTENT):
            preds.append(FileExistsWithClass(p, k))
    for d in [()] + vocab.DIR_PATHS:
        preds.append(CwdIs(d))
    return preds


def predicate_split(val_frac=0.2):
    """Hash-based holdout over the predicate universe; independent of layouts and seeds."""
    train, val = [], []
    threshold = int(val_frac * 1000)
    for pred in predicate_universe():
        key = json.dumps(pred.to_json(), sort_keys=True)
        h = int(hashlib.sha1(f"pred-split:{key}".encode()).hexdigest(), 16) % 1000
        (val if h < threshold else train).append(pred)
    return train, val
