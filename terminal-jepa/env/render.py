"""Observation renderers with distractor knobs (terminal-jepa.md §3). Rendering is a pure
function of (state, action, stdout, banner_id, noise_seed, step) so datagen can store
compact state and the training loader can pick regimes at load time."""

import random

from . import vocab


def _noise_line(noise_seed, step):
    rng = random.Random(f"noise:{noise_seed}:{step}")
    return f"[ts {rng.randrange(10**9)}] [pid {rng.randrange(10**5)}]"


def _header(banner_id, noise_seed, step):
    lines = []
    if banner_id is not None:
        lines.append(vocab.BANNERS[banner_id])
    if noise_seed is not None:
        lines.append(_noise_line(noise_seed, step))
    return lines


def render_full(state, banner_id=None, noise_seed=None, step=0):
    """Serialized tree + cwd: the full-obs regime. Content classes are shown so the state
    is exactly recoverable (parse.parse_full is the inverse; round-trip is tested)."""
    lines = _header(banner_id, noise_seed, step)
    lines.append(f"cwd: {vocab.path_to_str(state.cwd)}")
    lines.append("tree:")
    entries = [vocab.path_to_str(d) + "/" for d in state.dirs]
    entries += [
        f"{vocab.path_to_str(f)} [c{k}]" for f, k in state.files.items()
    ]
    lines.extend(sorted(entries))
    return "\n".join(lines)


def action_to_cmd(action):
    verb, a1, a2 = action
    return " ".join(x for x in (verb, a1, a2) if x)


def render_partial(cwd_before, action, stdout, banner_id=None, noise_seed=None, step=0):
    """Prompt + stdout of the last command only: the partial-obs regime."""
    lines = _header(banner_id, noise_seed, step)
    lines.append(f"user@sandbox:{vocab.path_to_str(cwd_before)}$ {action_to_cmd(action)}")
    if stdout:
        lines.append(stdout)
    return "\n".join(lines)


# -- verbose / lossy rendering (data redesign, finding 24; increment 1) -------------
# Moves the observation into JEPA's regime: content is no longer a literal [cK] token
# but a CLASS-CONDITIONAL byte snippet (the class is a needle recoverable only by
# aggregating characteristic tokens out of shared filler — abstraction, not a token
# read), lines carry in-band metadata nuisance (perms/size/mtime), and entries render
# in non-alphabetical order (kills the sorted-tree parser shortcut of findings 14/22).
# Ground truth still comes from the symbolic env; this only changes the observation.

# 8 disjoint class-characteristic vocabularies (index = content class 0..7).
CONTENT_LEXICON = [
    ["alpha", "apex", "amber", "arc", "atlas"],
    ["bravo", "basin", "birch", "bloom", "bison"],
    ["cobalt", "cedar", "crest", "cinder", "coral"],
    ["delta", "dune", "drift", "dawn", "dial"],
    ["ember", "echo", "elm", "eddy", "epoch"],
    ["flint", "fjord", "fern", "flux", "forge"],
    ["gulf", "grove", "gale", "glint", "gauss"],
    ["haze", "helix", "hollow", "hearth", "hertz"],
]
# Shared filler drawn into every body regardless of class — the nuisance the encoder
# must abstract THROUGH to recover the class.
FILLER_WORDS = [
    "the", "and", "run", "log", "tmp", "buf", "seq", "idx", "ptr", "val",
    "obj", "key", "row", "col", "bit", "map", "set", "job", "req", "res",
]


def _content_snippet(cls, path, n=9, n_sig=3):
    """Deterministic per (class, path): n_sig class-characteristic tokens sprinkled
    among filler in a path-seeded order. Class is recoverable only by noticing which
    lexicon dominates — not by reading a single token. Held-out (class, path) combos
    are unseen, so this forces abstraction, not memorization."""
    rng = random.Random(f"body:{cls}:{vocab.path_to_str(path)}")
    sig = [rng.choice(CONTENT_LEXICON[cls]) for _ in range(n_sig)]
    fill = [rng.choice(FILLER_WORDS) for _ in range(n - n_sig)]
    body = sig + fill
    rng.shuffle(body)
    return " ".join(body)


def _fake_meta(path, salt):
    """In-band nuisance: an ls -la-style perms/size/mtime triple, deterministic per
    (path, salt) so it is stable within a rendered state but carries no state info."""
    rng = random.Random(f"meta:{salt}:{vocab.path_to_str(path)}")
    perms = "".join(rng.choice("rwx-") for _ in range(9))
    size = rng.randrange(16, 999999)
    mtime = f"{rng.randrange(1,13):02d}-{rng.randrange(1,29):02d} {rng.randrange(24):02d}:{rng.randrange(60):02d}"
    return f"-{perms} {size} {mtime}"


def render_full_verbose(state, salt=0, banner_id=None, noise_seed=None, step=0):
    """Verbose lossy full-obs: same information as render_full (structure + content
    class + cwd) but as realistic, high-nuisance, non-alphabetical `ls -la`-style
    lines where content is a class-conditional snippet, not a [cK] token."""
    lines = _header(banner_id, noise_seed, step)
    lines.append(f"cwd: {vocab.path_to_str(state.cwd)}")
    order = random.Random(f"order:{salt}:{state.state_id()}")
    entries = [("dir", d) for d in state.dirs] + [("file", f) for f in state.files]
    order.shuffle(entries)
    for kind, p in entries:
        meta = _fake_meta(p, salt)
        if kind == "dir":
            lines.append(f"d{meta[1:]} {vocab.path_to_str(p)}/")
        else:
            snippet = _content_snippet(state.files[p], p)
            lines.append(f"{meta} {vocab.path_to_str(p)} :: {snippet}")
    return "\n".join(lines)
