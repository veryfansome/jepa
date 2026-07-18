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
