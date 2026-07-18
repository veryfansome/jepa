"""Phase 0 exit criterion: parse(render_full(state)) == state, under every distractor
regime, for hundreds of random states (terminal-jepa.md §3)."""

# Bootstrap: unittest discovery may import this file as a top-level module from any
# cwd, skipping tests/__init__.py — put the project root on sys.path here.
import pathlib as _pathlib
import sys as _sys

_ROOT = str(_pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import random
import unittest

from datagen.layouts import gen_layout
from env.parse import parse_full
from env.render import render_full

REGIMES = [
    {"banner_id": None, "noise_seed": None},
    {"banner_id": 7, "noise_seed": None},
    {"banner_id": None, "noise_seed": 123},
    {"banner_id": 63, "noise_seed": 9},
]


class TestRoundTrip(unittest.TestCase):
    def test_parse_inverts_render(self):
        rng = random.Random(0)
        for i in range(300):
            state = gen_layout(rng)
            for regime in REGIMES:
                text = render_full(state, step=i, **regime)
                self.assertEqual(parse_full(text), state, msg=text)

    def test_render_deterministic(self):
        rng = random.Random(1)
        state = gen_layout(rng)
        a = render_full(state, banner_id=3, noise_seed=42, step=5)
        b = render_full(state, banner_id=3, noise_seed=42, step=5)
        self.assertEqual(a, b)

    def test_dynamic_noise_changes_per_step(self):
        rng = random.Random(2)
        state = gen_layout(rng)
        a = render_full(state, noise_seed=42, step=1)
        b = render_full(state, noise_seed=42, step=2)
        self.assertNotEqual(a, b)
        self.assertEqual(parse_full(a), parse_full(b))


if __name__ == "__main__":
    unittest.main()
