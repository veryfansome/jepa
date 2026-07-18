"""Action semantics: invariants preserved by every transition, invalid actions never
mutate state, transition typing derived from actual state deltas, and the typed samplers
produce what they claim to."""

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
from env import actions


class TestActions(unittest.TestCase):
    def test_sampled_valid_actions_are_valid_and_preserve_invariants(self):
        rng = random.Random(10)
        for _ in range(200):
            state = gen_layout(rng)
            for _ in range(8):
                action = actions.sample_valid(state, rng)
                res = actions.apply(state, action)
                self.assertTrue(res.valid, msg=f"{action} -> {res.stdout}")
                self.assertEqual(res.state.invariant_violations(), [])
                state = res.state

    def test_sampled_invalid_actions_fail_as_declared_and_never_mutate(self):
        rng = random.Random(11)
        for _ in range(200):
            state = gen_layout(rng)
            action, declared = actions.sample_invalid(state, rng)
            res = actions.apply(state, action)
            self.assertEqual(res.ttype, actions.INVALID, msg=f"{action}")
            self.assertEqual(res.failure, declared, msg=f"{action}")
            self.assertEqual(res.state, state)

    def test_ttype_matches_state_delta(self):
        rng = random.Random(12)
        for _ in range(200):
            state = gen_layout(rng)
            action = actions.sample_valid(state, rng)
            res = actions.apply(state, action)
            changed = res.state != state
            self.assertEqual(
                res.ttype,
                actions.STATE_CHANGING if changed else actions.VALID_NO_OP,
                msg=f"{action}",
            )

    def test_apply_never_mutates_input_state(self):
        rng = random.Random(13)
        for _ in range(100):
            state = gen_layout(rng)
            snapshot = state.copy()
            actions.apply(state, actions.sample_valid(state, rng))
            actions.apply(state, actions.sample_invalid(state, rng)[0])
            self.assertEqual(state, snapshot)

    def test_malformed_args_are_invalid(self):
        rng = random.Random(14)
        state = gen_layout(rng)
        for action in [
            ("frobnicate", "/a", ""),
            ("cat", "not-a-path", ""),
            ("cd", "/nope/nope/nope/nope", ""),  # outside vocab
            ("write", "/a/notes.txt", "c99"),
        ]:
            res = actions.apply(state, action)
            self.assertEqual(res.ttype, actions.INVALID, msg=f"{action}")
            self.assertEqual(res.state, state)


if __name__ == "__main__":
    unittest.main()
