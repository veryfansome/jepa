"""Validated goal-exemplar constructor (terminal-jepa.md §5): per predicate type, every
generated exemplar satisfies both env invariants and the predicate — including the case
that motivated the requirement, a file predicate whose parent directory doesn't exist."""

# Bootstrap: unittest discovery may import this file as a top-level module from any
# cwd, skipping tests/__init__.py — put the project root on sys.path here.
import pathlib as _pathlib
import sys as _sys

_ROOT = str(_pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import random
import unittest

from datagen.layouts import gen_layout, predicate_universe
from env import vocab
from env.state import CwdIs, FileAbsent, FileExistsWithClass, make_satisfying


class TestConstructor(unittest.TestCase):
    def test_all_predicate_types_on_random_states(self):
        rng = random.Random(20)
        preds = predicate_universe()
        for _ in range(400):
            state = gen_layout(rng)
            pred = rng.choice(preds)
            for v in make_satisfying(state, pred, rng, n_variants=3):
                self.assertEqual(v.invariant_violations(), [])
                self.assertTrue(pred.check(v), msg=f"{pred!r}")

    def test_missing_parent_case_creates_parents(self):
        rng = random.Random(21)
        state = gen_layout(rng)
        # Deep file path whose parents are absent from this layout.
        target = None
        for p in vocab.FILE_PATHS:
            if len(p) == 3 and p[:-1] not in state.dirs and p[:1] not in state.dirs:
                target = p
                break
        self.assertIsNotNone(target)
        (v,) = make_satisfying(state, FileExistsWithClass(target, 3))
        self.assertEqual(v.invariant_violations(), [])
        self.assertEqual(v.files[target], 3)
        self.assertIn(target[:-1], v.dirs)

    def test_minimal_edit_preserves_irrelevant_features(self):
        rng = random.Random(22)
        for _ in range(100):
            state = gen_layout(rng)
            pred = FileAbsent(rng.choice(sorted(state.files)) if state.files
                              else vocab.FILE_PATHS[0])
            (v,) = make_satisfying(state, pred)
            self.assertEqual(v.cwd, state.cwd)
            self.assertEqual(v.dirs, state.dirs)
            others = {f: k for f, k in state.files.items() if f != pred.path}
            self.assertEqual({f: k for f, k in v.files.items()}, others)

    def test_variants_vary_only_cwd(self):
        rng = random.Random(23)
        state = gen_layout(rng)
        pred = CwdIs(())
        variants = make_satisfying(state, pred, rng, n_variants=3)
        # CwdIs pins cwd, so extra variants would break the predicate; constructor must
        # still return only predicate-satisfying states.
        for v in variants:
            self.assertTrue(pred.check(v))


if __name__ == "__main__":
    unittest.main()
