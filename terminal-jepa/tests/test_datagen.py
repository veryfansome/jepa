"""Datagen: determinism, split disjointness, transition-type mix, invalid quota, and
banner independence from layouts (terminal-jepa.md §3 exit criteria)."""

# Bootstrap: unittest discovery may import this file as a top-level module from any
# cwd, skipping tests/__init__.py — put the project root on sys.path here.
import pathlib as _pathlib
import sys as _sys

_ROOT = str(_pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import pathlib
import tempfile
import unittest

from datagen.generate import generate
from datagen.layouts import make_layout_split, predicate_split

ARGS = dict(n_train=30, n_val=10, steps=16, invalid_quota=0.15, epsilon=0.15, seed=7,
            n_layouts=60)


class TestDatagen(unittest.TestCase):
    def _run(self, out):
        return generate(out, **ARGS)

    def test_deterministic_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = pathlib.Path(td, "a"), pathlib.Path(td, "b")
            self._run(a)
            self._run(b)
            for name in ["train.jsonl", "val.jsonl", "manifest.json", "summary.json"]:
                self.assertEqual(
                    (a / name).read_bytes(), (b / name).read_bytes(), msg=name
                )

    def test_splits_disjoint(self):
        train_l, val_l = make_layout_split(60, 0.2, 7)
        self.assertFalse({l for l, _ in train_l} & {l for l, _ in val_l})
        self.assertTrue(train_l and val_l)
        train_p, val_p = predicate_split()
        self.assertFalse(set(train_p) & set(val_p))
        self.assertTrue(train_p and val_p)

    def test_layout_key_excludes_cwd(self):
        import random

        from datagen.layouts import gen_layout, layout_key

        rng = random.Random(30)
        for _ in range(50):
            st = gen_layout(rng)
            st2 = st.copy()
            st2.cwd = () if st.cwd != () else sorted(st.dirs)[0]
            self.assertEqual(layout_key(st), layout_key(st2))
            self.assertNotEqual(st.state_id(), st2.state_id())

    def test_summary_mix_and_quota(self):
        with tempfile.TemporaryDirectory() as td:
            summary = self._run(pathlib.Path(td, "d"))
            mix = summary["transition_mix"]
            for t in ["state-changing", "valid-no-op", "invalid"]:
                self.assertIn(t, mix)
                self.assertGreater(mix[t], 0.0)
            # Quota is enforced per step at the trajectory level, so realized rate
            # tracks it regardless of policy mix.
            self.assertGreater(mix["invalid"], ARGS["invalid_quota"] - 0.05)
            self.assertLess(mix["invalid"], ARGS["invalid_quota"] + 0.05)
            self.assertGreaterEqual(len(summary["invalid_coverage_by_failure_type"]), 4)
            self.assertIn("max_state_dirs", summary)
            self.assertIn("max_state_files", summary)

    def test_banner_independence_and_spread(self):
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td, "d")
            summary = self._run(out)
            self.assertGreater(summary["banners_used"], 10)
            # Banners must recur across layouts rather than pinning to them.
            self.assertGreater(
                summary["distinct_banner_layout_pairs"],
                summary["banners_used"],
            )
            # Same banner stream regardless of layout pool: banners keyed by
            # (seed, split, index) only, so trajectory i keeps its banner even if
            # layouts change (independence by construction).
            rows = [
                json.loads(l)
                for l in (out / "train.jsonl").read_text().splitlines()
            ]
            args2 = dict(ARGS)
            args2["n_layouts"] = 80  # different layout pool, same seed
            out2 = pathlib.Path(td, "d2")
            generate(out2, **args2)
            rows2 = [
                json.loads(l)
                for l in (out2 / "train.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [r["banner_id"] for r in rows], [r["banner_id"] for r in rows2]
            )

    def test_states_in_dataset_satisfy_invariants(self):
        from env.state import FsState

        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td, "d")
            self._run(out)
            for line in (out / "train.jsonl").read_text().splitlines():
                traj = json.loads(line)
                for step in traj["steps"]:
                    st = FsState.from_json(step["state_after"])
                    self.assertEqual(st.invariant_violations(), [])


if __name__ == "__main__":
    unittest.main()
