"""DG-4a at unit level: replay BOTH collected mints through the SST, docker-FREE.

Institutionalized from the round-4 adversarial review's 249K replay: every
recorded v1 (data/dockerfs) + v2 (data/dockerfs2) step is folded through
realenv.shell_state.ShellState in sst mode, asserting

  1. parser TOTALITY — every recorded command parses (zero ParseErrors);
  2. verbsig BIT-IDENTITY — sig(cmd) == the first-token verb, zero raises
     (the v1/v2 verb_of contract);
  3. the GOLDEN RULE — every determined predict() equals the record EXACTLY
     (ZERO wrong determined predictions — the mint-scale soundness gate);
  4. determinism — a sample of sequences double-folded, snapshots equal.

The determined-prediction count is printed so coverage regressions are visible
(soundness fixes must not silently collapse the determined surface).

Skips cleanly when a data root is absent (data is gitignored/regenerable); at
least one root must exist for the test to run.
"""

import json
import pathlib as _pathlib
import random
import sys as _sys
import unittest

_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from realenv import shell_state as M
from realenv import verbsig as V
from realenv.shell_state import BOT, ParseError, ShellState

ROOTS = [(_ROOT / "data" / "dockerfs", "v1"),
         (_ROOT / "data" / "dockerfs2", "v2")]
AVAILABLE = [(r, tag) for r, tag in ROOTS
             if any((r / f"{s}.jsonl").exists() for s in ("train", "val"))]


@unittest.skipUnless(AVAILABLE, "no mint data root present (data/ is regenerable)")
class TestMintReplayGoldenRule(unittest.TestCase):

    def test_full_replay_zero_wrong_determined(self):
        rng = random.Random(0)
        seqs = steps = det = wrong = twice = nondet = 0
        parse_fail, sig_fail, wrong_ex = [], [], []
        for root, tag in AVAILABLE:
            for split in ("train", "val"):
                path = root / f"{split}.jsonl"
                if not path.exists():
                    continue
                with open(path) as fh:
                    for line in fh:
                        seq = json.loads(line)
                        seqs += 1
                        st = ShellState(mode="sst")
                        folds = [st]
                        dbl = rng.random() < 0.03
                        if dbl:
                            folds.append(ShellState(mode="sst"))
                            twice += 1
                        for i, s in enumerate(seq["steps"]):
                            steps += 1
                            cmd = s["cmd"]
                            try:
                                M.parse_command(cmd)
                            except ParseError as e:
                                if len(parse_fail) < 5:
                                    parse_fail.append((cmd, str(e)[:120]))
                                continue
                            try:
                                if V.sig(cmd) != cmd.split()[0] and len(sig_fail) < 5:
                                    sig_fail.append(cmd)
                            except ValueError as e:
                                sig_fail.append((cmd, str(e)[:80]))
                            rec = {"output": s["output"], "exit": s["exit"],
                                   "cwd": s["cwd"]}
                            pred = st.predict(st.vt, cmd)
                            if pred is not BOT:
                                det += 1
                                if pred != rec:
                                    wrong += 1
                                    if len(wrong_ex) < 5:
                                        wrong_ex.append(
                                            (tag, seq.get("image"), i, cmd,
                                             (pred["output"][:90], pred["exit"]),
                                             (s["output"][:90], s["exit"])))
                            for f in folds:
                                f.fold({"cmd": cmd, **rec})
                        if dbl:
                            a, b = folds
                            if (a.cwd, a.fs, a.ws, a.jobs, a.touched, a.fs_clock) != \
                               (b.cwd, b.fs, b.ws, b.jobs, b.touched, b.fs_clock):
                                nondet += 1
        print(f"\n[mint-replay] roots={[t for _, t in AVAILABLE]} seqs={seqs} "
              f"steps={steps} determined={det} wrong={wrong} "
              f"double_folded={twice} nondet={nondet}")
        self.assertFalse(parse_fail, f"parser totality broke: {parse_fail}")
        self.assertFalse(sig_fail, f"verbsig bit-identity broke: {sig_fail}")
        self.assertEqual(wrong, 0,
                         f"GOLDEN-RULE violations on mint data: {wrong_ex}")
        self.assertEqual(nondet, 0, "double-fold determinism broke")
        self.assertGreater(det, 0, "determined surface collapsed to zero")


if __name__ == "__main__":
    unittest.main(verbosity=2)
