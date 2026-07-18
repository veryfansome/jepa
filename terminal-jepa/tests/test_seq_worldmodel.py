"""Sequence world model (realenv/seq_worldmodel.py): the causal no-future-leakage guarantee
and the retrieval metric's calibration. torch-only; skipped if torch is absent.

The leakage test is the regression guard for the core claim: the prediction at command
position t must depend ONLY on the exploration history + command t, never on observation t
(which doesn't exist yet) or any later token. A refactor of the cmd/obs interleaving or the
even-index slicing would silently reintroduce leakage; this catches it."""

import pathlib as _pathlib
import sys as _sys

_ROOT = str(_pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import unittest

try:
    import torch
    from realenv import seq_worldmodel as M
    HAVE_TORCH = True
except Exception:  # torch not installed in the stdlib test env
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "requires torch")
class TestNoLeakage(unittest.TestCase):
    def _one_seq(self, n=6, seed=0):
        torch.manual_seed(seed)
        return [{"z_obs": torch.randn(n, M.D), "z_cmd": torch.randn(n, M.D),
                 "cmds": ["ls /a"] * n, "image": "img"}]

    def test_cmd_prediction_independent_of_own_and_future_obs(self):
        """Corrupting observation k must NOT change any command-position prediction at t<=k, and
        MUST change at least one later prediction (t>k legitimately sees obs k)."""
        net = M.SeqWorldModel("jepa").eval()
        seq = self._one_seq(n=6)
        dev = torch.device("cpu")
        b0 = M.collate(seq, dev)
        with torch.no_grad():
            cmd0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone()  # [1,6,D]
        k = 3
        b1 = M.collate(seq, dev)
        b1["tok"][0, 2 * k + 1] = torch.randn(M.D) * 100.0  # corrupt obs_k (odd index)
        with torch.no_grad():
            cmd1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2]
        change = (cmd1 - cmd0).abs().amax(-1)[0]  # [6]
        for t in range(k + 1):
            self.assertLess(change[t].item(), 1e-5, f"LEAK: cmd pred at t={t} moved when obs_{k} changed")
        self.assertGreater(change[k + 1:].max().item(), 1e-4, "expected a later position to see obs_k")

    def test_masked_model_ignores_history(self):
        """no_history=True (self-only attention) must make every cmd prediction invariant to ALL
        other tokens — the matched-capacity history-free control."""
        net = M.SeqWorldModel("jepa", no_history=True).eval()
        seq = self._one_seq(n=5)
        dev = torch.device("cpu")
        b0 = M.collate(seq, dev)
        with torch.no_grad():
            cmd0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone()
        b1 = M.collate(seq, dev)
        b1["tok"][0, 0] = torch.randn(M.D) * 50.0   # corrupt cmd_0 (position 0)
        b1["tok"][0, 3] = torch.randn(M.D) * 50.0   # corrupt obs_1
        with torch.no_grad():
            cmd1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2]
        change = (cmd1 - cmd0).abs().amax(-1)[0]
        # cmd_0's own prediction may move (its own token changed); cmds 1..4 must be unchanged.
        self.assertLess(change[1:].max().item(), 1e-5, "self-only model leaked across positions")


@unittest.skipUnless(HAVE_TORCH, "requires torch")
class TestRetrievalCalibration(unittest.TestCase):
    def test_predict_mean_is_chance(self):
        """A constant (mean) prediction must score ~1/(1+n_foils) top-1 — the retrieval metric's
        calibration floor. Ties (mean vs mean) must not count as beating the true candidate."""
        torch.manual_seed(0)
        N, foils = 800, 63
        true = torch.randn(N, M.D)
        verbs = ["ls"] * N
        r = M.retrieval(torch.zeros(N, M.D), true, verbs, n_foils=foils, rounds=4, seed=0)
        self.assertLess(abs(r["top1_random"] - 1.0 / (1 + foils)), 0.02,
                        f"predict-mean top1 {r['top1_random']:.4f} not ~chance {1/(1+foils):.4f}")

    def test_perfect_prediction_is_top1(self):
        """Predicting the true embedding exactly must retrieve it (top-1 = 1.0)."""
        torch.manual_seed(1)
        true = torch.randn(300, M.D)
        r = M.retrieval(true.clone(), true, ["cat"] * 300, n_foils=31, rounds=2, seed=0)
        self.assertGreater(r["top1_sameverb"], 0.999)


if __name__ == "__main__":
    unittest.main()
