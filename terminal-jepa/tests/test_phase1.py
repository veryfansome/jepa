"""Phase 1 smoke tests: batch construction, encoder/action/predictor forward, one
backward pass through every loss arm, and the probe pipeline — all on CPU over a tiny
generated dataset. Catches import/path/tensor-shape regressions cheaply. Skipped when
torch is not installed (Phase 0 remains stdlib-only)."""

# Bootstrap: unittest discovery may import this file as a top-level module from any
# cwd, skipping tests/__init__.py — put the project root on sys.path here.
import pathlib as _pathlib
import sys as _sys

_ROOT = str(_pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import random
import tempfile
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed; Phase 1 tests need the .venv python")

from datagen.generate import generate

_TMP = tempfile.TemporaryDirectory()
generate(_TMP.name, n_train=4, n_val=2, steps=6, invalid_quota=0.2, epsilon=0.2,
         seed=3, n_layouts=12)
TRAIN = f"{_TMP.name}/train.jsonl"
VAL = f"{_TMP.name}/val.jsonl"


class TestPhase1Smoke(unittest.TestCase):
    def test_windows_and_all_loss_arms_backward(self):
        from models import losses, nets
        from models.data import TrajectoryData
        from train.train import HORIZON, encode_all

        data = TrajectoryData(TRAIN, "both", keep_states=False)
        b = data.sample_windows(2, HORIZON, random.Random(0))
        self.assertEqual(b["obs"].shape[:2], (2, HORIZON + 1))
        self.assertEqual(b["acts"].shape[:2], (2, HORIZON))

        torch.manual_seed(0)
        m = nets.build_models()
        z = encode_all(m["encoder"], b["obs"])
        bs, s, la = b["acts"].shape
        a = m["action_encoder"](b["acts"].reshape(bs * s, la)).reshape(bs, s, -1)
        zhat = m["predictor"](z[:, 0], a[:, 0])
        self.assertEqual(zhat.shape, z[:, 1].shape)

        z_flat = z.reshape(-1, z.shape[-1])
        loss = (
            ((zhat - z[:, 1]) ** 2).mean()
            + losses.sigreg(z_flat, n_directions=32)
            + losses.vicreg_var_cov(z_flat)
            + losses.temporal_similarity(z[:, 0], z[:, 1])
            + nets.IDMHead().loss(z[:, 0], z[:, 1], b["act_labels"][:, 0])[0]
            + nets.ReconDecoder().loss(z[:, 0], a[:, 0], b["obs"][:, 1])
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        grads = [p.grad for p in m["encoder"].parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_slot_variant_forward_backward(self):
        from models import losses, nets
        from models.data import TrajectoryData
        from train.train import HORIZON, encode_all

        data = TrajectoryData(TRAIN, "both", keep_states=False)
        b = data.sample_windows(2, HORIZON, random.Random(1))
        torch.manual_seed(0)
        m = nets.build_models("slot")
        self.assertEqual(m["encoder"].d_out, 16 * 64)
        z = encode_all(m["encoder"], b["obs"])
        self.assertEqual(z.shape[-1], 1024)
        bs, s, la = b["acts"].shape
        a = m["action_encoder"](b["acts"].reshape(bs * s, la)).reshape(bs, s, -1)
        zhat = m["predictor"](z[:, 0], a[:, 0])
        self.assertEqual(zhat.shape, z[:, 1].shape)
        loss = ((zhat - z[:, 1]) ** 2).mean() + losses.sigreg(
            z.reshape(-1, z.shape[-1]), n_directions=32
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        grads = [p.grad for p in m["encoder"].parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_per_index_sigreg_penalizes_static_slot_offsets(self):
        """Fidelity-audit regression: pooled per-slot SIGReg was 87%-satisfied by
        static slot-identity offsets. The per-index test must penalize a code whose
        variance comes from constant slot means rather than per-sample information."""
        from models import losses

        torch.manual_seed(0)
        n, k, d = 256, 16, 64
        centered = torch.randn(n, k, d)
        offsets = torch.randn(1, k, d) * 2.0
        shifted = centered * 0.1 + offsets
        self.assertGreater(
            losses.sigreg_per_index(shifted).item(),
            2 * losses.sigreg_per_index(centered).item(),
        )

    def test_standardize_uses_fit_stats_and_bounds_constant_columns(self):
        from probes.probe import standardize

        torch.manual_seed(0)
        z_fit = torch.randn(100, 8)
        z_fit[:, 3] = 7.0  # constant column: std clamps at eps, must not explode
        z_eval = torch.randn(50, 8) + 2.0
        s_fit, s_eval = standardize(z_fit, z_eval)
        self.assertTrue(torch.isfinite(s_fit).all() and torch.isfinite(s_eval).all())
        self.assertEqual(s_fit[:, 3].abs().max().item(), 0.0)
        # eval must be transformed with FIT statistics (mean shift preserved)
        self.assertGreater(s_eval[:, 0].mean().item(), 1.0)

    def test_slot_regularizer_prefers_unit_scale_over_contraction(self):
        """Adversarial-review regression: with VICReg's 25:1 weighting and cross-fitted
        covariance, a contracted code (z_std 0.35) must score clearly WORSE than an
        honest N(0,I) code. Under the 1:1-weighted plain estimator the contracted code
        scored 9.3x BETTER (rank floor ~7 at unit scale), making contraction the
        regularizer's own optimum."""
        from models import losses

        torch.manual_seed(0)
        n, k, d = 128, 16, 64

        def combo(z_slots):
            flat = z_slots.reshape(n, k * d)
            return (losses.sigreg_per_index(z_slots)
                    + losses.vicreg_var_cov(flat[: n // 2], flat[n // 2 :])).item()

        honest = torch.randn(n, k, d)
        contracted = honest * 0.35
        self.assertLess(combo(honest), 0.5 * combo(contracted))

    def test_crossfit_cov_detects_rotated_duplicate_slots(self):
        """Duplication is a population property: cross-fitting must still detect 16
        slots that are rotated copies of one source (rotation cannot evade a Frobenius
        penalty — ||R_i R_j^T||_F = ||I||_F)."""
        from models import losses

        torch.manual_seed(0)
        n, k, d = 128, 16, 64
        src = torch.randn(n, d)
        rots = [torch.linalg.qr(torch.randn(d, d))[0] for _ in range(k)]
        dup = torch.stack([src @ r for r in rots], dim=1).reshape(n, k * d)
        honest = torch.randn(n, k * d)

        def reg(z):
            return losses.vicreg_var_cov(z[: n // 2], z[n // 2 :]).item()

        self.assertGreater(reg(dup), reg(honest) + 1.0)

    def test_predictors_are_identity_at_init(self):
        """Load-bearing invariant (status doc finding 9): a freshly initialized
        predictor must map (z, a) -> z exactly, for every encoder variant. Violating
        this makes prediction loss large at init and encoder collapse the fastest
        descent path — SIGReg alone does not win that race."""
        from models import nets

        torch.manual_seed(0)
        for enc_type in nets.ENCODER_TYPES:
            m = nets.build_models(enc_type)
            d = m["encoder"].d_out
            z = torch.randn(4, d)
            a = torch.randn(4, nets.D_ACT)
            with torch.no_grad():
                zhat = m["predictor"](z, a)
            self.assertTrue(
                torch.allclose(zhat, z, atol=1e-6),
                msg=f"{enc_type} predictor is not identity at init",
            )

    def test_probe_pipeline(self):
        import torch.nn as nn

        from models import nets
        from models.data import TrajectoryData
        from probes.probe import (N_CWD, chance_floors, evaluate, extract, fit_head,
                                  make_head)

        torch.manual_seed(0)
        enc = nets.TokenEncoder().eval()
        d = TrajectoryData(VAL, "both")
        z, y = extract(enc, d.probe_examples(), torch.device("cpu"))
        self.assertEqual(z.shape[0], len(d.trajs) * 7)  # 6 steps -> 7 states
        floors = chance_floors(y)
        self.assertIn("cwd_majority", floors)

        head = fit_head(
            make_head("linear", nets.D_Z, N_CWD), z, y["cwd"],
            nn.functional.cross_entropy, torch.device("cpu"), steps=3, bs=64,
        )
        from probes.probe import N_BANNERS, N_CLS, N_FILES

        heads = {
            "cwd": head,
            "exists": fit_head(
                make_head("linear", nets.D_Z, N_FILES), z, y["exists"],
                nn.functional.binary_cross_entropy_with_logits,
                torch.device("cpu"), steps=3, bs=64,
            ),
            "cls": make_head("linear", nets.D_Z, N_FILES * N_CLS),
            "banner": make_head("linear", nets.D_Z, N_BANNERS),
        }
        out = evaluate(heads, z, y, torch.device("cpu"))
        for k in ["cwd_acc", "exists_balacc", "cls_acc_given_exists", "macro", "banner_acc"]:
            self.assertIn(k, out)


if __name__ == "__main__":
    unittest.main()
