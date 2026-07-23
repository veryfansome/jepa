"""MANDATORY PRE-MINT ROUND-TRIP SCORABILITY GATE (dockerfs3-runbook.md §0) — the anti-false-GO
tripwire that must pass green BEFORE the 2-hour v3 mint.

The gap that hid B1 was that NO test ran the whole mint -> reencode -> precompute -> resolve ->
score chain end to end on the mint's OWN stamped artifacts; every unit test stubbed one seam.
This test is that chain at TINY docker scale. It:

  1. mints a tiny two-arm v3 root (2 images incl. one bash; ~12 seqs/image; --arm both;
     collect_docker's real docker path, digest gate inert at n<100 per runbook §0),
  2. reencodes BOTH arms (full + train-only ablate) with `enc_e5_base`,
  3. precomputes the seven-arm SST/wtm baselines on the FULL root,
  4. and asserts the five GO-gate invariants (a)-(e) below on those artifacts.

Docker-gated (@skipUnless): needs docker up + alpine:latest (busybox) + fedora:latest (bash/GNU)
already present (never pulls). Runs the e5 encoder three times, so it takes a few minutes; that is
the price of exercising the EXACT path, not a stub.

  (a) the reencoded full root's summary carries classes_sha == the pinned 08b31dee... (B1 stamp
      travels through the mint -> reencode copy-forward);
  (b) bench_versions.resolve(root) returns a NON-EMPTY content-cell set (resolve no longer
      fail-closes / crashes on the mint's own stamps — the exact B1 gap);
  (c) require_v3_cache PASSES on the stamped root and RAISES on a stripped-stamp copy (B1
      falsy-stamp rejection — the guard fires per contract, not vacuously);
  (d) score_genome on a trivial identity-target genome returns a FINITE margin with
      guardrail=pass and the SEVEN baseline arms present (the whole eval path runs on the mint's
      artifacts, loading the precomputed sst/wtm arms);
  (e) the S2 coverage-demotion diagnostic reports the expected low-cov cells at this tiny scale.

TINY-SCALE FLOOR NOTE (why (d) scales the S2 floor, and only the floor): the S2 per-split
coverage floor is 500 SURVIVING content steps (harness._V3_GCOV_FLOOR — an explicitly
scoring-time, NON-frozen knob; the frozen classes.json is never touched). At ANY tiny scale every
content cell sits below 500, so the REAL floor demotes EVERY content cell -> the pooled content
subset is empty -> content_retrieval indexes an empty subset and the score returns
guardrail="exception" (harness is exception-safe: it never crashes the loop, but it also does NOT
falsely report pass — asserted in test (e2)). That empty-content regime cannot occur at the real
600-seqs/image mint (ls|hit|native / cat|hit|native clear 500 by orders of magnitude). So to make
(d) exercise the SCORING path on REAL surviving content cells at tiny scale, this test scales the
floor DOWN to the 2nd-highest content-cell count (keeping the top count-tier, demoting the rest) —
the honest predict-mean calibration guard (CHANCE_SLACK) is left UNMODIFIED and must still pass.
Assertion (e) uses the REAL 500 floor unmodified.
"""

import collections
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import realenv.collect_docker as C
from realenv import seq_worldmodel as M
from evolve import bench_versions as BV
from evolve import harness as H
from evolve import reencode as RE
from evolve import precompute_baselines as PB
from evolve.genome import baseline_genome
from evolve.splits import split_val

TRAIN_IMAGE = "alpine:latest"     # busybox (ash)
VAL_IMAGE = "fedora:latest"       # bash / GNU  -> matches splits.INNER_IMAGES ("fedora")
SEQS_PER_IMAGE = 12               # tiny; > 4 so the honest predict-mean calibration guard passes
PIN = "08b31deeb16269c2a9d2df338c35d6a6a2f6e733c36d34c7ee5e1c853a2c24e4"


def _docker_ready():
    """True iff docker is up AND both round-trip images are already present (never pulls)."""
    if not shutil.which("docker"):
        return False
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False
    except Exception:   # noqa: BLE001 (docker missing / daemon down / timeout)
        return False
    from realenv.docker_env import image_present
    return image_present(TRAIN_IMAGE) and image_present(VAL_IMAGE)


_DOCKER = _docker_ready()


def _cell_verbs_for(root, split="inner"):
    """Model-independent v3 cell pseudo-verbs + diag for a root's val `split` (the exact
    harness._v3_cell_verbs the score path runs), built from the raw val.jsonl with zero
    embeddings (only shapes + cmds + step meta are read). Returns (spec, verbs, diag)."""
    spec = BV.resolve(str(root))
    raw = [json.loads(l) for l in
           (pathlib.Path(root) / "val.jsonl").read_text().splitlines() if l]
    inner = split_val(raw, split)
    evalset = [{"z_obs": torch.zeros(len(s["steps"]), M.D),
                "z_cmd": torch.zeros(len(s["steps"]), M.D),
                "cmds": [x["cmd"] for x in s["steps"]], "image": s["image"]} for s in inner]
    verbs, _forced, diag = H._v3_cell_verbs(evalset, inner, spec)
    return spec, verbs, diag


@unittest.skipUnless(_DOCKER, f"docker + {TRAIN_IMAGE} + {VAL_IMAGE} required (round-trip mint)")
class TestV3RoundTrip(unittest.TestCase):
    """One shared mint -> reencode(x2) -> precompute (setUpClass); five GO-gate assertions."""

    tmp = None
    full = None          # raw full root (train alpine + val fedora)
    ablate = None        # raw ablate root (train-only alpine)
    full_e5 = None       # reencoded + precomputed full root (the scoring root)
    ablate_e5 = None     # reencoded train-only ablate root
    scaled_floor = None  # tiny-scale S2 floor for the scored assertion (d)

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="dockerfs3-rt-"))
        cls.full = cls.tmp / "rt"
        cls.ablate = cls.tmp / "rt-ablate"          # collect_docker --arm both appends -ablate
        cls.full_e5 = cls.tmp / "rt-e5"
        cls.ablate_e5 = cls.tmp / "rt-ablate-e5"

        # 1. THE TINY TWO-ARM MINT — real docker path; digest gate inert at n<100 (runbook §0).
        C.main(["--out", str(cls.full), "--policy", "v3", "--arm", "both",
                "--seqs-per-image", str(SEQS_PER_IMAGE), "--seq-len", "28",
                "--seed", "0", "--workers", "2",
                "--train-images", TRAIN_IMAGE, "--val-images", VAL_IMAGE])
        assert (cls.full / "summary.json").exists(), "full arm did not mint"
        assert (cls.ablate / "summary.json").exists(), "ablate arm did not mint"

        # 2. REENCODE BOTH ARMS (enc_e5_base). ablate is train-only -> reencode skips absent val (F6).
        RE.main(["--perception", "enc_e5_base", "--src", str(cls.full), "--out", str(cls.full_e5)])
        RE.main(["--perception", "enc_e5_base", "--src", str(cls.ablate), "--out", str(cls.ablate_e5)])

        # 3. PRECOMPUTE the seven-arm SST/wtm baselines on the FULL root (only it has a val split).
        PB.main(["--root", str(cls.full_e5)])

        # tiny-scale scored-floor: the real-floor lowcov map IS the full pre-demote content-cell
        # count distribution (at tiny scale every content cell is demoted). Keep the top count-tier.
        _spec, _verbs, diag = _cell_verbs_for(cls.full_e5)
        pre_counts = diag["lowcov"]
        assert pre_counts, "no content cells present in the mint's val split — cannot design the gate"
        distinct = sorted(set(pre_counts.values()), reverse=True)
        cls.scaled_floor = distinct[1] if len(distinct) > 1 else distinct[0]

    @classmethod
    def tearDownClass(cls):
        if cls.tmp and cls.tmp.exists():
            shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- (a) classes_sha stamp forward
    def test_a_reencoded_summary_carries_pinned_classes_sha(self):
        summ = json.loads((self.full_e5 / "summary.json").read_text())
        self.assertEqual(summ.get("classes_sha"), PIN,
                         f"reencoded full root classes_sha != pin: {summ.get('classes_sha')}")
        # the B1 stamp also travels into the cache_meta.json reencode wrote (require_v3_cache reads it)
        cm = json.loads((self.full_e5 / "cache_meta.json").read_text())
        self.assertEqual(cm.get("classes_sha"), PIN)
        self.assertEqual(cm.get("cache_format"), 3)
        self.assertTrue(cm.get("policy_sha") and cm.get("bench_version"))
        # ablate arm carries the same version identity
        asum = json.loads((self.ablate_e5 / "summary.json").read_text())
        self.assertEqual(asum.get("classes_sha"), PIN)
        self.assertTrue(asum.get("ablate") is True)

    # ---------------------------------------------------------------- (b) resolve -> non-empty content
    def test_b_resolve_returns_nonempty_content_cells(self):
        spec = BV.resolve(str(self.full_e5))
        self.assertTrue(spec.get("cell_based"), "v3 root did not resolve as cell_based")
        self.assertTrue(spec.get("content"), "resolve returned an EMPTY content-cell set")
        self.assertGreater(len(spec["content"]), 0)
        self.assertEqual(spec.get("classes_sha"), PIN)
        # the 7 baseline arms are declared on the resolved spec
        self.assertEqual(tuple(spec.get("arms", ())), BV.V3_ARMS)
        self.assertEqual(len(BV.V3_ARMS), 7)

    # ---------------------------------------------------------------- (c) require_v3_cache pass / raise
    def test_c_require_v3_cache_passes_and_strip_raises(self):
        # PASSES on the stamped, precomputed root (non-empty consistent stamps + perception stamp)
        cm = BV.require_v3_cache(str(self.full_e5))
        self.assertEqual(cm.get("cache_format"), 3)

        # RAISES on a stripped-stamp copy: falsy classes_sha in BOTH files (the exact B1 gap — a
        # pre-B1 guard passed this vacuously via None==None / ""=="")
        strip = self.tmp / "rt-e5-stripped-falsy"
        shutil.copytree(self.full_e5, strip)
        s = json.loads((strip / "summary.json").read_text()); s["classes_sha"] = ""
        (strip / "summary.json").write_text(json.dumps(s))
        c = json.loads((strip / "cache_meta.json").read_text()); c["classes_sha"] = ""
        (strip / "cache_meta.json").write_text(json.dumps(c))
        with self.assertRaises(ValueError) as ctx:
            BV.require_v3_cache(str(strip))
        self.assertIn("classes_sha", str(ctx.exception))
        self.assertIn("empty", str(ctx.exception))

        # RAISES on a copy with the cache_meta.json stamp removed entirely
        strip2 = self.tmp / "rt-e5-stripped-nocache"
        shutil.copytree(self.full_e5, strip2)
        (strip2 / "cache_meta.json").unlink()
        with self.assertRaises(ValueError):
            BV.require_v3_cache(str(strip2))

    # ---------------------------------------------------------------- (d) score -> finite / pass / 7 arms
    def test_d_score_genome_finite_pass_seven_arms(self):
        gen = baseline_genome()                      # trivial identity-target genome (no target chunk)
        self.assertNotIn("target", gen["chunks"])    # -> load_target defaults to identity

        base_cache = self.tmp / "rt-base-cache.json"
        orig_bc, orig_demote = H.BASE_CACHE, H._demote_lowcov
        floor = self.scaled_floor
        H.BASE_CACHE = base_cache
        H._demote_lowcov = lambda verbs, content, floor=None, _o=orig_demote, _f=floor: \
            _o(verbs, content, floor=_f)
        try:
            res = H.score_genome(gen, mode="proxy", data=str(self.full_e5),
                                 proxy_steps=200, split="inner")
        finally:
            H.BASE_CACHE, H._demote_lowcov = orig_bc, orig_demote

        self.assertEqual(res.get("guardrail"), "pass",
                         f"score did not pass its guardrails: {res.get('guardrail')}")
        fit = res.get("fitness")
        self.assertIsInstance(fit, float)
        self.assertTrue(math.isfinite(fit), f"fitness is not finite: {fit}")
        # the seven baseline arms were all computed (recorded into the base_cache entry)
        cache = json.loads(base_cache.read_text())
        self.assertEqual(len(cache), 1, "expected exactly one base_cache entry (proxy seed 0)")
        entry = next(iter(cache.values()))
        for arm in BV.V3_ARMS:
            self.assertIn(arm, entry, f"baseline arm {arm!r} missing from the 7-arm max")
        self.assertEqual(entry["base"], max(entry[a] for a in BV.V3_ARMS))
        # the v3 per-step diagnostic is surfaced on the (passing) result
        self.assertIsNotNone(res.get("v3_diag"))
        self.assertIn("lowcov", res["v3_diag"])

    # ---------------------------------------------------------------- (e) S2 coverage-demotion diag
    def test_e_s2_coverage_demotion_reports_lowcov_cells(self):
        # REAL 500 floor (unmodified): at tiny scale EVERY content cell is under-covered -> demoted.
        spec, verbs, diag = _cell_verbs_for(self.full_e5)
        lowcov = diag["lowcov"]
        self.assertTrue(lowcov, "S2 diagnostic reported no low-cov cells at tiny scale")
        # every demoted cell is a genuine content cell, below the floor
        for cell, cnt in lowcov.items():
            self.assertIn(cell, spec["content"], f"{cell} demoted but not a content cell")
            self.assertGreater(cnt, 0)
            self.assertLess(cnt, H._V3_GCOV_FLOOR)
        # the two guaranteed-frequent content cells appear in the low-cov report
        self.assertIn("cat|hit|native", lowcov)
        self.assertIn("ls|hit|native", lowcov)
        # tiny-scale property: NO content cell survives the real floor (pooled fitness margin empty),
        # and the demoted set == the content cells present (all routed to the report battery)
        self.assertFalse(any(v in spec["content"] for v in verbs),
                         "a content cell survived the 500 floor at tiny scale")
        self.assertFalse(any((v + "-lowcov") in spec["content"] for v in verbs),
                         "a '-lowcov' pseudo-verb leaked into the frozen content set")

    # ------ (e2) anti-false-GO: the real-floor score is exception-SAFE and does NOT falsely pass ----
    def test_e2_realfloor_score_is_exception_safe_not_false_pass(self):
        """Documents the tiny-scale real-floor regime (see the module TINY-SCALE FLOOR NOTE): with
        the unmodified 500 floor the content pool is empty, so the score must NOT return
        guardrail=pass (no false GO) and must NOT crash the loop (harness converts the empty-subset
        error to a recorded _fail). This is the anti-false-GO half of the tripwire."""
        base_cache = self.tmp / "rt-base-cache-realfloor.json"
        orig_bc = H.BASE_CACHE
        H.BASE_CACHE = base_cache
        try:
            res = H.score_genome(baseline_genome(), mode="proxy", data=str(self.full_e5),
                                 proxy_steps=50, split="inner")
        finally:
            H.BASE_CACHE = orig_bc
        # never a hard crash (returns a dict), and never a false pass on the empty content pool
        self.assertIsInstance(res, dict)
        self.assertNotEqual(res.get("guardrail"), "pass",
                            "real-floor tiny-scale score FALSELY reported pass on an empty content pool")
        self.assertEqual(res.get("fitness"), float("-inf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
