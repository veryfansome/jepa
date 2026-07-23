# dockerfs3 v3.0 mint runbook (the ONE command sequence; constitution §1/§3/§9; prereg §8)

Owner: prereg assembler. Models `benchmarks/mint-runbook.md` (v2) and extends it for the v3
two-arm mint, the seven-arm SST/wtm precompute, and the B1 classes/policy stamps.

**Pre-conditions (all required before step 1):**
- The review loop recorded CONVERGED GO (dated GO amendment in `benchmarks/dockerfs3-prereg.md` §11); the GO-WITH-FIXES fix set (B1/S2 code + S1/S4/S5 amendments + this runbook) has LANDED (§4.8).
- Working tree clean at the runbook's GO commit; `data/dockerfs3` and `data/dockerfs3-ablate` absent or empty.
- `benchmarks/dockerfs3-classes.json` matches its pinned sha `08b31deeb16269c2a9d2df338c35d6a6a2f6e733c36d34c7ee5e1c853a2c24e4` (the mint fail-closes on drift — B1, code-asserted, so this is a belt-and-suspenders manual check).
- Docker up; the 12 mint images present (the mint runs BY DIGEST via `--expect-digests`; a drift aborts).
- Mint host lightly loaded (the 8s per-command cap is all-or-nothing under load — MINOR-A; a load spike fails closed via DG-10b/G-EM, restart cost only).

Legend: **[CODE-ASSERTED]** = the harness/CLI enforces it and aborts fail-closed; **[MANUAL]** = an operator step or review with no code gate.

---

## 0. PRE-MINT ROUND-TRIP GATE (mandatory; the anti-false-GO tripwire — run BEFORE the 2h mint) [MANUAL to launch, CODE-ASSERTED within]

The gap that hid B1 was that no test ran mint→reencode→resolve→score end to end. This gate is that test at tiny scale; it MUST pass green before the real mint starts. It needs no digests (n_seqs<100 keeps the digest entry gate inert) and takes ~minutes.

```sh
# tiny two-arm mint (fresh containers; n<100 so the digest gate is inert)
uv run python -m realenv.collect_docker --out /tmp/dockerfs3-rt --policy v3 --arm both \
  --seqs-per-image 4 --seq-len 28 --seed 0 --workers 2 \
  --train-images "alpine:latest,debian:stable-slim" --val-images "fedora:latest,rockylinux:9"

# encode BOTH arms (ablate is train-only — reencode skips the absent val split, F6)
uv run python -m evolve.reencode --perception enc_e5_base --src /tmp/dockerfs3-rt         --out /tmp/dockerfs3-rt-e5
uv run python -m evolve.reencode --perception enc_e5_base --src /tmp/dockerfs3-rt-ablate  --out /tmp/dockerfs3-rt-ablate-e5

# the seven-arm SST/wtm precompute on the FULL root (emits sst-val.pt + wtm-val.pt)
uv run python -m evolve.precompute_baselines --root /tmp/dockerfs3-rt-e5

# resolve must return a NON-EMPTY content-cell set (B1: no crash on the mint's own stamps),
# and score >=1 seed must return a finite margin (proves the whole eval path runs)
uv run python -c "import sys; sys.path.insert(0,'.'); \
from evolve import bench_versions as BV; s=BV.resolve('/tmp/dockerfs3-rt-e5'); \
assert s['cell_based'] and s['content'], s; print('resolve OK, content cells:', len(s['content']))"
uv run python -m evolve.cli score --genome <a-registered-v3-genome.json> --mode proxy \
  --split inner --data /tmp/dockerfs3-rt-e5
```

Expected: `resolve OK`; the score returns `guardrail: pass` with a finite `fitness` (at this tiny scale the S2 coverage-demotion will route MOST/ALL content cells to `<cell>-lowcov` — the margin may be near-empty; the gate checks the pipeline RUNS and resolves, not the number). Any exception here BLOCKS the real mint — fix and re-run. Delete `/tmp/dockerfs3-rt*` after. Unit-scale versions of this gate live in `tests/test_collect_v3.py` (B1 stamps + resolve) and `tests/test_v3_scoring_gates.py` (S2 demotion + require_v3_cache).

---

## 1. THE MINT — single collection event, two arms, digest-gated [CODE-ASSERTED]

One command. `--arm both` mints the full arm (both splits) into `data/dockerfs3`, then the paired ablate arm (mutation/composition/time OFF, weights renormalized) TRAIN-ONLY into `data/dockerfs3-ablate`. `--pin-digests` + `--expect-digests` are REQUIRED at `collect()` entry (SystemExit otherwise); collection runs by digest and aborts on drift. B1: `summary["classes_sha"]` (fail-closed == the pin) and `summary["policy_sha"]` (the `lexicon_hashes()` content hash) are stamped AUTOMATICALLY into both roots — no manual stamp step.

```sh
uv run python -m realenv.collect_docker --out data/dockerfs3 --policy v3 --arm both \
  --pin-digests --expect-digests benchmarks/dockerfs3-digests.json \
  --seqs-per-image 600 --seq-len 28 --seed 0 --workers 6 \
  --train-images "alpine:latest,ubuntu:latest,debian:stable-slim,python:3.12-slim,redis:7-alpine,nginx:stable-alpine,postgres:16-alpine,node:22-slim" \
  --val-images "fedora:latest,rockylinux:9,mariadb:latest,httpd:2.4"
```

Budget (§7): full ≈1.7h + ablate ≈0.85h ≤ the 3h G-BUDGET. The ablate arm defaults to half the full seqs/image (`--ablate-seqs` overrides). Gates run as CODE ASSERTS during the mint (digest drift, MutGuard, DG-10a fail-fast on any skipped/dead image, per-field determinism); MINT-CONFIRMED gates (G-COV post-purge, DG-9 composed ≥6%, θ1 ejection-safety) are re-checked at the 600/900-seqs/image scale — **if DG-9 composed still breaches, the mint stays blocked** pending the §4.3 fallback fork or a dated GO amendment (prereg §11 criterion 4). Orphan sweep after any abort: `docker ps -aq --filter label=tj3-mint=0 | xargs docker rm -f`.

## 2. ENCODE the full scoring root [CODE-ASSERTED stamp/version binding]

Copies `summary.json` forward (version identity + the B1 classes_sha/policy_sha), adds the perception stamp `{impl, model, content_sha}`, and writes the root-level `cache_meta.json` (`cache_format:3` + policy_sha + classes_sha + built_summary_sha) that `require_v3_cache` enforces at scoring.

```sh
uv run python -m evolve.reencode --perception enc_e5_base --src data/dockerfs3 --out data/dockerfs3-e5
```

## 3. ENCODE the ablate arm — TRAIN-ONLY tolerance [CODE-ASSERTED]

The ablate raw root ships NO `val.jsonl`; `reencode` skips the absent split (F6) instead of raising. The ablate arm is scored against the FULL root's val split (§11.5 comparison arm), so it needs no val encode and no precompute of its own.

```sh
uv run python -m evolve.reencode --perception enc_e5_base --src data/dockerfs3-ablate --out data/dockerfs3-ablate-e5
```

## 4. PRECOMPUTE the seven-arm SST / within_traj_mut baselines (FULL root only) [CODE-ASSERTED alignment]

Emits `sst-val.pt` + `wtm-val.pt` over the FULL val split (inner AND final), rendered/encoded through the root's OWN perception module with `render_canon.canon` applied, aligned to `_data_tensors` step order (per-seq (image,len) manifest checked at scoring — a stale precompute fails closed). Only the full root has a val split, so precompute runs on `data/dockerfs3-e5` alone.

```sh
uv run python -m evolve.precompute_baselines --root data/dockerfs3-e5
```

Instrument caveat (MINOR C-4): `precompute_baselines.main` passes `error_templates=None`, so SST miss/error predictions are un-templated — D3-EXCLUDED (report channel only), not a fitness effect. Wiring `sst_error_templates` is a v3.1 instrument-honesty follow-up.

## 5. PUBLICATION SCAN (must exit 0; fail-closed on vacuous scans) [CODE-ASSERTED]

```sh
uv run python -m benchmarks.scan_publish --root data/dockerfs3
```

## 6. VERIFY version binding + cache staleness gate [CODE-ASSERTED]

```sh
uv run python -c "import sys; sys.path.insert(0,'.'); from evolve import bench_versions as BV; \
s=BV.resolve('data/dockerfs3-e5'); assert s['version']=='dockerfs3-v3.0' and s['content'], s; \
BV.require_v3_cache('data/dockerfs3-e5'); print('resolve+require_v3_cache OK:', len(s['content']), 'content cells')"
```

`resolve` asserts the root's `classes_sha` matches the frozen table verbatim (mint-vs-scoring class-table match, §1.1); `require_v3_cache` asserts the non-empty `cache_format:3` + policy_sha + classes_sha stamps agree between `cache_meta.json` and `summary.json` (B1 — falsy stamps now REJECTED, not passed vacuously).

## 7. RE-BASELINE + HARVEST + RECORD [MANUAL]

Re-baseline the incumbent on the v3 root (scores are comparable only within one environment — MPS vs CUDA differ beyond proxy noise; see `cloud/README.md`); the archived-margin regression must reproduce the v1 0.5848 and v2 r13 0.4781 bit-identically under the v3 harness (v1/v2 paths are structurally untouched by the v3 additions — §9 test). Harvest plangoals-v3; record the promoted numbers in `../README.md` + the `evolve-insights` memory. HF publication of the `dockerfs3/` + `dockerfs3-e5/` subtrees is canonical thereafter.

---

## ABORT DISPOSITION (pre-committed; constitution §1, v2 Amendment-5 precedent)

On ANY failure in step 1, delete `data/dockerfs3` AND `data/dockerfs3-ablate` wholesale and re-run step 1 in full. The completed run is the version's single collection event. No splicing, no resume, no `--train-only` patching. A failure in steps 2–6 is a derived-artifact rebuild (re-run that step and the ones after it), NOT a re-mint.
