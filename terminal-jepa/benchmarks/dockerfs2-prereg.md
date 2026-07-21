# dockerfs2 (v2.0) mint — pre-registration

STATUS: DRAFT pending adversarial review A (constitution + this prereg + the selected
collection policy), then review B (collector implementation + pilot data audit) before the
full mint. Governed by `bench-constitution.md`.

## Toolset (4 → 9 verbs) and predicted classes

Kept: `uname` (echo/const), `cat` (content), `ls` (content), `cd` (echo).
Added (busybox∩GNU flag subsets; per-image `command -v` availability probes, skips recorded):
- `head -n K` / `tail -n K`, K ∈ {3,5,10,20} — predicted **content** (pure file content).
- `stat -c '%n %s %F %a' <path>` — predicted **semi-echo** (%n echoes the path).
- `find <dir> -maxdepth {2,3} [-type f|d] -name '<glob>'` — predicted **semi-echo**.
- `grep -F [-i|-c] -m 8 <token> <file>` — predicted **semi-echo** (pattern in every hit).
Rejected: `file` (absent in slim images), `wc`/`du` (deferred, retryable).
Predicted fitness set: content = {ls, cat, head, tail}. FINAL classes are decided by the
calibrated classifier (below) on pilot data and frozen here by amendment BEFORE full collection.

## Verb-class protocol (thresholds to be calibrated on v1, then committed)

Two axes per verb: (1) cross-image output constancy (leave-one-out exact/near-dup rate of
same-command outputs across train images); (2) cmd↔obs lexical containment (char-3gram
containment of the command's argument text in its own output). Thresholds are chosen ONCE on
v1 data such that ls/cat classify content, cd classifies echo, uname classifies const —
recorded here with the measured v1 scores before any v2 data exists. Coverage per verb
reported; verbs under the coverage floor default to semi-echo (conservative).

## Collection recipe (the mint)

- Images: the 12 v1 images, PINNED BY DIGEST (table appended at ratification; resolved once).
- Budget: **600 seqs/image × seq-len 24 (±4 → max 28 steps = 56 tokens; battery imagination
  +2 steps = 60 ≤ 64) × 12 images ≈ 173K steps (~2.2× v1)**. Seed 0. Both splits, one run,
  one policy (constitution §3). Split assignment unchanged: train 8 / inner fedora+mariadb /
  final rockylinux+httpd.
- Policy: selected from competing inventor designs + review A (slot below, filled at
  ratification). Requirements it must meet: history-linkage targets for the new verbs
  (~60% of head/tail/stat targets and ~50% of grep queries entailed by earlier same-sequence
  observations, `meta.linked` recorded per step), ~20% deliberate grep misses (absence is an
  outcome), head/tail duplicate-foil control (length floor or per-file K draw), lexicons
  content-hashed into summary.json, old-verb mass ≥ ~60% (ls/cat competence stays trained).
- POLICY SPEC: __filled after review A__.
- Encoded root: `data/dockerfs2-e5` (enc_e5_base — the champion space). Raw + e5 published to
  HF under `dockerfs2/`. Instruments re-harvested per constitution §6 (plangoals-v2 stratified
  by depth × first component; batteries sha-pinned by amendment).

## Re-baseline & what "progress on v2" means

Per constitution §6 (incumbent + 5-rung probe ladder + noise band + one final-test reading).
The fitness margin uses the v2 baseline set (max gains retrieve_by_cmd_within_traj). Progress
on v2 = full-budget inner gains over the re-baselined incumbent line under the frozen class
set; the first new-champion claim additionally requires the standard sanity arms plus the
within-net zeroed-history ablation (the post-Stage-2 standard control).

## Review plan (three rounds, all pre-committed)

- **A (before any code):** attack constitution + this prereg + the candidate policies.
- **B (before the full mint):** attack the collector implementation + a PILOT collection
  (~20 seqs/image): echo-audit calibration transfer, head/tail dup rates, per-image verb
  availability, truncation rates per verb, leakage of obs into cmd renders, linkage stats.
  The class freeze happens here, by amendment.
- **C (before evolution reopens):** attack the re-baseline (ladder order, baseline sanity,
  noise band, instrument stratification).

---
## Amendment 1 (2026-07-21, review A applied — prereg now ACTIVE)

**Class protocol (corrected + committed):**
- Both axes on RENDERED obs text. v1 calibration recorded: containment (cmd-arg 3-grams in
  rendered obs) mean uname 0.000 / ls 0.127 / cat 0.331 / cd 0.980. Threshold RULE
  committed: max-margin midpoint of the v1 separating gap → containment ≥ **0.656**
  (midpoint of cat 0.331 ↔ cd 0.980) ⇒ semi-echo/echo. The raw-output constancy axis is
  RETIRED (measured coverage-biased: cat 0.833 "constant" via Docker-injected configs);
  axis 1 is instead the harness's own per-verb no-history-baseline vs WM measurement on
  pilot+probe data (predictable-from-command-alone, the well-posed form).
- grep classified per MODE (hit and miss separately); `grep -c` DROPPED from the toolset
  (count outputs are near-const). Coverage floor: ≥ 30 cross-image command pairs per verb,
  else a fixed calibration probe battery (same command run on all 12 images) supplies the
  measurement — never silent defaulting.
**Policy (selection rubric + adoptee):** rubric = requirement compliance, pilot-measured
linkage/miss/old-mass/truncation vs targets, cross-image command-identity preservation,
determinism hygiene; model scores NEVER consulted (constitution §7). Adopted: **SYNTHESIS,
adversarial-coverage base** with review-A fixes: K = crc32(path) % KSET globally
(image-independent; 2K+1 line floor enforced by REJECTION from the head/tail pool, never K
reassignment); linkage pools mined ONLY from the render-visible prefix (output[:1000]);
transplant-miss primary for grep (token observed in file A, grepped in file B) + small
audited lexicon-miss arm, no -i on intended misses; per-verb within-sequence used-sets;
explicit imports; availability probes in-band with recorded skips.
**Expected-shift note:** the v1-verbs-only continuity slice is EXPECTED to read lower on v2
(baseline lift from linked/coherent collection + image drift) — it is a report, never a
regression signal.
**Review-B checklist (named items):** (1) linked=True entailment verified in the ENCODED
obs text; (2) per-IMAGE linked-rate floors; (3) collector meta threading end-to-end;
(4) realized (not intended) grep miss rates per image; (5) per-verb near-dup-after-render
rates incl. head/tail; (6) DockerBox._exec contract pinned by a test; plus the six original
audit items (echo transfer, dup, availability, truncation, cmd-render leakage, linkage).

---
## Amendment 2 (2026-07-21, pilot audit → CLASS FREEZE; final pre-mint state)

**Frozen verb-class table (v2.0 fitness):**
- **content (fitness):** ls, cat, head, tail, stat, find, grep-HIT.
- **excluded:** uname (axis-1 rbc 0.791), cd (constitutional echo; render containment 0.98
  on v1, mode-mixed 0.625 on v2), grep-MISS (near-constant empty renders).
Evidence (pilot, 12 images, 5.7–5.8K steps, e5 space): axis-2 rendered containment — all
new verbs < 0.656 (head 0.253 / tail 0.158 / stat 0.472 / find 0.028 / grep 0.140); axis-1
per-verb cmd-only predictability (same-verb foils) — head 0.327 / tail 0.338 / stat 0.393
≈ cat 0.398 (content-like); find 0.136 / grep 0.172 vs **within_traj 0.400 / 0.442** — the
constitutional baseline (in the margin max) is what makes find/grep honestly creditable.
**Tuned-pilot stats (policy as of commit after tuning; the mint's policy identity):** grep
realized-miss 0.302 (arm windows 0.12 transplant / 0.10 lexicon-miss / 0.50 self-bind);
grep linked 0.686; old-verb mass 0.669. **Target deviation disclosed:** hts linkage
achieves 0.507 vs the ~0.60 design aim (pool-limited — 2K+1 floor + availability; the
controller saturates). 0.50±0.05 is ADOPTED as the v2.0 spec value (report slices are
unaffected; ~16K linked hts steps at mint scale). Per-image floors remain review-B checks.
**Dup-after-render (200-char prefix) rates recorded:** find 0.636 (shared-prefix listings —
a same-verb foil-ambiguity report for the find slice), grep 0.312, ls 0.334, cat 0.296,
head 0.128, tail 0.067, stat 0.023. Truncation >1000ch: cat 0.233, ls 0.169, others ≤ 0.04.

---
## Amendment 3 (2026-07-21, review B → NO-GO fixes; supersedes Amendment 2's class table)

Review B (3 empirical reviewers + synthesizer) returned NO-GO with three verified blockers.
Dispositions, all applied BEFORE any mint step:
- **F1 binary-decode artifact (blocker):** 5.9% of pilot steps (19.4% of cat, 18.4% of head,
  12.5% of tail) recorded the host-side Python string "executor error: 'utf-8' codec ..." as
  the observation. Fixed in DockerBox._exec (bytes + errors="replace"). All Amendment-2
  frozen-class numbers are re-verified on the post-fix re-pilot before GO.
- **F2 degenerate find arm (blocker):** 97.6% of pilot find outputs empty (glob independent
  of root). Fixed: probe-verified (dir, glob) hit pairs + a controlled ~20% deliberate-empty
  arm (meta.intended_empty), mirroring the grep-miss design.
- **F3 stat echo channel (blocker):** %n echoes the full path into 362/362 outputs (46.8% of
  chars; zero-parameter echo predictor top-1 0.81–0.85 vs best baseline 0.377). **stat is
  reclassified SEMI-ECHO** (the constitutional borderline default; Amendment 2's own original
  prediction). Frozen v2.0 table: **content = {ls, cat, head, tail, find, grep-hit}**;
  semi-echo = {stat}; excluded = {uname, cd, grep-miss}. grep/find mode label rule pinned:
  **exit != 0 or empty output ⇔ miss** (recoverable from the jsonl; meta.hit recorded).
- Further applied: fixed per-image --hostname (kills the container-ID nonce; restores
  same-seed reproducibility), container lifetime 86400 + fail-fast on daemon errors,
  --train-only val-truncation guard, V2_MINE_CAP 1000→500 (encoded-visible linkage),
  class table + bench_version written into summary.json (constitution §4 assert now binds),
  mint aborts on any skipped image, deterministic jsonl image order.
- **Axis-1 protocol committed:** benchmarks/axis1_measure.py (frozen protocol incl.
  grep-hit-only within-traj comparison for review C).
- **Environment note:** the local plumbing constant moved 0.3869 → 0.3844 with the
  uv/torch-2.13 migration (bisect-verified: predates all v2 eval edits; edited and pristine
  code agree at 0.3844 exactly). Within-environment bit-identity remains the standard.
GO decision deferred to the post-fix re-pilot audit (F11).

---
## Amendment 4 (2026-07-21, review round 2 → ITERATE; dispositions)

Round 2 (2 fresh reviewers + convergence judge) confirmed all 13 register items fixed and
every frozen number reproduced on pilot 3, then found: 1 NEW blocker, 1 NEW serious, 3 NEW
minors. Dispositions, all applied:
- **BLOCKER — derived roots lost version identity:** reencode/mv_encode never copied
  summary.json; the primary scoring root would silently resolve v1 (demonstrated end-to-end).
  Fixed: both encoders copy summary.json; resolve() now FAILS CLOSED (unparseable summary
  raises; meta-bearing train.jsonl without summary raises). v1 roots unaffected (verified).
- **SERIOUS — find-probe pool collapse on glob-sparse images** (fedora inner-val: 1 hit pair
  → 1 distinct observation for all find-hits). Fixed: anchor dirs (/etc, /usr/share,
  /usr/lib) probed first, budget 16 → 64 dirs, early-stop at ≥ 10 distinct hit pairs.
- **Minors:** stored-output cap 64KB + meta.trunc_stored (binary payloads were ~60% of mint
  bytes; render window unaffected) — a policy-identity change, hence this amendment;
  grep_mode_rule string corrected to "exit!=0 or empty output => miss (excluded)";
  aborted-mint roots now unresolvable rather than silently v1.
- Register carry-over (minor, open): ls -l runtime-mount timestamp jitter across container
  starts. Pilot 4 (post-fix) re-verifies all frozen numbers; round 3 review follows per the
  convergence rule.

---
## Amendment 5 (2026-07-21, review round 3 → ITERATE; fix batch applied)

Round 3 found 1 NEW serious + 6 NEW minors. Dispositions, all applied:
- **SERIOUS — digest-pinning commitment fulfilled:** the ratification promise ("table
  appended") is discharged: `benchmarks/dockerfs2-digests.json` (tracked) carries the
  12-image sha256 table measured at pilot 4 (0/12 drift vs local at commit time), and the
  mint runbook command REQUIRES `--expect-digests benchmarks/dockerfs2-digests.json` —
  collect() aborts on any drift, binding the mint to the audited bytes.
- resolve() sniff extended to val.jsonl; class-table mirror check now compares the FULL
  recorded table (content, semi_echo, excluded, grep_mode_rule); collect() unlinks any
  stale summary.json at start (aborted-into-reused-dir roots stay unresolvable).
- **Abort disposition pre-committed:** on ANY mint abort, delete the output dir wholesale
  and re-run the single one-command mint; the completed run is the version's sole
  collection event (constitution §1/§3). No splicing, no resume.
- **Publication scan committed:** `benchmarks/scan_publish.py` — mandatory runbook step
  between mint and HF upload; publication requires exit 0. Public CA CERTIFICATE blocks
  are deliberately not flagged (public documents; v1 precedent); private-key material,
  crypt hashes, cloud/API tokens, JWTs, secret assignments, host-path leakage block.
  Pilot 4 scan: clean (0 findings).
- Deferred (known-open minors): probe anchor-count constant (deterministic + conservative);
  ls -l runtime-mount timestamp jitter; per-image hts floors.
Round 4 review follows per the convergence rule.

---
## Amendment 6 (2026-07-21, review round 4 → ITERATE; fix batch applied)

Round 4 verified the entire round-3 batch by execution and found 2 NEW serious + 6 minors —
all in review/publication infrastructure, none in the collector, policy, or frozen numbers.
Dispositions, all applied:
- **SERIOUS — scanner vacuous pass:** scan_publish now FAILS CLOSED (exit 2) on a missing
  root or when no train.jsonl is among the scanned files; recursive rglob; scanned-file
  count printed. Verified: pilot-4 clean (exit 0), nonexistent root (exit 2), planted
  private-key/crypt/token fixture detected 3/3 (exit 1).
- **SERIOUS — runbook committed:** `benchmarks/mint-runbook.md` (the one-command mint with
  both digest flags, scan, encode, verify, publish steps + the abort disposition); and the
  prose rule is now CODE: a full-scale v2 mint without --expect-digests raises.
- Minors: resolve() raises on a v2 summary lacking verb_classes; sniff message names the
  triggering split; stale-unlink extends to emb-seq-*.pt; artifact sha256 manifest
  (train/val.jsonl) written into summary at mint (constitution §9 binding); scanner regexes
  extended (yescrypt, $rounds=, ENCRYPTED PRIVATE KEY — pilot 4 re-verified clean);
  stratified plangoals-v2 harvester implemented (plan_env harvest --stratify, round-robin
  depth × first-component strata; review-C precondition).
Round 5 (focused) follows per the convergence rule.
