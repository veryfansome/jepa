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
