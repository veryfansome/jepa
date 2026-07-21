# Benchmark constitution (durable rules; owns every dockerfs version from v2 on)

STATUS: DRAFT pending adversarial review A. Becomes binding at its ratification commit;
amendments only by a new dated section, never by editing ratified text.

## 1. Version identity

A benchmark version is the tuple: (image digests, collection policy + lexicon content-hashes,
toolset, seq-len, split assignment, verb-class table, encoder recipes of its published roots).
Any change to the tuple mints a NEW version. A version is minted ONCE: one collection event,
one re-baselining event. Scores are comparable ONLY within a version. Cross-version claims are
limited to declared continuity slices (e.g. the v1-verbs-only margin), reported, never gated.

## 2. v1 is closed

`data/dockerfs` / `data/dockerfs-e5` (HF: `veryfansome/terminal-jepa-dockerfs`, the canonical
artifact) are frozen. The 0.306 → 0.5848 progression is a completed v1 series. v1 is never
re-collected (its `:latest` tags have drifted; the seeded regen recipe is no longer faithful).

## 3. Collection integrity (the R5/R6 confound rule, structural)

Both splits of a version are collected in one run, under one policy, one seed protocol, one
image set. A train-side policy change without a paired val re-collection is not a version —
it is a Role-A experiment (§7) and can never redefine fitness.

## 4. Verb classes (what counts for fitness)

Every verb in a version's toolset is classified before its fitness is frozen:
- **content** — enters the fitness margin.
- **semi-echo** — argument text echoes into the observation; reported, excluded from fitness.
- **echo/const** — output derivable from the command or constant across images; excluded.
Classification is by a two-axis measured protocol (cross-image output constancy; cmd↔obs
lexical containment) whose thresholds are CALIBRATED ON v1 (where classes are known:
ls/cat = content, cd = echo) and COMMITTED in the version prereg BEFORE collection. Class
assignments are frozen in the prereg from pilot-data measurements before the full mint; the
prereg doc (not any data artifact) is authoritative, and the harness asserts the active
root's recorded classes match it before scoring. Borderline verbs go to semi-echo.

## 5. Baselines (the margin's denominator)

The fitness margin max is fixed per version and may change only at a version boundary,
program-wide. At the v2 boundary the max gains `retrieve_by_cmd_within_traj` (nearest earlier
same-command observation in the SAME trajectory, strictly causal) — history-linked collection
makes it mandatory (a model-free within-trajectory lookup must not be creditable as dynamics).
Each baseline's solo top-1 is a standing ledger column.

## 6. Re-baselining protocol (per mint)

Before evolution opens on a version: (a) the incumbent champion genome + all baselines, full
budget, 3 seeds, inner split, ONE environment; (b) a PROBE LADDER — architecturally diverse
frozen genomes (v2 ladder: mse+baseline-transformer, InfoNCE+baseline, hippo-era leader,
fastweights champion, chunked champion) scored the same way; the ladder's rank order is the
benchmark-sanity report; (c) the champion once on final-test = the version's declared
incumbent line; (d) the proxy noise band re-measured (triple incumbent re-sample, R8 protocol).
Planning instruments (goal sets, batteries) are re-harvested per version with stratification
by depth AND first path component (the R11 composition lesson), sha-pinned in the prereg.

## 7. Data-side evolution (the two legitimate roles)

- **Role A — collection-policy mutations within a version:** train-only swaps evaluated
  against the version's FIXED paired val (curriculum-value fitness). Legal as a normal evolve
  chunk. Can never change the eval, the classes, or the val set.
- **Role B — world/benchmark mutations:** proposals to mint a new version. Fitness for a
  Role-B proposal is benchmark quality, never model score: the probe ladder's rank order must
  be preserved (coarse gate), coverage/stratification requirements met, and the proposal is
  reviewed adversarially before minting. A genome or policy may NEVER score the version that
  scores it (the Darwin-Gödel guard, extended to the world).

## 8. Sacred invariants (unchanged from v1, restated as constitutional)

The 3-way split discipline (fit / inner-val / untouched final-test; final never scored for
selection); the causal no-leakage guard; frozen encoders per root; genomes never touch the
eval path; stats recorded neutrally, verdicts withheld; pre-registration before scoring for
every gated decision, amendments dated and disclosed, never silent.

## 9. Publication

Every minted version's raw + primary encoded roots are published to the HF dataset repo
(scanned before publication, as v1 was), with digests, policy hashes, class tables, and
sha256s in the tracked summary. The HF copy is the canonical artifact; local disks are caches.

---
## Ratification amendments (2026-07-21, from adversarial review A — applied before ratification)

- **§5 precise definition (fatal fix):** `retrieve_by_cmd_within_traj` = for each eval step,
  the observation of the nearest-earlier-command-BY-EMBEDDING within the same trajectory
  (strictly causal, any earlier step); fallback when the trajectory has no earlier step:
  predict_mean (zeros in standardized space). It enters the margin max as an aggregate
  column and gets a standing solo ledger column. Frozen BEFORE policy selection.
- **§6 probe-ladder gate quantified:** the mint-sanity gate is: mse < InfoNCE < every
  evolved-arch rung, strictly, and no non-adjacent inversions; adjacent-rung swaps within
  the re-measured noise band are reported, not failing.
- **§6 instrument arithmetic:** every instrument must satisfy prefix + imagination horizon
  ≤ 32 steps (64 tokens), enforced by an ASSERT (never a clamp — the hippo rung clamps
  silently); a positional-table extension is a version-identity change.
- **§4 measurement basis:** both class axes are measured on RENDERED observation text (the
  eval's actual space), never raw output.

RATIFIED as amended, 2026-07-21.
