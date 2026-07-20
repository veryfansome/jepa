# Plan-probe v1 (Phase 0, Stage 1) — pre-registration

Committed BEFORE any scoring run (R9-speed precedent). The first test of the JEPA program's
third leg: does the frozen champion world model support goal-conditioned ACTION SELECTION by
predicted-latent distance?

## Subject and environment

- Genome: `r9-arch-chunked-codex` (the validated champion), data root `data/dockerfs-e5`.
- One environment for everything: a single RunPod session (1x RTX 4090, cu126 torch — env tag
  `runpod-4090-cu126`). Checkpoints trained, fidelity-gated, and plan-evaluated in that session.
- Code: `evolve.cli score --save-dir` (checkpoint hook, saved AFTER scoring so training/eval are
  byte-identical to a hook-less run) + `realenv/plan_eval.py` @ the commit carrying this file.

## Blocking fidelity gate (before any plan number is read)

`cli score --mode full --split inner --save-dir` on the pod must reproduce the archived champion
inner margins (per-seed 0.6264 / 0.6289 / 0.6228, env runpod-4090-cu126): 3-seed mean within
±0.002 and every seed within ±0.004 of its archived value. Fail → stop, diagnose; no plan
numbers are reported from uncertified checkpoints.

## Stage-1 protocol (fixed)

K=8 candidates (true + 7 distractors); goals = seeded samples of eval positions on the split
(content: N=2000, verbs ls/cat, sample seed 1234; cd calibration slice: N=500, seed 1235);
eligibility: t ≥ 2 and ABSOLUTE-ARGUMENT commands only (both goals and distractors — bare
`ls`/relative targets have context-dependent counterfactuals); distractors same-image,
same-verb, other-sequence, distinct command text, logged-obs cosine to goal < 0.99
(standardized space); candidate draw seed = seed·1000003 + goal index. plan@1 strict (ties
fail). Planners: frozen WM (obs_t zeroed from input), masked self-only R4 twin trained under
the champion objective (4000 steps, per seed), lexical goal-match (cosine), retrieve-by-cmd
planner (fit-split nearest-command obs); random floor 1/8 = 0.125 analytic. Seeds 0,1,2.

## Pre-registered claims and decision rule

- **C1 (action-discriminability):** content plan-margin = plan@1(WM) − max(plan@1 lexical,
  plan@1 retrieve) ≥ **+0.05** (3-seed mean, every seed > 0).
- **C2 (history-driven):** history gap = plan@1(WM) − plan@1(masked) ≥ **+0.05** (3-seed mean).
- Validity checks (must hold or the run is diagnosed, not interpreted): cd-calibration slice
  has lexical plan@1 ≥ 0.5 (the echo planner works where echo exists); every planner ≥ the
  random floor on content goals minus noise.
- **Decision rule:** run on INNER only first. C1 ∧ C2 → score FINAL once with the identical
  protocol and unlock Phase 3 (Stage-2 latent MPC). C1 ∧ ¬C2 → record; investigate the history
  mechanism before Stage 2. ¬C1 → action-discriminability is the program's bottleneck; the v2
  mint (Phase 2) proceeds regardless, Stage 2 is deferred. All outcomes recorded neutrally in
  the ledger (stats, not verdicts).

No expected-value bands are registered (anchoring); thresholds above are the only gates.
