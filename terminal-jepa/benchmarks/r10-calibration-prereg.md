# R10-calibration round — pre-registration

Committed BEFORE any candidate is scored. The round evolves PREDICTION CALIBRATION under a
hard fitness non-regression constraint — the R9-speed pattern (fitness untouched; new gates)
applied to the defect Stage 2 + its adversarial review measured: the champion's predictions
rank superbly but are grossly off-manifold (norm_ratio ||pred||²/||true||² ≈ 4.7 at matched
cosine 0.70; matched pred-vs-true sqL2 ~2020 vs TRUE-vs-TRUE random-pair ~1430), which the
review localized to the contrastive row-softmax being invariant to per-prediction norm, and
which (with imagination infidelity) blocks latent-MPC planning.

## Instruments (frozen)

- `evolve/calib_bench.py` @ this commit. Battery artifact `data/plangoals-v1/dec0-battery-v1.pt`
  (sha256 prefix **6d1a2bf352a8bac6**; 100 inner goals, fixed opener, root candidate sets,
  horizon-2 write-policy imagination, SUM-of-cosine scoring).
- Champion baseline (certified Stage-1 checkpoints, seeds 0/1/2), measured before this file
  was frozen: first_move_acc 0.35 / 0.52 / 0.60 (mean 0.49 — note the large seed variance);
  matched_sqL2 2047 / 2024 / 2009; rand_pair 1436; norm_ratio 4.74 / 4.70 / 4.67; cosine 0.702.

## Pre-registered gates (promotion requires G1 AND G2)

- **G1 — fitness non-regression (sacred):** full-budget inner-val content-verb margin
  ≥ **0.6230** (champion 0.6260 − ε, ε = 0.003), 3 seeds, all standard guardrails; final-test
  scored once at promotion and must be ≥ **0.5818** (champion 0.5848 − ε).
- **G2 — calibration:** 3-seed means of BOTH: norm_ratio ∈ **[0.8, 1.5]** AND
  matched_sqL2 < rand_pair_sqL2 (i.e. predictions closer to their own targets than random
  true observations are to each other — the literal meaning of on-manifold; the champion
  fails both today).
- **G3 — planning direction (advisory, NOT sufficient/necessary):** battery first_move_acc
  3-seed mean ≥ 0.59 (champion 0.49 + 0.10); reported per-seed. Advisory because the
  champion's own seed spread (0.35–0.60) makes a mean gate noisy at n=3.
- **Payoff test (only for a G1∧G2 promotion):** re-run the Stage-2 episode probe
  (`plan_env`, inner images, wm-write + lexical + random, 3 seeds, sum score) — did
  calibration buy navigation? Recorded either way.

Screening: proxy-budget margin within the incumbent proxy band (±0.003 of the same-session
incumbent re-sample) AND directional improvement on norm_ratio at proxy — both required to
spend a full run. Selection on inner only; final never used for selection.

## Candidate space

One-chunk mutations of the champion genome (objective / target / head / arch-light), plus a
Codex objective slot. The eval/metric/split/leakage guards are immutable as always.
