# R11-planning round — pre-registration

Committed BEFORE any candidate is scored. The round's ONLY goal is planning capability,
via the three literature-validated directions (JEPA.md §5): rollout consistency
(exposure-bias training; 2512.24497, EB-JEPA's 97%→1% inverse-dynamics ablation),
value/reachability shaping of the latent cost landscape (2601.00844, temporal
straightening 2603.12231), and path-structured representations (the R8 pathkey thread —
retried because the R9 champion trunk is now a transformer with token attention).

## Instruments (frozen)

- `evolve/path_battery.py` @ this commit; battery `data/plangoals-v1/path-battery-v1.json`
  (sha256 prefix **66f8d905ac160118**; 100 inner goals, all intact, 33 decision nodes,
  389 decisions). Vectorized imagination verified ≡ sequential (1e-5). Modes: path_acc_real
  (teacher-forced stepwise) + path_acc_imag (self-imagined history) + by-remaining-depth.
- Battery-v1 (dec0, sha 6d1a2bf3) retained as a secondary report.
- Champion baseline (certified Stage-1 full ckpts, local MPS, measured pre-freeze):
  path_acc_real **0.4910 / 0.6401 / 0.6838** (mean 0.6050 — large seed variance, hence the
  paired gate); imag 0.4781 / 0.5733 / 0.6787; compounding_gap 0.0129 / 0.0668 / 0.0051.
  Structure: remaining-depth-1 is the WEAKEST slice (0.42–0.51) — the final-approach
  decision, not long-range compounding, is the binding constraint at this budget.

## Pre-registered gates (promotion requires G1 AND G2)

- **G1 — fitness non-regression (sacred, unchanged from R10):** full inner ≥ **0.6230**,
  final (once, at promotion) ≥ **0.5818**; all standard guardrails.
- **G2 — planning:** paired per-training-seed deltas of path_acc_real (candidate full ckpt
  seed s vs champion full ckpt seed s, same battery, ONE evaluation environment per
  comparison): mean delta ≥ **+0.08** AND ≥ 2/3 seeds positive.
- **G3 (advisory, reported):** compounding_gap, by-remaining-depth slices (rem-1
  highlighted), battery-v1 first_move.
- **Payoff test on any promotion:** the Stage-2 episode probe re-run (inner images,
  wm-write + lexical + random, sum score, 3 seeds) — the claim is episode success, the
  battery is only the screen.
- Screening (to spend a full run): proxy fitness ≥ same-session incumbent re-sample − 0.003
  AND proxy-ckpt path_acc_real > champion-proxy-ckpt path_acc_real (both in-session).

## Candidate space & disclosed limits

One-chunk mutations of the champion (head / objective / target / perception), plus
recombinations of survivors. Perception candidates are scorable on the battery via
`--percep` re-encoding (the battery stores text). The pathkey-stream × champion-arch
epistasis retry runs for FITNESS only this round — the battery does not yet support
multi-vector streams (disclosed; plumbing follows only if it clears the fitness band).
The eval/metric/split/leakage guards are immutable as always.
