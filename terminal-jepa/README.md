# terminal-jepa

Implementation of the plan in [../terminal-jepa.md](../terminal-jepa.md). Current state, findings so far, and reproduction notes: [../terminal-jepa-status.md](../terminal-jepa-status.md).

- **Phase 0** (env + datagen): a synthetic in-process filesystem sandbox with ground-truth access, a validated state constructor, the shared full-obs parser, distractor knobs, and typed trajectory generation with layout/predicate splits. Stdlib only.
- **Phase 1** (world model + probing): from-scratch token encoder, compositional action encoder, AdaLN predictor, SIGReg/VICReg/IDM/temporal-similarity arms, the generative reconstruction twin, and the probing harness (linear pass bar, MLP gap, banner-swap audit). Requires PyTorch — `python3 -m venv .venv && .venv/bin/pip install torch`.

## Layout

- `env/vocab.py` — shared enumerations (dir/file names, content classes, path universe, banner vocabulary); every layout, probe label space, and action argument draws from these.
- `env/state.py` — `FsState`, invariants, ground-truth probe features, goal predicates, and `make_satisfying` (the validated goal-exemplar constructor).
- `env/actions.py` — factored `(verb, arg1, arg2)` actions, validity semantics, transition typing (state-changing / valid-no-op / invalid + failure taxonomy), typed valid/invalid samplers.
- `env/render.py` — full-obs and partial-obs renderers with banner / dynamic-noise knobs; pure functions so the training loader picks the regime at load time.
- `env/parse.py` — full-obs text → `FsState`; the shared nonprivileged parser (baseline, validity filter, goal exemplars, Phase-3 belief tracker).
- `env/world.py` — `Sandbox` stepped-episode wrapper.
- `datagen/layouts.py` — layout generator, hash-stable layout split, predicate-universe split.
- `datagen/policies.py` — typed-random policy (invalid quota) and scripted goal-reacher (ε-noise, chained goals).
- `datagen/generate.py` — dataset CLI; writes `train.jsonl`, `val.jsonl`, `manifest.json`, `summary.json`.
- `models/data.py` — compositional tokenizer (closed vocab built from `env.vocab` + UNK), regime-aware trajectory loader, training windows, probe examples.
- `models/nets.py` — `TokenEncoder` (CLS latent, z∈R²⁵⁶), action encoder, AdaLN `Predictor` (zero-init gates), `ReconDecoder` (generative twin), `IDMHead`.
- `models/losses.py` — SIGReg (Epps–Pulley over random 1-D projections), VICReg variance/covariance, temporal similarity.
- `train/train.py` — training CLI: arms `sigreg | vicreg | sigreg+idm | sigreg+tempsim | recon`; 1-step teacher-forced + 3-step rollout latent L2, no EMA/stop-gradient.
- `probes/probe.py` — probing CLI: fits on train-layout states, evaluates on held-out layouts; cwd accuracy, pooled existence balanced-accuracy, content accuracy given existence, macro average, banner-identity probe, banner-swap ‖Δz‖ audit.
- `probes/frozen_probe.py` — A1 day-zero probe (zero training): frozen pretrained HF encoder over rendered observation text, pooled + path-keyed readouts, protocol-v2 head fitting reused from `probe.py`, pooled and line-level banner-swap audits. Requires `transformers`.
- `probes/target_noise.py` — zero-training gate-2 target-noise measurement over frozen features (noise-line nuisance vs changed-line signal, pre-registered power-ratio criterion → clean-vs-raw target decision). Requires `transformers`.
- `evals/dynamics.py` — dynamics-gate battery (the shared referee for Track B tiers and Track A gate 2): rollout error by transition type, change-magnitude calibration, violation-of-expectation (copy-resistant alt-action foil primary), goal-distance ranking along `plan_for` plans vs `make_satisfying` exemplars. Adapter interface; ships oracle/oracle and oracle/copy self-test adapters (stdlib only) plus `--adapter tier2 --ckpt ...` for learned predictors (torch).
- `models/tier2.py` — tier-2 predictor bake-off: slot-transformer trunk over the fixed 301-slot key universe with tied action-arg/slot embeddings; `slotpred` (full re-prediction, identity-at-init copy-margin logits — the bake-off winner) vs `editpred` (structural copy + edit heads — failed calibration); training CLI + battery adapter.
- `models/gate2.py` — gate 2 over frozen ModernBERT features: `--mode precompute` (dual-regime feature cache), `--mode train` (embedding-space slotpred — failed rounds 1–2, kept as the negative-result artifact), `--mode codebook` (linear-decoder symbol grounding — the passing round-3 configuration), plus battery adapters (`gate2`, `gate2-copy`, `gate2-codebook`).
- `plan/cem.py` — Planner A: factored discrete CEM + random shooting (matched budget) + scripted `plan_for` ceiling, over the adapter interface (tier-1 oracle now; tier-2/gate-2 checkpoints swap in via `--adapter`/`--ckpt`); unfiltered primary + position-0 validity-filter arms.

## Commands

```sh
python3 -m unittest discover -s tests        # property-based test suite (any cwd works)
python3 -m datagen.generate --out data/v0    # generate a dataset + summary report

# Phase 1 (needs .venv with torch):
.venv/bin/python -m train.train --data data/v0 --arm sigreg --out runs/sigreg-s0
.venv/bin/python -m probes.probe --data data/v0 --ckpt runs/sigreg-s0/ckpt.pt --out runs/sigreg-s0/probe.json
.venv/bin/python -m probes.probe --data data/v0 --ckpt untrained --seed 0 --out runs/untrained-probe.json

# Track A1 (needs transformers: .venv/bin/python -m pip install transformers):
.venv/bin/python -m probes.frozen_probe --data data/v1 --model answerdotai/ModernBERT-base --out runs/frozen-modernbert-v1/probe.json
.venv/bin/python -m probes.target_noise --data data/v1 --out runs/frozen-modernbert-v1/target-noise.json

# Dynamics-gate battery (stdlib only for oracle adapters):
python3 -m evals.dynamics --data data/v1 --adapter oracle --out runs/dynamics-battery/oracle-oracle.json
python3 -m evals.dynamics --data data/v1 --adapter oracle-copy --out runs/dynamics-battery/oracle-copy.json

# Track B tier 2 (predictor bake-off over oracle features):
.venv/bin/python -m models.tier2 --data data/v1 --head slotpred --out runs/tier2/slotpred-v2
.venv/bin/python -m evals.dynamics --data data/v1 --adapter tier2 --ckpt runs/tier2/slotpred-v2/ckpt.pt --out runs/tier2/slotpred-v2/battery.json

# Gate 2 (frozen features; codebook = the passing configuration):
.venv/bin/python -m models.gate2 --mode precompute --data data/v1 --out runs/gate2
.venv/bin/python -m models.gate2 --mode codebook --cache runs/gate2 --data data/v1
.venv/bin/python -m evals.dynamics --data data/v1 --adapter gate2-codebook --ckpt runs/tier2/slotpred-v2/ckpt.pt --input-regime both --out runs/gate2/codebook-battery-dirty.json

# Track B tier 1 planner validation:
python3 -m plan.cem --data data/v1 --adapter oracle --episodes 100 --out runs/tier1/cem-oracle.json
```

## Notes

- Trajectories store compact states/actions/stdout, not rendered text; observations are rendered on demand so the renderer stays the single source of truth and distractor regimes are chosen at load time (`banner_id` and `noise_seed` are stored per trajectory).
- Banner ids come from an RNG stream keyed by (seed, split, trajectory index) only — independent of layout and policy by construction; the datagen test verifies banners are unchanged when the layout pool changes.
- The invalid-action quota is enforced per step at the trajectory level (the policy's choice is overridden with probability q), so the realized invalid rate tracks `--invalid-quota` regardless of policy mix; `summary.json` reports the realized mix.
- The ≤12-dir / ≤20-file caps bound layout generation only; runtime states can grow past them (bounded by the vocabulary: 42 dirs / 258 files) and `summary.json` reports the realized maximum. Runtime caps are deliberately not invariants — they would break minimal-edit goal exemplars.
- Layout identity for splits is keyed on tree + contents only (cwd excluded): cwd is initial state, not layout.
- `malformed-arg` never appears in generated data by construction (typed samplers stay in-vocabulary); it exists for Phase 2's LLM-proposal lowering path and is covered by tests.
