# terminal-jepa

The JEPA shell world model (Phase R4) + the ShinkaEvolve evolutionary search over it. Working context (env, data regen, the evolve loop + its replication manual) is in the `CLAUDE.md` files (`../CLAUDE.md`, `CLAUDE.md`, `evolve/CLAUDE.md`); current direction and results are in [../README.md](../README.md).

**Environment:** `uv sync`, then run everything as `uv run python -m <module>`. Needs Docker for data collection; first encoder use downloads the HF model.

## Architecture

The same shape as I-JEPA (see `../JEPA.md` §3), transposed to a real shell: the "image context" becomes an exploration history on a real Docker filesystem, the "masked block" becomes the next command's observation, and prediction happens in embedding space — the model never generates output text.

```
    x = exploration history + the command about to run     y = that command's observation
    cmd₁ obs₁ cmd₂ obs₂ … cmd_t                            obs_t  (what the shell prints)
                   │                                          │
         ┌─────────▼─────────┐                     ┌──────────▼────────┐
         │  frozen text      │                     │  frozen text      │ ← same encoder, FROZEN
         │  encoder          │                     │  encoder          │   (retrieval-tuned, no
         │  (e5-base-v2)     │                     │  (e5-base-v2)     │   gradients — stronger
         └─────────┬─────────┘                     └──────────┬────────┘   than EMA here; the
                   │ renders + pools each                     │            synthetic-era finding)
                   │ cmd/obs → 768-d, standardized            │
         ┌─────────▼─────────┐                                │
         │ causal transformer│                                │
         │ trunk over the    │                                │
         │ interleaved       │                                ▼
         │ z(cmd),z(obs)     │── prediction ẑ(obs_t) ──►  loss = contrastive objective
         │ stream            │   read at the cmd_t slot   in EMBEDDING space (in-batch
         └───────────────────┘   (before obs_t is seen)   negatives; no text decoded)
```

Every named piece is an **evolvable chunk** (see `evolve/CLAUDE.md` for contracts): the encoder + rendering/pooling is `perception`; the token layout of the interleave is `stream`; the trunk is `arch` (champion: system-identity FiLM + file/path delta memories); the readout at the cmd slot is `head` (champion adds a cued-recall aux probe); what ẑ must match — raw `z(obs_t)`, a delta, or a learned invertible transform — is `target`; the loss is `objective` (champion: anti-retrieval ring negatives); batch composition (= the negative pool) is `batcher`; the optimizer is `optim`. The eval is immutable: ẑ(obs_t) ranks the true observation against hard same-verb foils on held-out *images*, and fitness is the content-verb top-1 **margin** over honest baselines (retrieve-by-command / no-history / copy-prev / within-trajectory).

## Modules

- `realenv/seq_worldmodel.py` — **R4, the foundation**: a causal transformer over interleaved `cmd,obs,cmd,obs,…` frozen-encoder embeddings; the hidden state at each command position predicts that command's next-observation embedding (latent, standardized). Owns the eval the search reuses — next-obs retrieval with hard same-verb foils, the honest baselines (predict-mean / copy-prev / retrieve-by-command), the content-verb margin — plus the optional generative twin and the history ablation.
- `realenv/docker_env.py` — `DockerBox`: per-command `docker exec` with tracked `cd`, path enumeration, system-id readers.
- `realenv/collect_docker.py` — sequence generator + parallel collection over Docker images; held-out-*image* split → `data/dockerfs/`; exploration policies incl. the 9-verb dockerfs2 mint policy `v2` (history-linked targets with per-step `meta`, content-hashed lexicons, availability probes, `--pin-digests`; spec: `benchmarks/dockerfs2-prereg.md` Amendment 1); guards in `tests/test_collect_v2.py`.
- `evolve/` — the chunk-based evolutionary design search over the R4 model. See `evolve/CLAUDE.md` (working context + replication manual).
- `realenv/plan_eval.py` — Phase-0 planning probe (Stage 1): goal-conditioned action ranking by predicted-latent distance on the frozen champion; prereg in `benchmarks/plan-probe-v1-prereg.md`; guards in `tests/test_plan_eval.py`.
- `tests/test_seq_worldmodel.py` — no-future-leakage + retrieval-calibration guards for the R4 model.
- `cloud/` — RunPod offload for GPU-heavy scoring: `runpod.sh` (provider wrapper: launch/bootstrap/sync-data/pull/terminate) + `runpod_score.sh` (idempotent batch orchestrator over `cloud/jobs.tsv`); pulled results enter the archive via `uv run python -m evolve.cli ingest`. See `cloud/README.md`.

## Commands

```sh
uv sync                                              # build/refresh the locked env

# Data (needs Docker; full seeded recipe is in CLAUDE.md):
uv run python -m realenv.collect_docker --out data/dockerfs ...      # collect raw sequences
uv run python -m evolve.reencode --perception enc_e5_base --src data/dockerfs --out data/dockerfs-e5

# R4 sequence world model (baseline + sanity arms):
uv run python -m realenv.seq_worldmodel --data data/dockerfs --seeds 0,1,2 --gen-twin --out runs/dockerfs/seq-worldmodel.json
uv run python -m realenv.seq_worldmodel --data data/dockerfs --seeds 0,1,2 --ablation history --out runs/dockerfs/seq-history-ablation.json
uv run python -m unittest tests.test_seq_worldmodel

# Evolve search (score a genome; see evolve/CLAUDE.md):
uv run python -m evolve.cli score --genome <g.json> --mode proxy --data data/dockerfs-e5
uv run python -m evolve.cli leaderboard --top 12
```

## Notes

- Data roots (`data/dockerfs*`) are derived and gitignored (only `summary.json` per root is tracked); regenerate via the recipe in `CLAUDE.md`. The collection RNG is seeded, so the dataset reproduces when the local Docker images match.
- The retired synthetic Phase 0–1 sandbox and the R1–R3 real-shell prototypes were removed 2026-07-18 (git history retains them); the project is now the R4 world model + the evolve search only. Their empirical record survives as "prior work" in `../README.md`.
