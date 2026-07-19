# terminal-jepa

The JEPA shell world model (Phase R4) + the ShinkaEvolve evolutionary search over it. Working context (env, data regen, the evolve loop + its replication manual) is in the `CLAUDE.md` files (`../CLAUDE.md`, `CLAUDE.md`, `evolve/CLAUDE.md`); current direction and results are in [../README.md](../README.md).

**Environment:** `uv sync`, then run everything as `uv run python -m <module>`. Needs Docker for data collection; first encoder use downloads the HF model.

## Modules

- `realenv/seq_worldmodel.py` — **R4, the foundation**: a causal transformer over interleaved `cmd,obs,cmd,obs,…` frozen-encoder embeddings; the hidden state at each command position predicts that command's next-observation embedding (latent, standardized). Owns the eval the search reuses — next-obs retrieval with hard same-verb foils, the honest baselines (predict-mean / copy-prev / retrieve-by-command), the content-verb margin — plus the optional generative twin and the history ablation.
- `realenv/docker_env.py` — `DockerBox`: per-command `docker exec` with tracked `cd`, path enumeration, system-id readers.
- `realenv/collect_docker.py` — sequence generator + parallel collection over Docker images; held-out-*image* split → `data/dockerfs/`; exploration policies.
- `evolve/` — the chunk-based evolutionary design search over the R4 model. See `evolve/CLAUDE.md` (working context + replication manual).
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
