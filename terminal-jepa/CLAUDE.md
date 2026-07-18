# terminal-jepa/ — the code project

The Python project (packages `realenv`, `evolve`). Run as `uv run python -m <module>` from this directory. Repo-wide identity, doc index, and rules are in the root `../CLAUDE.md`; the file inventory + full command list is in `README.md`; current results/direction are in `../terminal-jepa-status.md`.

## Environment

- `uv sync` builds/refreshes the locked `.venv` from `pyproject.toml` + `uv.lock` (torch 2.13 / transformers 5.14, Apple MPS). `.venv/` is gitignored; `pyproject.toml` + `uv.lock` are tracked.
- Run everything via `uv run python -m ...` (or the `.venv/bin/python` uv creates). Docker (CLI) is required for data collection; first encoder use downloads the HF model.

## Active code surface

- **`realenv/seq_worldmodel.py`** — R4, the foundation: a causal transformer over interleaved `cmd,obs,cmd,obs,…` frozen-encoder embeddings; the hidden state at each command position predicts that command's next-observation embedding (latent). Owns the **eval** the whole search reuses: next-obs retrieval, hard same-verb foils, the honest baselines (predict-mean / copy-prev / retrieve-by-command), the content-verb margin. Immutable from a genome's point of view.
- **`realenv/docker_env.py`** (`DockerBox`: per-command `docker exec`, tracked `cd`, path enumeration) + **`realenv/collect_docker.py`** (sequence generator + parallel collection, held-out *image* split, exploration policies).
- **`evolve/`** — the ShinkaEvolve search over the R4 foundation. **See `evolve/CLAUDE.md`.**
- `tests/test_seq_worldmodel.py` — no-future-leakage + retrieval-calibration guards for the R4 model.

The retired synthetic Phase 0–1 sandbox and R1–R3 prototypes were removed 2026-07-18 (git history + the `terminal-jepa-status.md` "prior work" section retain them). The tree is R4 + evolve only.

## Data + regeneration (data is gitignored; only `summary.json` per root is tracked)

`data/dockerfs/` = the raw collected sequences; `data/dockerfs-e5/` = re-encoded with e5-base-v2 (what the current champion uses). Both are **derived and regenerable** — the raw `*.jsonl` and all `emb-seq-*.pt` are gitignored (they were purged from history once for exceeding GitHub's size limit; keep it that way). To rebuild from scratch (needs Docker + the target images; the collection RNG is seeded, so it reproduces the dataset when the local images match):

```sh
uv run python -m realenv.collect_docker --out data/dockerfs \
  --seqs-per-image 400 --seq-len 16 --seed 0 --workers 4 \
  --train-images "alpine:latest,ubuntu:latest,debian:stable-slim,python:3.12-slim,redis:7-alpine,nginx:stable-alpine,postgres:16-alpine,node:22-slim" \
  --val-images "fedora:latest,rockylinux:9,mariadb:latest,httpd:2.4"
uv run python -m evolve.reencode --perception enc_e5_base --src data/dockerfs --out data/dockerfs-e5
```

Other perception roots (bge / codebert / e5-large / multi-vector / exploration-policy variants) are rebuilt on demand via `evolve/reencode.py` / `evolve/mv_encode.py`; `base_cache.json` (gitignored) rebuilds on the first score against a root.

## Update triggers

- Add/rename/retire a module or run command → update `README.md` (inventory) and, if it moves the active-vs-retired boundary, the root `../CLAUDE.md` code map.
- Change the R4 eval, foils, baselines, or the leakage/calibration guards → this is the fitness the search trusts: update `../terminal-jepa-status.md` (R4 section) and note the ripple in `evolve/CLAUDE.md` (the guardrails reuse this code).
