# cloud/ — RunPod offload for GPU-heavy evolve scoring

Ported from the battle-tested `sandbox/runs/runpod*.sh` orchestration (see that repo's `runs/README.md` for the full lesson history). Two layers:

- **`runpod.sh`** — provider wrapper (pure curl+jq+ssh+rsync, no CLI install): `types --available`, `launch` (prints `<podId> <ip> <port>` as the parseable last line), `host`, `ssh`, `bootstrap` (apt deps → rsync repo code → `uv sync` → CUDA smoke), `sync-data <pod> <root>...` (rsyncs only `emb-seq-*.pt` + `summary.json` per data root), `pull`, `terminate`, `status`, `list`. Auth: `RUNPOD_API_KEY` from `~/.runpod.env` (falls back to `~/.lambda.env`). Defaults: 1x RTX 4090, SECURE cloud, 60GB disk.
- **`runpod_score.sh`** — idempotent stage pipeline (`provision → bootstrap → syncdata → jobs → verify → score → poll → download → verifydl → terminate`; `STAGE=<name>` runs one). Jobs come from `cloud/jobs.tsv` (`<gid> <mode> <split> <data_root> <genome_json_path>` per line); each runs `evolve.cli score` on the pod inside a tmux session, skipped if its result JSON already exists — so **fix → re-run resumes** only unfinished work. Results land in `cloud/podresults/`; `verifydl` gates `terminate` (never terminate before results are verified local). `YES=1 AUTO_TERMINATE=1` for unattended runs.

Ingest pulled results into the archive locally (never scored twice):

```sh
uv run python -m evolve.cli ingest --genome <g.json> --result cloud/podresults/<gid>.<mode>.<split>.json --env runpod-4090
```

## Rules (methodology — these are load-bearing)

- **One environment per comparison.** CUDA numbers are not comparable to local MPS numbers at the ±0.001 noise level, and `base_cache.json` is per-environment (never synced, pod recomputes its own). When moving environments, **re-baseline the incumbent first** (score the unchanged champion genome, like the R8 `r8-incumbent-resample` pattern) and keep a round's selection comparisons inside one environment. Tag ingested entries with `--env`.
- **The archive stays local.** Pods append to their own throwaway copy of `genomes.jsonl`; the authoritative archive only grows via local `cli score` or `cli ingest`.
- **Serialize by default.** `CONCURRENCY=1`; raise it only knowingly (GPU contention adds nondeterminism).
- **First trip on new hardware: run a cheap probe** (incumbent proxy + one full seed) to measure wall-clock speedup and the environment's fitness offset before committing a round to it. Budget ~$15-25 of learning incidents for any new tier (sandbox lesson).

## Cost ballpark

An R8-sized round (17 proxies + 4 full split-scores + sanity arms) took ~14 h serialized on the local MPS machine. On a 1x RTX 4090 (~$0.35–0.70/hr) the same queue is a few GPU-hours → **single-digit dollars per round**; even an A100/H100 pod stays <$20. The dominant win is that full promotions (3 seeds × 4000 steps on the sequential fastweights archs) stop monopolizing the dev machine.
