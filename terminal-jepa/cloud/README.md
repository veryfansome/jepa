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

## Probe results (2026-07-19, 1x RTX 4090 SECURE $0.34/hr, torch 2.13.0+cu126, pod driver CUDA 12.8)

Measured with the champion genome (`rp0-cuda-probe-champion` in the archive, env `runpod-4090-cu126`):

- **Fitness agreement is essentially exact**: proxy 0.6150 (local MPS 0.6150); full inner 0.6143, per-seed 0.6136/0.6103/0.6191 (local 0.6142: 0.6137/0.6098/0.6190). The one-environment rule stays as cheap insurance, but the measured MPS↔CUDA offset on this workload is ≈±0.0005.
- **A 4090 is NOT faster per job**: full inner 64 min vs 69 min local (~1.08×); proxy ~7.5 min vs ~6 min. The fastweights-family archs are a sequential python loop over positions with tiny tensors — kernel-launch-bound, not FLOPs-bound, so a bigger GPU doesn't help. **The offload win is parallelism, not speed**: N pods (or `CONCURRENCY>1` on one under-utilized pod) divide the round's wall-clock at ~$0.35/hr each, and the dev machine stays free. Per-job speedups would need harness-side work (torch.compile/CUDA-graphs on the sequential archs) — an environment change requiring re-baselining.
- **Data upload is the tax**: 16 min for the 650MB e5 root from the home connection (~5.4 Mbps effective) — motivates hosting the roots (HF dataset repo or a volume) rather than re-uploading per pod.
- Bootstrap ≈40s warm / ~3 min cold. Whole probe incl. 3 debug re-provisions: **<$1.50**.

Setup facts encoded in the scripts by the probe's failures: account-level console key must be offerable (no `IdentitiesOnly` hard-pin by default); volume deploys are opt-in (`RUNPOD_USE_VOLUME=1`) so stale env-file volume ids can't hijack; locked cu13 torch wheels don't run on CUDA-12.x drivers — bootstrap swaps in `torch==2.13.0+cu126` (same version as the lock; cu128 index stops at torch 2.11) and `UV_NO_SYNC=1` keeps `uv run` from undoing it.
