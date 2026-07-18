# evolve/ — the ShinkaEvolve search (working context)

A Claude-Code-native evolutionary design search over the R4 world model: decompose it into swappable **chunks**, have heterogeneous LLM inventors (**Claude** via the Workflow tool at effort=high + **Codex** via `codex exec`, arxiv/cross-domain research encouraged) invent/recombine chunk implementations, keep whatever beats the current best on a trustworthy held-out margin.

**`DESIGN.md` is the replication manual** — the full method, contracts, contract extensions, proxy→full ladder, inventor-dispatch pattern, recombination/epistasis, and the step-by-step "how to run a round" recipe. Read it before running a round. Results/progression live in `../../terminal-jepa-status.md`; the neutral per-genome stat ledger is the `evolve-insights` memory. The `/evolve` skill drives a generation.

## The eight evolvable chunks

`objective`, `arch`, `optim`, `target` (+ R6/R7 additions:) `perception` (encoder "eyes" + render + pooling, scored via a data-side re-encode), `batcher` (train-batch composition = the in-batch negative pool), `stream` (single- vs multi-vector token layout), `head` (readout + optional aux task). Each `chunks/<chunk>/<impl>.py`; each has a `baseline`/`identity` impl that reproduces prior behavior bit-for-bit (the plumbing check every new axis must pass). Loaders + `validate()` are in `genome.py`. Two harness **contract extensions** (learned targets; the head-wrap) are documented in DESIGN.md — mind the head-wrap module-cycle / forward-recursion hazards.

## Scoring

A genome is a JSON dict selecting one impl (+ params) per chunk. Score via the backend CLI:

```sh
uv run python -m evolve.cli score --genome <g.json> --mode proxy|full --split inner|final --data data/dockerfs-e5
uv run python -m evolve.cli leaderboard --top 12
uv run python -m evolve.cli impls --chunk arch        # list registered impls for a chunk
```

- **proxy** = 1000 steps / seed 0 (inner-loop screen); **full** = 4000 / seeds 0,1,2 (promotion). Only full-budget 3-seed results are reliable — **MPS proxy is nondeterministic ~±0.001** for the identical genome, so proxy deltas < ~0.002 are noise.
- Perception/stream recipes need a re-encoded data root first (`evolve/reencode.py` or `mv_encode.py`); score with `--data <that root>`.
- Fitness = the held-out **content-verb margin** (see DESIGN.md). Hard filters → −inf: future-leakage, predict-mean-vs-chance calibration, NaN. Never score `final` for selection.

**Current champion** (`archive/genomes.jsonl`, `r7-arch-fastweights`, final-test 0.5702): e5-base-v2 encoder (`data/dockerfs-e5`) + `r7_path_delta_fastweights_codex` arch + `r6_free_energy_precision_l2_contrastive` objective + `r6_sysblock_hardneg_curriculum` batcher + `go_warmup_holdcos_floor` optim.

## State

- `archive/genomes.jsonl` — **tracked**, append-only program archive (every scored genome). Survives across sessions; the ground truth for the leaderboard.
- `archive/base_cache.json` — **gitignored**, objective-independent baselines cached per `data_root|split|seed|steps`; rebuilds on first score.
- `evolve-insights` **auto-memory** — the neutral running stat ledger (per-genome margins, negatives, process facts). Record stats, not verdicts.

## Update triggers

- New chunk impl → drop the file in `chunks/<chunk>/`; nothing else (auto-discovered). Smoke-test on CPU first (leakage/anti-collapse/invertibility per DESIGN.md).
- New chunk **axis**, or a change to fitness/guardrails/split/scoring-CLI → update `DESIGN.md` (the chunk table + method) and the scoring section above; note eval ripples in `../CLAUDE.md`.
- A promoted full result / new champion → update `../../terminal-jepa-status.md` + the `evolve-insights` memory (same round).
