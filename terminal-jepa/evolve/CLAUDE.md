# evolve/ — the ShinkaEvolve search (working context + replication manual)

A Claude-Code-native evolutionary design search over the R4 world model (`../realenv/seq_worldmodel.py`): decompose it into swappable **chunks**, have heterogeneous LLM inventors (**Claude** via the Workflow tool at effort=high + **Codex** via `codex exec`, arxiv/cross-domain research encouraged) invent/recombine chunk implementations, keep whatever beats the current best on a trustworthy held-out margin. This file is both the working context and the full replication manual — read it before running a round. Results/progression live in `../../README.md`; the neutral per-genome stat ledger is the `evolve-insights` auto-memory note. The `/evolve` skill drives a generation.

## The two layers

**Python backend (this package) — deterministic, no LLM calls.** A genome is a JSON dict selecting one impl (+ params) per chunk; `genome.py` resolves impl names to code (`load_objective/arch/optim/target/batcher/stream/head`) and `validate()`s structure. `harness.py::score_genome` assembles → trains → guardrail-filters → scores the content-verb margin, reusing the *validated* R4 eval from `../realenv/seq_worldmodel.py` (same retrieval metric, same baselines) so fitness means what the R4 result meant. `archive.py` = the append-only `archive/genomes.jsonl` + leaderboard + parent sampling. `reencode.py`/`mv_encode.py` build re-encoded data roots for the perception/stream chunks. `cli.py` = `seed | score | leaderboard | sample-parent | impls`. `inventors.py` builds the chunk brief (the same brief to Claude and Codex = genuine model diversity). `sanity.py` = the champion validation arms.

**Claude-Code layer — the control loop.** Inventors are the mutation operators; the archive + `evolve-insights` memory are the meta-scratchpad. A round: pick target chunks → dispatch inventors → register + CPU-smoke-test each proposal → score at proxy → promote the best to full → recombine winners → validate the champion → record neutral stats.

## The eight evolvable chunks

`objective`, `arch`, `optim`, `target`, `perception` (encoder "eyes" + render + pooling, scored via a data-side re-encode), `batcher` (train-batch composition = the in-batch negative pool), `stream` (single- vs multi-vector token layout), `head` (readout + optional aux task). Each is `chunks/<chunk>/<impl>.py`, auto-discovered by filename; each has a `baseline`/`identity` impl that reproduces prior behavior **bit-for-bit** (the plumbing check every new axis must pass). Immutable from a genome's view: the frozen encoding, the causal no-leakage contract, and the entire eval path (metric, baselines, 3-way split) — a genome may never edit the thing that scores it.

Contract per chunk (the reference baseline's docstring is authoritative — read it before writing a new impl):

| chunk | loader | an impl exposes | reference baseline |
|---|---|---|---|
| objective | `load_objective` | `loss(pred, tgt) -> scalar` | `objective/mse.py` |
| arch | `load_arch` | `build(**p) -> Module` with `forward(tok_emb, types, key_pad) -> (pred, h)` | `arch/baseline_transformer.py` |
| optim | `load_optim` | `make(params, steps) -> (opt, sched)` | `optim/baseline_adamw.py` |
| target | `load_target` | pure `make_target(z_obs, z_prev)` + `to_obs(pred, z_prev)`, or `LEARNED` `make(D) -> Module` | `target/identity.py` |
| perception | `reencode.py` / `mv_encode.py` | `MODEL` + `render_obs` / `render_cmd` / `pool` (+ `render_obs_multi` / `K` for multi-vector) | `perception/baseline.py` |
| batcher | `load_batcher` | `make_batcher(fit, bs, seed) -> next_batch(step, total)` | `batcher/baseline_uniform.py` |
| stream | `load_stream` | `collate` / `extract_cmd_pred` / `flatten_predictions` / `leakage_ok` | `stream/baseline_interleave.py` |
| head | `load_head` | `wrap` / `aux_loss` / `leak_safe` | `head/baseline_passthrough.py` |

**Contract extensions** (two, load-bearing):
- **LEARNED targets** — a `target` impl may set `LEARNED=True` + `make(D)->nn.Module` (make_target/to_obs/reg); the harness registers its params on the net (trained jointly) and evaluates via the module's **exact inverse in the fixed obs space** — so a collapsed learned target can't reconstruct and is scored down. Honest by construction.
- **`head` axis** — `wrap(net, D)` may re-point `net.forward` + register readout/aux params (run *before* the optimizer is built); `aux_loss(...)` is a train-only term (0.0 for passthrough); `leak_safe(...)` is asserted before scoring. Two hazards a wrapper must avoid (both hit in R7): a parent↔child **module cycle** (store the base net non-registered, else `.to(device)` recurses) and **forward recursion** in a trunk fallback (save the original bound forward before re-pointing). The head recomputes pred from the trunk `h`, so on an arch whose pred isn't a per-position function of `h` (e.g. the hippo episodic read) it silently bypasses that mechanism — isolate the head on a `pred=head(h)` arch.

**Adding a new axis:** one loader in `genome.py` + one `validate()` line + thread it through `_train`/`score_genome`, then ship a `baseline`/`identity` impl verified **bit-identical** to pre-axis behavior (a unit test + an end-to-end proxy that reproduces the incumbent's recorded margin). R6 added `batcher` + `stream` this way; R7 added `head`.

## Fitness (the part that must be trustworthy)

`margin = content_top1(WM) − max(content_top1 of retrieve_by_cmd, wm_no_history, copy_prev)` on **content verbs only** (ls+cat; `cd` is a trivial `cwd=<target>` echo — the toolset is a starting point, so this metric grows as the toolset does). The *margin* (not absolute top-1) is fitness because the signal partly rides on shared cross-distro filesystem structure the lexical baseline already banks — a variant that lifts the WM and the baselines equally has discovered nothing (this is why a bigger encoder can *lose*: e5-large lifted the baselines as much as the WM). Hard filters → −inf: future-leakage (perturb obs_t, the cmd_t prediction must not move — lives in each stream's `leakage_ok`), predict-mean ≈ chance (calibration), NaN/collapse; any inventor-code exception → −inf, never a crash.

**The 3-way split (sacred — the anti-overfitting rail):** `fit` = train images (minus a dev slice); `inner-val` = fedora+mariadb (**fitness scored here**); `final-test` = rockylinux+httpd (**never scored for selection** — only a promoted champion, once, + the sanity arms).

## Proxy → full (sample efficiency)

**proxy** = 1000 steps / seed 0 (screen every proposal); **full** = 4000 / seeds 0,1,2 (promote proposals that beat the incumbent, on inner + final). Frozen encodings are cached per data root; objective-independent baselines cached in `archive/base_cache.json` (gitignored) keyed by `data_root|split|seed|steps`. **Dev-environment proxy nondeterminism is ~±0.001** for the identical genome (float-accumulation order on the current GPU backend; CPU training is bit-exact), so **proxy deltas < ~0.002 are noise** — the full-budget 3-seed result is the reliable signal. Proxy→full shrinkage is normal (e5: +0.155 proxy → +0.043 full).

## Scoring

```sh
uv run python -m evolve.cli score --genome <g.json> --mode proxy|full --split inner|final --data data/dockerfs-e5
uv run python -m evolve.cli leaderboard --top 12
uv run python -m evolve.cli impls --chunk archuv run python -m evolve.bench --arch <impl> [--ref <impl> --eq]   # speed bench + Tier-A equivalence (R9-speed rounds)
```

Perception/stream recipes need a re-encoded root first (`reencode.py` / `mv_encode.py`); score with `--data <that root>`. **Current champion** (`r7-arch-fastweights`, final-test 0.5702): e5-base-v2 encoder (`data/dockerfs-e5`) + `r7_path_delta_fastweights_codex` arch + `r6_free_energy_precision_l2_contrastive` objective + `r6_sysblock_hardneg_curriculum` batcher + `go_warmup_holdcos_floor` optim.

## Inventor dispatch (the mutation operators)

**Effort is load-bearing, and the Agent tool has no effort knob.** At default effort inventors finish fast and ungrounded (0/3 beat the incumbent); dispatch Claude inventors via the **Workflow** tool at `effort:'high'` with a brief that **mandates reading the actual chunk code + the results ledger before designing** (2/3 beat the incumbent). Pattern: one Workflow, ~10 Claude inventors in `parallel()` (one per chunk/lens), each returning a structured proposal (name, chunk, idea, mechanism, external_refs, code, risks); in parallel, 3 **Codex** inventors (`codex exec -s read-only --output-last-message`) on the same briefs. `inventors.py` builds the brief and already packages the chunk contract, the leaderboard + current frontier, the top impls' CODE for that chunk (inspiration exemplars), and STANDING_RULES — including the **external-research** encouragement (arxiv + beyond ML: neuroscience, biology, physics, information theory; several winners came from cross-domain lenses). So both Claude and Codex inventors get identical grounding from the one channel both receive — the brief itself.

**CPU-smoke-test every proposal before spending GPU:** objectives (finite loss, gradient, anti-collapse — a constant prediction must not minimize it, n=1 edge); archs (shapes, strict leakage across every obs perturbation, params, fwd time); targets (invertibility, or LEARNED identity-at-init); optim (a short loop, LR schedule, finiteness); batcher (deterministic, in-bounds); perception (768-d, deterministic). **Harden inventor integration bugs yourself** — the inventor can't test against the live harness.

Standing rules in every brief: metric/eval/split are fixed and off-limits; be causal/leak-free/NaN-safe/anti-collapse-safe; novelty over safety; previously-failed traits may be retried recombined.

## Recombination & epistasis

The biggest gains after the first objective/encoder wins came from **stacking** marginal per-chunk winners (R6: arch+objective+optim, ~+0.005 each, composed +0.045 at full). So after single-chunk mutations, **recombine the winners** and full-score. But epistasis cuts both ways — always test it: in R7 fusing an objective onto the fast-weights arch scored *below* fast-weights alone (the fast-weights target-space read subsumed what the objective added). And **retry failed traits recombined** — `target_space` failed 3× early, then came within noise of identity on the e5+contrastive context. Deprioritize, keep live, retry in the changed context.

## Culture & state

**Record stats, not verdicts** — the `evolve-insights` ledger logs neutral numbers + setup, not conclusions; editorializing biases the next round. Negatives are recorded with the same weight as wins. `archive/genomes.jsonl` is **tracked** (append-only, every scored genome — the leaderboard ground truth); `archive/base_cache.json` is **gitignored** (rebuilds on first score).

## How to run a round

1. Re-ground: `cli impls --chunk <c>` for each chunk; `cli leaderboard`; the `evolve-insights` memory. Pick the parent to mutate with **`cli sample-parent`** (balanced exploration/exploitation — the offspring-penalty avoids over-mutating one parent), not always the champion; take the champion directly only for a deliberate focused push.
2. Brief: `python -m evolve.inventors <chunk>` — the brief now **auto-injects** the current frontier (champion recipe), the top impls' CODE for that chunk (inspiration exemplars), and STANDING_RULES (retry-failed + cross-domain), so an addendum only needs round-specific framing (e.g. a target hypothesis).
3. Dispatch: one Workflow of ~10 high-effort Claude inventors + 3 background Codex.
4. Register + CPU-smoke-test each proposal; harden integration bugs; stage new-axis / re-encode specs.
5. Proxy queue (serialize GPU — one background queue): score each as a one-chunk mutation of the best on the best data root; a new axis also runs its bit-identical plumbing check.
6. Promote anything beating the incumbent by > ~0.002 proxy to full (inner+final); recombine winners; test epistasis.
7. Validate the champion (`sanity.py` gen-twin + history ablation on final-test; a revisited-split if an arch has a memory/history mechanism).
8. Record neutral stats in `evolve-insights` + commit `archive/genomes.jsonl`; update the progression in `../../README.md`.

Ops: run heavy GPU jobs one at a time via tracked background tasks (not detached `&`); invoke as `uv run python -m ...`; score via redirect-to-file + JSON-extract; derived data roots are gitignored (only `summary.json` tracked). Big queues (full promotions, long proxy batches) can be offloaded to a RunPod pod via `../cloud/runpod_score.sh` + `cli ingest` — but scores are only comparable **within one environment** (MPS vs CUDA differ beyond proxy noise; re-baseline the incumbent when moving; see `../cloud/README.md`).

## Update triggers

- New chunk impl → drop it in `chunks/<chunk>/` (auto-discovered); CPU-smoke-test first.
- New chunk **axis**, or a change to fitness/guardrails/split/scoring-CLI → update this file (chunks + fitness + scoring) and note eval ripples in `../CLAUDE.md`.
- A promoted full result / new champion → update `../../README.md` + the `evolve-insights` memory (same round).
