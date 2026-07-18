# evolve/ — a Claude-Code-native, chunk-based evolutionary design search

A ShinkaEvolve-inspired loop specialized for the terminal-JEPA sequence world model
(`realenv/seq_worldmodel.py`, the R4 result). See `../../ShinkaEvolve.md` for the method it
draws on. The bet: decompose our working foundation into swappable **chunks**, have
heterogeneous LLM **inventors** (Claude subagents *and* Codex/OpenAI, for genuine model
diversity) propose novel implementations of a chunk, recombine them, and keep whatever
beats the current best on a **trustworthy fitness** — so exploration finds designs a single
model or a hand-search would miss.

Unlike the reference ShinkaEvolve (a Python framework driving an LLM API), the control loop
here is **native to Claude Code**: subagents are the mutation operators, auto-memory is the
meta-scratchpad, the Workflow tool fans them out, the Python package is only the deterministic
backend that assembles a genome, scores it, and records it.

**This doc is the replication manual.** It is kept current with the code. For the running
results ledger see the auto-memory note `evolve-insights` and `archive/genomes.jsonl`; for the
narrative and headline numbers see `../../terminal-jepa-status.md` (Phase R5–R7).

---

## The result this loop produced (as of R7, 2026-07-17)

Held-out **final-test** content-verb margin (untouched rockylinux+httpd; full budget, 3 seeds),
the search progression:

| step | change | final margin | wm content top-1 |
|---|---|---|---|
| R4 baseline | MSE + ModernBERT | 0.306 | 0.31 |
| R5 objective | + L2-InfoNCE (Claude) | 0.459 | 0.72 |
| R6 encoder | + e5-base-v2 "eyes" | 0.5017 | 0.80 |
| R6 recombine | + recency-ALiBi arch / focal obj / warmup optim | 0.5471 | 0.84 |
| R6 arch | + hippocampal episodic-memory arch | 0.5607 | 0.85 |
| R6 batcher/obj | + system-blocked batcher + free-energy objective | 0.5641 | 0.857 |
| **R7 arch** | **+ path-delta fast-weights arch (Codex)** | **0.5702** | **0.863** |

**Current best genome** (`archive/genomes.jsonl`, id `r7-arch-fastweights`): e5-base-v2 encoder
(data root `data/dockerfs-e5`) + `r7_path_delta_fastweights_codex` arch +
`r6_free_energy_precision_l2_contrastive` objective + `r6_sysblock_hardneg_curriculum` batcher
(block2/hard75/ramp30) + `go_warmup_holdcos_floor` optim. Validated with the champion sanity
arms (latent ≫ generative twin; history-driven; gains hold on novel context).

---

## The two layers

**Python backend (this package) — deterministic, no LLM calls.**
- `genome.py` — a genome is a JSON dict selecting one implementation per chunk (+ params).
  A registry maps chunk-impl names to code (`load_objective`, `load_arch`, `load_optim`,
  `load_target`, `load_batcher`, `load_stream`, `load_head`). `baseline_genome()` is the R4
  world model; `validate()` structurally checks a genome before a run is spent on it.
- `chunks/<chunk>/<impl>.py` — the swappable implementations. Inventors drop new files here;
  `list_impls(chunk)` discovers them by filename.
- `splits.py` — the mandatory **3-way split** (see below).
- `harness.py` — `score_genome(genome, mode, data, split)`: assemble → train → **guardrail
  hard-filter** → evaluate the **content-verb margin** → return a scalar fitness + metrics.
  Reuses the *validated* eval code from `realenv/seq_worldmodel.py` (same retrieval metric,
  same baselines) so fitness means exactly what the R4 result meant.
- `reencode.py` / `mv_encode.py` — the data-side re-encode for the `perception`/`stream`
  chunks: re-embed the corpus under a new encoder/render recipe into a **new data root**, then
  score genomes on it via `--data <root>`. `reencode.py` builds single-vector roots;
  `mv_encode.py` builds multi-vector roots (adds `z_obs_multi`/`obs_valid`, copying the
  single-vector caches verbatim so the target/eval space is unchanged).
- `archive.py` — append-only `archive/genomes.jsonl`; leaderboard; fitness+novelty-weighted
  **parent sampling**.
- `sanity.py` — the champion validation arms (gen-twin + history ablation) on final-test.
- `cli.py` — `seed | score | leaderboard | sample-parent | impls` (the backend the loop calls).
- `inventors.py` — builds the inventor brief for a chunk (contract + leaderboard + what's
  tried); the same brief goes to Claude and Codex, so mixing them is genuine model diversity.

**Claude-Code layer — the evolutionary control loop.**
- **Inventors (mutation operators):** for a chosen chunk, dispatch subagents given the chunk
  contract + the results ledger + external-research encouragement, asked to invent a NEW
  implementation. Two harnesses for diversity: **Claude** (via the **Workflow** tool at
  `effort: 'high'` with mandatory code-reading — see "Inventor dispatch" below) and **Codex**
  (`codex exec -s read-only --output-last-message`).
- **Archive = memory.** The JSONL is the program archive; durable *stats* (per-genome margins,
  what failed) go to the `evolve-insights` auto-memory note so they survive across sessions and
  steer future rounds — the meta-scratchpad. Record **stats, not verdicts** (see "Culture").
- **A round:** pick target chunks → dispatch inventors → register + smoke-test each proposal →
  `cli score` each at cheap **proxy** → promote the best to **full** → recombine winners →
  validate the champion → record all stats.

---

## The chunks (genes of the foundation)

Eight evolvable axes. Each has a `baseline`/`identity` impl that reproduces current behavior
**bit-for-bit** (the plumbing check every new axis must pass).

| chunk | what it controls | genome key | backend surface | notes |
|---|---|---|---|---|
| `objective` | the training loss `loss(pred, tgt)` | `chunks.objective` | `_train` loss line | biggest early lever (InfoNCE, free-energy) |
| `arch` | the causal predictor module | `chunks.arch` | `load_arch` → `build(**params)` | biggest overall lever (hippo, fast-weights) |
| `optim` | optimizer + LR schedule | `chunks.optim` | `load_optim` → `make(params, steps)` | warmup/hold/cosine-floor best |
| `target` | what is predicted (a transform of z_obs) | `chunks.target` | `load_target` | pure (invertible) or **LEARNED** (see extensions) |
| `perception` | the "eyes": encoder model + render + pooling | `chunks.perception` | `reencode.py`/`mv_encode.py` → new data root | encoder swap = biggest data-side lever (e5) |
| `batcher` | training-batch composition (the in-batch negative pool) | `chunks.batcher` | `load_batcher` | system-blocked hard-negative curriculum beats uniform |
| `stream` | how (cmd,obs) steps are laid out as tokens | `chunks.stream` | `load_stream` (collate/extract/flatten/leakage) | single-vector interleave (baseline) vs multi-vector |
| `head` | the readout `h → prediction` (+ optional aux task) | `chunks.head` | `load_head` (wrap/aux_loss/leak_safe) | new in R7; no gain yet |

Immutable no matter what: `cached_encode` of the frozen embeddings, the **no-future-leakage
causal contract**, and the **entire eval path** (`content_retrieval`, the baselines, the
retrieval metric, the 3-way split). A genome may never edit the thing that scores it.

### Contract extensions (how axes were added without breaking archived genomes)

The harness was extended twice beyond simple chunk-swaps; both are load-bearing patterns:

- **LEARNED targets.** A `target` impl may set `LEARNED = True` and expose `make(D) -> nn.Module`
  with `make_target/to_obs/reg`. The harness registers the module's params on the net (trained
  jointly by the genome's optimizer, `reg()` added to the loss) and evaluates via the module's
  **exact inverse in the fixed obs space** — so a collapsed learned target cannot reconstruct
  and is scored down. Honest by construction. Example: `tgt_space_diag_gate`.
- **`head` axis.** A `head` impl exposes `wrap(net, D, **p)` (may re-point `net.forward` and
  register readout/aux params — run **before** the optimizer is built so aux params are
  optimized), `aux_loss(state, batch, net, device)` (a train-only self-supervised term, `0.0`
  for passthrough), and `leak_safe(mod, params)` (a required predicate asserted before scoring).
  Two integration hazards a wrapper must avoid, both caught in R7: a **parent↔child module
  cycle** (store the base net as a non-registered reference, else `net.to(device)` recurses),
  and **forward recursion** in a trunk fallback (save the original bound `forward` before
  re-pointing). Note the `head` recomputes the prediction from the trunk hidden state `h`, so on
  an arch whose prediction is *not* a per-position function of `h` alone (e.g. the hippo episodic
  read), it silently bypasses that mechanism — test the head on a `pred = head(h)` arch to
  isolate its effect.

Every new axis adds one loader in `genome.py`, one `validate()` line, and threads through
`_train`/`score_genome`; its baseline impl is verified **bit-identical** to pre-axis behavior
(unit test + an end-to-end proxy that must reproduce the incumbent's recorded margin).

---

## Fitness — the part that must be trustworthy

Fitness = the **content-verb margin**, mirroring the R4 headline exactly:

    margin = content_top1(WM) − max(content_top1(retrieve_by_cmd),
                                    content_top1(wm_no_history),
                                    content_top1(copy_prev))

on **content verbs only** (ls+cat; `cd` is a trivial `cwd=<target>` echo). The *margin* (not
absolute top-1) is the fitness because R4's signal partly rides on shared cross-distro
filesystem structure the lexical baseline already banks — a variant that lifts the WM and the
baselines equally has discovered nothing. (This is why bigger encoders can *lose*: e5-large
lifted the baselines as much as the WM, so the margin fell even though wm top-1 held.)

**Hard filters (fitness = −inf if any fails):**
- the per-genome **no-future-leakage** check (perturb obs_t, the cmd_t prediction must not
  move) — so a genome can never "win" by leaking the answer. Lives in each `stream` impl's
  `leakage_ok` (the multi-vector stream perturbs *all* segments of obs_t);
- `predict_mean` retrieval must stay ≈ chance (the metric-calibration guard);
- training must not NaN / collapse; any inventor-code exception → −inf, never a crash.

**The 3-way split (non-negotiable — the anti-overfitting rail).** Evolving against a metric
overfits it. So (`splits.py`):
- `fit` — the 8 train images (minus a dev slice) → train on these;
- `inner-val` — **fedora + mariadb** (held-out images) → **fitness is scored here**;
- `final-test` — **rockylinux + httpd** (held-out images) → the loop **never** scores these for
  selection; only a promoted champion is judged here, once, plus the `sanity.py` arms.

---

## The proxy → full ladder (sample efficiency)

- **proxy** = `steps≈1000`, `seeds=[0]` — the inner-loop score every proposal gets.
- **full** = `steps=4000`, `seeds=[0,1,2]` — only proposals that beat the incumbent at proxy are
  promoted, on both `inner` and `final` splits, before entering the archive as scored survivors.

The frozen encoding is cached per data root (`emb-seq-{split}.pt`) and the objective-independent
baselines are cached in `archive/base_cache.json` keyed by `data_root|split|seed|steps`, so a
score is just the small transformer train + retrieval — affordable on Apple Silicon.

**MPS nondeterminism (measured, R7).** The *identical* genome scores within a **~±0.001** band
across MPS proxy runs (fixed seed, fixed foil draws) because float-accumulation order varies
run-to-run; CPU training is bit-exact. **Implication: proxy deltas smaller than ~0.002 are
noise** — do not promote on a sub-0.002 proxy win alone; the full-budget 3-seed result is the
reliable signal. Proxy→full shrinkage is also normal (e.g. e5: +0.155 proxy → +0.043 full).

---

## Inventor dispatch (the mutation operators)

**Effort matters, and the Agent tool has no effort knob.** At default effort, inventors finish
too fast and ungrounded — 0/3 grounded objectives beat the incumbent. Dispatch Claude inventors
via the **Workflow** tool at `effort: 'high'` with a brief that **mandates reading the actual
chunk code and the results ledger before designing**; that measurably lifted grounding (2/3 beat
the incumbent). The current pattern (see `scratchpad/r*/` and the R6/R7 workflow scripts):

1. One Workflow, ~10 Claude inventors in `parallel()`, one per chunk/lens, each returning a
   **structured proposal** (schema: name, chunk, idea, mechanism, external_refs, code, risks).
2. Give each brief: the exact chunk **contract**, the **full stats ledger**, the standing rules
   (below), and **explicit external-research encouragement** — arxiv, and beyond ML
   (neuroscience, biology, ecology, physics, information theory). Cross-domain lenses produced
   several winners (the hippocampal episodic-memory arch, the free-energy objective).
3. In parallel, 3 **Codex** inventors (`codex exec -s read-only --output-last-message`) on the
   same briefs → genuine model/harness diversity. Codex produced the R7 fast-weights arch (the
   current best) and the R6 Householder targets.
4. **Register + smoke-test every proposal on CPU before spending GPU:** objectives (finite loss,
   gradient, **anti-collapse** — a constant prediction must not minimize it, n=1 edge case);
   archs (shapes, **strict leakage** across every obs perturbation, param count, forward time);
   targets (invertibility, or LEARNED identity-at-init + reg); optim (a real short loop, LR
   schedule, finiteness); batcher (deterministic, in-bounds); perception (768-d, deterministic).
   **Harden inventor integration bugs yourself** — the inventor couldn't test against the live
   harness (the head module cycle/recursion, a Codex `torch.where` float-condition crash salvaged
   by a Reflexion retry).

**Standing rules in every brief:** the metric/eval/split are fixed and off-limits; be
causal/leak-free/NaN-safe/anti-collapse-safe; novelty over safety (a safe tweak is a wasted
slot); **previously-failed traits may be retried recombined** (see epistasis).

---

## Recombination & epistasis

The single biggest source of gains after the first objective/encoder wins was **stacking**
marginal per-chunk winners: arch+objective+optim wins that were ~+0.005 each individually
composed for +0.045 at full budget (positive epistasis, confirmed at full). So after a round of
single-chunk mutations, **recombine the winners** into one genome and full-score it.

But epistasis cuts both ways — always test it, never assume it:
- **Positive:** the R6 stack; the batcher + free-energy + arch all composing into 0.5641.
- **Negative:** in R7, fusing the fe-mutprox objective onto the fast-weights arch scored *below*
  fast-weights alone (the fast-weights target-space read partly subsumes what a hubness/geometry
  objective adds). So free-energy stayed the objective.
- **Retry failed traits recombined.** Don't foreclose a trait because it lost once — evolution
  doesn't. The `target_space` idea failed 3× early; retried on the e5 encoder + contrastive
  objective it came within noise of identity (a much smaller gap). Deprioritize, keep live, retry
  in the changed context. (See the `evolve-retry-failed-traits` memory.)

---

## Culture: record stats, not verdicts

The ledger records **neutral stats** (per-genome margins, what was tried), not conclusions.
Editorializing a result ("objective X is the answer") biases the next round and foreclosed
exploration. Log the number and the setup; let the next round decide. Negative and null results
are recorded with the same weight as wins — several "failures" (multi-vector readouts, learned
targets, the head axis, Lévy exploration) are load-bearing knowledge about where the margin does
*not* live.

---

## How to run a round (recipe)

1. **Re-ground.** `cli impls --chunk <c>` for every chunk; `cli leaderboard`; read the current
   best genome from `archive/genomes.jsonl`; read the `evolve-insights` memory.
2. **Brief.** Regenerate the Codex briefs (`python -m evolve.inventors <chunk>`), append an
   addendum with the round's incumbent + the prior round's deltas.
3. **Dispatch.** One Workflow of ~10 high-effort Claude inventors (one per chunk/lens, arxiv +
   cross-domain encouraged) + 3 background Codex inventors.
4. **Register + smoke-test** every returned proposal on CPU (contracts above). Harden integration
   bugs. Stage new-axis / re-encode / multi-file specs for review.
5. **Proxy queue** (serialize GPU work — one background queue, not parallel): score each proposal
   as a one-chunk mutation of the current best on the best data root; a new axis also runs its
   **bit-identical plumbing check**.
6. **Promote** anything beating the incumbent by > ~0.002 proxy to **full** (inner+final).
   **Recombine** the full winners; test epistasis explicitly.
7. **Validate the champion** (`sanity.py` gen-twin + history ablation on final-test; a
   revisited-split if an arch has a memory/history mechanism).
8. **Record** every result as neutral stats in `evolve-insights` + commit `archive/genomes.jsonl`.
   Update the progression table in `../../terminal-jepa-status.md`.

Operational notes: run heavy GPU jobs one at a time via tracked background tasks (not detached
`&`); `.venv/bin/python` (not the console scripts — stale shebangs); score via a redirect-to-file
+ JSON-extract (nested grep escaping breaks capture); derived data roots (`data/dockerfs-*/`) are
gitignored, only their `summary.json` is tracked for provenance.

---

## Why Claude-Code-native (vs. porting the shinka package)

Model/harness diversity for free (Claude + Codex + any Agent model), the archive and insights
live in auto-memory across sessions, no API-key plumbing or task-contract adaptation, and the
loop is inspectable/steerable at every generation. The cost is that we implement the loop
ourselves — but the loop is small; the deterministic backend here is the only code that has to be
exactly right, and it reuses the already-validated R4 eval.
