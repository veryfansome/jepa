# evolve/ — a Claude-Code-native, chunk-based evolutionary design search

A ShinkaEvolve-inspired loop specialized for the terminal-JEPA sequence world model
(`realenv/seq_worldmodel.py`, the R4 result). See `../../ShinkaEvolve.md` for the method it
draws on. The bet: decompose our working foundation into swappable **chunks**, have
heterogeneous LLM **inventors** (Claude subagents *and* Codex/OpenAI, for genuine model
diversity) propose novel implementations of a chunk, recombine them, and keep whatever
beats the baseline on a **trustworthy fitness** — so exploration finds designs a single
model or a hand-search would miss.

Unlike the reference ShinkaEvolve (a Python framework driving an LLM API), the control loop
here is **native to Claude Code**: subagents are the mutation operators, auto-memory is the
meta-scratchpad, a skill + `/loop` drive the generations. The Python package is only the
deterministic backend: assemble a genome, score it, and record it.

## The two layers

**Python backend (this package) — deterministic, no LLM calls.**
- `genome.py` — a genome is a JSON dict selecting one implementation per chunk (+ params).
  A registry maps chunk-impl names to code.
- `chunks/<chunk>/<impl>.py` — the swappable implementations. Inventors drop new files here.
- `splits.py` — the mandatory **3-way split** (see below).
- `harness.py` — `score_genome(genome, mode)`: assemble → train → **guardrail hard-filter**
  → evaluate the **content-verb margin** on inner-val → return a scalar fitness + metrics.
  Reuses the *validated* eval code from `realenv/seq_worldmodel.py` (same retrieval metric,
  same baselines) so fitness means exactly what the R4 result meant.
- `archive.py` — append-only `archive/genomes.jsonl`; leaderboard; ShinkaEvolve-style
  fitness+novelty-weighted **parent sampling**.
- `cli.py` — `seed | score | leaderboard | sample-parent` (the backend the loop calls).

**Claude-Code layer — the evolutionary control loop (a skill + `/loop`).**
- **Inventors (mutation operators):** for a chosen chunk, dispatch subagents given the chunk
  contract + the archive leaderboard + what's already been tried, asked to invent a NEW
  implementation. A mix of **Claude** (Agent tool) and **Codex** (`codex exec -s read-only
  --output-last-message`) → the ShinkaEvolve "ensemble of frontier LLMs", across harnesses.
- **Archive = memory.** The JSONL is the program archive; durable *insights* ("cosine
  objective beat MSE by +X margin", "the SSM arch collapsed") go to auto-memory so they
  survive across sessions and steer future generations — the meta-scratchpad.
- **Generation:** sample parent(s) → pick a chunk → dispatch N inventors (Claude+Codex) →
  write each proposal as a new chunk impl → `cli score` each (cheap proxy) → record →
  promote the best to a full score → update memory. Driven by the `evolve` skill / `/loop`.

## The chunks (genes of the foundation)

| chunk | what it controls | seq_worldmodel surface | risk |
|---|---|---|---|
| `objective` | the training loss | `train_model` loss line | LOW — start here |
| `arch` | the predictor (depth/width/heads/dropout, later module swaps) | `SeqWorldModel` | MED (module swaps can break the leakage mask) |
| `optim` | lr / wd / steps / batch | `train_model` optimizer | LOW |
| `history` | how context is used (full attn / summary token / decay) | `encode` mask | MED |
| `target` | what is predicted (obs / delta-from-prev / decomposed cwd+content) | `collate` tgt | MED |
| `exploration` | the data-gen policy | `collect_docker.py` | HIGH — forces re-collect+re-encode (slow outer loop) |

Immutable no matter what: the frozen encoder + `cached_encode`, `collate`'s cmd/obs
interleaving and the **no-future-leakage causal mask**, and the **entire eval path**
(`content_retrieval`, the baselines, the retrieval metric). A genome may never edit the
thing that scores it.

## Fitness — the part that must be trustworthy

Fitness = the **content-verb margin**, mirroring the R4 headline exactly:

    margin = content_top1(WM) − max(content_top1(retrieve_by_cmd),
                                    content_top1(wm_no_history),
                                    content_top1(copy_prev))

on **content verbs only** (ls+cat; `cd` is a trivial `cwd=<target>` echo). The *margin* (not
absolute top-1) is the fitness because R4's signal partly rides on shared cross-distro
filesystem structure the lexical baseline already banks — a variant that lifts the WM and the
baseline equally has discovered nothing.

**Hard filters (fitness = −inf if any fails):**
- the per-genome **no-future-leakage** check (perturb obs_t, the cmd_t prediction must not
  move) — so a genome can never "win" by leaking the answer;
- `predict_mean` retrieval must stay ≈ chance (the metric-calibration guard);
- training must not NaN / collapse.

**The 3-way split (non-negotiable — the anti-overfitting rail).** Evolving against a metric
overfits it. So:
- `fit` — the 8 train images (minus a dev slice) → train on these;
- `inner-val` — **fedora + mariadb** (held-out images) → **fitness is scored here**;
- `final-test` — **rockylinux + httpd** (held-out images) → the loop **never** scores these;
  only the champion is judged here, once, with `run_gen_twin` + `run_history_ablation`.

Inner-loop uses a cheap **proxy** (`steps≈1000`, `seeds=[0]`, optional `--limit`); only the
top proposals are **promoted** to a full score (`steps=4000`, `seeds=[0,1,2]`) before they
enter the archive as scored survivors — this is the sample-efficiency lever that makes an
LLM-driven loop affordable on Apple-Silicon (the frozen encoding is already cached, so a
score is just the small transformer train + retrieval).

## Why Claude-Code-native (vs. porting the shinka package)

Model/harness diversity for free (Claude + Codex + any Agent model), the archive and insights
live in auto-memory across sessions, no API-key plumbing or task-contract adaptation, and the
loop is inspectable/steerable at every generation. The cost is that we implement the loop
ourselves — but the loop is small; the deterministic backend here is the only code that has to
be exactly right, and it reuses the already-validated R4 eval.
