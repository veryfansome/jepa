# terminal-JEPA — Context

Research program testing whether Yann LeCun's **JEPA** world-model recipe (predict in latent space, anti-collapse, plan by latent distance) transfers to a **real-shell agent**. Active substrate: a sequence world model over real Linux **Docker** filesystems that, given an exploration history, predicts `ls`/`cat`/`cd` outcomes on **unseen systems** — improved by a Claude-Code-native **ShinkaEvolve** evolutionary search. Python + PyTorch on Apple **MPS**; Docker for data collection. Deps via **uv**.

Git root is `jepa/`; all code lives under `terminal-jepa/`.

## Documentation index — where context lives (single source of truth per topic)

| doc | owns | status |
|---|---|---|
| `CLAUDE.md` (this file) | repo identity, navigation, repo-wide rules, doc-sync triggers | live, auto-loaded |
| `terminal-jepa/CLAUDE.md` | the code project: env/run, active-vs-retired module map, data + regen recipe | live, loaded in-subtree |
| `terminal-jepa/evolve/CLAUDE.md` | the evolve loop working context: chunks, scoring, guardrails, archive/memory | live, loaded in-subtree |
| `terminal-jepa-status.md` | **the living status** — current direction, R4→R7 results, findings (synthetic negative as prior work) | **current — read for "what's true now"** |
| `terminal-jepa/evolve/DESIGN.md` | the evolve **replication manual** — how the search works and how to run/iterate a round | current (deep methodology) |
| `terminal-jepa/README.md` | module/file inventory + reproduction commands | current for R4/evolve; Phase-0/1 sections describe retired code |
| `JEPA.md` | background: LeCun's JEPA research program (2022–2026) | external reference (stable) |
| `ShinkaEvolve.md` | background: LLM-driven evolutionary program search | external reference (stable) |

Cross-session working **state** (per-genome result stats, user preferences) lives in **auto-memory** (`~/.claude/projects/-Users-fanzhu-PyCharmProjects-jepa/memory/`, indexed by `MEMORY.md`), not in these docs. Rule of thumb: **versioned repo facts → CLAUDE.md / docs; evolving cross-session state → auto-memory.** The `evolve-insights` memory is the neutral running stat ledger.

## Code map

The whole tree is now the active direction: `terminal-jepa/realenv/{seq_worldmodel,docker_env,collect_docker}.py` (the R4 world model + Docker data collection), all of `terminal-jepa/evolve/` (the search), `terminal-jepa/tests/test_seq_worldmodel.py`, and the `data/dockerfs*` roots. The synthetic Phase 0–1 sandbox and the R1–R3 real-shell prototypes were **removed 2026-07-18** — git history retains them; their empirical record is the "prior work" section of `terminal-jepa-status.md`. Don't resurrect that substrate: it cannot scale past the retired 301-slot ontology / held-out-tool framing.

## Repo-wide rules

- **Dependencies: uv.** `cd terminal-jepa && uv sync` rebuilds the exact env from `pyproject.toml` + `uv.lock`. Run tools as `uv run python -m <module>`. Never add `requirements.txt`. See the `uv-dependency-management` memory.
- **Record stats, not verdicts.** In the running ledger and result docs, log neutral numbers + setup; don't editorialize or foreclose exploration. (`evolve-record-stats-not-conclusions` memory.)
- **Honest evaluation is non-negotiable.** Honest baselines, multi-seed, adversarial review of large changes/results before writing them up, conclusive over convenient. The evolve **3-way split** (fit / inner-val / untouched final-test) and its guardrails are sacred — never score final-test for selection, never let a genome edit the eval. (`working-style-adversarial-review` memory.)
- **Failed traits stay live.** Deprioritize, don't foreclose; retry recombined in changed contexts (epistasis). (`evolve-retry-failed-traits` memory.)

## Keeping context in sync (do this in the SAME change, so docs don't drift)

Each `CLAUDE.md` is a contract with future sessions: the context you need to work in this subtree that the code itself doesn't provide. Be concise, not didactic — maximize relevance per token.

**Update triggers** — when a change touches one of these, update the named doc in the same commit:
- **`terminal-jepa/CLAUDE.md`** — you add/rename/retire a module or run command, add a data root, or move the active-vs-retired boundary.
- **`terminal-jepa/evolve/CLAUDE.md`** + **`evolve/DESIGN.md`** — you add a new evolvable **chunk axis**, change the fitness/guardrails/split, or change the scoring CLI. (DESIGN.md owns the deep how; the CLAUDE.md owns the quick operational pointer.)
- **`terminal-jepa-status.md`** — a promoted full-budget result, a new champion genome, or a finding. Record it here (and the neutral stat in the `evolve-insights` memory). This is the living status.
- **`terminal-jepa/README.md`** — you add/rename a module or its run command (the file inventory).
- **the `evolve-insights` memory** — any scored genome (neutral stats).

**Conventions.**
- Keep each `CLAUDE.md` under ~200 lines. Approaching the cap → push detail down a level (or into the owning deep doc) and link, don't compress prose.
- Every `CLAUDE.md` has a sibling `AGENTS.md` symlink (`ln -s CLAUDE.md AGENTS.md`, committed as a symlink) so the **Codex** inventors and other `AGENTS.md` readers load identical context. Creating a new `CLAUDE.md` → add its `AGENTS.md` symlink and index it in this table, same commit.
- Point to the deep docs for detail (status / DESIGN / README); **never duplicate** them here — a fact with two homes drifts. If a fact is specific to one file, put it in a code comment, not a CLAUDE.md.
- Delete a `CLAUDE.md` (or a line) that stops earning its tokens; a future session recreates it if the need recurs.
