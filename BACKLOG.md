# Backlog — deferred ideas

Forward-looking ideas we've decided not to do *yet* (distinct from `README.md`, which records what's true now). Pull an item up when its cost/benefit flips. Delete items that stop being worth their line.

## Evolve framework — turning it inward (meta-evolution)

- **Brief / prompt evolution (the feasible inward-turn).** Evolve the inventor brief (`evolve/inventors.py`) Promptbreeder-style, scored by the **proxy margin of the proposals a brief-variant elicits** (dispatch a few inventors with brief B, proxy-score their chunk code). This is the one framework component with a cheap, hard-to-game meta-fitness (proposals still face the JEPA guardrails + 3-way split, so a "better brief" can only mean better-scoring chunk code). Treat the brief as a text chunk: fitness = mean proxy margin of its N proposals; evolve on inner-val, validate the winning brief on a held-out chunk/round. Deferred 2026-07-18 in favor of the cheap framework wins (inspiration exemplars, auto-meta-scratchpad, sample-parent — now done). Design sketch owed.
- **Full harness / policy meta-evolution — deferred until the JEPA search plateaus.** Evolving the harness, parent-sampling policy (λ), dispatch config, promotion threshold, chunk-selection policy, scored by whole-loop outcomes. Premature now: a meta-eval = a whole evolution loop (hours, noisy), the framework isn't the bottleneck (search still climbing 0.306→0.5702), and the meta-fitness is gameable (cf. Darwin Gödel Machine learning to falsify its own eval, `ShinkaEvolve.md` §6/§9). Revisit when (a) the search plateaus and (b) a cheap, un-gameable meta-fitness proxy exists. Until then hand-tune policies via the record-stats loop.

## Evolve framework — ShinkaEvolve/lineage mechanisms not yet adopted (see `ShinkaEvolve.md`)

Adopted 2026-07-18: inspiration exemplars, automated meta-scratchpad (champion-frontier digest), and using `sample_parent` in the round. Still on the table, lower-leverage at our scale:

- **Bandit LLM/lens allocation (§4.3, UCB1, reward = relative improvement).** Adaptively shift inventor budget toward whichever *model* (Claude vs Codex) or *lens* (grounded / cross-domain / recombiner) or *chunk* is currently producing improvements, instead of a fixed fan-out. Modest with 2 model-arms; more interesting as a bandit over lenses or which-chunk-to-target across many rounds.
- **Diff / intra-chunk crossover patch types (§3.2; AlphaEvolve).** Beyond full rewrites: a "mutate this winning impl slightly" diff mode and an intra-chunk crossover (combine two winning objectives/archs). We already do genome-level stacking; code-level crossover of two impls of the *same* chunk is untried.
- **Embedding novelty-rejection (§4.2) — low priority here.** Cosine-sim dedup of proposals before evaluation. Its payoff scales with eval cost; ours are cheap and few (~13/round), so a near-duplicate proposal is cheap to just score. Reconsider if eval cost rises.
- **Islands + migration (§3.1) — low priority here.** Independent subpopulations to avoid premature convergence. Our multi-chunk fan-out + heterogeneous inventors already supply diversity.

## JEPA model / research directions

### World/action expansion roadmap (2026-07-19 design; phases 0–3 COMPLETED, phase 5 superseded — see below)

Phases 0–3 of the original roadmap are done and recorded in `README.md` (Stage-1 planning probe: positive; v2 constitution + mint: `terminal-jepa/benchmarks/bench-constitution.md` + dockerfs2-v2.0; Stage-2 latent MPC: honest localized negative). The old Phase-5 write-tools sketch is **superseded** by the ratified v3 design (`terminal-jepa/benchmarks/dockerfs3-design-draft.md`, decided 2026-07-22), which resolves its central worry (the image-independent scratch slice bypassing the held-out-images bar) with image-grounded payload provenance + the symbolic shell-state-tracker baseline + measured `sim`-class exclusion. Still deferred:

- **v4 PTY spike (UD-1 deferral, 2026-07-22).** A persistent-PTY substrate bringing `jobs`/`fg`/`bg`/`%n`/`wait`, literal `^C`/`^Z`, bare `sleep N &`, `kill -INT`, and `<<<` herestrings into the world. Deferred from v3 because the exec-substrate virtual step-clock delivers the same 3-state process automaton race-free, while the PTY was the pre-mortem's top nondeterminism surface (measured: "Done"-notification placement wall-clock-nondeterministic with >200-step spread on identical seeds; unreachable timeout-recovery ladder). The spike charter inherits the infra design memo AND its verdict's mandated fixes (sentinel framing, stty preamble, tiered SIGINT→kill ladder, zero-timeout gate) — both archived in the v3 design provenance (draft §15).
- **Live services + wider held-out pool (the old Phase 4).** Services actually running (redis/nginx/postgres are collected as dead filesystems today): per-image specs, readiness probes, `--hostname` pinning; step-class taxonomy incl. a dynamic-service slice entering fitness only at its own version boundary; container-restart-every-N-sequences discipline; new held-out images with train-alias tagging. Partial predictability becomes ON-thesis here (the JEPA bet: the generative twin degrades more than the latent model on the partially-predictable slice). Candidate for v4/v5 alongside or after the PTY spike.
- Man-page/tool-inference stays an eventual goal, not a gate (the R1-R3 lesson) — revisit as an exploratory arm once v3 verbs are solid.

- **Expand the toolset beyond the initial `uname`/`cat`/`ls`/`cd`** (e.g. `man`/`less`/more tools) once the current bar is solid — the content-verb eval grows with it. Superseded in detail by the roadmap above; kept as the one-line intent.
- **Structured / path-keyed multi-vector readout** — the earlier finding pointed at a path-keyed readout; the R6 line-strip and R7 role-canonical multi-vector streams both lost to their single-vector controls, but a per-directory-entry or path-keyed variant is untried (a one-file `stream` + `perception` impl on existing infra).
