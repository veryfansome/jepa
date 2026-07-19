---
name: evolve
description: Run one round (or a /loop of rounds) of the chunk-based evolutionary design search over the terminal-JEPA sequence world model — heterogeneous Claude + Codex inventor subagents propose new chunk implementations, scored by the held-out content-verb-margin fitness. Invoke when the user asks to evolve/improve the world model, run an evolution round, or explore new designs for a chunk (objective / arch / perception / batcher / …).
---

# evolve — chunk-based evolutionary design search (Claude-Code-native ShinkaEvolve)

**`terminal-jepa/evolve/CLAUDE.md` is the manual and the single source of truth** — the 8 chunks + their contracts, the fitness + guardrails + 3-way split, the proxy→full ladder, the inventor-dispatch pattern, recombination/epistasis, and the full step-by-step "how to run a round" recipe live there and stay current. Read it, then run the round from it. This skill is the entry point + the decisions that are easy to get wrong. All commands run from `terminal-jepa/` as `uv run python -m …`; the current champion trains on `data/dockerfs-e5`.

## The round, at a glance

Follow the "How to run a round" recipe in `evolve/CLAUDE.md`. The shape: re-ground (leaderboard + the `evolve-insights` memory; pick the parent with `cli sample-parent`, not always the champion) → brief (`inventors.py` auto-injects the frontier + top-impl code + standing rules) → **dispatch inventors** → **CPU-smoke-test + register** → **proxy queue** → **promote + recombine** → **validate champion** → record neutral stats. A round fans out across *several* chunks at once (one inventor per chunk/lens), not just one.

## Don't get these wrong (the traps R5–R7 taught us)

- **Never score `final-test` for selection.** Fitness is the inner-val (fedora+mariadb) content-verb margin; the untouched final-test (rockylinux+httpd) is scored only on a promoted champion, once. Inventors never touch the eval / metric / leakage mask — the harness re-checks (−inf on a violation), but don't hand them those files.
- **Proxy deltas < ~0.002 are noise** (measured dev-environment run-to-run nondeterminism; CPU is bit-exact). Promote to full only on a clear proxy win, and trust only the **full-budget 3-seed** result — don't burn a full run on a sub-0.002 proxy "win."
- **Inventors: Workflow at `effort:'high'`, code-grounded, cross-domain.** Dispatch Claude inventors via the **Workflow** tool (`agent(prompt, {effort:'high', schema})`), NOT the Agent tool (no effort knob → they underperform: at default effort 0/3 beat the incumbent; at effort=high with mandatory code-reading, 2/3 did). Require each to read the real harness/eval code + `archive/genomes.jsonl` first, and encourage external research (arxiv + neuroscience / biology / physics — the hippo, fast-weights, and free-energy winners all came from cross-domain lenses). Add 3 **Codex** inventors (`codex exec -s read-only --output-last-message`) for genuine model diversity.
- **CPU-smoke-test every proposal before spending GPU** (finite/anti-collapse for objectives, strict leakage for archs, invertibility for targets, …), and **harden inventor integration bugs yourself** — they can't test against the live harness (the R7 head axis had a module cycle + forward recursion; a Codex `torch.where` crash needed a Reflexion retry).
- **Serialize GPU work.** One background scoring queue at a time, via **tracked** background tasks — not detached `&` (loses tracking) and not foreground (times out at 2 min on the slow fast-weights-style archs).
- **Recombine, and test epistasis.** After single-chunk mutations, stack the winners into one genome and full-score it — the biggest gains came from stacking (+0.045 at full in R6). But epistasis can be **negative** (a win on one arch can hurt on another — R7 fe-mutprox), so always test the combination, never assume it. Failed traits may be retried recombined in a changed context.

## Running a /loop

For unattended generations, `/loop` fires one round per iteration (sample a parent from the archive, mutate/recombine, score, record). Keep the inner loop on proxy; promote sparingly. Stop when K consecutive rounds produce no new archive-beating full-budget margin (plateau) or a budget is hit. **Before claiming any improvement**, validate the champion on the untouched final-test with `evolve/sanity.py` (gen-twin + history ablation) — plus a revisited-split if the arch has a memory/history mechanism — and record the result as a neutral stat in the `evolve-insights` memory.

## Cost

A proxy score ≈ a few minutes (frozen encodings are cached; a score is just the small transformer train + retrieval); a full score ≈ the R4 run (~30 min × 3 seeds). Codex calls cost OpenAI credits. Run inventor dispatch + scoring in the background.
