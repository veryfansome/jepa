---
name: evolve
description: Run one generation (or a /loop of generations) of the chunk-based evolutionary design search over the terminal-JEPA sequence world model — heterogeneous Claude + Codex inventor subagents propose new chunk implementations, scored by the trustworthy content-verb-margin fitness. Invoke when the user asks to evolve/improve the world model, run an evolution generation, or explore new designs for a chunk (objective/arch/optim/…).
---

# evolve — chunk-based evolutionary design search (Claude-Code-native ShinkaEvolve)

Backend + design: `terminal-jepa/evolve/` (read `evolve/DESIGN.md` once per session). Method it
draws on: `ShinkaEvolve.md`. All commands run from the `terminal-jepa/` directory with
`.venv/bin/python`. Frozen embeddings must be cached (`data/dockerfs/emb-seq-*.pt`); if missing,
they build on first score.

## The invariant you must never break

Inventors mutate exactly ONE chunk file. They may NEVER touch: the frozen encoder / `cached_encode`,
`collate`'s cmd/obs interleaving and the no-future-leakage causal mask, or the eval path
(`content_retrieval`, the baselines, the retrieval metric). The harness re-checks per genome —
a leaking or mis-calibrated genome scores −inf — but do not hand inventors those files to edit.
Fitness is the held-out **content-verb margin** on inner-val (fedora+mariadb); the **final-test**
split (rockylinux+httpd) is never scored except on a promoted champion.

## One generation

1. **Seed the baseline once** (if the archive is empty): `.venv/bin/python -m evolve.cli seed --mode proxy`.
2. **Pick the parent + chunk.** `.venv/bin/python -m evolve.cli sample-parent` gives a
   fitness+novelty-weighted parent; for the first generations just use the baseline. Choose a
   chunk to mutate (start with `objective` — lowest risk, cannot touch the leakage mask).
3. **Build the inventor brief:** `.venv/bin/python -m evolve.inventors objective` — a
   self-contained brief (contract + leaderboard + what's been tried). Use its text verbatim.
4. **Dispatch a diverse ensemble of inventors** (this is the point — model/harness diversity):
   - **Claude inventors:** spawn 1–2 via the Agent tool, passing the brief. Ask each for a new
     objective module; take its returned code.
   - **Codex inventor (OpenAI, different harness):** run
     `codex exec -s read-only --skip-git-repo-check -C "$PWD" --output-last-message /tmp/codex_obj.md "<brief>"`
     then read `/tmp/codex_obj.md`. (Read-only sandbox: it proposes code, it does not write.)
   Give every inventor the identical brief so the comparison is fair.
5. **Land each proposal.** Extract the module with
   `.venv/bin/python -c "import sys;from evolve.inventors import extract_code;open(sys.argv[2],'w').write(extract_code(open(sys.argv[1]).read()))" <reply.md> evolve/chunks/objective/<name>.py`
   Name files `g<gen>_<idea>_<model>.py` (e.g. `g1_infonce_claude.py`, `g1_ranking_codex.py`) —
   the filename stem IS the objective impl name.
6. **Build a genome per proposal** (copy the parent, set `chunks.objective.impl` to the new stem;
   set `id`, `parent`, `generation`, `inventor` (`claude`/`codex`), `chunk_changed:"objective"`,
   `rationale`). Write to `/tmp/g<gen>_<name>.json`.
7. **Score proxy, then promote.** `.venv/bin/python -m evolve.cli score --genome /tmp/g….json --mode proxy`
   for each (records to the archive, −inf if it fails a guardrail). Promote any proxy winner that
   **beats the parent's margin** to a full score: `--mode full` (steps 4000 × 3 seeds).
8. **Leaderboard + memory.** `.venv/bin/python -m evolve.cli leaderboard`. Write a durable
   **insight** to auto-memory (e.g. `evolve-insights`): what beat the baseline and by how much,
   what collapsed and why — this is the meta-scratchpad that steers future generations across
   sessions. Report the generation's outcome to the user.

## Running a loop

For unattended generations use `/loop`: each firing = one generation (steps 2–8), parent sampled
from the archive, stopping when K consecutive generations produce no new archive-beating margin
(plateau) or a budget is hit. Keep proxy for the inner loop; promote sparingly. Always report the
champion's margin on the untouched **final-test** split (with `run_gen_twin` + `run_history_ablation`)
before claiming an improvement — an evolved champion must clear the program's own adversarial bar.

## Cost note

One proxy score ≈ a few minutes on MPS (the frozen encoding is already cached, so a score is just
the small transformer train + retrieval). Prefer proxy for exploration; a full score is ≈ the R4
run (~30 min). Codex calls cost OpenAI credits. Run inventor dispatch + scoring in the background
with waiters when possible.
