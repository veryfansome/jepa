# Adversarial Review — `research/07-stream.md` (the `stream` chunk)

## Bottom-line verdict
**Substantially sound and unusually well-grounded.** All 14 citations are real and correctly attributed; every project-specific number in §3.2 verifies *exactly* against the evolve archive; the arch-dependence direction (helps attention, neutral on the token-attention-less fast-weights arch) is reported correctly, not backwards. The defects are all Minor: two proxy-only numbers presented without a budget label alongside a full-budget result, one mechanism detail overstated (`mv_obs_k4` segment count), and a "champion" reference that is stale relative to the current R9 transformer-trunk arch.

---

## Findings by severity

### Critical
None.

### Major
None. (No fabricated citation, no unsupported empirical claim, no reversed mechanism. The report's central story survives every cross-check.)

### Minor

**M1 — Proxy-only numbers presented as if comparable to the full-budget result (precision/labeling).**
- Location: §3.2, `mv_obs_k4` "Result **0.5361 vs control 0.5386**" and `r7_role_multivec` "Result **0.5880 vs control 0.5885**"; contrasted with r8 "the first multi-vector stream to beat its control **at full budget**."
- Problem: All four of these numbers are **proxy** runs — 1000 steps, seed 0, inner-val only — while the r8 hero numbers are full-budget/3-seed. The report gives no budget label, so a reader assumes a like-for-like comparison. The implied "the earlier two lost at full budget" is not what happened: they were screened out at proxy and never promoted to full, so there is no full-budget head-to-head for the losers.
- Evidence (evolve archive `terminal-jepa/evolve/archive/genomes.jsonl`):
  - `mv-stream-k4` fitness 0.5361, `mode: proxy, seeds:[0], steps:1000, split:inner`.
  - `mv-arch-control` fitness 0.5386, `mode: proxy … inner` (the matched single-vector control — verifies).
  - `r7-stream-rolemv` fitness 0.588, `mode: proxy … inner`.
  - `r7-stream-mvarch-ctrl` fitness 0.5885, `mode: proxy … inner` (control — verifies).
- Note in the report's favor: it honestly labels the `mv_obs_k4` result "a loss" (Δ −0.0025, marginally outside the ±0.001 proxy noise band) and `r7` "a wash" (Δ −0.0005, inside noise) — the latter is *more* careful than the README, which flatly says "failed." So this is a labeling gap, not a misstatement of outcome.
- Fix: annotate the four numbers as "(proxy, seed 0, inner-val)"; and either drop "at full budget" from the contrast or add "the earlier two never cleared the proxy screen to reach a full-budget comparison."

**M2 — `mv_obs_k4` mechanism overstated: "K=4 contiguous line strips."**
- Location: §3.2, first bullet: "split each observation into K=4 contiguous line strips, *replacing* the aggregate vector."
- Problem: The impl is **1 header segment + 3 line strips**, not 4 line strips. Docstring of `chunks/stream/mv_obs_k4.py` (lines 1–4): "segment 0 = the cwd/exit status header; segments 1..3 = contiguous strips of the output lines … layout `[cmd_i, obs_i^0, obs_i^1, obs_i^2, obs_i^3]`, stride 1+K." So K=4 obs tokens total, but only 3 are line strips. Counting the header as a "line strip" mildly overstates the "arbitrary line-index boundaries" characterization (which the argument leans on).
- The "*replacing* the aggregate" half is correct (there is no mean-pooled aggregate token in that layout).
- Evidence: `terminal-jepa/evolve/chunks/stream/mv_obs_k4.py:1-4, 28-34`.
- Fix: "K=4 tokens per observation (a cwd/status header + 3 contiguous line strips), replacing the aggregate vector."

**M3 — "the fast-weights champion … has no token attention" is stale relative to the current champion.**
- Location: Abstract, §1, §2.3, §3.2, and design-guidance point 3 ("Extra tokens help attention trunks and do nothing for pure recurrences without token attention").
- Problem: The R8 path-key experiment's "fast-weights champion" was the R7 `r7_path_delta_fastweights_codex` arch (a pure delta-rule recurrence — the statement is true of it, confirmed by archive `r8-stream-pathkey` arch field). But the **current** champion per `README.md` (lines 44, 46) is `r9_chunked_delta_fastweights_codex` = "chunkwise-WY delta memories **on a transformer trunk**," which *does* have token attention. A reader taking "the fast-weights champion has no token attention" as present-tense would wrongly conclude the current champion can't exploit the path-key stream (in fact it is untested on it).
- Evidence: `README.md:44,46` (R9 champion is a transformer trunk); archive `r8-stream-pathkey` uses `arch: r7_path_delta_fastweights_codex`.
- Fix: scope the phrase — "the R7/R8-era path-delta fast-weights champion (a pure recurrence with no token attention)"; optionally add that the R9 transformer-trunk champion has attention and the path-key stream is untested on it (this also strengthens open question (a)).

**M4 — "single most actionable lesson" rests on a sub-0.003 margin.**
- Location: §3.2 / §4 point 3; the headline win is "+0.0015 inner / +0.0027 final."
- Problem: The full-budget win is genuine and correctly quoted from `README.md:57`, but tiny — inner +0.0015 is barely above the stated ±0.001 proxy-noise floor, final +0.0027 modestly above. The *arch-dependence* (help vs. exact neutrality) is the real signal; leaning the "most actionable lesson" on the magnitude slightly over-weights a marginal effect. The report does show the numbers, so the reader is not misled about size — this is framing, not misstatement.
- Fix: foreground the *qualitative* arch-dependence as the lesson and note the magnitude is small (the value is that a multi-vector stream stopped *losing*, and did so only where attention could address the keys).

**M5 — Citation title trimmed (trivial).**
- Location: Ref [5]. Report title "Poly-encoders: Architectures and Pre-training Strategies…"; the arXiv title is "Poly-encoders: **Transformer** Architectures…". The shorter form matches the published ICLR 2020 version, so this is at most cosmetic. Authors, ID (1905.01969), venue all correct.

---

## Verified clean

**Citations — 14/14 real, correctly attributed** (title / authors / arXiv ID / venue all confirmed via WebSearch):
- [1] ColBERT — Khattab & Zaharia, arXiv:2004.12832, SIGIR 2020. ✓
- [2] ColBERTv2 — Santhanam, Khattab, Saad-Falcon, Potts, Zaharia, arXiv:2112.01488, NAACL 2022. ✓ (residual-compression 6–10× storage claim supported.)
- [3] DPR — Karpukhin et al., arXiv:2004.04906, EMNLP 2020. ✓ (canonical ID.)
- [4] E5 — Wang et al., arXiv:2212.03533, 2022. ✓ (single-vector general-purpose embedder; matches usage.)
- [5] Poly-encoders — Humeau, Shuster, Lachaux, Weston, arXiv:1905.01969, ICLR 2020. ✓ (title trimmed — see M5; "learned code vectors that attend context then query" description correct.)
- [6] Sentence-BERT — Reimers & Gurevych, arXiv:1908.10084, EMNLP-IJCNLP 2019. ✓ (canonical ID.)
- [7] BERT — Devlin et al., arXiv:1810.04805, NAACL 2019. ✓ ("segment embeddings" claim correct.)
- [8] ViT — Dosovitskiy et al., arXiv:2010.11929, ICLR 2021. ✓ (16×16 patch tokenization correct.)
- [9] Perceiver — Jaegle et al., arXiv:2103.03206, ICML 2021. ✓ ("iterative cross-attention into a latent bottleneck" correct.)
- [10] Deep Sets — Zaheer et al., arXiv:1703.06114, NeurIPS 2017. ✓ (ρ(Σφ(x)) permutation-invariance correct; standard shorthand.)
- [11] ColPali — Faysse et al., arXiv:2407.01449, ICLR 2025. ✓ (patch-level multi-vector late interaction correct.)
- [12] Shwartz-Ziv & Tishby — "Opening the Black Box…", arXiv:1703.00810, 2017. ✓ (information-bottleneck framing correct.)
- [13] Weller, Boratko, Naim, Lee — "On the Theoretical Limitations of Embedding-Based Retrieval," arXiv:2508.21038, 2025 (Google DeepMind & JHU). ✓ **This was the flagged likely-fabrication point; it is real.** Top-k-subset capacity bound tied to embedding dimension, and the LIMIT dataset, both confirmed.
- [14] Flamingo — Alayrac et al., arXiv:2204.14198, NeurIPS 2022. ✓ (Perceiver Resampler → fixed token set correct.)

**Project-data fidelity — verified against `archive/genomes.jsonl` + `README.md:57` + `evolve/CLAUDE.md`:**
- `mv_obs_k4` 0.5361 vs control 0.5386 — **exact match** (archive `mv-stream-k4`, `mv-arch-control`).
- `r7_role_multivec` 0.5880 vs control 0.5885 — **exact match** (archive `r7-stream-rolemv`, `r7-stream-mvarch-ctrl`).
- r8 additive path-keyed "+0.0015 inner / +0.0027 final on the hippo (attention) leader, noise-neutral on the fast-weights champion" — matches `README.md:57` verbatim; archive confirms `r8-pk-hippo-pathkey` at full budget (inner 0.6092 / final 0.5668, 3 seeds) and a separate hippo control.
- Mechanism of r8 (additive `[cmd, obs, p^1..p^K]`, stride 2+K, delta-rule write fires only on the untouched aggregate obs token, path-prefixed segments in the encoder's own passage/query space) — matches `chunks/stream/r8_pathkey_entries.py` docstring and code line-for-line.
- Impl names `mv_obs_k4` / `r7_role_multivec` / `r8_pathkey_entries` / `baseline_interleave` all exist in `chunks/stream/`.
- "Multi-vector universally helped" — **not** claimed; report correctly says two failed and the third helped only under attention. Arch-dependence **not** reversed.
- "pathkey stream stayed neutral in a third (planning) context" — matches `README.md:70`.

Citations checked: 14/14 resolved. Issues: 0 Critical, 0 Major, 5 Minor.
