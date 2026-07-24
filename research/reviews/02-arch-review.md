# Adversarial Review — `research/02-arch.md` (the `arch` chunk)

**Bottom-line verdict:** Substantially clean and unusually well-sourced. All 22 references resolve to real papers with correct titles, authors, years, venues, and identifiers; every terminal-JEPA number matches the project's own README / evolve docs; the high-risk recent sequence-model citations (DeltaNet, Gated DeltaNet, DeltaProduct, minGRU, RetNet, Mamba, modern Hopfield, TEM, CLS) are all accurate. No fabrications, no misattributed data. Only one substantive nit (a slightly garbled DeltaProduct mechanism sentence) plus trivial venue-labeling items — none rising above Minor.

---

## Findings by severity

### Critical
None found.

### Major
None found.

### Minor

**M1 — DeltaProduct online-GD sentence conflates "one step" (DeltaNet) with DeltaProduct's multi-step generalization.**
- Location: §2.3 — "**DeltaProduct** [16] generalizes the single delta step per token to `n_h` steps … and, notably, casts the delta recurrence as *one step of online gradient descent per token on an associative-recall loss*, tying fast weights back to online learning."
- Problem: The "one step of online gradient descent per token on an associative-recall loss" framing is the *DeltaNet* interpretation that DeltaProduct **builds on and then generalizes to `n_h` steps** — it is not something DeltaProduct originates, and DeltaProduct's own contribution is precisely to take *multiple* steps, not one. As written, the sentence attributes the one-step-GD casting to DeltaProduct and sits awkwardly beside the immediately preceding clause about `n_h` steps.
- Evidence: The paper's abstract states it is "Building on the interpretation of DeltaNet's recurrence as performing one step of online gradient descent per token on an associative recall loss, [DeltaProduct] instead takes multiple (n_h) steps per token." (https://arxiv.org/abs/2502.10297)
- Fix: Reword to attribute the framing correctly, e.g. "DeltaProduct builds on the interpretation of *DeltaNet's* single-step recurrence as one step of online gradient descent per token on an associative-recall loss, and generalizes it to `n_h` such steps (a product of `n_h` Householder transforms)."

**M2 — DeltaProduct labeled preprint-only; it is now a NeurIPS 2025 poster.**
- Location: Ref 16 — "arXiv:2502.10297 (2025 preprint)."
- Problem: Under-credit only (the report errs conservatively, so this is not a peer-review overclaim). Given the July-2026 vantage, it has since been accepted.
- Evidence: NeurIPS 2025 poster page (https://neurips.cc/virtual/2025/poster/117900); OpenReview (https://openreview.net/forum?id=nvb60szj5C).
- Fix: Optionally update to "arXiv:2502.10297; NeurIPS 2025."

---

## Attack-surface notes

**1. Citation existence & attribution — 22/22 verified, all correct.** Precisely checked the high-fabrication-risk recent papers; every identifier, author list, year, and venue is right:
- [14] DeltaNet — Yang, Wang, Zhang, Shen, Kim; arXiv:2406.06484; NeurIPS 2024. ✓ (title/authors/ID/venue all confirmed)
- [15] Gated DeltaNet — S. Yang, J. Kautz, A. Hatamizadeh; arXiv:2412.06464; ICLR 2025. ✓
- [16] DeltaProduct — Siems, Carstensen, Zela, Hutter, Pontil, Grazzi; arXiv:2502.10297. ✓ (see M2 for venue update)
- [10] "Were RNNs All We Needed?" (minGRU/minLSTM) — Feng, Tung, Ahmed, Bengio, Hajimirsadeghi; arXiv:2410.01201. ✓
- [17] Modern Hopfield — Ramsauer et al.; ICLR 2021; arXiv:2008.02217. ✓
- [18] Neural Episodic Control — Pritzel, Uria, Srinivasan, Puigdomènech, Vinyals, Hassabis, Wierstra, Blundell; ICML 2017; arXiv:1703.01988. ✓
- [19] CLS — McClelland, McNaughton, O'Reilly; Psychological Review 102, 419–457, 1995; PMID 7624455. ✓ (PMID and page range confirmed)
- [20] TEM — Whittington … Behrens; Cell 183(5):1249–1263, 2020; DOI 10.1016/j.cell.2020.10.024. ✓ (issue, pages, DOI, authors confirmed)
- [11] Schmidhuber, fast-weight memories; Neural Computation 4(1):131–139, 1992; DOI 10.1162/neco.1992.4.1.131. ✓
- [12] Ba, Hinton, Mnih, Leibo, Ionescu; NIPS 2016; arXiv:1610.06258. ✓
- [3] RoFormer/RoPE — Neurocomputing 568, art. 127063, 2024; DOI 10.1016/j.neucom.2023.127063. ✓ (volume, article number, DOI all confirmed)
- Well-established refs [1] Vaswani, [2] Shaw (NAACL 2018), [4] ALiBi (ICLR 2022), [5] Katharopoulos linear-attn (ICML 2020), [6] S4 (ICLR 2022), [7] Mamba (COLM 2024 — correct; it was not an ICLR paper), [8] RWKV (Findings of EMNLP 2023), [9] RetNet (preprint, honestly not claimed peer-reviewed), [13] Schlag (ICML 2021), [21] PlaNet (ICML 2019), [22] LeCun position paper — titles/IDs/venues consistent with the record. The report's careful "(preprint)" tags on arXiv IDs alongside the real venue are honest, not mislabels.

**2. Claim–source support — spot-checked the riskiest mechanism claims; all faithful.**
- "linearized self-attention *is* a fast-weight programmer … [Schlag] fix it with a delta-rule write" [13] — matches the paper's thesis.
- "chunkwise algorithm … WY representation … recovering parallel training while computing the exact delta recurrence, and outperforming Mamba/GLA at 1.3B scale" [14] — matches the DeltaNet abstract (hardware-efficient WY/Householder-product training scaled to standard LM; the 1.3B comparison is in the paper).
- "Gated DeltaNet adds a gated (decaying) memory so the store can also *erase*, combining Mamba-2's gating with the delta rule" [15] — matches abstract ("gating enables rapid memory erasure while the delta rule facilitates targeted updates").
- Modern-Hopfield "retrieves one in a single update — and its update rule is *equivalent to Transformer attention*" [17] — verbatim-consistent with the paper's stated result.
- minGRU: "stripping the hidden-state dependence out of the gates makes classic RNNs trainable by parallel scan" [10] — matches the paper.
- No distortions found in the S4/Mamba/RWKV/RetNet descriptions.

**3. Fidelity to this project's data — every number matches.** Cross-checked against `/home/user/jepa/README.md` (Recorded-chunk `arch` line + progression table + champion validations) and `evolve/CLAUDE.md`:
- history vs self-only 0.558 / 0.296 (§1) — README:23. ✓
- progression baseline→recency-ALiBi/sysid→hippo 0.5607→path-delta 0.5702→chunkwise-WY champion; "~+0.005–0.01 each, stack" (§2.1, §3) — README:52, README:39–44. ✓
- champion 0.6260 inner / 0.5848 final at 6.95× (§2.3) — README:52; evolve/CLAUDE.md:53. ✓
- minGRU variant beat old champion at 7.17× (§2.2) — README:52. ✓
- R8 RLS/Kaczmarz DeltaProduct + CLS dual-store "beat proxy (+0.006/+0.004), fell below at full" (§2.3) — README:52. ✓
- R9 champion sanity: gen-twin 0.884 vs 0.572, history-ablation 0.836 vs 0.441 (§3) — README:62 (0.8839/0.5715; 0.8362/0.4414). ✓
- "0.5471 → 0.5607 → 0.5702 final margin" (§3.1) — README progression table. ✓
- "proxy→full direction is inconsistent across arch families" (abstract, §3) — README:52. ✓
- code-level claims verified against source: `gh_recency_alibi_perhead.py`, `r6_hippo_episodic_place_read.py`, `r7_path_delta_fastweights_codex.py`, `r9_chunked_delta_fastweights_codex.py`, `r9_pscan_mingru_wy_fastweights.py` all exist; the r9 champion's own docstring confirms "causal Transformer trunk + chunkwise lower-triangular delta-rule memories for command content and path state, gated recency channel" and cites DeltaNet/Gated DeltaNet chunkwise/WY, online-least-squares, and RLS/Kalman exponential forgetting — exactly as §2.3 characterizes. ✓ (Note: the *code's* inline arXiv IDs are inventor-supplied and differ from the report's [14]/[15]; the report does not reproduce those IDs, so no defect in the report.)
- The `(pred, h)` contract, no-future-leakage guard, and head-bypass hazard (§1, §4 Q4) match `evolve/CLAUDE.md:20,30,36`. ✓

**4. Reasoning quality.** No internal contradictions. The report consistently and correctly hedges (proxy→full inconsistency is stated as a caution, mechanism plausibility framed as "hypothesis generator, not a verdict," §3). The "on-manifold read" thesis, the delta-rule-edit-vs-additive-saturation argument, and the chunkwise-exactness point are all mechanically sound and correctly grounded. Aside from M1's phrasing, no hand-waving-as-fact or garbled mechanisms.

---

## Verified-clean note
**Citations checked: 22 / 22 resolved correctly** (all identifiers, authors, years, venues verified; the 9 highest-risk recent/neuroscience papers each independently confirmed via web search). **Data facts: all terminal-JEPA numbers reconciled** against README.md and evolve/CLAUDE.md with zero discrepancies. **Code claims: 5/5 referenced arch impls exist and match their descriptions.** Issues: 0 Critical, 0 Major, 2 Minor (one mechanism-attribution rewording, one trivial venue update).
