# Adversarial Review — `05-perception.md`

**Bottom-line verdict:** Substantially sound — all 19 citations exist and are correctly attributed (title/authors/year/ID checked), and the project-data claims are faithful to the ledger, *including* the subtle e5-large baseline-lift mechanism the report gets right; the only substantive defect is an unverifiable/likely-misattributed "cosine near 0.99" figure pinned on Li et al. (2020), plus a few minor overstatements and one imprecise mechanism description.

---

## Findings ranked by severity

### Critical
None found. The two highest-risk technical claims — the differential-margin fitness definition and the **e5-large "lost margin because it lifted the baselines, not because the WM got worse"** mechanism — are both stated correctly and match ground truth (see Verified, below).

### Major

**M1. "cosine near 0.99" attributed to Li et al. (2020) — unverifiable, likely a misattributed number.**
- Location: §2.1, "Li et al. (2020) showed this directly harms sentence similarity: **mean-pooled BERT sentence embeddings have pairwise cosine near 0.99**, and cosine loses discriminative power in such a space [9]." Echoed in the abstract's framing.
- Problem: I could not confirm the figure **0.99** appears in the BERT-flow paper (Li et al., 2020, "On the Sentence Embeddings from Pre-trained Language Models"). The paper's own contribution is qualitative anisotropy + the word-frequency/narrow-cone analysis; it does not headline a 0.99 sentence-cosine number. The famous ~0.99 average-cosine figure is Ethayarajh (2019) for **GPT-2's upper layers between random words** — a different model and a different measurement than "mean-pooled BERT sentence embeddings." Independent anisotropy literature bounds *unrelated-sentence* BERT cosine at roughly ~0.9, not 0.99.
- Evidence: WebSearch of the BERT-flow paper (https://aclanthology.org/2020.emnlp-main.733/ ; arXiv:2011.05864) surfaces the anisotropy/narrow-cone claim but no 0.99 value; a targeted search returned "average cosine similarity for all pairs of unrelated sentence embeddings is upper bounded by approximately 0.9" from the surrounding literature, and the 0.99 figure tracks to Ethayarajh's GPT-2 result (https://aclanthology.org/D19-1006/). Direct PDF fetches of the BERT-flow paper 403'd here, so this is "could not verify," not "confirmed false" — but per the skeptical default it should be flagged.
- Fix: Either quote the exact number from Li et al. with a page/section, or drop the "0.99" and attribute the high-cosine anisotropy figure to Ethayarajh (2019) [18] (GPT-2), keeping Li et al. [9] for the qualitative "anisotropy harms cosine similarity" claim it actually makes.

### Minor

**M2. BGE "helped but less than e5" overstates the ground truth.**
- Location: §4, "BGE (also retrieval-tuned, CLS-pooled) **helped** but less than e5."
- Problem: The project ledger only records **`bge < e5`** (README:53; evolve/CLAUDE.md:53). It does *not* establish that BGE scored **above** the ModernBERT baseline (i.e., that it "helped"). "Helped" is an inference the recorded stat doesn't support; BGE could sit below baseline for all the provided data shows.
- Evidence: `/home/user/jepa/README.md:53` ("bge < e5"); `/home/user/jepa/terminal-jepa/evolve/CLAUDE.md:53`. No "bge > ModernBERT" statement exists in either.
- Fix: Say "BGE < e5 (also retrieval-tuned, CLS-pooled)" and drop "helped," or cite the specific BGE margin if it exists in `evolve-insights`.

**M3. CodeBERT training objective is described imprecisely.**
- Location: §2.6, "CodeBERT is trained for code *understanding* (**replaced-token detection, NL-PL alignment**)".
- Problem: CodeBERT (Feng et al., 2020) is trained with **MLM + Replaced Token Detection (RTD)**. "NL-PL alignment" is a *property* of its bimodal data, not a named training objective. Minor, but the parenthetical reads as a list of objectives.
- Evidence: CodeBERT paper (arXiv:2002.08155) — objectives are MLM and RTD.
- Fix: "(masked-LM + replaced-token detection over paired NL–PL data)".

**M4. Mechanistic rationale for mean-pooling over-attributed to [1].**
- Location: §2.5, "**Mean-pooling** tends to win … because their training signal is distributed across the sequence and mean-pooling is the aggregation those objectives implicitly optimized [1]."
- Problem: SBERT [1] is cited, but SBERT only *empirically* found mean best among mean/CLS/max; the "implicitly optimized" causal story is the author's reasoning, not [1]'s claim. (The cleaner, correct reason — mean is E5/GTE/Contriever's *training-time* readout — is given elsewhere in the same section, so this is a redundant over-reach rather than a wrong conclusion.)
- Fix: Attribute the empirical result to [1] and present the "implicitly optimized" line as inference, or lean on the training-time-readout argument already made.

**M5. Section-5 guidance generalizes an n=1-domain result to a rule.**
- Location: §5 Guidance #1, "**Prefer retrieval-tuned encoders … over general or code-specialized ones.**"
- Problem: The transfer of "retrieval-tuning is the helpful property" is established on exactly one domain (this shell corpus, this eval). As *design guidance for this project's `perception` chunk* it is fine; phrased as a bare imperative it reads more universal than the evidence. The report is otherwise careful ("confirmed here"), so this is borderline.
- Fix: Scope it — "In this project's retrieval eval, prefer …".

---

## Verified clean

**Citations checked: 19 / 19 exist with correct title, authors, year, and identifier.**

| # | Ref | Verified |
|---|---|---|
| 1 | Sentence-BERT, Reimers & Gurevych 2019 | ✓ D19-1410 / arXiv:1908.10084; "65 h → ~5 s" and mean-pooling-best both confirmed |
| 2 | E5, Wang et al. 2022 | ✓ arXiv:2212.03533; weakly-supervised contrastive on CCPairs, first to beat BM25 zero-shot on BEIR, MTEB "40× more params" all confirmed |
| 3 | MTEB, Muennighoff et al. 2022 | ✓ arXiv:2210.07316; "field has yet to converge on a universal method" confirmed |
| 4 | BGE / C-Pack, Xiao et al. 2023 | ✓ arXiv:2309.07597; authors Xiao/Liu/Zhang/Muennighoff confirmed |
| 5 | Contriever, Izacard et al. 2021 | ✓ arXiv:2112.09118; "beats BM25 on most BEIR" (11/15) confirmed |
| 6 | GTE, Li et al. 2023 | ✓ arXiv:2308.03281; "GTE-base beats 10× larger models" confirmed |
| 7 | Instructor, Su et al. 2022 | ✓ arXiv:2212.09741; instruction-conditioned embedding confirmed |
| 8 | ModernBERT, Warner et al. 2024 | ✓ arXiv:2412.13663; RoPE/GeGLU/8192-ctx/2T-tokens/code confirmed |
| 9 | BERT-flow, Li et al. 2020 | ✓ ACL 2020.emnlp-main.733 / arXiv:2011.05864; normalizing-flow-to-Gaussian confirmed (but see M1 for the 0.99 figure) |
| 10 | Whitening, Su et al. 2021 | ✓ arXiv:2103.15316; authors Su/Cao/Liu/Ou confirmed |
| 11 | CodeBERT, Feng et al. 2020 | ✓ arXiv:2002.08155 (objective wording — see M3) |
| 12 | DPR, Karpukhin et al. 2020 | ✓ arXiv:2004.04906; dual-encoder/inner-product framing accurate |
| 13 | SimCSE, Gao et al. 2021 | ✓ arXiv:2104.08821; dropout-aug contrastive + anisotropy-flattening accurate |
| 14 | I-JEPA, Assran et al. 2023 | ✓ arXiv:2301.08243; predict-in-representation-space accurate |
| 15 | R3M, Nair et al. 2022 | ✓ arXiv:2203.12601; "frozen perception module" accurate |
| 16 | Repr. Degeneration, Gao et al. 2019 | ✓ arXiv:1907.12009; likelihood + weight-tying → narrow cone confirmed |
| 17 | LeCun 2022, "A Path Towards…" | ✓ OpenReview BZ5a1r-kVsf |
| 18 | Ethayarajh 2019 | ✓ D19-1006 / arXiv:1909.00512; anisotropy in all layers confirmed |
| 19 | Parisi et al. 2022 | ✓ arXiv:2203.03580; frozen features competitive-with/better-than ground-truth state confirmed |

**Project-data fidelity — verified against code + README + evolve/CLAUDE.md (all accurate):**
- `margin = content_top1(WM) − max(retrieve_by_cmd, wm_no_history, copy_prev)` on ls+cat — matches evolve/CLAUDE.md:36. ✓
- e5-base-v2 = **+0.043** full-final over ModernBERT, first genome >0.50, top-1 ~0.72→~0.80 — matches README:38,53. ✓
- **CodeBERT < ModernBERT** — matches README:53. ✓
- **e5-large lost margin via baseline-lift (WM top-1 comparable, baselines rose)** — the flagged high-risk mechanism is stated correctly; matches README:53 ("lifted the objective-independent baselines as much as the WM… differential help, not absolute retrievability"). ✓
- e5-large = e5-large-v2, 1024-d → 768 via **fixed orthonormal (QR Q-factor) projection** — matches `enc_e5_large_proj768.py:12-16`. ✓
- Asymmetric `query:`/`passage:` ≈ tied; champion uses `passage:` on both sides — matches `enc_e5_base.py:10-12` (both `passage:`) and `r6_enc_e5_asym_query_passage.py:29` (`query:` on cmd). ✓
- `OBS_CAP = 1600` — matches `baseline.py:8`. ✓
- e5 tokenizer head-truncates at **256 tokens** — matches `reencode.py:52` (`max_length=256`). ✓
- KMV `ls` render = **−0.0073, WM itself dropped** (not baseline-lift), ~29% cat / ~15% ls truncated — matches README:58 and `r8_kmv_sketch_render.py:7,46-48`. ✓
- e5_multivec K=4 — matches `e5_multivec.py:16`. ✓
- bge pools CLS — matches `enc_bge_base.py:5-6`. ✓

**Issue counts:** Critical 0 · Major 1 · Minor 4.
