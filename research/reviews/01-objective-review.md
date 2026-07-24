# Adversarial Review — `research/01-objective.md` (the "objective" chunk report)

**Bottom-line verdict:** Solid and well-grounded — 0 fabricated/misattributed citations, 0 wrong project numbers; 5 Minor issues (one mislabeled baseline name, some editorial over-attribution/causal slippage). Publishable after small wording fixes.

## Verified-clean note

All **21 references** were checked for existence, title, authors, year, venue, and identifier (arXiv ID / DOI / page range) via WebSearch / Semantic Scholar / publisher pages. **All 21 resolved correctly**, including every falsifiable page range and identifier:

- IDs spot-checked exactly: CPC 1807.03748, SimCLR 2002.05709, MoCo 1911.05722, Wang&Isola 2005.10242, BYOL 2006.07733, Barlow Twins 2103.03230, VICReg 2105.04906, W-MSE **2007.06346** (confirmed), FaceNet 1503.03832, Focal Loss 1708.02002, I-JEPA 2301.08243, CSLS/Word-Translation **1710.04087** (confirmed), Millidge PC review **2107.12979** (confirmed), SwAV 2006.09882, LeCun blueprint OpenReview **BZ5a1r-kVsf** (confirmed).
- Page ranges verified exactly: Wang&Isola PMLR 119:**9929–9939** ✓; LMNN JMLR **10:207–244**, 2009 ✓; Radovanović "Hubs in Space" JMLR **11:2487–2531**, 2010 ✓; Friston NRN **11:127–138**, 2010 (DOI 10.1038/nrn2787) ✓; NCA NIPS 2004:**513–520** ✓; N-pair NIPS 2016:**1849–1857** ✓.
- No preprint was mislabeled as peer-reviewed; venue tags (ICML/CVPR/NeurIPS/ICLR/ICCV/JMLR/NRN) all correct.

Project-data cross-checks against `README.md` and `terminal-jepa/evolve/CLAUDE.md` — **all correct**: MSE 0.306 → L2-InfoNCE 0.459 → focal-listwise/sameverb ~0.461–0.463 → free-energy precision 0.5641; NCA-hinge negative; Sinkhorn 0.34 proxy; fe-mutual-proximity neutral-to-negative; manifold-capacity/local-geometry underperformers. Every cited genome filename exists in `terminal-jepa/evolve/chunks/objective/` (`g1_l2infonce_claude.py`, `g5_l2_nca_hinge_retrieval.py`, `g5_l2_top1_focal_listwise.py`, `r6_sinkhorn_l2_assignment_codex.py`, `r6_free_energy_precision_l2_contrastive.py`, `r6_csls_hub_listwise.py`, `r7_free_energy_mutual_proximity_listwise.py`, `r8_voronoi_geometric_margin.py`). The two mechanistic τ claims were verified against source: `g1_l2infonce_claude.py:53` sets `temperature = float(d)**0.5` (= √768 ≈ 27.7 ✓) and `r6_free_energy_precision_l2_contrastive.py:49` sets `_TEMP = 0.25` on per-dim-mean sqL2 (✓).

## Findings (ranked by severity)

### Critical
None.

### Major
None.

### Minor

**M1 — Margin's third baseline mislabeled "history-free MLP".**
- Location: §1, line 11: fitness is `content_top1(WM) − max(content_top1 of retrieve-by-command, history-free MLP, copy-prev)`.
- Problem: The authoritative fitness formula uses `wm_no_history`, not the "history-free MLP". These are plausibly *distinct* R4 baselines — `README.md:22` lists **both** "history-free MLP 0.21" **and** the masked self-only world model (`wm_no_history`, the "matched-capacity control", content-verb 0.296) as separate objects. The margin subtracts the max over `{retrieve_by_cmd, wm_no_history, copy_prev}`.
- Evidence: `terminal-jepa/evolve/CLAUDE.md:36` — "`margin = content_top1(WM) − max(content_top1 of retrieve_by_cmd, wm_no_history, copy_prev)`".
- Fix: Replace "history-free MLP" with "history-masked WM (`wm_no_history`)" in the §1 fitness definition (or confirm in code they are the same object before keeping the current name).

**M2 — τ=√D framed as an "internal bug"/"accidental" when the source code frames it as deliberate; mild internal tension.**
- Location: §2, line 24: "the first InfoNCE used τ = √D ≈ 27.7 … an accidental single-nearest-negative loss."
- Problem: The genome that produced the program's *largest* objective jump (0.306→0.459) is `g1_l2infonce_claude.py`, whose own comment presents √D as intentional ("Temperature that normalizes for dimensionality so logits are O(1) regardless of d", `g1_l2infonce_claude.py:52`). Calling it "accidental"/"a bug" is the report's inference, not something the record supports. It also creates a mild tension: the same file is credited with the biggest gain **and** with a "zero-entropy hardmax" loss, unexplained.
- Evidence: `terminal-jepa/evolve/chunks/objective/g1_l2infonce_claude.py:52-53`.
- Fix: Soften to "a large τ that pushed the softmax toward near-hardmax; later champions reduced it (τ=0.25 on per-dim-mean distance)" — drop "bug/accidental" unless the ledger records it as a diagnosed defect.

**M3 — Overstated "logits differ by thousands."**
- Location: §2, line 24: "makes logits differ by thousands and the softmax a hardmax with zero entropy."
- Problem: In `g1`, `logits = -dist2 / temperature` with `temperature ≈ 27.7`. Raw sqL2 over standardized 768-d is O(10^3), but *after* dividing by τ the logit spread is tens-to-hundreds, not "thousands." Directionally correct (still a peaky softmax), but the number is off by ~1–2 orders once the temperature it is describing is applied.
- Evidence: `g1_l2infonce_claude.py:44-55` (dist2 is the full sum, then `/temperature`).
- Fix: "logits differ by tens-to-hundreds after the temperature" or drop the magnitude.

**M4 — Causal slippage on why target transforms lost to identity.**
- Location: §6, line 56: "made a learned target-space diagonal transform redundant … which is why every pure target transform lost to identity."
- Problem: The ground truth scopes the free-energy-redundancy explanation to the **learned** transforms (diagonal gate / Givens), which "came within noise of identity" — i.e. they did *not* "lose." The **pure** transforms (partial-residual, delta, Householder-frame, radial companding) "lost to identity" and are not attributed to free-energy redundancy. The report fuses two separate facts into one causal claim.
- Evidence: `README.md:56` — "every pure transform lost to identity … learned target-space transforms … came within noise of identity — plausibly because the free-energy objective already does the per-dim reweighting a diagonal target would add."
- Fix: Split the claim: pure transforms lost; the *learned diagonal* only reached parity, plausibly because the free-energy objective already supplies per-dim reweighting.

**M5 — "LeCun … lists the four anti-collapse regularizer families" is over-crisp attribution.**
- Location: §5, line 50: "[LeCun's blueprint] lists the four anti-collapse regularizer families (contrastive, distillation/asymmetry, variance-covariance, clustering)."
- Problem: LeCun's *Path Towards Autonomous Machine Intelligence* frames collapse-prevention primarily as two broad strategies (contrastive vs. regularized/information-maximization methods); the clean four-way enumeration is closer to how later SSL surveys taxonomize it than a verbatim list in the blueprint. Not wrong in spirit, but presented as a direct enumeration from [10].
- Evidence: OpenReview BZ5a1r-kVsf (the paper discusses contrastive and regularized methods; the "four families" is a common secondary framing).
- Fix: "which surveys group into roughly four anti-collapse families …" or attribute the four-way split to the SSL literature rather than to [10] specifically.

## Spot-checked claims that hold up (no defect)
- §2 anti-collapse mechanism ("constant pred → uniform softmax → InfoNCE pinned at log n") matches `g1` docstring/structure. ✓
- §4 Sinkhorn "underperformed badly (0.34 proxy)" and "metric-mismatch" reasoning consistent with the SwAV/Sinkhorn-Knopp mechanism and the recorded 0.34. ✓
- §3 SimSiam stop-gradient, BYOL EMA-target, Barlow-Twins cross-correlation-to-identity, VICReg variance/covariance/invariance descriptions all match the cited papers. ✓
- §6 hubness (Radovanović) and CSLS (Conneau et al.) mechanism descriptions are accurate to the sources. ✓
- Champion objective labeling: `r6_free_energy_precision_l2_contrastive` at 0.5641 called "the best objective" (not the best *genome*, which is 0.5848 via the R9 arch) — correctly distinguished. ✓

**Citations checked:** 21 / 21 resolved correctly (0 fabricated, 0 misattributed).
**Issues:** Critical 0 · Major 0 · Minor 5.
