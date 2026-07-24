# Adversarial Review — `research/03-optim.md` (optim chunk)

## Bottom-line verdict
Citations are clean and the central Muon-win story is faithful to the code; the one substantive defect is a repeated, self-contradictory misdescription of the incumbent schedule's hyperparameters (β₂ and weight-decay split) — fixable without touching the thesis.

---

## Findings ranked by severity

### MAJOR

**M1 — The champion's inherited schedule is mis-specified (wrong β₂, wrong weight-decay rule), and the report contradicts itself about it.**

- Quotes:
  - §2: *"the project's tuned schedule uses β₂=0.98 (≈50-step memory) rather than 0.999 … (`go_warmup_cosine_floor_beta98_decoupled.py`)."*
  - §4 headline bullet: *"warmup → hold → cosine-to-floor AdamW (β₂ tuned, decoupled ndim-split weight decay) was the strongest hand-built optimizer and remains the schedule backbone every evolved impl inherits."*
  - §2: *"the best hand-designed one goes further with the standard no-decay-on-norm/bias rule … improves generalization for adaptive methods."*
- Problem: The schedule the champion actually inherits is **`go_warmup_holdcos_floor.py`** (the Muon file rebuilds it verbatim: `_incumbent_lambda` is *"Exactly go_warmup_holdcos_floor's multiplier"*, and the AdamW group is plain `AdamW(rest, …)`). That incumbent uses **β₂ = 0.95** and **plain, uniform weight decay — no ndim split**. The β₂ = 0.98 *and* the decoupled ndim-split ("no-decay-on-norm/bias") both belong to a **different** file, `go_warmup_cosine_floor_beta98_decoupled.py`, which has **no hold phase** and is *not* what any evolved impl (Muon / SWA / homeostatic) inherits. So "warmup→hold→cosine-floor" and "β₂=0.98 + ndim-split decoupled decay" are fused into one optimizer that does not exist.
- Self-contradiction: §4 bullet 1 states *"The decoupled-by-ndim variant actually scored **lower** than plain AdamW … (0.3575 vs 0.4014)"* — i.e. the ndim-split was measured worse and **not adopted** — which flatly contradicts the §4 headline claim that the "strongest hand-built optimizer" and "schedule backbone" used "decoupled ndim-split weight decay," and undercuts §2's "goes further … improves generalization."
- Evidence:
  - `terminal-jepa/evolve/chunks/optim/go_warmup_holdcos_floor.py:26-32` — `AdamW(params, lr=5e-4, wd=5e-4, betas=(0.9,0.95))`, single group, no split.
  - `terminal-jepa/evolve/chunks/optim/r8_muon_key_orthogonal_addressing.py:172-218` — reuses that exact schedule; `make(..., beta2=0.95)` default (line 193); AdamW group is `AdamW(rest, weight_decay=wd, betas=(0.9,0.95))` with no ndim split (line 214).
  - `terminal-jepa/evolve/chunks/optim/r8_fastweight_homeostatic_consolidation_codex.py:23` — independently calls the incumbent *"AdamW(lr 5e-4, wd 5e-4, beta2 .95)"*.
  - Champion genome (`evolve/archive/genomes.jsonl`, `r9-arch-chunked-codex` line): `"optim": {"impl": "r8_muon_key_orthogonal_addressing", "bs": 64}` — **no `beta2` override**, so β₂ = 0.95 is what runs.
  - `go_warmup_cosine_floor_beta98_decoupled.py:28-32` — the *only* file with β₂=0.98 + ndim split; it has 6% warmup, cosine to 0.1 floor, **no hold**.
- Fix: In §2 and §4, describe the actual incumbent as β₂ = 0.95 with **plain** (uniform) decoupled AdamW weight decay. Keep β₂=0.98 and the ndim-split as properties of the *separate, non-incumbent* `beta98_decoupled` candidate, and note (consistent with §4 bullet 1) that the ndim split scored lower and was **not** carried into the champion lineage. Remove the "β₂ tuned, decoupled ndim-split weight decay" clause from the "schedule backbone every evolved impl inherits" sentence.

---

### MINOR

**m2 — "runs stably in low precision" applied to the champion's Newton–Schulz.**
- Quote (§3.4): *"the polar factor, computed by a quintic Newton–Schulz iteration that runs stably in low precision."*
- Problem: True of Muon *in general* (the blog runs it in bf16), but the champion impl explicitly computes in **float32** (`_ns_orth`: `x = g.float()`, `r8_muon_key_orthogonal_addressing.py:80-93`). The clause reads as if describing this code.
- Fix: Attribute the low-precision property to Muon-the-method, or note this impl runs the iteration in float32.

**m3 — Homeostatic optimizer "motivated by [13][14][16]" over-attributes EWC.**
- Quote (§4): *"the homeostatic-consolidation optimizer (`r8_fastweight_homeostatic_consolidation_codex`, which imposed synaptic-scaling row homeostasis + Oja orthogonalization + tail-EMA on the key matrices, motivated by [13][14][16])."*
- Problem: The mechanism description is accurate (verified in the file), but the file's own grounding cites synaptic-scaling papers (arXiv:1304.2266, arXiv:1709.05633), Oja's rule, and weight normalization (=[16]) — **not** EWC ([13] Kirkpatrick). [13] is a real, correctly-described paper elsewhere, but "motivated by [13]" for this specific optimizer is a mild stretch.
- Evidence: `r8_fastweight_homeostatic_consolidation_codex.py:9-20`.
- Fix: Drop [13] from this optimizer's motivation list, or soften to "in the spirit of synaptic-consolidation work [13]."

**m4 — `InfoNCE 0.39→0.47` presented as ledger evidence but sourced from a design note.**
- Quote (§1): *"The recorded proxy→full improvement (e.g. InfoNCE 0.39→0.47) signals the regime is under-optimized."*
- Problem: This number is not from the neutral per-genome ledger; it is copied from a docstring design note in the **non-incumbent** `go_warmup_cosine_floor_beta98_decoupled.py`. Not fabricated, but presented with more evidential weight than a code comment warrants.
- Evidence: `go_warmup_cosine_floor_beta98_decoupled.py:7-8`.
- Fix: Mark it as an illustrative design-note figure, not a ledger stat.

---

## Verified-clean note

**All 16 references verified** (title / authors / year / venue / ID), none fabricated or misattributed:
[1] Adam 1412.6980 (ICLR 2015) ✓ · [2] AdamW/Decoupled WD 1711.05101 (ICLR 2019) ✓ · [3] SGDR 1608.03983 (ICLR 2017) ✓ · [4] Goyal 1-Hour ImageNet 1706.02677 ✓ · [5] LARS 1708.03888 ✓ · [6] RAdam 1908.03265 (ICLR 2020) ✓ · [7] SWA 1803.05407 (UAI 2018) ✓ · [8] **Muon blog, Keller Jordan, 2024** — correctly labeled a *blog writeup* and **not** dressed up as a paper ✓ · [9] **Muon is Scalable for LLM Training, Liu et al. (Moonshot AI), 2502.16982, 2025** — correctly attributed, correctly split from [8] ✓ · [10] Shampoo 1802.09568 (ICML 2018) ✓ · [11] K-FAC 1503.05671 (ICML 2015) ✓ · [12] Schlag/Irie/Schmidhuber Fast Weight Programmers 2102.11174 (ICML 2021) ✓ · [13] EWC PNAS 114(13):3521–3526, doi:10.1073/pnas.1611835114 ✓ · [14] Turrigiano & Nelson, Nat Rev Neurosci 5:97–107 (2004), doi:10.1038/nrn1327 ✓ · [15] CPC/InfoNCE 1807.03748 ✓ · [16] Weight Normalization 1602.07868 (NeurIPS 2016) ✓.

**Mechanism claims spot-checked and supported:** Muon = Newton–Schulz → nearest semi-orthogonal / polar factor of the momentum, flattening the update's singular-value spectrum to equal magnitude (verified vs blog + `_ns_orth` quintic coeffs 3.4445/−4.7750/2.0315); RMS-match factor `0.2·√(max(n,m))` (verified vs code line 129 and arXiv:2502.16982's weight-decay + per-parameter update-scale fixes); decoupled-weight-decay rationale (penalty gradient escaping Adam's 1/√v rescaling); RAdam warmup = early-variance reduction; SWA constant-LR spread → flatter-minimum average; LARS per-layer LR = ‖w‖/‖g‖; K-FAC = Kronecker-factored inverse-Fisher; Shampoo = per-dimension preconditioner — all accurate.

**Project-data fidelity confirmed:** first evolved-optimizer win = Muon scoped to exactly the four `(key_d=64, d=176)` addressing matrices (`content/path_read/write`) with the incumbent schedule elsewhere; strict-generalization fallback returns the exact incumbent when no key matrices are present; landed in the fast-weights context after the hippo-targeted R7 optimizer (`r7_cls_twotimescale_trust_memory`) failed; SWA/tail-averaging underperformed in both contexts; `0.3575 vs 0.4014` ndim-split comparison — all corroborated in code and the champion genome. The champion arch description (delta-rule associative memory, unit keys, two memories) matches the current R9 chunked-delta champion, not merely R7.
