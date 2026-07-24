# Adversarial Review — `research/04-target.md` ("Designing the Prediction Target")

**Bottom-line verdict:** Substantially sound. All 18 citations exist and are correctly attributed (including the very recent LeJEPA and the two risky numeric claims), and the report faithfully represents this project's `target`-chunk results (no claim that any transform beat identity; honesty mechanism described correctly). One code-verified **false universal claim** about the harness (`to_obs(zeros)==zeros for every impl`) and a few minor omissions/hand-waves are the only defects. Publishable after fixing the Major item.

---

## Findings by severity

### Critical
None.

### Major

**M1. False universal claim: `to_obs(zeros) == zeros for every impl`.**
- **Location / quote (§1, line 13):** "Because `to_obs(zeros) == zeros` for every impl, the predict-mean calibration guard is unaffected."
- **Problem:** This is false for the *pure residual* target family that the report itself discusses in §5. `to_obs` for those impls re-injects `z_prev`, so `to_obs(zeros, z_prev) = z_prev ≠ 0` (except trivially at step 0 where `z_prev=0`).
- **Evidence (code-verified):**
  - `terminal-jepa/evolve/chunks/target/delta_prev.py:15-16` — `def to_obs(pred, z_prev): return z_prev + pred` → `to_obs(zeros) = z_prev`.
  - `terminal-jepa/evolve/chunks/target/r6_householder_prev_residual_codex.py:44-48` — `to_obs(zeros, z_prev) = H(‖z_prev‖·e0) = z_prev` (H reflects `z_prev` onto the axis and back), ≠ 0.
  - The invariant *does* hold for `identity.py`, `tgt_space_diag_gate.py` (`pred/g`), and the orthogonal Givens map — i.e. exactly the LEARNED/identity impls, not "every impl."
- **Impact:** Peripheral (a supporting parenthetical, not a headline result), but it is a concrete, verifiable misstatement about the project's own honesty/calibration machinery — precisely the class of claim this review is charged to catch.
- **Fix:** Scope the claim to the impls it holds for, e.g. "Because `to_obs(zeros)==zeros` for identity and the norm-preserving learned transforms (the residual impls instead map `0 ↦ z_prev`), …" — or drop the sentence.

### Minor

**m1. Gen-twin result stated without the project's documented caveat.**
- **Quote (§2, line 17):** "the bet is identical and already validated at the substrate level: latent next-observation retrieval beats a compute-matched generative (token-bag reconstruction) twin."
- **Problem:** `README.md:24` reports this result explicitly "*with a caveat* — the two live in different geometries (768-d MSE vs 4000-d BCE), so this is **suggestive, not a clean apples-to-apples** objective comparison." The report upgrades "suggestive" to "validated" and omits the geometry caveat.
- **Evidence:** `/home/user/jepa/README.md:24`.
- **Fix:** Soften to "suggestive (the two objectives live in different geometries)," mirroring the README.

**m2. "Commute and are redundant" asserted as mechanism without derivation.**
- **Quote (§6, line 53):** "a diagonal target gate is the *same kind of operator* as the free-energy objective's diagonal precision weighting `Π_d = 1/MSE_d` in the fixed basis; the two commute and are redundant."
- **Problem:** Stated as established fact. A learnable diagonal gate `g_d` and a *fixed* precision `Π_d` compose multiplicatively into `Π_d·g_d²` and are not simply "redundant"; the redundancy argument only goes through if `Π_d` is re-estimated adaptively from residuals (which absorbs the gate). The report's own §7 correctly downgrades this to "most likely explanation" / "plausibly," so §6 overstates relative to §7. (This mirrors the `r7_learned_givens_rotation_decorrelate.py` docstring's framing, so it is faithful to the impl — but the impl's claim is itself a hypothesis, not a proof.)
- **Fix:** Hedge §6 to match §7 ("would be redundant to the extent the objective's precision is estimated adaptively").

**m3. Internal number inconsistency for the norm ratio.**
- **Quote:** §6 line 51 "`‖pred‖²/‖true‖² ≈ 4.7`"; §7 line 60 "toy `k` from 4.78 → ~1.46." `README.md:70` gives "≈ 4.7×."
- **Problem:** Cosmetic 4.7 vs 4.78 drift within the same document; harmless but avoidable.
- **Fix:** Use one figure consistently.

**m4. EMA momentum stated as a single value for two papers.**
- **Quote (§3, line 23):** "the target encoder is an EMA copy of the context encoder (momentum ≈ 0.996) … [3, 4]."
- **Problem:** Both I-JEPA and V-JEPA use a momentum *schedule* (≈0.996 → 1.0), not a fixed 0.996; presenting a single value for both is a mild imprecision. Not wrong at the low end.
- **Fix:** "momentum starting ≈ 0.996, ramped toward 1.0."

---

## Verified clean

**Citations checked: 18/18 exist and are correctly attributed. 0 fabricated, 0 misattributed.** Eight were verified in depth via web search (all task-flagged risk items), including the two specific numeric claims and the very recent LeJEPA:

- **[18] LeJEPA (arXiv:2511.08544)** — REAL. Balestriero & LeCun, Nov 2025; SIGReg / isotropic-Gaussian embedding target. Matches the report exactly.
- **[15] Kalapos & Gyires-Tóth, "Whitening Consistently Improves SSL" (arXiv:2408.07519)** — REAL; ZCA-as-last-layer; "**1–5%** linear/kNN improvement" claim (§4 line 35) verified verbatim against the abstract.
- **[3] I-JEPA (arXiv:2301.08243, CVPR 2023)** — "**roughly 5× fewer iterations** than MAE-class pixel methods" (§2) verified ("converges in 5× fewer iterations," ~7% slower/iter).
- **[4] V-JEPA (arXiv:2404.08471)** — "latent beats pixel by **~+5 points** on frozen K400" (§2) verified: the paper states latent prediction "outperforms by 5+ points" under frozen attentive probing. (Exact per-cell K400 numbers not extractable — ar5iv 403'd — but the paper's own summary corroborates the magnitude.)
- **[6] BYOL (arXiv:2006.07733, NeurIPS 2020)** — "**74.3%** ImageNet linear top-1, no negatives" (§3) verified exactly.
- **[2] Dawid & LeCun, LVEBM tutorial (arXiv:2306.02572)** — REAL; also J. Stat. Mech. 2024, DOI 10.1088/1742-5468/ad292b — both identifiers correct.
- **[1] LeCun, "A Path Towards Autonomous Machine Intelligence" (OpenReview BZ5a1r-kVsf, v0.9.2, 2022)** — REAL; ID and version correct. (The quoted phrase in §2 is a faithful paraphrase of the position paper's argument; exact wording not independently re-verified but plausible.)
- **[10] Tian, Chen & Ganguli, DirectPred (arXiv:2102.06810, ICML 2021)** — REAL; authors/venue correct; "eigenspectrum tracks input correlation structure" is an accurate paraphrase of the gradient-free predictor result.

The remaining ten are canonical papers whose authors/venue/year/arXiv-ID as printed match the established record: [5] MoCo 1911.05722, [7] DINO 2104.14294, [8] DINOv2 2304.07193, [9] SimSiam 2011.10566, [11] data2vec 2202.03555, [12] Barlow Twins 2103.03230, [13] VICReg 2105.04906, [14] W-MSE 2007.06346, [16] Flow Matching 2210.02747, [17] Progressive Distillation / v-parameterization 2202.00512. The v-parameterization formula `v = α_t ε − σ_t x` (§5) is stated correctly.

**Fidelity to project data — verified accurate:**
- "Every *pure* transform lost to identity (partial-residual, delta, Householder, radial companding)" — matches `README.md:56` / `evolve/CLAUDE.md`.
- "Learned diagonal/Givens transforms tied identity within noise; did not beat it" — matches ground truth; **the report nowhere claims any target transform beat identity.**
- Diagonal-gate "−0.002 (within noise)" — corroborated by the `r7_learned_givens_rotation_decorrelate.py:10` docstring ("the prior tgt_space_diag_gate, -0.002"); not fabricated.
- Impl names all exist: `delta_prev`, `r6_householder_prev_residual_codex`, `r10_radial_power_compand_sched`, `r7_learned_givens_rotation_decorrelate`, `tgt_space_diag_gate`, `r6_free_energy_precision_l2_contrastive`.
- Mechanism descriptions (§6) match code exactly: diag gate `g=D·softmax(θ)`, `make_target=g·z_obs`, `to_obs=pred/g`, KL stabilizer (`tgt_space_diag_gate.py`); Givens `Q@z_obs` / `Qᵀ@pred`, exactly orthogonal, identity-at-init, ZCA framing (`r7_learned_givens_rotation_decorrelate.py`).
- **Exact-inverse honesty mechanism** (§1, §6) — described correctly and consistently with `evolve/CLAUDE.md:29` ("evaluated via the module's exact inverse in the fixed obs space — so a collapsed learned target can't reconstruct and is scored down").
- "Free-energy objective already does the per-dim reweighting a diagonal target would add" (§7) — matches `README.md:56`'s "plausibly," and the report preserves the hedge.
- Note in the report's favor: the report correctly separates W-MSE (Ermolov, [14]) from the Kalapos whitening paper ([15]) — the underlying impl docstring actually conflates the two ("Ermolov et al.; 'Whitening Consistently Improves SSL' arXiv:2408.07519"); the report is *more* accurate than the code it summarizes.
