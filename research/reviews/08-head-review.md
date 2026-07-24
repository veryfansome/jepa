# Adversarial Review — `research/08-head.md` (the `head` chunk)

## Bottom-line verdict
Substantively sound and faithful to the project's data, with strong code-level fidelity — but it contains **one critical citation misattribution** (ref [18] authors are wrong) and **one major misrepresentation of a cited source's headline number** (UNREAL's 87% is Labyrinth, not Atari), plus a few minor venue/attribution and unverifiable-number issues. Fixable without changing any conclusion.

---

## Findings ranked by severity

### CRITICAL

**C1 — Reference [18] authors are fabricated/misattributed.**
- Location: References, entry 18; cited in text §3 ("[5, 18]") and §6 ("Ni et al.'s recent analysis of self-prediction auxiliaries").
- Quote: *"18. Ni, T., Eysenbach, B., Levine, S., Salakhutdinov, R., et al. (2024). When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning. RLC/arXiv preprint arXiv:2406.17718."*
- Problem: The title, arXiv ID (2406.17718), and venue (RLC 2024) are correct, but the **authors are wrong**. arXiv:2406.17718 is by **Claas A. Voelcker, Tyler Kastner, Igor Gilitschenski, Amir-massoud Farahmand** — not Ni/Eysenbach/Levine/Salakhutdinov. This looks like a conflation with the real but *different* paper by Tianwei Ni, Benjamin Eysenbach et al., "Bridging State and History Representations: Understanding Self-Predictive RL" (arXiv:2401.08898). Right paper + right claim, wrong author list = textbook misattribution.
- Evidence: https://arxiv.org/abs/2406.17718 ; OpenReview https://openreview.net/forum?id=izAJ8sHF5q ; author page https://cvoelcker.de/publications/ ; RLC PDF https://tisl.cs.toronto.edu/publication/202407-rlc-aux_tasks_in_rl/rlc2024-aux_tasks_in_rl.pdf
- Fix: Replace author list with "Voelcker, C. A., Kastner, T., Gilitschenski, I., Farahmand, A." and change in-text "Ni et al." to "Voelcker et al." (The self-prediction-helps-when claim it supports is genuinely made by this paper, so only the attribution needs correcting.)

### MAJOR

**M1 — UNREAL's headline number is attributed to the wrong benchmark, and a baseline figure is unsupported.**
- Location: §3.
- Quote: *"reaches ~87% of human-normalized Atari score versus 54% for A3C and learns ~10× faster [5]."*
- Problem: In the UNREAL paper the **87% expert-human score is on Labyrinth (DeepMind Lab 3D tasks), not Atari.** On **Atari**, UNREAL averaged ~**880%** expert-human performance. So the sentence swaps the benchmark and pairs the Labyrinth number with the word "Atari." The "**54% for A3C**" comparison does not appear in the paper's abstract and could not be verified — it reads as invented or, at best, an unsourced Labyrinth-baseline figure. The "~10× faster" speedup is correct (Labyrinth).
- Evidence: UNREAL abstract — "significantly outperformed previous state-of-the-art on Atari, averaging 880% expert human performance … on Labyrinth … achieving a mean speedup in learning of 10× and averaging 87% expert human performance." https://arxiv.org/abs/1611.05397
- Fix: "reaches ~87% of expert-human score on **Labyrinth** (and ~880% on Atari), learning ~10× faster than A3C." Drop the unverifiable "54% for A3C" unless a source is added.

### MINOR

**m1 — Ref [4] venue claim "(subsequently in IJCAI 2024)" is a likely conflation.**
- Location: References, entry 4.
- Quote: *"arXiv:2212.11491 (subsequently in IJCAI 2024)."*
- Problem: The IJCAI-2024 projection-head paper by this group is a **different paper** — "Deciphering the Projection Head: Representation Evaluation Self-Supervised Learning" (arXiv:2301.12189, IJCAI 2024, pp. 4724–4732) — a distinct title and arXiv ID from 2212.11491 ("Understanding and Improving the Role of Projection Head"). I could not confirm that 2212.11491 itself was published at IJCAI 2024; it appears the two were merged. The preprint 2212.11491 is real and correctly described otherwise.
- Evidence: https://arxiv.org/abs/2212.11491 (the cited preprint) vs https://www.ijcai.org/proceedings/2024/522 and https://arxiv.org/pdf/2301.12189 (the actual IJCAI-2024 paper, different title/ID).
- Fix: Drop the "(subsequently in IJCAI 2024)" tag, or cite the distinct IJCAI-2024 paper separately.

**m2 — The "head absorbs invariances / sacrificial preconditioner" mechanism is loosely pinned to [4].**
- Location: §2.
- Quote: *"The dominant explanation is that the head absorbs the invariances the pretext loss demands … a disposable projection head acts as a sacrificial preconditioner that soaks up the extreme invariance/distortion demand and shields the backbone [4]."*
- Problem: [4] (Gupta et al.) actually frames the mechanism as **implicit data-dependent subspace selection** (the head picks a feature subspace on which to enforce the contrastive loss, addressing *sub-optimal augmentations*), not as "absorbing invariances." The report's *next* sentence states the subspace-selection claim correctly, so [4] is fine there — but the "absorbs invariance" narrative is a distinct line of analysis (closer to guillotine-regularization / Bordes-style arguments the report does not cite) being folded under a single citation.
- Evidence: Paper finding — "the projection head implicitly learns to choose a subspace of features to apply the contrastive loss … data-dependent subspace selection … bilevel optimization." https://arxiv.org/abs/2212.11491
- Fix: Attribute "absorbs invariances/shields backbone" to a source that argues it, or reword to [4]'s own subspace-selection framing.

**m3 — Project-specific figures not present in the supplied ground-truth docs (unverifiable here).**
- Location: §4 ("random-pair ≈ 1430"), §5 ("up to ~237 cd siblings share one history"), §3 ("the champion lineage already tolerates a small MSE anchor without margin damage").
- Problem: README gives random-pair distance as **~1,478** (Stage-2 review) / matched-distance 2020→790 (R10); the report's "≈ 1430" is close but doesn't match either figure exactly. The "~237 cd siblings" count and the "champion tolerates a small MSE anchor" claim do not appear in README or `evolve/CLAUDE.md`. These may be sourced from the `evolve-insights` ledger / `runs/plan/` artifacts not provided for this review, so they are flagged as **unverified**, not wrong.
- Evidence: README lines on R10/Stage-2 ("~1,478 random-pair"; "2020 → 790 sqL2"). No mention of 237 siblings or an MSE-anchor tolerance.
- Fix: Cite the ledger/artifact source for these three numbers, or align "≈ 1430" with the recorded ~1,478.

**m4 — TCN mechanism gloss is the single-view variant only.**
- Location: §5. *"Time-Contrastive Networks learn embeddings where temporal neighbors attract and distant frames repel."*
- Problem: TCN's primary (multi-view) loss *attracts co-occurring viewpoints and **repels** temporal neighbors*; only the single-view variant matches the report's "temporal neighbors attract." Minor imprecision, not a misattribution.
- Evidence: https://arxiv.org/abs/1704.06888
- Fix: Note it's the single-view TCN objective, or state the multi-view formulation.

---

## Verified clean

**Citations checked (16 of 19 web-verified; standard ML canon spot-accepted):**
- **[4] Gupta et al., 2212.11491** — title/authors/preprint correct; core "subspace-selection + bilevel" claim in §2 is accurately represented (venue tag flagged in m1). ✓ (with caveats)
- **[11] Agrawal et al., "Learning to Poke by Poking"** — authors, NeurIPS 2016, arXiv **1606.07419** all correct; §5 forward+inverse-model description accurate. ✓
- **[12] Dayan 1993, SR** — Neural Computation 5(4):613–624; title and definition ("similarity of successors") correct. ✓
- **[13] Barreto et al., Successor Features** — authors, NeurIPS 2017, arXiv **1606.05312** correct; reward-independent SF + reward-weight factorization accurately described. ✓
- **[16] Sermanet et al., TCN** — authors, ICRA 2018, arXiv **1704.06888** correct (see m4 on the gloss). ✓
- **[17] Wang, Torralba, Isola, Zhang, QRL** — authors, ICML 2023 (PMLR v202), arXiv 2304.01203; quasimetric/shortest-path/triangle-inequality description correct; PMLR URL in report resolves. ✓
- **[18] title/ID/venue correct, authors WRONG** — see C1.
- **[19] Lyle, Rowland, Ostrovski, Dabney** — authors, AISTATS 2021, arXiv **2102.13089** correct; "auxiliaries act on representation dynamics" claim accurate. ✓
- **[1] SimCLR, [2] BYOL, [3] SimSiam, [5] UNREAL, [6] PCGrad, [7] Dreamer, [8] MuZero, [9] TD-MPC, [10] ICM, [14] Scheduled Sampling, [15] DAgger** — standard, well-known works; titles/authors/venues/arXiv IDs as listed match established records (UNREAL's *number* misused per M1, but the citation itself is valid). Mechanism descriptions spot-checked and accurate: BYOL predictor+EMA-target anti-collapse ✓; SimSiam predictor+stop-grad, EM framing ✓; MuZero reward/policy/value readouts, no obs reconstruction ✓; Dreamer latent-imagination value-gradient ✓; PCGrad gradient projection onto conflicting-gradient normal plane ✓.

**Project-data fidelity — verified accurate against README / `evolve/CLAUDE.md` / head code:**
- Passthrough head is the strong incumbent baseline ✓ (README L59; `baseline_passthrough.py` returns `wrap→None`, `aux_loss→0.0`).
- R7 head "no gain, below clean-isolation control, isolated on a `pred=head(h)` arch because it bypasses the hippo episodic read" ✓ (README L59; `evolve/CLAUDE.md` head contract; `r7_normres_multihorizon_aux.py` uses `object.__setattr__(self,"base",…)`, k=2,3 multi-horizon aux, normalized-residual readout — all as described).
- R10/R11 auxiliaries: early-budget planning gains **inverted at full budget, no promotion**; "remaining-depth-4 0.011→0.286 at proxy, 0/3 seeds positive at full" ✓ (README Phase-0 section, verbatim).
- "Budget itself was the largest planning lever; champion planning accuracy 0.36→0.60 with budget" ✓ (README).
- **No overclaim detected**: the report nowhere states any auxiliary head was promoted, beat the champion at full budget, or raised the fitness margin — consistent with ground truth.
- All six R10/R11 head files it names exist on disk (`r10_rollout_write_consistency_aux.py`, `r11_invdyn_action_witness_aux.py`, `r11_reachability_horizon_rank_aux.py`, `r11_replay_successor_rollout_codex.py`, `r11_rollout_chain_cosnce_aux.py`).
