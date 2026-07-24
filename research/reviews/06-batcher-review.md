# Adversarial Review — `research/06-batcher.md` (batcher chunk)

## Bottom-line verdict
Substantially clean: all 18 citations exist and are correctly attributed, mechanism descriptions are faithful, and the project-data claims match README/evolve ground truth (including the trap-avoidance: it correctly reports collision-clique and the replay curricula as at-or-below, not winners). Only a handful of minor paraphrase embellishments; no Critical or Major defects found.

## Findings ranked by severity

### Critical
None.

### Major
None.

### Minor

**M1. ANCE "not model capacity" is an added gloss.**
- Location: §2.3, "ANCE [12] states the alignment principle most sharply: the training negatives are not representative of the irrelevant documents seen at test time, and this train/test mismatch — *not model capacity* — is the bottleneck."
- Problem: ANCE's stated bottleneck is "the domination of uninformative negatives sampled in mini-batch training, which yield diminishing gradient norms, large gradient variances, and slow convergence," plus resolving "the discrepancy between the data distribution used in training and testing." The train/test-mismatch framing is well supported; the explicit contrast "not model capacity" is the report's editorial addition, not language ANCE uses.
- Evidence: ANCE abstract/paper, https://arxiv.org/abs/2007.00808 ; https://www.microsoft.com/en-us/research/publication/approximate-nearest-neighbor-negative-contrastive-learning-for-dense-text-retrieval/
- Fix: drop "— not model capacity —" or rephrase to "attributed to uninformative in-batch negatives rather than a stated capacity limit."

**M2. DPR "large lift" from one BM25 negative is mildly overstated.**
- Location: §2.3, "adding *one* BM25 hard negative per question gave a large lift."
- Problem: DPR reports adding a single BM25 hard negative as its best in-batch configuration and a meaningful improvement, but the top-k accuracy gain is a few points, not obviously "large." Characterization is defensible but leans generous.
- Evidence: DPR, https://aclanthology.org/2020.emnlp-main.550/
- Fix: soften to "gave a consistent improvement" or cite the actual top-k delta.

**M3. Lu/Hasson/Norman characterization is specific and paraphrased (but internally consistent).**
- Location: §2.5 / ref [18], "show a memory-augmented network learns *when* to retrieve only when the training distribution actually contains recurring items."
- Problem: the paper's headline finding is that the network learns to retrieve *selectively as a function of uncertainty* (and that encoding at event boundaries aids prediction); the "only when the training distribution contains recurring items" conditional is a reasonable but tighter gloss than the paper states. Note this exact phrasing is lifted from the project's own `r8_loop_closure_memory_curriculum.py` docstring, so it is at least consistent with how the code authors read the paper.
- Evidence: eLife 11:e74445, https://elifesciences.org/articles/74445 ; project file `terminal-jepa/evolve/chunks/batcher/r8_loop_closure_memory_curriculum.py`
- Fix: attribute more loosely ("retrieval is useful only when past situations reoccur, motivating…") or note it as motivation rather than a demonstrated result.

## Fidelity to project data — checked, no misstatements

Cross-checked against `README.md` line 54 and `terminal-jepa/evolve/CLAUDE.md` line 53, plus the actual chunk code:

- "Uniform → system-blocked hard-negative curriculum beat uniform; harder blocking trended up; single-image blocking (block1) is in the champion" — matches README exactly. Champion genome `r6_sysblock_hardneg_curriculum` (block1/hard75/ramp30) confirmed in both README and evolve/CLAUDE.md.
- Collision-clique reported as *at or below* image-blocking (not a winner) — correct. The report describes it as keyed on `(verb, image, subtree-prefix)`; this is more precise than README's shorthand "(verb,system)" and is verified against `r7_collision_clique_curriculum.py` line 113 (`(v, img, _prefix_key(...))`). No overclaim.
- Both R8 memory-exercising curricula (loop-closure, SWIL path-replay) reported as at-or-below — correct. Descriptions ("oversample loop-closure-rich sequences, tilt annealed to zero"; "similarity-weighted route-neighbor packets") match the docstrings of `r8_loop_closure_memory_curriculum.py` and `r8_swil_path_replay_curriculum_codex.py`. (Report drops the `_codex` filename suffix; the file's internal `NAME` is `r8_swil_path_replay_curriculum`, so this is not an error.)
- Eval description ("63 same-verb foils from the two-image inner-val pool," inner-val = fedora+mariadb, fitness = content-verb margin) matches README R4 section and evolve/CLAUDE.md fitness definition.
- The report makes **no fabricated numeric batcher margins** — consistent with README, which also reports these only qualitatively. Good discipline.

## Verified clean — citations checked (18/18 resolved)

All references verified via web search for existence, authorship, venue, and year; all correct:

1. van den Oord/Li/Vinyals, CPC (arXiv:1807.03748, 2018) — verified; InfoNCE MI lower-bound + log-N ceiling claim accurate.
2. Poole/Ozair/van den Oord/Alemi/Tucker, Variational Bounds of MI (ICML 2019, PMLR 97:5171) — verified; "bounds degrade (high bias/variance) when MI is large" accurate.
3. Chen/Kornblith/Norouzi/Hinton, SimCLR (ICML 2020) — verified; large-batch/long-training claim accurate.
4. He/Fan/Wu/Xie/Girshick, MoCo (CVPR 2020, arXiv:1911.05722) — verified; queue decouples dictionary from batch, momentum encoder — accurate.
5. Wang/Isola, Alignment & Uniformity (ICML 2020) — verified; negatives drive uniformity — accurate.
6. Schroff/Kalenichenko/Philbin, FaceNet (CVPR 2015) — verified; hardest-negative → collapse, use semi-hard — accurate (paper notes hardest negatives can yield a collapsed model f(x)=0).
7. Hermans/Beyer/Leibe, In Defense of the Triplet Loss (arXiv:1703.07737, 2017) — verified; batch-hard (P×K, hardest pos/neg in-batch) — accurate.
8. Robinson/Chuang/Sra/Jegelka, Hard Negative Samples (ICLR 2021) — verified; controllable-hardness unsupervised sampling — accurate.
9. Kalantidis/Sariyildiz/Pion/Weinzaepfel/Larlus, Hard Negative Mixing / MoCHi (NeurIPS 2020) — verified; feature-space synthesis of hard negatives — accurate (MoCHi is the paper's own method acronym).
10. Chuang/Robinson/Lin/Torralba/Jegelka, Debiased Contrastive Learning (NeurIPS 2020) — verified; false-negative correction to true-negative marginal — accurate.
11. Karpukhin et al., DPR (EMNLP 2020) — verified; in-batch + BM25 hard negatives — accurate (see M2 on "large").
12. Xiong et al., ANCE (ICLR 2021) — verified; async ANN-mined global hard negatives, train/test alignment — accurate (see M1 gloss).
13. Qu et al., RocketQA (NAACL 2021) — verified; cross-batch negatives + denoised hard negatives via cross-encoder — accurate.
14. Bengio/Louradour/Collobert/Weston, Curriculum Learning (ICML 2009) — verified; continuation-method framing — accurate.
15. Kumar/Packer/Koller, Self-Paced Learning (NIPS 2010) — verified; select by current loss — accurate.
16. Schaul/Quan/Antonoglou/Silver, Prioritized Experience Replay (ICLR 2016) — verified; TD-error priority — accurate.
17. Saxena/Shobe/McNaughton, SWIL (PNAS 119(27):e2115229119, 2022) — verified; similarity-weighted interleaving vs catastrophic interference — accurate.
18. Lu/Hasson/Norman, when-to-retrieve episodic memory (eLife 11:e74445, 2022) — verified (see M3 on characterization tightness).
