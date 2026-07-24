# Research reports — chunk design theory

Per-chunk literature reviews for the eight evolvable **chunks** of the terminal-JEPA
sequence world model (see the root [`README.md`](../README.md) for the project and
[`terminal-jepa/evolve/CLAUDE.md`](../terminal-jepa/evolve/CLAUDE.md) for the search).
Each report surveys **the main theories for how to design that component**, grounds
them in the literature, and ties the theory to what this project's evolutionary search
actually measured for that axis.

One report per chunk, eight in all — ~23.5k words and **147 references** total.

| # | chunk | report | refs | what it designs |
|---|---|---|---|---|
| 01 | objective | [The Training Objective in a JEPA-Style Latent-Prediction World Model](01-objective.md) | 21 | the loss: contrastive/InfoNCE, variance–covariance anti-collapse, metric/ranking losses, free-energy precision |
| 02 | arch | [Architectural Theories for a Causal Sequence World Model](02-arch.md) | 22 | the network: transformers, SSM/linear-attention, fast-weights/delta-rule, Hopfield & hippocampal memory |
| 03 | optim | [Optimizers and Learning-Rate Schedules for a Small, Contrastively-Trained Sequence World Model](03-optim.md) | 16 | optimizer + schedule: AdamW, warmup/cosine/SWA, Muon/Shampoo structured updates |
| 04 | target | [Designing the Prediction Target in a JEPA-Style Latent World Model](04-target.md) | 18 | the target space: EMA/stop-grad targets, latent-vs-pixel, whitening, residual/rotation transforms |
| 05 | perception | [Choosing the Eyes of a Latent World Model: the Frozen Text-Embedding Front-End](05-perception.md) | 19 | the frozen encoder: retrieval-tuned embeddings (E5/BGE/ModernBERT), pooling, the "bigger-isn't-better" margin |
| 06 | batcher | [Negative-Pool Composition, Hard-Negative Mining, and Curriculum](06-batcher.md) | 18 | the in-batch negatives: in-batch/hard negatives, DPR/ANCE, curriculum learning |
| 07 | stream | [Single- vs Multi-Vector Token Layouts for a Sequence World Model](07-stream.md) | 14 | the token layout: single-vector pooling vs multi-vector late interaction (ColBERT) |
| 08 | head | [Prediction Heads and Auxiliary Self-Supervised Objectives on a Shared Trunk](08-head.md) | 19 | the readout + aux task: SSL projection/prediction heads, RL auxiliaries, rollout/successor readouts |

## How these were produced

- **One reviewer per chunk.** Each report was researched independently against the
  same brief: read the chunk's actual contract + baseline + top evolved impls, read
  the project's recorded results for that axis, then survey the design literature.
- **Structure.** Every report follows: role of the chunk in this world model →
  theoretical frameworks (mechanism + evidence) → domain-specific findings (the
  relevant sub-literature **and** this project's own measured results) → design
  guidance & open questions → references.
- **Citation integrity.** Every reference was located and verified via web search
  while writing; each carries a stable identifier (arXiv ID, DOI, or venue URL) so a
  later reader can check it. arXiv preprints and non-peer-reviewed writeups (e.g. the
  Muon blog post) are marked as such. A spot-check of the most recent / highest-risk
  citations (Muon, DeltaProduct, minGRU, Gated DeltaNet, ColPali, LeJEPA, ModernBERT)
  confirmed titles, authors, and IDs resolve to the real papers.

## Adversarial review

Every report was put through a one-skeptic-per-report adversarial pass (the project's
standard for any large write-up). Each reviewer attacked four surfaces: **citation
existence & attribution** (web-verify every reference), **claim–source support** (do the
cited sources actually say what's claimed), **fidelity to the project's own recorded
numbers** (cross-check against `README.md` / `evolve/CLAUDE.md`), and **reasoning
quality**.

Result: **141/141 citations verified real and correctly attributed; zero misrepresented
project numbers on the substantive claims.** The pass surfaced 1 Critical + 4 Major +
~26 Minor defects, all in prose rather than in the reference lists, and the corrections
below are reflected in the reports:

- **Critical (fixed):** `08-head` ref [18] (arXiv:2406.17718) was attributed to the wrong
  authors — corrected to Voelcker, Kastner, Gilitschenski, Farahmand.
- **Major (fixed):** `08-head` mis-benchmarked UNREAL's 87% (Labyrinth, not Atari);
  `03-optim` mis-stated the champion's inherited schedule hyperparameters (it is β₂=0.95 +
  uniform decay, not β₂=0.98 + ndim-split); `04-target` made a false universal claim about
  `to_obs(zeros)`; `05-perception` mis-pinned an anisotropy "≈0.99" figure on the wrong
  paper. All corrected.
- **Minor:** the substantive/factual minors were applied; a handful of purely editorial
  hedging/paraphrase notes were judged not worth actioning.

## Caveats

- These are **design surveys**, not empirical additions to the project. Where a report
  cites a terminal-JEPA result, that number comes from the root `README.md` / the
  `evolve-insights` ledger; the reports do not re-run the search.
- Some cited works are very recent arXiv preprints (2024–2025) chosen because they are
  the primary source for a mechanism in the current champion (e.g. chunkwise-WY
  delta-rule training, Muon). Treat preprint claims as preprint claims.
- Coverage is deliberately theory-first: the goal is the *space of design options and
  why each works*, to inform future inventor proposals — not a ranked recommendation.
