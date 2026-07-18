# JEPA: Joint-Embedding Predictive Architecture

This document explains Yann LeCun's JEPA line of research — what the architecture is, how it's trained, why it's built the way it is, and how it has evolved across the ~28 papers LeCun co-authored on it from 2022 through mid-2026. It assumes you know roughly how LLMs and standard supervised ML work, but nothing about self-supervised vision. Every claim is cited to a specific paper so you can drill into details.

## 1. The one-paragraph version

A JEPA learns by predicting missing parts of its input — like an LLM predicting a masked or next token — but with one decisive change: **the prediction happens in an abstract representation (embedding) space, not in the input space**. An LLM must output the exact next token; a masked autoencoder (MAE) must output the exact missing pixels. A JEPA instead encodes the visible part of the input, encodes the hidden part with a second encoder, and trains a predictor to map the first embedding to the second. Because the target is an embedding rather than raw input, the target encoder can *discard unpredictable, irrelevant detail* (the exact texture of grass, the exact ripples on a pond), and the model spends its capacity on the predictable, semantic structure of the world ([I-JEPA, arXiv 2301.08243](https://arxiv.org/abs/2301.08243); [position paper](https://openreview.net/forum?id=BZ5a1r-kVsf)).

## 2. Why LeCun proposed this: the 2022 position paper

The research program starts with LeCun's 2022 position paper, ["A Path Towards Autonomous Machine Intelligence"](https://openreview.net/forum?id=BZ5a1r-kVsf) (OpenReview, v0.9.2). Its argument, briefly:

- Humans and animals learn enormous amounts about how the world works largely by **observation** — infants learn intuitive physics (object permanence, gravity) before they can act on the world. Current AI does not: supervised learning needs labels, RL needs huge numbers of trials, and LLMs need orders of magnitude more text than a human sees.
- The core missing piece is a **world model**: a module that predicts plausible future states of the world, optionally conditioned on actions the agent might take. With a world model, an agent can *plan* — simulate candidate action sequences internally and pick the one that leads toward a goal — instead of learning a policy for every task by trial and error.
- The paper proposes a full cognitive architecture built around this: a perception module, the world model, a cost module (a hard-wired "intrinsic cost" plus a trainable critic), a short-term memory, an actor, and a "configurator" that sets the other modules up for the task at hand. All modules are differentiable. Planning ("Mode-2", analogous to System 2 deliberation) is done by optimizing action sequences against the world model; solutions can be distilled into fast reactive policies ("Mode-1"/System 1).
- **Why not generative prediction?** Predicting video at the pixel level forces the model to commit to unpredictable detail — which way each leaf blows, the exact texture of a carpet. The paper argues generative models "are not capable of eliminating irrelevant details, other than by pushing them into a latent variable," whereas a JEPA's target encoder can simply abstract those details away. This is the paper's central architectural bet, and the reason the whole line avoids reconstruction.
- The paper also sketches **H-JEPA**: JEPAs stacked hierarchically, where a low level predicts short-horizon detail and a higher level, operating on the lower level's representations, predicts coarse long-horizon structure — intended to support hierarchical planning (commute-to-work decomposes into drive-to-station decomposes into muscle commands). H-JEPA remains largely a proposal; the paper itself lists it as unvalidated.

A companion tutorial, [Dawid & LeCun 2023 (arXiv 2306.02572)](https://arxiv.org/abs/2306.02572), works out the formal framing: JEPA is a **latent-variable energy-based model (EBM)**. An EBM assigns a scalar energy F(x, y) that is low when y is a plausible completion of x and high otherwise; inference is energy minimization, and no normalized probability distribution is ever needed. In JEPA terms, the energy is the prediction error in embedding space: E = distance(s_y, Predictor(s_x, z)), where z is an optional latent variable capturing whatever about y is genuinely unpredictable from x (e.g., which way a car turns at a fork).

## 3. The core architecture

Using I-JEPA (the first large-scale instantiation, on images) as the reference design ([Assran et al. 2023, arXiv 2301.08243](https://arxiv.org/abs/2301.08243)):

```
                 x = visible context           y = hidden target
                 (e.g. 85% of image)           (e.g. a few masked blocks)
                        │                             │
                ┌───────▼────────┐            ┌───────▼────────┐
                │ context encoder│            │ target encoder │  ← same architecture,
                │   (ViT, θ)     │            │ (EMA copy of θ)│    weights = moving avg
                └───────┬────────┘            └───────┬────────┘    of context encoder,
                        │                             │             no gradients
                ┌───────▼────────┐                    │
                │   predictor    │── prediction ──►  loss = L2/L1 distance
                │ (small ViT)  + │                   in EMBEDDING space
                │ position tokens│
                └────────────────┘
```

The three components:

1. **Context encoder** — a standard Vision Transformer (ViT) that sees only the visible patches.
2. **Target encoder** — the same architecture, but its weights are an exponential moving average (EMA) of the context encoder's weights, updated with momentum ~0.996 and never trained by gradient. It encodes the *full* image, and the representations of the masked blocks are taken as prediction targets. Because targets are the output of a deep encoder, they are semantic abstractions, not pixels.
3. **Predictor** — a narrow transformer that takes the context embeddings plus positional "mask tokens" saying *where* the hidden blocks are, and outputs predicted embeddings for those blocks.

The loss is simply the distance (L2 in I-JEPA, L1 in V-JEPA) between predicted and target embeddings. There are no labels, no text, no contrastive negative pairs, and no hand-crafted augmentations (no random crops/color jitter as in SimCLR/DINO — masking is the only corruption).

### 3.1 The collapse problem

If you train two encoders to produce embeddings that predict each other, there is a trivial solution: every input maps to the same constant vector, prediction error is exactly zero, and nothing is learned. Preventing this **representation collapse** is *the* central engineering problem of the JEPA family, and different papers solve it differently:

- **Architectural asymmetry (EMA + stop-gradient):** the target encoder receives no gradients and lags the context encoder. Used by I-JEPA, V-JEPA, V-JEPA 2. Works empirically but was long considered a heuristic without theory.
- **Regularization (VICReg-style):** add explicit loss terms that keep the variance of each embedding dimension above a threshold and decorrelate dimensions, making constant outputs impossible. Used by MC-JEPA ([2307.12698](https://arxiv.org/abs/2307.12698)), VJ-VCR ([2412.10925](https://arxiv.org/abs/2412.10925)), PLDM ([2502.14819](https://arxiv.org/abs/2502.14819)). The position paper frames this as "maximize the information content of the embeddings."
- **Distribution matching (SIGReg / LeJEPA):** force the whole embedding distribution toward an isotropic Gaussian, which LeJEPA proves is the optimal embedding distribution for downstream tasks — see §6.
- **Contrastive negatives:** push embeddings of non-matching pairs apart (SimCLR-style). The position paper argues this scales poorly with embedding dimension, and the JEPA line mostly avoids it, though Crys-JEPA and VL-JEPA use InfoNCE variants where it fits the task.
- **Frozen targets:** when the target encoder is a frozen pretrained model (DINOv2 in DINO-WM-style world models, or Causal-JEPA's frozen slot encoder), collapse is impossible by construction because targets can't move.

### 3.2 What this buys you versus generative training (the LLM comparison)

An LLM is trained by input-space reconstruction: cross-entropy on the exact next token. That works well for text — tokens are discrete and low-bandwidth, so the "unpredictable detail" problem is manageable. The JEPA papers argue it breaks down for continuous, high-bandwidth signals like video, and report consistent empirical advantages for embedding-space prediction:

- **Compute and sample efficiency:** V-JEPA trains ~2× faster in wall-clock than equivalent pixel-reconstruction training and degrades far less with few labels ([2404.08471](https://arxiv.org/abs/2404.08471)); I-JEPA pretrains a ViT-H in <72 GPU-hours ×16 A100s, ~5× fewer iterations than MAE-class methods ([2301.08243](https://arxiv.org/abs/2301.08243)).
- **Better semantics per FLOP:** V-JEPA's ablation shows latent-space prediction beats pixel-space prediction by +5.1 points on frozen Kinetics-400 classification, all else equal ([2404.08471](https://arxiv.org/abs/2404.08471)).
- **More physically informative features:** on PDE/fluid-dynamics benchmarks, JEPA-pretrained encoders recover governing physical parameters substantially better than a VideoMAE trained on the same data (e.g., 51% lower regression error on active-matter systems) ([2603.13227](https://arxiv.org/abs/2603.13227)).

Note the trade-off is real, not one-sided: a JEPA produces representations, not samples — it cannot generate an image or a sentence by itself. Where output is needed, JEPA systems bolt on a separate lightweight decoder (VL-JEPA, §7) or a second reconstruction stage (speech tokenizer, §8).

## 4. Scaling it up: images → video → a 1B-parameter world model

The core recipe scaled along a clear line:

- **I-JEPA** (2023, [2301.08243](https://arxiv.org/abs/2301.08243)) — images. Multi-block masking; ImageNet linear probe 79.3–81.1% with ViT-H, beating MAE/data2vec without any augmentations.
- **MC-JEPA** (2023, [2307.12698](https://arxiv.org/abs/2307.12698)) — adds motion: one shared ConvNeXt encoder jointly trained on optical-flow estimation (flow acts as the predictor, warping frame-t features to frame-t+1) and VICReg content learning. Showed motion and content objectives help each other, though training was reportedly delicate.
- **V-JEPA** (2024, [2404.08471](https://arxiv.org/abs/2404.08471)) — video. Same masked-prediction recipe on 16-frame clips with 3D "tubelet" patches and multi-block masks spanning time, trained on ~2M videos. Frozen-encoder results beat pixel-reconstruction video models (e.g., 72.2% on Something-Something-v2, ~+6 over prior video models); strong on motion-centric tasks, slightly behind image-specialist models (DINOv2) on pure appearance.
- **VJ-VCR** (2024, [2412.10925](https://arxiv.org/abs/2412.10925)) — a smaller-scale study showing a video JEPA can be trained *without* EMA/stop-gradient at all, using variance-covariance regularization instead, and that adding an explicit latent variable z captures stochastic future factors (its hidden information can predict a moving object's trajectory switch with 99.5% accuracy on synthetic data).
- **V-JEPA 2** (2025, [2506.09985](https://arxiv.org/abs/2506.09985)) — scale: ViT-g encoder (~1B params), 22M videos / ~1M hours (VideoMix22M), 3D rotary position embeddings, progressive-resolution training (8.4× GPU-time saving). Results: 77.3% SSv2, 87.3% K400, state-of-the-art action anticipation on Epic-Kitchens-100 (+44% relative), and — when its frozen features are aligned to an 8B LLM — state-of-the-art video QA among models with no language-supervised pretraining. Its action-conditioned variant is covered in §5.
- **V-JEPA 2.1** (2026, [2603.14482](https://arxiv.org/abs/2603.14482)) — fixes V-JEPA 2's weakness: noisy *dense* (per-patch) features. Diagnosis: supervising the predictor only on masked tokens destroys local spatial structure. Fix: supervise on all tokens (with distance-weighted loss near masks), add deep supervision at intermediate layers, and train jointly on images + video (163M samples). Dense tasks improve dramatically (ADE20K segmentation +23.4 mIoU; NYUv2 depth error 0.642→0.307, edging out DINOv3-7B) while global performance is preserved, and robot grasping improves +20% over V-JEPA 2.

## 5. The payoff: world models and planning

This is where the position paper's agenda gets cashed out. The pattern shared by all the planning papers: **encode the current observation and a goal image into embeddings, roll out a learned latent dynamics model under candidate action sequences, and pick the sequence whose predicted final embedding is closest to the goal embedding** — energy minimization over actions, executed as model-predictive control (MPC, i.e., replan after each step). No reward function, no task-specific training.

- **IWM** (2024, [2403.00504](https://arxiv.org/abs/2403.00504)) — first paper to treat the JEPA predictor itself as a reusable world model over image transformations ("actions" = color jitter, masking). Established three levers that make a predictor a real (equivariant) world model rather than collapsing to invariance: condition it on the action, use complex transformations, and give it capacity. Also showed the trained predictor can be cheaply fine-tuned for downstream tasks in place of the encoder.
- **PLDM** (2025, [2502.14819](https://arxiv.org/abs/2502.14819)) — the flagship "JEPA + planning" demonstration. A JEPA latent dynamics model trained on reward-free, suboptimal offline trajectories, planned over with MPPI. Against goal-conditioned RL baselines it is the only method that generalizes to unseen maze layouts (trained on just 5 layouts) and works from a few thousand transitions.
- **V-JEPA 2-AC** (2025, [2506.09985](https://arxiv.org/abs/2506.09985)) — the scale version: freeze the 1B V-JEPA 2 encoder, train a 300M action-conditioned predictor on only 62 hours of unlabeled robot video, then plan with the cross-entropy method over end-effector actions. Deployed zero-shot on Franka arms in labs it never saw: 100% reach, 65–80% pick-and-place on cups, vs ~15% for the Octo baseline. Planning takes ~16 s/action vs 4 min for a generative (Cosmos) world model.
- **"What Drives Success in Physical Planning with JEPA World Models?"** (2026, [2512.24497](https://arxiv.org/abs/2512.24497)) — a large ablation study unifying DINO-WM and V-JEPA 2-AC into one recipe and testing 7 design axes. Headline findings: sampling-based planners (CEM) beat gradient descent on contact-rich tasks; training with multi-step rollouts is essential against compounding error; frozen DINO features beat frozen V-JEPA features as the world-model backbone (sharper object boundaries); proprioception is needed for metric precision near goals; scaling model size only pays off on complex real-world data.
- **Value-guided planning** (2026, [2601.00844](https://arxiv.org/abs/2601.00844)) — identifies the planning bottleneck as the *cost landscape*, not the dynamics model: raw embedding distance has local minima (a wall between you and the goal looks "close"). Shaping the latent space with an offline-RL value function (IQL) so that Euclidean distance encodes *reachability* substantially improves planning success.
- **Temporal Straightening** (2026, [2603.12231](https://arxiv.org/abs/2603.12231)) — adds a regularizer that straightens latent trajectories (consecutive velocity vectors should align), provably conditioning the planning objective better; enables cheap gradient-based planning (~10× faster than CEM) with large success gains, e.g. 44→94% on PointMaze.
- **Causal-JEPA** (2026, [2602.11389](https://arxiv.org/abs/2602.11389), ICML) — moves masking from patches to *object slots*: mask an object's entire history (keeping one identity anchor) so the model can only fill it in by reasoning about interactions with other objects. Big gains on counterfactual video QA (+21 points on CLEVRER counterfactuals) and near-DINO-WM planning using ~1% of the tokens.
- **SkyJEPA** (2026, [2606.23444](https://arxiv.org/abs/2606.23444)) — first JEPA in a real-time control loop: a ~99K-parameter latent dynamics model + MPPI running at 100 Hz on an embedded GPU flies a real quadrotor, trained entirely in simulation with domain randomization (zero-shot sim-to-real). Latent prediction cuts long-horizon prediction error ~84% versus a state-space baseline, supporting the claim that predicting in latent space mitigates compounding error.
- **LeWorldModel** (2026, [2603.19312](https://arxiv.org/abs/2603.19312)) — a stable *end-to-end* JEPA world model from raw pixels: a two-term loss (prediction + SIGReg) replaces the usual pile of heuristics (EMA, stop-gradient, pretrained encoders, multi-term losses), trains a ~15M-param model on one GPU, and plans ~48× faster than DINO-WM.

An honest early data point on failure modes belongs here too: **"JEPAs Focus on Slow Features"** (2022, [2211.10831](https://arxiv.org/abs/2211.10831)) showed that JEPA objectives implement *slow-feature learning* — which is exactly what you want when task-relevant features change slowly, but which fails hard when a distractor is *slower* than the signal (a fixed noise background gets copied into the representation and the objective is satisfied without learning dynamics). This inductive bias, strength and weakness, underlies the whole line.

## 6. The theory turn: removing the heuristics

Through 2024, JEPA training rested on empirically-tuned tricks (EMA schedules, stop-gradients, multi-term losses). Three papers put it on firmer ground:

- **Gaussian Embeddings** (2025, [2510.05949](https://arxiv.org/abs/2510.05949)) — proves that a JEPA trained to optimality with an anti-collapse constraint *implicitly learns the training-data density*: the encoder's local volume distortion (singular values of its Jacobian) encodes p(x). This yields JEPA-SCORE, a training-free density estimate from any pretrained JEPA (validated on I-JEPA, DINOv2, MetaCLIP), usable for outlier detection and data curation — and it means "non-generative" JEPAs are secretly closer to generative models than assumed.
- **LeJEPA** (2025, [2511.08544](https://arxiv.org/abs/2511.08544)) — the capstone theory paper (Balestriero & LeCun). Proves the isotropic Gaussian is the *unique* embedding distribution minimizing worst-case downstream risk for both linear and nonlinear probes, then introduces **SIGReg** (Sketched Isotropic Gaussian Regularization): project embeddings onto ~1024 random directions and apply a differentiable 1-D statistical test (Epps–Pulley) pushing each projection toward N(0,1) — by the Cramér–Wold theorem, matching all 1-D projections matches the full distribution. Result: training loss = prediction + λ·SIGReg, one hyperparameter, no stop-gradient, no EMA teacher, no schedulers, ~50 lines of PyTorch, validated across 50+ architectures up to ~1.8B params (79% ImageNet linear probe with ViT-H). Training loss now correlates with downstream accuracy, so models can be selected without labels. SIGReg was immediately adopted by LeWorldModel and SkyJEPA above.
- **Rectified LpJEPA** (2026, [2602.01456](https://arxiv.org/abs/2602.01456)) — generalizes LeJEPA's Gaussian target to a rectified/sparse family (maximum-entropy under an Lp constraint), making embedding *sparsity* an analytically controllable dial (e.g., 85.1% ImageNet-100 linear probe at 73% zeros).
- **EB-JEPA** (2026, [2602.03604](https://arxiv.org/abs/2602.03604)) — an open-source pedagogical library ([github.com/facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa)) casting images → video → action-conditioned world models in one energy-based framework, with single-GPU recipes and ablations showing which regularizers are load-bearing (e.g., removing the inverse-dynamics loss drops planning success from 97% to 1%).

## 7. JEPA meets language

Three papers test whether the "predict in embedding space" idea transfers to the LLM domain:

- **LLM-JEPA** (2025, [2509.14252](https://arxiv.org/abs/2509.14252)) — keeps the standard next-token loss and *adds* a JEPA term: for data with two natural views of the same content (a natural-language description and its code/regex/SQL), the LLM's own hidden states embed both views, and the model is trained so the text embedding predicts the code embedding. Reported gains are consistent across Llama/Gemma/OpenELM fine-tuning (e.g., 57.3→71.5% on NL→regex) with better overfitting resistance, at ~2× training compute.
- **VL-JEPA** (2025, [2512.10942](https://arxiv.org/abs/2512.10942)) — a vision-language model that predicts continuous *text-semantic embeddings* instead of generating tokens: frozen V-JEPA 2 vision encoder → predictor (Llama-3.2-1B layers) → target embeddings from a frozen text encoder; a small decoder converts embeddings to text only when needed. With 1.6B total params it beats token-space training in a controlled comparison (2× CIDEr at equal data) and supports streaming video with ~2.85× fewer decode operations.
- **Semantic Tube Prediction** (2026, [2602.22617](https://arxiv.org/abs/2602.22617)) — drops the need for paired views entirely: it hypothesizes token sequences trace near-linear geodesics in hidden-state space, and adds a loss keeping hidden-state trajectories locally collinear (a "semantic tube"). The paper reports matching next-token-only baselines with 16× less training data on its main benchmark (NL-RX-SYNTH, Llama-3-1B), with consistent gains at 3B/8B — presented as evidence that embedding-space geometric priors can beat brute-force data scaling. (Claims are benchmark-specific; treat the 16× figure as the paper's own headline result.)

## 8. JEPA beyond vision and language

- **Speech:** [JEPA as a Neural Tokenizer (2512.07168)](https://arxiv.org/abs/2512.07168) uses JEPA masked latent prediction as the semantic stage of a speech codec (2.5 Hz frame rate vs 75 Hz for EnCodec-class codecs), with waveform reconstruction bolted on afterward. [S-JEPA (2606.19398)](https://arxiv.org/abs/2606.19398) replaces continuous regression targets with soft cluster posteriors from an online GMM, beating HuBERT-style pipelines at sub-90M parameter scale (12.1% WER frozen-encoder on LibriSpeech). (An earlier version appears as [2602.09040](https://arxiv.org/abs/2602.09040).)
- **Materials science:** [Crys-JEPA (2605.14759)](https://arxiv.org/abs/2605.14759) builds views from formation-energy-preserving crystal augmentations (instead of masking) and uses the resulting energy-aware latent space to screen generated crystal candidates ~17× faster than ML force fields, nearly doubling the DFT-validated stable-unique-novel rate of a MatterGen generator (26.4→47.9%).
- **Physics:** [Representation Learning for Spatiotemporal Physical Systems (2603.13227)](https://arxiv.org/abs/2603.13227) shows JEPA-style latent block prediction recovers governing PDE parameters far better than pixel-reconstruction pretraining on fluid/active-matter benchmarks, with strong low-data behavior (JEPA at 10% fine-tuning data beats VideoMAE at 100%).

Across these, essentially every component varies — targets (continuous latents, soft cluster posteriors, augmented-view embeddings), anti-collapse mechanism (EMA, VICReg, SIGReg, contrastive), context construction (spatial masks, temporal blocks, physics-preserving augmentations) — which the papers collectively read as evidence that "predict in representation space" is the portable core idea rather than any single recipe.

## 9. Recurring findings across the program

Design principles that show up repeatedly, with the papers that establish them:

1. **Latent prediction beats input reconstruction for representation quality and compute**, in vision ([2404.08471](https://arxiv.org/abs/2404.08471), [2301.08243](https://arxiv.org/abs/2301.08243)), physics ([2603.13227](https://arxiv.org/abs/2603.13227)), and control ([2606.23444](https://arxiv.org/abs/2606.23444)); pixel models still win when the "distractor" is slower than the signal ([2211.10831](https://arxiv.org/abs/2211.10831)).
2. **Masking design is load-bearing:** large semantic blocks, not random scatter ([2301.08243](https://arxiv.org/abs/2301.08243); V-JEPA's multi-block vs random-tube ablation: 72.9 vs 51.5 on K400 [2404.08471](https://arxiv.org/abs/2404.08471)); masking only ever the targets damages dense features, fixed by supervising all tokens ([2603.14482](https://arxiv.org/abs/2603.14482)); masking *objects* rather than patches induces interaction reasoning ([2602.11389](https://arxiv.org/abs/2602.11389)).
3. **The anti-collapse mechanism migrated from heuristic to principled:** EMA/stop-gradient (2023–25) → variance-covariance regularization ([2412.10925](https://arxiv.org/abs/2412.10925), [2502.14819](https://arxiv.org/abs/2502.14819)) → provably-grounded SIGReg with one hyperparameter ([2511.08544](https://arxiv.org/abs/2511.08544)), now reused across the newest world-model papers.
4. **Planning = energy minimization over actions in latent space**, and its practical recipe is now well-mapped: sampling-based optimizers for contact-rich tasks, multi-step rollout training against compounding error, short context windows, proprioception for metric precision ([2512.24497](https://arxiv.org/abs/2512.24497)); the residual bottleneck is the cost landscape, addressable by value shaping ([2601.00844](https://arxiv.org/abs/2601.00844)) or trajectory straightening ([2603.12231](https://arxiv.org/abs/2603.12231)).
5. **A small amount of interaction data goes a long way once passive pretraining is done:** 62 hours of unlabeled robot video turns V-JEPA 2 into a zero-shot manipulation planner ([2506.09985](https://arxiv.org/abs/2506.09985)) — the position paper's observation-first thesis in practice.

## 10. Open problems, as stated in the papers themselves

- **Hierarchy (H-JEPA) is still mostly unbuilt.** The position paper's multi-timescale stacked JEPA has no full implementation in this corpus; LeWorldModel and V-JEPA 2 both cite short planning horizons and compounding rollout error as the motivating gap ([2603.19312](https://arxiv.org/abs/2603.19312), [2506.09985](https://arxiv.org/abs/2506.09985)).
- **Stochastic futures.** Deterministic predictors learn the conditional mean of multimodal futures ([2512.24497](https://arxiv.org/abs/2512.24497)); the latent-variable-z machinery from the position paper is only lightly explored ([2412.10925](https://arxiv.org/abs/2412.10925)).
- **Goal specification.** Robot planning currently requires goal *images*; language goals are unsupported ([2506.09985](https://arxiv.org/abs/2506.09985)).
- **The slow-feature failure mode** — spurious slow distractors satisfying the objective — was demonstrated on toy data ([2211.10831](https://arxiv.org/abs/2211.10831)) and has no general solution beyond data diversity.
- **Language transfer is early-stage.** LLM-JEPA needs paired views and 2× compute; Semantic Tube Prediction's efficiency claims rest on relatively narrow benchmarks; VL-JEPA explicitly does not attempt reasoning/tool-use where token-generative VLMs excel.

## 11. Paper index

Foundations:

| Year | Paper | Link |
|---|---|---|
| 2022 | A Path Towards Autonomous Machine Intelligence (position paper) | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) |
| 2022 | Joint Embedding Predictive Architectures Focus on Slow Features | [2211.10831](https://arxiv.org/abs/2211.10831) |
| 2023 | I-JEPA: Self-Supervised Learning from Images with a JEPA | [2301.08243](https://arxiv.org/abs/2301.08243) |
| 2023 | Introduction to Latent Variable Energy-Based Models | [2306.02572](https://arxiv.org/abs/2306.02572) |

Video:

| Year | Paper | Link |
|---|---|---|
| 2023 | MC-JEPA: motion + content features | [2307.12698](https://arxiv.org/abs/2307.12698) |
| 2024 | V-JEPA: Revisiting Feature Prediction for Video | [2404.08471](https://arxiv.org/abs/2404.08471) |
| 2024 | VJ-VCR: Video JEPA with variance-covariance regularization | [2412.10925](https://arxiv.org/abs/2412.10925) |
| 2025 | V-JEPA 2: Understanding, Prediction and Planning | [2506.09985](https://arxiv.org/abs/2506.09985) |
| 2026 | V-JEPA 2.1: Unlocking Dense Features | [2603.14482](https://arxiv.org/abs/2603.14482) |

World models and planning:

| Year | Paper | Link |
|---|---|---|
| 2024 | IWM: Learning and Leveraging World Models | [2403.00504](https://arxiv.org/abs/2403.00504) |
| 2025 | PLDM: Planning with Latent Dynamics Models | [2502.14819](https://arxiv.org/abs/2502.14819) |
| 2025 | What Drives Success in Physical Planning with JEPA World Models? | [2512.24497](https://arxiv.org/abs/2512.24497) |
| 2025 | Value-guided action planning with JEPA world models | [2601.00844](https://arxiv.org/abs/2601.00844) |
| 2026 | Causal-JEPA: object-level latent masking | [2602.11389](https://arxiv.org/abs/2602.11389) |
| 2026 | Temporal Straightening for Latent Planning | [2603.12231](https://arxiv.org/abs/2603.12231) |
| 2026 | LeWorldModel: stable end-to-end JEPA from pixels | [2603.19312](https://arxiv.org/abs/2603.19312) |
| 2026 | SkyJEPA: zero-shot sim-to-real quadrotor control | [2606.23444](https://arxiv.org/abs/2606.23444) |

Theory and tooling:

| Year | Paper | Link |
|---|---|---|
| 2025 | Gaussian Embeddings: JEPAs secretly learn data density | [2510.05949](https://arxiv.org/abs/2510.05949) |
| 2025 | LeJEPA: provable SSL without the heuristics (SIGReg) | [2511.08544](https://arxiv.org/abs/2511.08544) |
| 2026 | Rectified LpJEPA: sparse maximum-entropy embeddings | [2602.01456](https://arxiv.org/abs/2602.01456) |
| 2026 | EB-JEPA: lightweight energy-based JEPA library | [2602.03604](https://arxiv.org/abs/2602.03604) |

Language, speech, and science:

| Year | Paper | Link |
|---|---|---|
| 2025 | LLM-JEPA: LLMs meet JEPAs | [2509.14252](https://arxiv.org/abs/2509.14252) |
| 2025 | VL-JEPA: vision-language JEPA | [2512.10942](https://arxiv.org/abs/2512.10942) |
| 2025 | JEPA as a Neural Tokenizer (speech) | [2512.07168](https://arxiv.org/abs/2512.07168) |
| 2026 | Semantic Tube Prediction: beating LLM data efficiency | [2602.22617](https://arxiv.org/abs/2602.22617) |
| 2026 | Crys-JEPA: crystal discovery | [2605.14759](https://arxiv.org/abs/2605.14759) |
| 2026 | Representation Learning for Spatiotemporal Physical Systems | [2603.13227](https://arxiv.org/abs/2603.13227) |
| 2026 | S-JEPA: soft clustering anchors for speech | [2606.19398](https://arxiv.org/abs/2606.19398) |

A note on authorship: LeCun is a co-author on every paper above; first authors are researchers at Meta FAIR, NYU, Brown, Mila, and collaborating labs. The dates given are first arXiv submission dates.
