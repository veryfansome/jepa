# World Models Beyond JEPA — A Field Map for terminal-JEPA

Companion to `JEPA.md`. That doc covers LeCun's JEPA line in depth; this one maps the **rest** of the world-model landscape (~2018–2026) so a terminal-JEPA session knows what else is out there, who the neighbors are, and which lines are worth borrowing from vs. explicitly arguing against.

This survey was assembled by a fan-out research round (multi-source search → primary-source fetch → 3-vote adversarial verification). **Verification convention** used throughout:
- **✓** — the characterization was adversarially verified 3-0 against the primary source in this round.
- **○** — title/authors/venue confirmed from the primary source, but the JEPA-contrast characterization was *not* independently vote-verified here (treat as a lead to read, not a settled claim).

Per the repo's honest-evaluation rule, unverified threads are flagged, not smoothed over.

## The one coordinate system

JEPA's recipe is three orthogonal bets. Every system below can be placed on these three axes, and that placement is the useful thing:

1. **Prediction target** — abstract latent (JEPA) vs. reconstructed pixels/tokens (generative). *The* defining JEPA axis: does the model predict a representation it never has to decode, or must it regenerate the observation?
2. **Anti-collapse mechanism** — what stops the representation from trivially collapsing? JEPA uses architectural asymmetry + VICReg-style variance/covariance regularization and *no reward*. Others avoid collapse "for free" by grounding the latent in reconstruction, reward, or value — a target that can't collapse because it's tied to real signal.
3. **Planning / control** — plan by **latent distance to a goal** (JEPA's distinctive proposal) vs. reward/value maximization (RL) vs. rolling out and scoring imagined observations (generative) vs. no planning at all (pure representation learning).

terminal-JEPA sits at: non-reconstructive latent prediction · anti-collapse without reward · action-outcome-given-history · (aspirationally) plan by latent distance. Almost nothing else occupies exactly that corner — the value of the map is seeing *which* neighbor shares *which* axis.

---

## Thread 1 — Latent-imagination RL (the closest kin)

These predict forward in a **learned latent space** and derive behavior by rolling out *inside* that latent model. Shared with JEPA: prediction happens in latent space, control comes from latent rollouts. Different from JEPA: the latent is grounded by reconstruction and/or reward, and planning maximizes return rather than minimizing latent distance to a goal.

- **PlaNet** — *Learning Latent Dynamics for Planning from Pixels*, Hafner et al., ICML 2019 ([1811.04551](https://arxiv.org/abs/1811.04551)). ✓ Purely model-based; learns image dynamics and picks actions by **fast online planning in latent space** (CEM, no policy/value net). Introduces **latent overshooting**, a multi-step objective training predictions across future *latent* steps. **Why it matters:** the purest "plan by latent rollout" instance — closest in spirit to planning-in-latent — but its RSSM is still reconstruction-trained, and it plans by reward, not latent distance.
- **Dreamer (v1)** — *Dream to Control*, Hafner et al., ICLR 2020 ([1912.01603](https://arxiv.org/abs/1912.01603)). ✓ Solves long-horizon tasks "purely by latent imagination": optimizes behavior by propagating analytic value gradients back through trajectories imagined in the compact latent space. RSSM trained with pixel reconstruction. **Why it matters:** canonical "learn behavior entirely inside the world model" — the loop terminal-JEPA's action-outcome prediction is a prerequisite for.
- **DreamerV3** — Hafner et al., 2023 ([2301.04104](https://arxiv.org/abs/2301.04104), later *Nature* 2025). ✓ Same recipe, scaled and made robust across 150+ tasks with fixed hyperparameters; RSSM encodes inputs into **32 one-hot categorical latents** and predicts future representations + rewards given actions — but still uses an observation decoder as learning signal. **Why it matters:** shows the latent-imagination recipe generalizes across domains without per-task tuning; the categorical-latent design is a concrete alternative to continuous JEPA embeddings.
- **DayDreamer** — Wu, Escontrela, Hafner, Goldberg, Abbeel; CoRL 2022 ([2206.14176](https://arxiv.org/abs/2206.14176)). ✓ Dreamer applied to **4 physical robots learning online in the real world, no simulator**; actor-critic trained on latent trajectories imagined by the world model. **Why it matters:** existence proof that latent-imagination world models train on real, expensive-to-sample dynamics — the regime terminal-JEPA lives in (real Docker filesystems, not cheap synthetic rollouts).
- **TD-MPC** — Hansen, Wang, Su; ICML 2022 ([2203.04955](https://arxiv.org/abs/2203.04955)). ✓ Learns a **Task-Oriented Latent Dynamics (TOLD)** model that **omits the decoder** — no observation reconstruction — trained by TD learning; plans by local trajectory optimization (MPC) over a short horizon plus a learned terminal value. Collapse is avoided by reward/value grounding + a latent self-consistency loss, *not* a VICReg-style term. **Why it matters:** the **tightest structural analogue to JEPA's decoder-free predict-and-plan-in-latent-space recipe** — differs mainly in *what grounds the latent* (value vs. anti-collapse) and *what planning optimizes* (return vs. latent distance).
- **TD-MPC2** — Hansen, Su, Wang; ICLR 2024 ([2310.16828](https://arxiv.org/abs/2310.16828)). ✓ Scales TD-MPC: one algorithm, one set of hyperparameters, 104 continuous-control tasks; MPPI/CEM MPC in the latent space of a learned **implicit (decoder-free)** world model. **Why it matters:** the state-of-the-art proof that a *reconstruction-free* latent world model scales and plans — the strongest evidence the JEPA half of terminal-JEPA's bet (drop the decoder) is sound, even though it grounds via value rather than anti-collapse.

> **Read-first for terminal-JEPA:** TD-MPC2 (decoder-free latent + planning) and PlaNet (plan-by-latent-rollout). These are the neighbors whose recipe you can most directly borrow.

---

## Thread 2 — Autoregressive / generative world models (the contrast pole)

These predict the **next observation** — image tokens or pixels — and regenerate it. They are the explicit *opposite* of JEPA's "predict an abstract latent you never decode." Recent momentum in the field is overwhelmingly here, which makes this the pole terminal-JEPA is implicitly betting against. Their **action-conditioned autoregressive rollout structure** is nonetheless directly instructive for "predict outcome given action + history."

- **IRIS** — *Transformers are Sample-Efficient World Models*, Micheli, Alonso, Fleuret; ICLR 2023 ([2209.00588](https://arxiv.org/abs/2209.00588)). ✓ Discrete VQ autoencoder tokenizes frames; a GPT-style transformer autoregressively predicts next image-token + reward + termination. Dynamics-learning as **sequence modeling over reconstructed image tokens**. **Why it matters:** the token-reconstruction paradigm terminal-JEPA contrasts against — *but the sequence-model-over-tokens framing is exactly terminal-JEPA's substrate* (shell states as token sequences), so the architecture is a live baseline even though the target differs.
- **Genie** — Bruce et al., Google DeepMind; ICML 2024 Best Paper ([2402.15391](https://arxiv.org/abs/2402.15391)). ✓ First generative interactive environment trained **unsupervised from unlabelled internet videos, with no ground-truth action labels**. Learns a **latent action space of only 8 discrete VQ codes** unsupervised; spatiotemporal tokenizer + latent-action model + MaskGIT dynamics generating next-frame tokens. **Why it matters:** the single most relevant idea for terminal-JEPA's *expanding, non-fixed toolset* — Genie **discovers a compact action ontology from observation alone**. It's generative (contrasts with JEPA), but the unsupervised-latent-action mechanism is directly portable.
- **DIAMOND** — Alonso et al., NeurIPS 2024 ([2405.12399](https://arxiv.org/abs/2405.12399)). ✓ Trains an RL agent entirely inside a **diffusion** world model operating **in image space**, explicitly arguing that "compression into a compact discrete representation may ignore visual details important for RL." **Why it matters:** the sharpest published **counter-argument to JEPA's compress-to-abstract-latent thesis** — it claims latent compression *discards* task-relevant detail. terminal-JEPA should take this as the steelman to beat: is command-outcome-relevant detail preserved in an abstract latent, or lost?
- **GameNGen** — Valevski et al., Google, 2024 ([2408.14837](https://arxiv.org/abs/2408.14837)). ✓ Diffusion model producing the **next frame conditioned on past frames + actions**, stable enough to run DOOM interactively (next-frame PSNR ~29.4). Denoises in an SD VAE latent then decodes to pixels — target is the full observation. **Why it matters:** canonical action-conditioned generative world model; the reconstruction pole in its most polished form.
- **Sora — "Video generation models as world simulators"** — OpenAI technical report, Feb 2024 ([openai.com](https://openai.com/index/video-generation-models-as-world-simulators/)). ✓ Compresses video to a lower-dimensional latent, decomposes into **spacetime patches** as transformer tokens, diffusion-denoises them, decodes to pixels; argues scaling generative video is "a promising path towards … general purpose simulators of the physical world." **Why it matters:** the highest-profile statement of the **generation-as-world-model thesis JEPA argues against**. *(Caveat: non-peer-reviewed corporate report; page was bot-blocked, quotes corroborated via search.)*

> **Read-first for terminal-JEPA:** Genie (unsupervised action-ontology discovery) and DIAMOND (the anti-latent-compression argument to rebut). IRIS as the token-sequence baseline architecture.

---

## Thread 3 — LLM / agentic & text world models (the task-level analogue)

Structurally the closest to terminal-JEPA's *task*: predict what an action does to an environment, then plan by scoring imagined outcomes. The difference is the medium — these predict in **natural-language / token space** using a prompted or fine-tuned LLM, with no trained latent embedding and no anti-collapse.

- **WebDreamer** — *Is Your LLM Secretly a World Model of the Internet?*, OSU-NLP-Group; TMLR 2025 ([2411.06559](https://arxiv.org/abs/2411.06559)). ✓ Uses an LLM as a world model to **simulate the outcome of each candidate action** ("what would happen if I click this?") in natural language, scores the imagined outcomes, picks the best. Beats reactive baselines with GPT-4o (VisualWebArena 17.6→23.6%, Mind2Web-live 20.2→25.0%). **Why it matters:** the **most structurally analogous system to terminal-JEPA** — model-based lookahead by simulating action-outcomes — and the natural baseline to beat: *does a trained latent shell-state predictor outperform just prompting an LLM to imagine the command's output?* (Caveat: absolute success rates remain low.)
- **WMA (Web agents with world models)** — *Learning and Leveraging Environment Dynamics in Web Navigation*, 2024 ([2410.13232](https://arxiv.org/abs/2410.13232)). ○ Web agent that learns environment dynamics to predict the outcome of navigation actions before committing. **Why it matters:** a second, independent data point that lightweight learned dynamics improve web agents — same lookahead thesis as WebDreamer.
- **RAP** — *Reasoning with Language Model is Planning with World Model*, 2023 ([2305.14992](https://arxiv.org/abs/2305.14992)). ○ Repurposes the LLM as both reasoner and world model, doing MCTS-style planning over imagined states. **Why it matters:** frames LLM reasoning itself as world-model planning — a conceptual bridge from JEPA-style planning to the text-agent setting.
- **AutoManual** — *Constructing Instruction Manuals by LLM Agents via Interactive Environmental Learning*, 2024 ([2405.16247](https://arxiv.org/abs/2405.16247)). ○ Agents interactively probe an environment and distill reusable rules/manuals about its dynamics. **Why it matters:** directly analogous to terminal-JEPA's "explore an unseen system, build a model of how it responds" — but as explicit text rules rather than a latent.

> **Read-first for terminal-JEPA:** WebDreamer — it *is* the LLM-shaped version of what terminal-JEPA does, and the honest baseline the latent approach must justify itself against.

---

## Thread 4 — Robotics / large-scale foundation simulators

Scaled, action-conditioned world models aimed at real-world / robot dynamics. Mostly generative (pixel/video), so they share Thread 2's contrast with JEPA, but they matter for the *scale* and *unlabeled-action* questions terminal-JEPA faces.

- **UniSim** — *Learning Interactive Real-World Simulators*, Yang, Du et al. (UC Berkeley / Google DeepMind / MIT); ICLR 2024 Outstanding Paper ([2310.06114](https://arxiv.org/abs/2310.06114)). ○ A single generative **video-diffusion** simulator absorbing heterogeneous data (images, robot actions, navigation) to predict visual outcomes of actions. **Why it matters:** the pixel-reconstruction pole at foundation scale — action-conditioned outcome prediction across many domains, the generative mirror-image of terminal-JEPA's cross-system generalization goal.
- **PWM** — *Policy Learning with Multi-Task World Models*, Georgiev et al. (Georgia Tech / NVIDIA), 2024 ([2407.02466](https://arxiv.org/abs/2407.02466)). ○ Learns multi-task world models and extracts control via first-order gradients through them. **Why it matters:** multi-task generalization of a learned world model — the "one model, many systems" axis terminal-JEPA cares about.
- **1X World Model** — 1X Technologies, 2024–2025 ([1xgpt repo](https://github.com/1x-technologies/1xgpt); report [arXiv:2510.07092](https://arxiv.org/abs/2510.07092)). ○ A world-model challenge/benchmark predicting future frames of a humanoid robot's egocentric video conditioned on actions. **Why it matters:** an industrial action-conditioned prediction benchmark; useful reference for how outcome-prediction quality gets measured at scale.
- **NVIDIA Cosmos** — world-foundation-model platform, 2025. ○ *(Named in the brief; no primary source verified in this round.)* Treat as a lead: a large-scale generative world-model platform for physical AI/robotics. Verify before citing.

> Note: this thread was **not** independently vote-verified in this round (all ○). Characterizations are from source titles/abstracts; confirm before relying on specifics.

---

## Thread 5 — Implicit / value-equivalent models & the theory side

An alternative philosophy: don't model the world faithfully at all — model only what's needed to act or predict return. This is the conceptual cousin of JEPA's "discard unpredictable detail," but the discarding criterion is *value* (reward-relevance), not *predictability*.

- **MuZero** — *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*, Schrittwieser et al. (DeepMind), Nature 2020 ([1911.08265](https://arxiv.org/abs/1911.08265)). ○ Learns a latent dynamics model trained **only** to predict reward, value, and policy — never the observation — and plans with MCTS in that latent space. **Why it matters:** the original **decoder-free, value-equivalent** world model. Philosophically: JEPA drops the decoder to keep *predictable* structure; MuZero drops it to keep *reward-relevant* structure. terminal-JEPA (no dense reward) can't take MuZero's route — which is *why* it needs an anti-collapse objective instead of a value objective. This contrast is the sharpest framing of terminal-JEPA's core bet.
- **The Value Equivalence Principle** — Grimm, Barreto, Singh, Silver; NeurIPS 2020 ([2011.03506](https://arxiv.org/abs/2011.03506)). ○ Formalizes *why* an implicit model like MuZero works: two models are "value-equivalent" if they induce the same value updates, so you only need to match the world on value-relevant quantities. **Why it matters:** the theory that legitimizes non-reconstructive world models generally — the closest existing formal grounding for "you don't need to reconstruct to have a useful world model," which is JEPA's and terminal-JEPA's whole premise.
- **Emergent World Representations (Othello-GPT)** — Kenneth Li et al.; ICLR 2023 ([2210.13382](https://arxiv.org/abs/2210.13382)). ○ Probes a next-move sequence model trained on Othello and finds a **linearly-recoverable latent representation of the board state** — evidence a pure sequence predictor builds an internal world model. **Why it matters:** the template for terminal-JEPA's *own* probing question — does the shell-state sequence model internally represent the filesystem/system state? This is the methodology to reuse for "does terminal-JEPA have a world model?"
- **Evaluating the World Model Implicit in a Generative Model** — Vafa, Chen, Rambachan, Kleinberg, Mullainathan; NeurIPS 2024 ([2406.03689](https://arxiv.org/abs/2406.03689)). ○ Proposes metrics (adapting the Myhill-Nerode theorem) to test whether a next-token model's latent state truly captures the environment's state, and shows models can score well on next-token accuracy while having an incoherent underlying world model. **Why it matters:** directly relevant **evaluation methodology** — a rigorous way to ask whether terminal-JEPA's predictor has learned the *system's* state vs. just surface next-token statistics. A caution that outcome-prediction accuracy alone doesn't prove a coherent world model.

> **Read-first for terminal-JEPA:** MuZero + Value Equivalence (why decoder-free works, and why the no-reward setting forces anti-collapse), and the two probing/eval papers (Othello-GPT, Vafa et al.) as the methodology for *evaluating* whether terminal-JEPA has a world model.

---

## The map, in one table

| System | Target (latent vs. recon) | Anti-collapse via | Planning | Nearest to terminal-JEPA on… |
|---|---|---|---|---|
| **JEPA / terminal-JEPA** | abstract latent, no decode | VICReg-style, no reward | (aspirational) latent distance | — |
| PlaNet | latent (recon-trained) | reconstruction | latent rollout + reward (CEM) | plan-in-latent |
| Dreamer v1/V3 | latent (recon-trained) | reconstruction + reward | value grads in imagination | learn-in-imagination |
| **TD-MPC / TD-MPC2** | **latent, decoder-free** | **value/TD + consistency** | MPC in latent (reward+value) | **decoder-free latent** ← tightest |
| IRIS | reconstructed tokens | reconstruction | autoregressive rollout | token-sequence substrate |
| Genie | generated frames | reconstruction | — (interactive) | **unsupervised action ontology** |
| DIAMOND | pixels (diffusion) | reconstruction | in-model RL | the anti-latent argument |
| GameNGen / Sora / UniSim | pixels/latent-decoded | reconstruction | — / rollout | action-conditioned generation |
| **WebDreamer / WMA / RAP** | **NL tokens (LLM)** | n/a (pretrained) | **score imagined outcomes** | **action-outcome lookahead** ← task twin |
| MuZero | latent, decoder-free | **value equivalence** | MCTS in latent | decoder-free (but value-grounded) |
| Othello-GPT / Vafa et al. | (probing, not a WM) | — | — | **how to evaluate "has a WM"** |

## Where terminal-JEPA actually sits

The distinctive corner — **non-reconstructive latent prediction · anti-collapse without reward · plan by latent distance** — is shared *in spirit only* by the TD-MPC/PlaNet/Dreamer family, and even they ground the latent with reward/value and plan by return rather than latent distance. The **"plan by latent distance to a goal"** component is genuinely distinctive to LeCun's formulation and is **not directly matched by any surveyed system**. Meanwhile the dominant recent momentum (Genie, Sora, GameNGen, DIAMOND, UniSim) runs the *opposite*, generative direction — and DIAMOND explicitly argues latent compression throws away task-relevant detail. That tension is terminal-JEPA's core empirical question, not a settled point.

## Open questions this round surfaced (for the backlog)

1. **Value-equivalence vs. anti-collapse.** Is JEPA's reconstruction-free latent best understood as value-equivalence *without a reward signal*? terminal-JEPA has no dense reward, so it's forced onto a purely self-supervised anti-collapse objective — MuZero's route is closed. Is that a strength (task-agnostic) or a handicap (no grounding signal)?
2. **The DIAMOND challenge, head-to-head.** Does non-reconstructive latent prediction actually beat token-generation for *action-outcome prediction on unseen systems*? The generative camp bet the other way and cites concrete wins. Is there any head-to-head on *this* axis, or does terminal-JEPA have to produce it?
3. **Genie's discovered action ontology.** Genie learns 8 latent actions unsupervised. terminal-JEPA has an expanding, non-fixed toolset — could a Genie-style latent-action model *discover* the command ontology from exploration traces rather than fixing it?
4. **WebDreamer as the honest baseline.** Under what data/compute regime does a trained latent shell-state predictor beat simply prompting an LLM to simulate command outcomes in text? If it never does, that's the finding.
5. **Probing terminal-JEPA.** Reuse the Othello-GPT / Vafa-et-al. methodology: does the shell-state predictor internally represent filesystem/system state, or just surface next-token statistics? Next-token accuracy alone won't answer this.

## Sources & verification status

Verified 3-0 this round (primary arXiv unless noted): PlaNet [1811.04551], Dreamer [1912.01603], DreamerV3 [2301.04104], DayDreamer [2206.14176], TD-MPC [2203.04955], TD-MPC2 [2310.16828], IRIS [2209.00588], Genie [2402.15391], DIAMOND [2405.12399], GameNGen [2408.14837], Sora (OpenAI report, corroborated-not-fetched), WebDreamer [2411.06559].

Title/venue confirmed, characterization **not** vote-verified this round (○): WMA [2410.13232], RAP [2305.14992], AutoManual [2405.16247], UniSim [2310.06114], PWM [2407.02466], 1X World Model [1xgpt / 2510.07092], MuZero [1911.08265], Value Equivalence [2011.03506], Othello-GPT [2210.13382], Vafa et al. [2406.03689]. Named but unverified: NVIDIA Cosmos, Genie 2/3, TWM.

*One claim was refuted and excluded: that PlaNet's deterministic+stochastic RSSM split was motivated specifically by multi-step reward prediction (1-2 vote). One citation note: DayDreamer is arXiv:2206.14176 (an upstream source mis-cited 2203.00580; content correct).*
