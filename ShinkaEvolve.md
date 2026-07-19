# ShinkaEvolve: LLM-Driven Evolutionary Program Search

This document explains ShinkaEvolve — what it is, how the evolutionary loop works, why it's built the way it is, and how it fits into the LLM-driven program-search family that runs from FunSearch through AlphaEvolve. It assumes you know roughly how LLMs and standard optimization work, but nothing about evolutionary algorithms or quality-diversity search. Every claim is cited to a specific paper, blog, or repo so you can drill into details. Where a number appears only in a blog/press writeup and not the paper's main text, or where a source flagged something as uncertain, that is called out at the point of use.

## 1. The one-paragraph version

ShinkaEvolve is an open-source framework that uses an ensemble of LLMs as **mutation operators** in an evolutionary loop over source code, searching for programs that maximize an automated scalar fitness function ([arXiv 2509.19349](https://arxiv.org/abs/2509.19349); [Sakana blog](https://sakana.ai/shinka-evolve/)). It keeps an archive of previously evaluated programs with their scores; each generation it samples a parent program from the archive, asks an LLM to propose an edit (a diff, a full rewrite, or a crossover of two parents), evaluates the resulting candidate, and inserts it back — the same "LLM proposes, evaluator scores, best survive" loop pioneered by [FunSearch](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/) and generalized by DeepMind's [AlphaEvolve](https://arxiv.org/abs/2506.13131). Its decisive contribution over those predecessors is **sample efficiency**: where FunSearch used on the order of 10⁶ LLM samples and AlphaEvolve "thousands," ShinkaEvolve finds a new state-of-the-art circle-packing solution in **~150 evaluations** ([abs](https://arxiv.org/abs/2509.19349); [blog](https://sakana.ai/shinka-evolve/)). It gets there with three mechanisms — balanced parent sampling, code-novelty rejection sampling, and a bandit-based LLM-ensemble selector — and, unlike the closed, API-gated AlphaEvolve, it ships under Apache-2.0 ([GitHub](https://github.com/SakanaAI/ShinkaEvolve)).

## 2. Why this exists: the lineage from FunSearch to AlphaEvolve to ShinkaEvolve

The whole family rests on one shift: instead of using an LLM to emit an *answer*, use it to write and iteratively improve a *program* that an automated evaluator then scores. That reframing is what [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) introduced, and it is why these systems can discover things their base LLM does not "know."

- **FunSearch (2023, DeepMind, Nature 625:468–475).** Pairs "a pre-trained LLM with an automated evaluator" in an iterative loop; the LLM proposes new versions of a `priority` function inside a fixed program skeleton, and the evaluator runs and scores them ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/)). The key framing is that it evolves **programs, not solutions** — you get an interpretable function, not just a number. It used Codey (built on PaLM 2), an island-based database with periodic resets, best-shot prompting with **k = 2** in-context exemplars, and Boltzmann/temperature cluster sampling; results were obtained "using a total number of samples on the order of 10⁶," parallelized over "typically 15 samplers and 150 CPU evaluators" ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/)). It found a cap set of size **512** in dimension n = 8 (previous best known 496) and bin-packing heuristics beating best-fit ([blog](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)).
- **AlphaEvolve (2025, DeepMind).** Generalizes FunSearch from a single evolvable function to whole code files in any language, edited via **SEARCH/REPLACE diffs**, driven by a Gemini 2.0 Flash + Gemini 2.0 Pro ensemble, over an evolutionary database "inspired by a combination of the MAP elites algorithm and island-based population models" ([white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf); [arXiv 2506.13131](https://arxiv.org/abs/2506.13131)). It improved the SOTA for 14 matrix-multiplication algorithms (48 scalar multiplications for 4×4 complex matrices, the first improvement over Strassen in this setting in 56 years), matched best-known constructions on ~75% of 50+ open math problems and surpassed SOTA on ~20%, and recovered on average 0.7% of Google's fleet-wide compute via an evolved Borg scheduling heuristic ([white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)). AlphaEvolve's own Table 1 frames the cost delta against its predecessor: FunSearch used "millions of LLM samples," AlphaEvolve "thousands suffice."
- **ShinkaEvolve (2025, Sakana AI).** Takes aim at exactly that cost. The blog states prior LLM-evolution systems "often require thousands of attempts to find good solutions," making them "slow, expensive, and inaccessible," and positions ShinkaEvolve as the "open-source framework leveraging large language models to advance scientific discovery with state-of-the-art performance and unprecedented efficiency" ([blog](https://sakana.ai/shinka-evolve/); [abs](https://arxiv.org/abs/2509.19349)). It pushes the sample count from thousands-to-millions down to **hundreds** while remaining model-agnostic and open ([arXiv 2509.19349](https://arxiv.org/abs/2509.19349)). ("Shinka" is Japanese for "evolution.")

The sample-efficiency problem is the through-line: each fitness evaluation can be expensive (AlphaEvolve budgets "on the order of 100 compute-hours to evaluate any new solution" [white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)), so the number of evaluations needed to reach a good program is the dominant cost, and the thing ShinkaEvolve's three innovations attack.

## 3. The core loop

Using ShinkaEvolve's own pipeline (Sec. 3.1–3.3 of [arXiv 2509.19349](https://arxiv.org/html/2509.19349v1)) as the reference design:

```
        ┌──────────────────────────────────────────────────────┐
        │   PROGRAM ARCHIVE (per-island, fixed size ~40)        │
        │   evaluated programs + fitness + text feedback        │
        └───────┬───────────────────────────────▲──────────────┘
                │ sample island (uniform)        │ insert if valid
                │ then parent + inspirations      │ (novel + scored)
                ▼                                 │
        ┌───────────────┐                 ┌───────┴────────┐
        │ PARENT +      │                 │  EVALUATE       │  ← domain-supplied
        │ INSPIRATIONS  │                 │  fitness F(P)   │    evaluate() returns
        │ (top-K +      │                 │  + public       │    scalar combined_score
        │  random)      │                 │  metrics + text │    + metrics + feedback
        └───────┬───────┘                 └───────▲────────┘
                │ context to LLM                   │ execute (local/Slurm)
                ▼                                  │
        ┌───────────────┐   candidate    ┌────────┴────────┐
        │ LLM MUTATION  │───────────────►│ NOVELTY REJECT  │  ← embed EVOLVE-BLOCKs,
        │ diff / full / │   (bandit-     │ cosine sim gate │    cosine sim; if > η
        │ crossover     │    selected    │ + LLM-as-judge  │    query LLM judge,
        └───────────────┘    LLM)        └─────────────────┘    reject near-dups
                ▲                                  │  before evaluation
                │ update bandit reward ◄───────────┘
                │ (relative improvement, normalized)
```

The loop walks through six boxed stages:

1. **Program archive.** A fixed-size database of previously evaluated programs with their fitness scores, split into independent **island subpopulations** each seeded from the initial solution; islands evolve in parallel to avoid premature convergence, with occasional **migration** between them (each island's best performer is exempt from migration) (Sec. 3.1, [HTML](https://arxiv.org/html/2509.19349v1)).
2. **Parent + inspiration sampling.** Sample an island uniformly, then sample a **primary parent** plus **inspiration programs** (top-K performers + random archive members) from that island (Sec. 3.1).
3. **LLM mutation.** A bandit-selected LLM proposes a candidate as a diff edit, a full rewrite, or a crossover of two archive programs (Sec. 3.2).
4. **Novelty rejection.** Before the (expensive) evaluation, embed the candidate's mutable code and reject near-duplicates (Sec. 3.2).
5. **Evaluate.** Run the candidate; collect scalar fitness, public metrics, and textual feedback (Sec. 3.3).
6. **Insert + update.** Register valid, novel, scored candidates back into the archive; update the bandit's LLM-selection distribution from the observed improvement; periodically append meta-scratchpad insights to future prompts (Sec. 3.3).

Mutable vs. immutable code is marked with `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` markers — only code inside those blocks is evolved, and only those blocks are embedded for the novelty check ([HTML](https://arxiv.org/html/2509.19349v1)). A **meta-scratchpad** summarizes successful programs every T generations, extracts recurring optimization patterns, and appends "recommendations" to mutation prompts (Sec. 3.3, Fig. 4).

## 4. ShinkaEvolve's three innovations

The abstract names exactly three ([abs](https://arxiv.org/abs/2509.19349)): balanced parent sampling, code-novelty rejection sampling, and bandit-based LLM ensemble selection. Each is a distinct answer to "how do we spend a small evaluation budget well."

### 4.1 Balanced parent sampling (exploration vs. exploitation)

The archive offers two selectable strategies for picking which program to mutate (Sec. 3.1, [HTML](https://arxiv.org/html/2509.19349v1)):

- **Power-law sampling** by fitness rank r_i (best = rank 1): p_i = r_i^(−α) / Σ_j r_j^(−α). Setting α = 0 gives uniform sampling (pure exploration); α → ∞ gives greedy hill-climbing (pure exploitation).
- **Weighted sampling (performance + novelty).** It combines a sigmoid-scaled performance term against the median-fitness baseline α₀ = median{F(P₁),…,F(Pₙ)} — s_i = σ(λ·(F(P_i) − α₀)) with σ(x)=1/(1+e^(−x)) — with a novelty term h_i = 1/(1 + N(P_i)) that penalizes parents that have already produced many offspring N(P_i). The combined weight is w_i = s_i · h_i and the sampling probability is p_i = w_i / Σ_j w_j. λ sets the selection pressure; the circle-packing config uses **λ = 10.0** (Table 1 / App. B.1).

The blog frames this as an "intelligent program parent sampling technique that balances exploiting known good solutions and exploring new ideas" ([blog](https://sakana.ai/shinka-evolve/)). This contrasts with AlphaEvolve's MAP-Elites + islands database and FunSearch's island-reset + Boltzmann cluster sampling — same goal (keep the population diverse while pursuing the best), different mechanism.

### 4.2 Novelty-based rejection sampling (code-embedding dedup)

Wasting an evaluation on a near-duplicate of something already scored is pure cost. After the LLM proposes a candidate, ShinkaEvolve **embeds the mutable code blocks** with OpenAI `text-embedding-3-small` (the confirmed repo default, configurable) and computes **cosine similarity** against programs in the island subpopulation ([HTML](https://arxiv.org/html/2509.19349v1); [GitHub](https://github.com/SakanaAI/ShinkaEvolve)). If the max similarity exceeds a threshold — the paper reports **η = 0.95** (the shipped repo default `code_embed_sim_threshold` is 0.99) — the candidate is treated as a likely near-duplicate and an **LLM-as-novelty-judge** is queried to decide whether it is "meaningfully different"; if not, it is rejected **before** evaluation. Invalid or duplicate patches trigger Reflexion-style feedback and re-sampling up to `max_patch_resamples` ([HTML](https://arxiv.org/html/2509.19349v1)). Neither AlphaEvolve nor FunSearch has this explicit dedup gate. The ablation (below) shows the embedding gate does most of the work and the LLM judge adds only marginal further gains.

### 4.3 Bandit-based LLM ensemble selection

Rather than a fixed model split (AlphaEvolve's Flash + Pro) or a single model (FunSearch's Codey), ShinkaEvolve treats "which LLM should propose the next mutation?" as a multi-armed bandit and solves it with **UCB1** (the repo confirms `"ucb"` selection with a `cost_aware_coef: 0.5` parameter) (Sec. 3.3, [HTML](https://arxiv.org/html/2509.19349v1); [GitHub](https://github.com/SakanaAI/ShinkaEvolve)). Each LLM carries a visit count and an expected-score estimate; the model to query is chosen by UCB1's mean-reward-plus-exploration-bonus rule.

The reward is **relative improvement**, not absolute fitness: r_i^u = exp(max(r_i − r_i^b, 0)) − 1, where the baseline r_i^b = max(parent fitness, initial-program fitness). The paper states the exp(·) and max(·,0) "precisely promote LLMs able to come up with bold, high-risk, high-reward mutations," and rewards are normalized using tracked statistics over observed rewards to give "invariance to the fitness scale" across domains ([HTML](https://arxiv.org/html/2509.19349v1)). *(Unverified: the paper names UCB1 but the fetched HTML did not transcribe the explicit c·√(ln N / n_i) exploration-bonus formula; the reward equation above is high-confidence, the exact acquisition rule should be checked against the PDF/appendix if it is load-bearing.)* The ablation reports the bandit ensemble "significantly outperforms both single LLM and fixed ensemble" (Fig. 9).

## 5. Results

All four headline tasks run on tiny evaluation budgets (note eval count ≈ generations, since roughly one program is produced per generation); per-task budgets are in App. B ([HTML](https://arxiv.org/html/2509.19349v1)).

- **Circle packing (n = 26 in the unit square) — the flagship.** ShinkaEvolve reaches a sum-of-radii ≈ **2.635983283**, beating **AlphaEvolve's 2.63586276**, using **only ~150 generations/samples** vs AlphaEvolve's "thousands"; it stays ahead even under exact verification (~2.6360 > AlphaEvolve's 2.63586) ([HTML](https://arxiv.org/html/2509.19349v1); [blog](https://sakana.ai/shinka-evolve/)). The discovered algorithm is a golden-angle spiral initialization + gradient refinement + simulated annealing; baselines compared were AlphaEvolve, OpenEvolve, and LLM4AD. Budget: ~150 generations, 2 islands, archive 40, elite ratio 0.3, weighted sampling with λ=10, an ensemble of claude-sonnet-4 / o4-mini / gpt-4.1 / gpt-4.1-mini / gpt-4.1-nano at temperatures [0.0, 0.5, 1.0] (Table 1).
- **MoE load-balancing loss (vs DeepSeek "Global LBL").** After only **30 generations** (capped by pretraining cost), ShinkaEvolve discovered a new load-balancing loss (App. B.4 / Eq. 6): L_LBL = N_E·(1/L)Σ_ℓ Σ_i f_{ℓ,i} P_{ℓ,i} + (0.1/L) Σ_ℓ s(P_ℓ) Σ_i max(0, τ − f_{ℓ,i}), with τ = 0.064/N_E and s(P_ℓ) = 0.5 + (1 − H(P_ℓ)/log N_E). It was evolved on a 556M-param (82M active) model over 2B fineweb tokens and validated on a 2.7B-param (404M active) model over ~30B tokens across 7 benchmarks (CommonSenseQA, HellaSwag, OpenBookQA, PIQA, SIQA, WinoGrande, ARC). *Source discrepancy (flag): the blog and MarkTechPost state exact figures — "-5.81%" less inefficient token routing and "+1.73% on average" higher task performance ([blog](https://sakana.ai/shinka-evolve/)) — but the paper main text shows this graphically (Fig. 8) with qualitative wording ("consistent edge... growing larger with the λ coefficient") and no in-text percentage. Attribute the −5.81% / +1.73% to Sakana's reported headline numbers, not a specific paper table.*
- **AIME math-reasoning agent scaffold.** Over **75 generations** (30 problems/eval, 3 runs each), it discovered a 3-stage scaffold ("diverse expert personas, critical peer review") operating under a 10-LLM-query cap, with the Pareto-optimal solution using **~7 queries/problem** (Fig. 6). It generalizes across 2023/2024/2025 AIME (largest gains on the unseen 2025 set) and transfers to gpt-4.1-mini, gpt-4.1, and o4-mini ([HTML](https://arxiv.org/html/2509.19349v1); [blog](https://sakana.ai/shinka-evolve/)).
- **AtCoder ALE-Bench (competitive programming).** Over **50 generations** on 10 LITE problems, ShinkaEvolve improves ALE-Agent solutions by **~2.3% on average** across the 10 tasks; on **ahc039** the combined ShinkaEvolve+ALE-Agent solution would counterfactually have reached **2nd place** (performance 2880 → 3140; the blog frames it as 5th → 2nd). A private-test overfitting check moved a "negligible" 1923.5 → 1927.0 on top-5 public→private; the discovered operators include advanced caching and a "targeted edge move" ([HTML](https://arxiv.org/html/2509.19349v1); [blog](https://sakana.ai/shinka-evolve/)).

**Ablation evidence (circle packing, Fig. 9, Sec. 5).** Each innovation is isolated:

- *Parent selection:* weighted sampling > hill-climbing (fast then plateaus) > random search (worst).
- *LLM ensembling:* bandit ensemble >> fixed ensemble > single LLM.
- *Novelty rejection:* embedding-based rejection (sim > 0.95) gives "substantial" gains; adding the LLM-as-novelty-judge gives only "marginal" further gains — i.e., the embedding similarity is already an effective near-duplicate proxy ([HTML](https://arxiv.org/html/2509.19349v1)).

## 6. Relationship to AlphaEvolve and FunSearch, and the wider lineage

What is **shared** across all three: an LLM proposes code edits (diffs), an automated evaluator returns scalars, and an evolutionary database of scored programs balances exploitation of the best against diversity ([FunSearch PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/); [AlphaEvolve white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf); [ShinkaEvolve HTML](https://arxiv.org/html/2509.19349v1)). AlphaEvolve already generalized FunSearch's single evolvable `priority` function to whole codebases via SEARCH/REPLACE diffs and an LLM ensemble ([arXiv 2506.13131](https://arxiv.org/abs/2506.13131)).

What ShinkaEvolve **adds** on top of that shared skeleton: the three innovations of §4 (fitness+novelty-weighted parent sampling, embedding-based novelty rejection, and a UCB1 bandit over the LLM ensemble), all aimed at driving the sample count from thousands-to-millions down to hundreds ([abs](https://arxiv.org/abs/2509.19349)). Neither predecessor has the explicit embedding-dedup gate or the improvement-reward bandit.

The **openness difference is concrete.** AlphaEvolve is a Google-internal, Gemini-2.0-only, API-gated system described in a white paper, with no open-source release ([white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)). FunSearch's method and framework code are public ([github.com/google-deepmind/funsearch](https://github.com/google-deepmind/funsearch)) but it depends on Codey/PaLM 2. ShinkaEvolve ships under **Apache-2.0** and is model-agnostic across GPT, Gemini, Claude, and DeepSeek ([GitHub](https://github.com/SakanaAI/ShinkaEvolve)). The community reimplementation **OpenEvolve** ("Open-source implementation of AlphaEvolve," MAP-Elites + island-based population + LLM ensemble) is a separate open effort in the same space ([github.com/codelion/openevolve](https://github.com/codelion/openevolve)).

The **wider lineage** ShinkaEvolve draws on and sits within:

- **ELM — "Evolution through Large Models"** (2022, [arXiv 2206.08896](https://arxiv.org/abs/2206.08896), Lehman, Stanley et al.; predates Sakana). The origin of "LLMs trained to generate code can vastly improve the effectiveness of mutation operators applied to programs" — the LLM-as-intelligent-mutation-operator idea, demonstrated on Sodarace walking-robot programs, with **MAP-Elites** as the archive. Open-sourced as [OpenELM](https://github.com/CarperAI/OpenELM) (CarperAI).
- **MAP-Elites — "Illuminating search spaces by mapping elites"** (2015, [arXiv 1504.04909](https://arxiv.org/abs/1504.04909), Mouret & Clune). The quality-diversity root: keep "a set of high-performing, yet diverse solutions" rather than a single optimum — the archive-of-diverse-elites idea that ShinkaEvolve's island database (and AlphaEvolve's MAP-Elites+islands) echoes.
- **Eureka** (2023, [arXiv 2310.12931](https://arxiv.org/abs/2310.12931), NVIDIA-led). Uses GPT-4 for "evolutionary optimization over reward code" for RL, outperforming human-engineered rewards on 83% of 29 environments (average +52% normalized) and training a Shadow Hand to spin a pen — LLM×evolution applied to reward functions.
- **Promptbreeder** (2023, [arXiv 2309.16797](https://arxiv.org/abs/2309.16797), DeepMind). Self-referential prompt evolution: an LLM mutates a population of task-prompts *and* the mutation-prompts themselves evolve; the same "evolve the prompts too" idea ShinkaEvolve's optional prompt-evolution knob exposes. (Compare [EvoPrompt](https://arxiv.org/abs/2309.08532), Microsoft, ICLR 2024, which connects LLMs with EAs for discrete prompt optimization.)
- **Sakana's own evolution/agent program**, which ShinkaEvolve's README explicitly cites as inspiration (AI Scientist, AlphaEvolve, Darwin Gödel Machine, [GitHub](https://github.com/SakanaAI/ShinkaEvolve)):
  - **Evolutionary Model Merge** (2024, [arXiv 2403.13187](https://arxiv.org/abs/2403.13187)) — evolution auto-discovers model-merging recipes "in both parameter space and data flow space."
  - **The AI Scientist v1** (2024, [arXiv 2408.06292](https://arxiv.org/abs/2408.06292)) — end-to-end automated research (idea → code → experiments → paper → review) at ~$15/paper — and **v2** (2025, [arXiv 2504.08066](https://arxiv.org/abs/2504.08066)), agentic tree search that produced the first fully AI-generated paper to pass workshop peer review.
  - **Darwin Gödel Machine** (2025, [arXiv 2505.22954](https://arxiv.org/abs/2505.22954), a Sakana-affiliated collaboration) — a self-improving coding agent that maintains an archive of agents and evolves improved variants (SWE-bench 20.0% → 50.0%; Polyglot 14.2% → 30.7%), notable for learning to game its own evaluation.

## 7. The reference implementation

The repo is `github.com/SakanaAI/ShinkaEvolve` (Apache-2.0, package imported as `import shinka`; install `pip install shinka-evolve`, Python ≥3.10). All facts below are tied to source files.

**Defining a task** requires two files in a task directory ([README](https://github.com/SakanaAI/ShinkaEvolve)):

1. **`initial.<ext>` — the seed program.** The evolvable region is delimited by `# EVOLVE-BLOCK-START` … `# EVOLVE-BLOCK-END`; everything outside is immutable. It must define the experiment entry function named by `experiment_fn_name`. In `examples/circle_packing/initial.py` that function is `run_packing()` returning `(centers, radii, sum_radii)`. Supported languages (from `shinka/cli/run.py`) include Python, Julia, Go, Verilog, Rust, Swift, C++, CUDA, JSON, and Fortran.
2. **`evaluate.py` — the fitness contract.** It calls `shinka.core.run_shinka_eval(...)`. The verified signature (`shinka/core/wrap_eval.py`) is `run_shinka_eval(program_path, results_dir, experiment_fn_name, num_runs, get_experiment_kwargs=None, aggregate_metrics_fn=None, validate_fn=None, ...) -> Tuple[Dict, bool, Optional[str]]` returning `(metrics, correct, err)`. The aggregated metrics dict **must contain a scalar `"combined_score"`** (which is maximized). Optional keys: `"public"` (shinka-visible metrics), `"private"` (hidden), `"extra_data"` (pickled), and `"text_feedback"` (a string fed back to the LLM). The circle-packing example uses `experiment_fn_name="run_packing"`, `num_runs=1`, and a `validate_fn` that hard-codes n=26 and checks unit-square containment + pairwise non-overlap.

**Key knobs** live in three dataclasses (verified defaults from source):

- `EvolutionConfig` (`shinka/core/config.py`): `num_generations=50`; `patch_types=["diff","full","cross"]` with `patch_type_probs=[0.6,0.3,0.1]` (diff = targeted edit, full = rewrite, cross = crossover); `llm_dynamic_selection="ucb"` with `{"cost_aware_coef":0.5}`; `llm_kwargs={"temperatures":[0.0,0.5,1.0],"max_tokens":16384}`; `embedding_model="text-embedding-3-small"`; `code_embed_sim_threshold=0.99` (novelty gate); `max_novelty_attempts=3`; `max_patch_resamples=3`; `meta_rec_interval=10`; `max_api_costs=None` (USD budget cap); `evolve_prompts=False`.
- `DatabaseConfig` (`shinka/database/dbase.py`): `num_islands=2`; `archive_size=40`; `elite_selection_ratio=0.3`; `migration_interval=10`; `parent_selection_strategy="weighted"` (or `power_law` / `beam_search`); `parent_selection_lambda=10.0`; `exploitation_ratio=0.2`; `island_selection_strategy="uniform"`.
- `LocalJobConfig` (`shinka/launch/scheduler.py`): `eval_program_path="evaluate.py"`, optional `activate_script`/`conda_env`; Slurm variants (`SlurmDockerJobConfig`, etc.) add `image`/`partition`/`gpus`/`mem`. Concurrency (`max_evaluation_jobs`, `max_proposal_jobs`, `max_db_workers`) lives on the runner.

*Note: model-id strings in the shipped configs (e.g. `gpt-5-mini`, `gemini-3.1-pro-preview`, `us.anthropic.claude-sonnet-4-6-v1:0`) are verbatim from the current repo and reported as-is.*

**How to run it:**

- Hydra: `shinka_launch variant=circle_packing_example`, or compose groups: `shinka_launch task=circle_packing database=island_large evolution=small_budget cluster=local evo_config.num_generations=20`.
- Agent CLI (no Hydra): `shinka_run --task-dir examples/circle_packing --results_dir results/circle --num_generations 20`, with overrides via `--set evo.llm_models='[...]'`.
- Python API: construct `EvolutionConfig`, `DatabaseConfig`, `LocalJobConfig`, pass to `ShinkaEvolveRunner(...).run()` (see `examples/circle_packing/run_evo.py`).
- WebUI: `shinka_visualize --port 8888 --open` shows real-time progress and genealogy/lineage trees.

Shipped example tasks (`examples/`): circle packing (n=26), a 2048 policy, Julia prime counting, Fortran heat diffusion, Wolfram GCD-sum, Go Collatz steps, an LLM-judged novelty generator, a headless-agent sine approximation, and RTLLM Verilog PPA. **There is no MoE / load-balancing-loss example in the repo — that is a paper result only** ([GitHub](https://github.com/SakanaAI/ShinkaEvolve)).

## 8. Recurring principles: what makes these systems work

Design principles that show up across FunSearch, AlphaEvolve, and ShinkaEvolve, with the papers that establish them:

1. **The automated evaluator is the bottleneck, not the LLM.** The whole family only works when a program's quality can be scored automatically and cheaply; FunSearch requires "an evaluation procedure plus a seed program" ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/)), AlphaEvolve an `evaluate` function returning a dictionary of scalars ([white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)), and ShinkaEvolve a `combined_score` scalar ([GitHub](https://github.com/SakanaAI/ShinkaEvolve)). Sample efficiency matters *because* each evaluation is expensive (AlphaEvolve, ~100 compute-hours per solution [white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)).
2. **Diffs over rewrites.** Editing existing code via SEARCH/REPLACE-style diffs (AlphaEvolve, [white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf); ShinkaEvolve's `diff` patch type at 0.6 probability, [config](https://github.com/SakanaAI/ShinkaEvolve)) makes small, evaluable increments the default and reserves full rewrites for short code or a fresh direction.
3. **Archive diversity beats a single elite.** All three keep a population, not a best-so-far — islands + resets in FunSearch ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/)), MAP-Elites+islands in AlphaEvolve ([white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)), islands + novelty-weighted parent sampling + embedding dedup in ShinkaEvolve ([HTML](https://arxiv.org/html/2509.19349v1)) — traceable to MAP-Elites ([1504.04909](https://arxiv.org/abs/1504.04909)).
4. **An ensemble of frontier LLMs, not one model.** AlphaEvolve uses fast+capable (Flash+Pro) ([white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)); ShinkaEvolve's bandit adaptively allocates queries to whichever model is currently producing improvements ([HTML](https://arxiv.org/html/2509.19349v1)).
5. **Sample efficiency is the frontier.** The measured trajectory is FunSearch ~10⁶ → AlphaEvolve "thousands" → ShinkaEvolve ~150 for a comparable circle-packing result ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/); [white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf); [blog](https://sakana.ai/shinka-evolve/)) — every ShinkaEvolve innovation is in service of this number.

## 9. Open problems and limitations

- **Fitness must be cheap and is gameable if narrow.** The entire loop optimizes exactly the scalar you hand it; a narrow or exploitable fitness gets exploited. ShinkaEvolve's own guardrail is illustrative — the AtCoder run adds a public→private overfitting check (1923.5 → 1927.0) precisely because public-test fitness can be over-fit ([HTML](https://arxiv.org/html/2509.19349v1)) — and the Darwin Gödel Machine in the same lineage learned to falsify its own test results ([2505.22954](https://arxiv.org/abs/2505.22954)).
- **You need a strong evaluator, and it caps applicability.** No cheap automated scorer, no evolution. The MoE task was capped at ~30 generations *by pretraining cost* ([HTML](https://arxiv.org/html/2509.19349v1)) — a direct example of the evaluator's cost bounding the search.
- **Compute and cost.** Even at hundreds of evaluations, each candidate is a full run; the framework exposes `max_api_costs` and a `cost_aware_coef` on the bandit precisely because LLM-plus-evaluation spend is the binding constraint ([GitHub](https://github.com/SakanaAI/ShinkaEvolve)).
- **Generalization of the discovered artifact is not guaranteed.** The paper flags possible training-data contamination/saturation on older (2023) AIME benchmarks ([HTML](https://arxiv.org/html/2509.19349v1)); an artifact that wins on the evaluation distribution has not been shown to transfer off it.
- *Unverified: the fetched paper HTML did not contain an enumerated dedicated "Limitations" section, and the blog omits failure-mode and wall-clock analysis; treat the above as the caveats surfacing in the fetched text, and check the paper's conclusion directly for an exhaustive list ([HTML](https://arxiv.org/html/2509.19349v1); [blog](https://sakana.ai/shinka-evolve/)).*

## 10. Applying ShinkaEvolve to terminal-JEPA

The terminal-JEPA program has an evolvable program (a world-model training recipe), an automated fitness (held-out-image retrieval), and a small trainable model on top of a cached frozen encoder — exactly the shape ShinkaEvolve is efficient at. The asset to push is the R4 model in `terminal-jepa/realenv/seq_worldmodel.py`: a causal transformer (`d=192, layers=4, heads=4`) over the interleaved `cmd_0,obs_0,cmd_1,obs_1,…` stream of frozen ModernBERT embeddings, whose `cmd`-position hidden states predict the standardized `z_obs[t]` under MSE (`train_model`, `aux="jepa"`). Its headline is `content_retrieval` top-1 on held-out images (fedora/rocky/mariadb/httpd): **WM ≈0.56** on ls+cat vs lexical `retrieve_by_cmd_baseline` 0.25, history-free `CmdOnlyMLP` 0.21, `copy_prev` 0.10, `predict_mean` 0.02, with `run_history_ablation` showing +0.262 content-verb gain from attending over history.

**The mutable program (start with Target 1).** Wrap the training recipe in `SeqWorldModel.__init__` + `train_model` in `EVOLVE-BLOCK` markers: `d, layers, heads, dropout`; `steps, bs, lr, weight_decay`, grad-clip; and especially the **objective** (currently `((pred-tgt)**2).mean()`, which the LLM could extend with a cosine/InfoNCE term aligned to the retrieval geometry the eval actually uses, or an anti-collapse SIGReg term ([LeJEPA, arXiv 2511.08544](https://arxiv.org/abs/2511.08544)) — the anti-collapse mechanisms catalogued in JEPA.md §3.1 are natural mutations here). Keep **immutable**: the frozen encoder + `cached_encode`, `collate`'s interleaving and the no-future-leakage causal mask, `cmd_hidden`'s even/odd convention, and the entire eval path. (Two higher-variance targets exist — the predictor architecture in `encode`/`forward`, and the exploration policy in `collect_docker.py` — but the predictor risks breaking the leakage guardrail, and the exploration policy forces a re-collect+re-encode per variant, so both are slow outer loops, not the inner one.)

**The fitness function.** The primary scalar is held-out **content-verb** top-1 from `content_retrieval(pred, true, verbs, content=("ls","cat"))` — not overall top-1, because `cd`'s observation (`cwd=<target>`) is trivially echoable and inflates the number (per-verb `cd` gain is +0.00). Crucially, fitness should be a **margin**, not an absolute: `content_top1(wm) − max(content_top1(retrieve_by_cmd), content_top1(wm_no_history), content_top1(copy_prev))`, all of which `main` already computes. R4's signal partly rides on shared cross-distro filesystem structure that `retrieve_by_cmd` already banks, so a loop that raises absolute top-1 while the lexical baseline rises equally has discovered nothing — the margin is the only fitness worth optimizing (`content_retrieval` in the local file `terminal-jepa/realenv/seq_worldmodel.py`). Optionally add the `run_history_ablation` content-verb gain (full − masked) as a secondary term so the loop is rewarded specifically for *using* history.

**The overfitting risk and the third split (mandatory).** This is the §9 "fitness is gameable if narrow" problem made concrete. There are only two held-out image types' worth of val; selecting variants on that split for hundreds of generations will overfit the four-distro idiosyncrasies and the number stops estimating unseen-systems performance. The mitigation is a **third split — held-out-of-held-out** images the loop never scores against (reserve some current held-out distros, or add fresh distros/databases), used **once at the end** to validate the evolved champion. Inner-loop fitness uses one held-out set; the final claim uses the untouched third set. Keep the generative twin (`run_gen_twin`) as a standing sanity arm on the champion — the JEPA-vs-generative comparison is the actual research bet and must not be silently lost during optimization.

**The eval-cost reality.** One fitness eval is `train_model` (`steps=4000` on MPS) + `train_cmd_only` + `flatten_predictions`/`retrieval` over ~25,660 held-out steps, per seed × 3 seeds — multi-minute to tens-of-minutes per candidate on Apple Silicon. What makes hundreds of these affordable is that **the frozen encoding is already amortized**: `cached_encode` writes `emb-seq-{split}.pt` once, so per-eval cost is only the tiny transformer train + retrieval, not re-encoding 77K steps. This is exactly why ShinkaEvolve's sample efficiency (§5) is the enabling property — random/grid search over the same space would not fit the budget. Use a cheap-proxy inner loop (`--steps 1000 --seeds 0`, subsample with `--limit`), and promote only the top fraction to the full `steps=4000`, 3-seed eval before they enter the population as scored survivors.

**A concrete first experiment.** Evolve Target 1 restricted to objective + optimizer + shape (leave `collate`/mask/eval fixed):

1. **Fitness** = mean over seeds `{0,1,2}` of ls+cat `content_retrieval` top-1 on held-out **minus** `max(retrieve_by_cmd, wm_no_history)` content-verb top-1 — computed directly from the dict `main` returns.
2. **Inner-loop proxy** `--steps 1000 --seeds 0`; champions re-scored at `--steps 4000 --seeds 0,1,2`.
3. **Guardrails as hard filters:** any variant failing `tests/test_seq_worldmodel.py` (no-future-leakage, self-only invariance, `predict_mean` calibration ≈0.016) scores −∞ and is discarded — so the loop cannot "win" by leaking `obs_t` into the `cmd_t` prediction.
4. **Search space** for the LLM to mutate: the loss expression in `train_model`, `lr`, `d/layers/heads`, `dropout`, and the standardization.
5. **Final verdict:** judge the champion **once** on the third split, re-running `run_gen_twin` and `run_history_ablation` on it, so the claim "evolution improved the *world-model margin*, not the val-fit" survives the program's own adversarial bar.

Relevant local files: `/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa/realenv/seq_worldmodel.py` (`SeqWorldModel`, `train_model`, `content_retrieval`, `retrieval`, `run_history_ablation`, `run_gen_twin`, `retrieve_by_cmd_baseline`, `CmdOnlyMLP`, `cached_encode`, `main`); `/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa/realenv/collect_docker.py` (exploration-policy target); `/Users/fanzhu/PyCharmProjects/jepa/README.md` (§ "Phase R — real shell world model").

## 11. Index

The core ShinkaEvolve sources:

| Year | Work | Link |
|---|---|---|
| 2025 | ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution (abstract) | [2509.19349](https://arxiv.org/abs/2509.19349) |
| 2025 | ShinkaEvolve (full HTML) | [HTML](https://arxiv.org/html/2509.19349v1) |
| 2025 | ShinkaEvolve announcement | [Sakana blog](https://sakana.ai/shinka-evolve/) |
| 2025 | ShinkaEvolve reference implementation (Apache-2.0) | [GitHub](https://github.com/SakanaAI/ShinkaEvolve) |

Direct predecessors (LLM×program-search):

| Year | Work | Link |
|---|---|---|
| 2023 | FunSearch: Mathematical discoveries from program search with LLMs (Nature) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/) |
| 2023 | FunSearch framework code | [github.com/google-deepmind/funsearch](https://github.com/google-deepmind/funsearch) |
| 2025 | AlphaEvolve: A coding agent for scientific and algorithmic discovery (abstract) | [2506.13131](https://arxiv.org/abs/2506.13131) |
| 2025 | AlphaEvolve white paper (full PDF) | [PDF](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) |
| 2025 | OpenEvolve: open-source AlphaEvolve reimplementation | [github.com/codelion/openevolve](https://github.com/codelion/openevolve) |

The LLM × evolution lineage:

| Year | Work | Link |
|---|---|---|
| 2015 | MAP-Elites: Illuminating search spaces by mapping elites | [1504.04909](https://arxiv.org/abs/1504.04909) |
| 2022 | ELM: Evolution through Large Models | [2206.08896](https://arxiv.org/abs/2206.08896) |
| 2022 | OpenELM library | [github.com/CarperAI/OpenELM](https://github.com/CarperAI/OpenELM) |
| 2023 | Eureka: Human-Level Reward Design via Coding LLMs | [2310.12931](https://arxiv.org/abs/2310.12931) |
| 2023 | EvoPrompt: LLMs + evolutionary algorithms as prompt optimizers | [2309.08532](https://arxiv.org/abs/2309.08532) |
| 2023 | Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution | [2309.16797](https://arxiv.org/abs/2309.16797) |

Sakana's evolution / agent program:

| Year | Work | Link |
|---|---|---|
| 2024 | Evolutionary Optimization of Model Merging Recipes | [2403.13187](https://arxiv.org/abs/2403.13187) |
| 2024 | The AI Scientist (v1) | [2408.06292](https://arxiv.org/abs/2408.06292) |
| 2025 | The AI Scientist-v2: Agentic Tree Search | [2504.08066](https://arxiv.org/abs/2504.08066) |
| 2025 | Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents | [2505.22954](https://arxiv.org/abs/2505.22954) |

A note on dates and attribution: dates are first arXiv submission dates (or Nature publication year for FunSearch). ShinkaEvolve is Lange, Imajuku, Cetin (Sakana AI). AlphaEvolve and FunSearch are Google DeepMind. ELM/OpenELM (Lehman/Stanley circle, CarperAI), MAP-Elites (Mouret & Clune), Eureka (NVIDIA-led), Promptbreeder (DeepMind), and EvoPrompt (Microsoft) predate or sit outside Sakana; the Darwin Gödel Machine is a Sakana-affiliated collaboration. Where a number is attributed to a blog/press rather than the paper's main text (notably the −5.81% / +1.73% MoE figures), or where the fetched sources did not confirm a detail (the exact UCB1 acquisition formula, an enumerated Limitations section), that is flagged inline.
