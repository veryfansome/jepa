# Plan-probe v1 Stage 2 (live latent-MPC navigation) — pre-registration

Committed before the full scoring runs. Disclosure: the episode/scoring mechanics were
iterated during a smoke phase on small fedora-only goal subsets (5 depth-4 + 12 depth-2
goals) — that exploration selected `--dist cos` (measured: contrastive-era predictions carry
off-manifold magnitudes, in-dist pred-vs-true sqL2 1859 > random-pair 1478, and magnitude
artifacts do not cancel when comparing different predictions) and `--score-mode min`
(imagined cd-obs OR imagined ls-obs proximity), and produced the decision-0 geometry
diagnostic (even GROUND-TRUTH cd-obs embeddings carry no cosine gradient toward deep goals;
command-TEXT space does). Smoke numbers are exploratory and are superseded by this run.

## Protocol (fixed)

`realenv/plan_env.py` @ this commit. Goals: `data/plangoals-v1` (50/image, depth 2–4, seed 0,
`ls -la` oracle views). Inner = fedora+mariadb (100 goals), final = rockylinux+httpd (100
goals), scored ONCE after inner. Episodes: opener (seeded uname+cat-config) + real `ls -la`
probes; agent-visible candidates (child dirs parsed from the real listing, + `..` off-root);
H = min(depth+3, 7) cd-decisions. Planners on identical candidate sets: wm (Stage-1-certified
checkpoints, horizon-2 imagination, PRIMARY write-policy = write, withhold reported), masked
(self-only twin), lexical (cosine of cd-command embedding to goal), random. Seeds 0,1,2 for
all planners. Distance: cosine; score: min(imagined-cd, imagined-ls). Environment: local
(MPS inference + local Docker); all planners share it.

## Pre-registered claims

- **C3 (planning):** success(wm) − max(success(random), success(lexical)) ≥ **+0.10** on inner
  (3-seed mean).
- **C4 (history-driven):** success(wm) − success(masked) ≥ **+0.10** on inner.
- Validity: random ≪ lexical (the echo navigator must clearly beat random, else the goal set
  is degenerate); wm successes' mean decisions ≤ oracle depth + 1 slack reported.

## Pre-registered reporting (regardless of outcome)

Depth-stratified success (2/3/4); fidelity-by-decision (imagined-vs-real ls distance);
write vs withhold; the decision-0 geometry table (imagined-cd / ground-truth-cd /
command-text cosines to goal) as the mechanism artifact. Final-test scored once, all
planners, identical protocol, whatever the inner outcome — the probe's value is the
measurement. A C3 failure with the measured flat-field mechanism routes to: path-structured
observation/goal representations (the R8 pathkey-stream thread; v2 design) and
goal-conditioned readout lenses for future evolve rounds — recorded as stats, not verdicts.
