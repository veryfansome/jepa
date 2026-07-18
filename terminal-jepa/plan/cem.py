"""Planner A: factored discrete CEM over the adapter interface (plan §5; Track B
tiers — status doc "Path to a working agent").

Completes Track B tier 1: with the oracle adapter, this validates search + cost +
goal exemplars end-to-end (the battery already validated cost/exemplar/rollout in
isolation, finding 15b). The SAME planner runs tiers 2-3 by swapping the adapter
(--adapter tier2/gate2 --ckpt ...), so nothing here is representation-specific.

Mechanics:
- Goal exemplars are rebuilt from the CURRENT state at every replan via
  make_satisfying (plan §5's plan-time construction); cost of a candidate action
  sequence = min over exemplars of adapter.distance(rollout final latent, exemplar).
- CEM maintains per-position independent categoricals over (verb, arg1, arg2) —
  the factored action space, sampled unfiltered (invalid-heavy, the honest §7
  primary condition). Elites (top 10%) refit each factor with smoothing.
- Receding horizon: plan, execute the FIRST action in the true env, replan.
  Success = goal predicate satisfied (ground truth), within --max-steps.
- Baselines: random shooting at the SAME total rollout budget (samples × iters
  one-shot), and the scripted plan_for ceiling (constructive, reported as
  context per §5 — the parser+search analogue for full obs).

Budget note: §7 commits CEM = 300 samples x 30 iterations for the formal Phase-2
RQ3 runs. The tier-1 machinery check defaults to 300 x 10 with CEM and random
shooting matched at the same total budget (3x cheaper, documented here); pass
--iters 30 to run the committed budget.

Usage:
  python3 -m plan.cem --data data/v1 --adapter oracle --episodes 100 --out runs/tier1/cem-oracle.json
"""

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from datagen.policies import plan_for  # noqa: E402
from env import actions, vocab
from env.state import Predicate, make_satisfying
from evals.dynamics import ADAPTERS, load_transitions

VERBS = ["cd", "ls", "cat", "mkdir", "touch", "rm", "cp", "mv", "write"]
ARGS = ([""] + ["/"] + [vocab.path_to_str(p) for p in vocab.DIR_PATHS]
        + [vocab.path_to_str(p) for p in vocab.FILE_PATHS]
        + [f"c{k}" for k in range(vocab.N_CONTENT)])


def _sample_cat(probs, rng):
    r, acc = rng.random(), 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def _refit(probs, elite_ids, damp=0.5, smooth=0.25):
    counts = [smooth * len(probs) ** -0.5] * len(probs)
    for i in elite_ids:
        counts[i] += 1.0
    z = sum(counts)
    return [damp * p + (1 - damp) * c / z for p, c in zip(probs, counts)]


def rollout_cost(adapter, z0, seq, goal_d, step_weight=0.1):
    """final goal distance + step_weight * mean per-step distance. The second term
    front-loads useful actions: garbage prefixes are no-ops in this domain, so a
    final-distance-only cost is indifferent to WHERE in the sequence the useful
    action sits — and receding horizon executes seq[0], which would then usually
    be garbage (measured: 0% episode success despite CEM finding improving
    candidates internally)."""
    z, acc = z0, 0.0
    for a in seq:
        z = adapter.predict(z, a)
        acc += goal_d(z)
    final = goal_d(z)
    return final + step_weight * acc / len(seq), final


def cem_plan(adapter, z0, goal_d, horizon, samples, iters, rng, valid_pool=None):
    """Returns (best sequence, its shaped cost). Factored categoricals per position.
    Elites are STRICT IMPROVERS only (final distance below the start distance): at
    ~1-in-500 improvement rates, top-K elites are otherwise cost-tied noise and
    refitting on them concentrates the sampler on garbage — measured to make CEM
    lose to same-budget random shooting. With no improvers the distribution is left
    untouched (sampling stays uniform, never worse than random shooting)."""
    pv = [[1 / len(VERBS)] * len(VERBS) for _ in range(horizon)]
    p1 = [[1 / len(ARGS)] * len(ARGS) for _ in range(horizon)]
    p2 = [[1 / len(ARGS)] * len(ARGS) for _ in range(horizon)]
    best_seq, best_cost = None, float("inf")
    start_d = goal_d(z0)
    n_elite = max(1, samples // 10)
    for _ in range(iters):
        cands = []
        for _ in range(samples):
            ids = [(_sample_cat(pv[t], rng), _sample_cat(p1[t], rng),
                    _sample_cat(p2[t], rng)) for t in range(horizon)]
            seq = [(VERBS[v], ARGS[a1], ARGS[a2]) for v, a1, a2 in ids]
            if valid_pool is not None:  # position-0 validity filter (plan §5)
                seq[0] = valid_pool[rng.randrange(len(valid_pool))]
            cands.append((*rollout_cost(adapter, z0, seq, goal_d), ids, seq))
        cands.sort(key=lambda c: c[0])
        if cands[0][0] < best_cost:
            best_cost, best_seq = cands[0][0], cands[0][3]
        elites = [c for c in cands if c[1] < start_d][:n_elite]
        if not elites:
            continue
        for t in range(horizon):
            pv[t] = _refit(pv[t], [e[2][t][0] for e in elites])
            p1[t] = _refit(p1[t], [e[2][t][1] for e in elites])
            p2[t] = _refit(p2[t], [e[2][t][2] for e in elites])
    return best_seq, best_cost


def random_shooting_plan(adapter, z0, goal_d, horizon, budget, rng, valid_pool=None):
    """Same rollout budget as CEM (samples x iters), single uniform shot."""
    best_seq, best_cost = None, float("inf")
    for _ in range(budget):
        seq = [(VERBS[rng.randrange(len(VERBS))], ARGS[rng.randrange(len(ARGS))],
                ARGS[rng.randrange(len(ARGS))]) for _ in range(horizon)]
        if valid_pool is not None:
            seq[0] = valid_pool[rng.randrange(len(valid_pool))]
        shaped, _ = rollout_cost(adapter, z0, seq, goal_d)
        if shaped < best_cost:
            best_cost, best_seq = shaped, seq
    return best_seq, best_cost


def build_valid_pool(state, rng, draws=300):
    """Approximate the valid-action set by repeated typed sampling (the plan's
    validity filter, position-0 form: nonprivileged in full obs — sample_valid
    reads exactly the state a parser would recover from the observation).
    Sorted for cross-process determinism (finding 19: set-hash order made the
    filtered arms irreproducible). Coverage caveat (finding 19): ~300 draws cover
    a minority of the valid space; pool coverage bounds filtered-arm success."""
    return sorted({actions.sample_valid(state, rng) for _ in range(draws)})


def run_episode(adapter, planner, s0, pred, args, rng):
    state, steps = s0, 0
    filtered = planner.endswith("-filtered")
    while steps < args.max_steps:
        if pred.check(state):
            return True, steps
        exemplars = [adapter.encode(e, None)
                     for e in make_satisfying(state, pred, rng=rng, n_variants=3)]
        goal_d = lambda z: min(adapter.distance(z, e) for e in exemplars)
        z0 = adapter.encode(state, None)
        pool = build_valid_pool(state, rng) if filtered else None
        if planner.startswith("cem"):
            seq, _ = cem_plan(adapter, z0, goal_d, args.horizon, args.samples,
                              args.iters, rng, valid_pool=pool)
        elif planner.startswith("random"):
            seq, _ = random_shooting_plan(adapter, z0, goal_d, args.horizon,
                                          args.samples * args.iters, rng,
                                          valid_pool=pool)
        else:  # scripted ceiling
            seq = plan_for(pred, state) or [("ls", "", "")]
        state = actions.apply(state, seq[0]).state
        steps += 1
    return pred.check(state), steps


def sample_episodes(data_root, n, max_trajs, rng):
    """Stratified round-robin over predicate kinds (finding 19: the
    unsatisfied-at-s0 filter otherwise skews ~90% of episodes to one family)."""
    trajs = load_transitions(pathlib.Path(data_root) / "val.jsonl", max_trajs)
    manifest = json.loads((pathlib.Path(data_root) / "manifest.json").read_text())
    by_kind = {}
    for p in manifest["val_predicates"]:
        by_kind.setdefault(p["kind"], []).append(Predicate.from_json(p))
    kinds = sorted(by_kind)
    eps = []
    while len(eps) < n:
        pool = by_kind[kinds[len(eps) % len(kinds)]]
        tr = trajs[rng.randrange(len(trajs))]
        s0 = tr["states"][rng.randrange(len(tr["states"]))]
        pred = pool[rng.randrange(len(pool))]
        if pred.check(s0) or not plan_for(pred, s0):
            continue
        eps.append((s0, pred))
    return eps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--adapter", default="oracle",
                    choices=sorted(ADAPTERS) + ["tier2", "gate2", "gate2-codebook"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--codebook", default="runs/gate2/codebook.pt",
                    help="gate2-codebook: path to the built codebook")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-val-trajs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.adapter == "tier2":
        from models.tier2 import Tier2Adapter
        adapter = Tier2Adapter(args.ckpt)
    elif args.adapter == "gate2":
        from models.gate2 import Gate2Adapter
        adapter = Gate2Adapter(args.ckpt)
    elif args.adapter == "gate2-codebook":
        from models.gate2 import Gate2CodebookAdapter
        adapter = Gate2CodebookAdapter(args.ckpt, args.codebook)
    else:
        adapter = ADAPTERS[args.adapter]()

    rng = random.Random(f"cem:{args.seed}")
    episodes = sample_episodes(args.data, args.episodes, args.max_val_trajs, rng)
    report = {"adapter": adapter.name, "ckpt": args.ckpt,
              "data": args.data, "seed": args.seed,
              "episode_stratification": "round-robin over predicate kinds",
              "budget": {"samples": args.samples, "iters": args.iters,
                         "horizon": args.horizon, "max_steps": args.max_steps},
              "episodes": len(episodes), "planners": {}}
    # "scripted-privileged": plan_for reads true state — an upper bound, NOT the
    # plan §5 nonprivileged parser+BFS ceiling, which remains unimplemented.
    for planner in ("cem", "random", "cem-filtered", "random-filtered",
                    "scripted-privileged"):
        p_rng = random.Random(f"cem-run:{args.seed}:{planner}")
        succ, steps_used = [], []
        by_kind = {}
        for i, (s0, pred) in enumerate(episodes):
            ok, steps = run_episode(adapter, planner, s0, pred, args, p_rng)
            succ.append(ok)
            by_kind.setdefault(pred.to_json()["kind"], []).append(ok)
            if ok:
                steps_used.append(steps)
            if (i + 1) % 20 == 0:
                print(f"  {planner}: {i + 1}/{len(episodes)} "
                      f"(success so far {sum(succ)}/{len(succ)})", flush=True)
        report["planners"][planner] = {
            "success_rate": sum(succ) / len(succ),
            "success_by_kind": {k: {"rate": sum(v) / len(v), "n": len(v)}
                                for k, v in sorted(by_kind.items())},
            "mean_steps_on_success": (sum(steps_used) / len(steps_used)
                                      if steps_used else float("nan")),
        }
        print(planner, json.dumps(report["planners"][planner]), flush=True)
    report["cem_minus_random_points"] = round(
        100 * (report["planners"]["cem"]["success_rate"]
               - report["planners"]["random"]["success_rate"]), 1)
    report["cem_minus_random_points_filtered"] = round(
        100 * (report["planners"]["cem-filtered"]["success_rate"]
               - report["planners"]["random-filtered"]["success_rate"]), 1)
    print("deltas:", report["cem_minus_random_points"],
          report["cem_minus_random_points_filtered"], flush=True)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
