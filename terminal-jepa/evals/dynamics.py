"""Dynamics-gate evaluation battery (status doc "Path to a working agent", gate 2;
plan §4.3-4 rollout drift + violation-of-expectation, §5 goal-distance machinery).

The shared referee for every (representation, predictor) pair: Track B tier 1
(oracle/oracle — machinery self-test), tier 2 (oracle representation + learned
predictor — the architecture bake-off), and Track A gate 2 (frozen features +
learned predictor). Built and pre-registered BEFORE any learned predictor exists
so architectures are never selected on training loss.

Components:
- rollout error by transition type (state-changing / valid-no-op / invalid),
  horizons 1..3 — the copy-predictor run supplies the floor to beat;
- change-magnitude calibration: ||predict(z,a) - z|| per transition type + AUC
  discriminating state-changing from non (the no-op miscalibration readout);
- violation-of-expectation: prediction ranks the true next state above (a) an
  alternative-action outcome — copy-resistant, the primary foil — and (b) a
  two-edit state ("a file appears without a creating command"); the two-edit
  foil is copy-passable (a 0-edit prediction is nearer a 1-edit truth than a
  2-edit foil) and is reported for completeness, never as the bar;
- goal-distance ranking along constructive optimal plans (plan_for) against
  make_satisfying exemplars: the optimal suffix must outrank random-valid and
  invalid continuations by latent-distance margin (RANKING, not per-step
  monotonicity), plus executed-plan distance improvement as an encoder readout.

Adapters implement: encode(state, ctx=None) -> latent; predict(latent, action)
-> latent; distance(a, b) -> float. ctx carries nuisance rendering context
(banner_id/noise_seed/step) for text encoders; oracle adapters ignore it.

Expected self-test outcomes (recorded before first run):
- oracle/oracle: zero rollout error, calibration AUC 1.0, VoE 1.0 on both
  foils, ranking ~1.0 with final optimal distance 0.
- oracle/copy: zero error on no-op/invalid but nonzero on state-changing;
  AUC 0.5 (all magnitudes zero); VoE ~0.5 on alt-action, ~1.0 on two-edit;
  ranking ~0.5 everywhere (all continuations map to the same latent).

Usage:
  python3 -m evals.dynamics --data data/v1 --adapter oracle --out runs/dynamics-battery/oracle-oracle.json
  python3 -m evals.dynamics --data data/v1 --adapter oracle-copy --out runs/dynamics-battery/oracle-copy.json
"""

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from datagen.policies import plan_for  # noqa: E402
from env import actions, vocab
from env.state import FsState, Predicate, make_satisfying


# -- adapters ---------------------------------------------------------------------


class OracleAdapter:
    """Tier-1 representation and dynamics: the latent IS the symbolic state; the
    predictor IS the env's transition function; distance is symbolic edit count."""

    name = "oracle"

    def encode(self, state, ctx=None):
        return state

    def predict(self, latent, action):
        return actions.apply(latent, action).state

    def distance(self, a, b):
        d = float(a.cwd != b.cwd)
        d += sum((p in a.dirs) != (p in b.dirs) for p in vocab.DIR_PATHS)
        d += sum(a.files.get(p, -1) != b.files.get(p, -1) for p in vocab.FILE_PATHS)
        return d


class CopyDynamics:
    """Same representation and distance as the base adapter; predict = identity.
    The floor every learned predictor must beat on state-changing transitions and
    must MATCH on no-ops (copy is optimal there — that's the calibration bar)."""

    def __init__(self, base):
        self.base = base
        self.name = f"{base.name}-copy"

    def encode(self, state, ctx=None):
        return self.base.encode(state, ctx)

    def predict(self, latent, action):
        return latent

    def distance(self, a, b):
        return self.base.distance(a, b)


ADAPTERS = {
    "oracle": lambda: OracleAdapter(),
    "oracle-copy": lambda: CopyDynamics(OracleAdapter()),
}


# -- data -------------------------------------------------------------------------


def load_transitions(jsonl_path, max_trajs=None):
    trajs = []
    with open(jsonl_path) as fh:
        for line in fh:
            if max_trajs is not None and len(trajs) >= max_trajs:
                break
            t = json.loads(line)
            states = [FsState.from_json(t["layout"])]
            for s in t["steps"]:
                states.append(FsState.from_json(s["state_after"]))
            trajs.append({
                "states": states,
                "actions": [tuple(s["action"]) for s in t["steps"]],
                "ttypes": [s["ttype"] for s in t["steps"]],
                "banner_id": t["banner_id"],
                "noise_seed": t["noise_seed"],
            })
    return trajs


def _ctx(tr, t):
    return {"banner_id": tr["banner_id"], "noise_seed": tr["noise_seed"], "step": t}


def _median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), exact over all pairs."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


# -- battery components -----------------------------------------------------------


def rollout_eval(adapter, trajs, rng, n=500, max_h=3):
    by_type = {actions.STATE_CHANGING: [], actions.VALID_NO_OP: [], actions.INVALID: []}
    by_h = {h: [] for h in range(1, max_h + 1)}
    for _ in range(n):
        tr = trajs[rng.randrange(len(trajs))]
        t = rng.randrange(len(tr["actions"]) - max_h + 1)
        z = adapter.encode(tr["states"][t], _ctx(tr, t))
        for h in range(1, max_h + 1):
            z = adapter.predict(z, tr["actions"][t + h - 1])
            err = adapter.distance(z, adapter.encode(tr["states"][t + h], _ctx(tr, t + h)))
            by_h[h].append(err)
            if h == 1:
                by_type[tr["ttypes"][t]].append(err)
    return {
        "n": n,
        "h1_err_median_by_type": {k: _median(v) for k, v in by_type.items()},
        "h1_err_mean_by_type": {k: (sum(v) / len(v) if v else float("nan"))
                                for k, v in by_type.items()},
        "err_median_by_horizon": {h: _median(v) for h, v in by_h.items()},
    }


def calibration_eval(adapter, trajs, rng, n=500):
    mags = {actions.STATE_CHANGING: [], actions.VALID_NO_OP: [], actions.INVALID: []}
    for _ in range(n):
        tr = trajs[rng.randrange(len(trajs))]
        t = rng.randrange(len(tr["actions"]))
        z = adapter.encode(tr["states"][t], _ctx(tr, t))
        m = adapter.distance(adapter.predict(z, tr["actions"][t]), z)
        mags[tr["ttypes"][t]].append(m)
    non = mags[actions.VALID_NO_OP] + mags[actions.INVALID]
    return {
        "n": n,
        "change_magnitude_median_by_type": {k: _median(v) for k, v in mags.items()},
        "auc_state_changing_vs_non": _auc(mags[actions.STATE_CHANGING], non),
    }


_SYM = OracleAdapter()


def voe_eval(adapter, trajs, rng, n=300):
    hits_alt, hits_two, errs_true = [], [], []
    tries = 0
    while len(errs_true) < n and tries < n * 20:
        tries += 1
        tr = trajs[rng.randrange(len(trajs))]
        t = rng.randrange(len(tr["actions"]))
        if tr["ttypes"][t] != actions.STATE_CHANGING:
            continue
        s, s_next = tr["states"][t], tr["states"][t + 1]
        pred = adapter.predict(adapter.encode(s, _ctx(tr, t)), tr["actions"][t])
        err_true = adapter.distance(pred, adapter.encode(s_next, _ctx(tr, t + 1)))
        errs_true.append(err_true)
        for _ in range(20):  # alt-action foil: a different one-edit outcome
            alt = actions.sample_valid(s, rng)
            r = actions.apply(s, alt)
            if r.ttype == actions.STATE_CHANGING and r.state != s_next:
                e = adapter.distance(pred, adapter.encode(r.state, _ctx(tr, t + 1)))
                hits_alt.append((err_true < e) + 0.5 * (err_true == e))
                break
        for _ in range(20):  # two-edit foil: >=2 symbolic edits from s (finding 19:
            # a same-coordinate second edit is one edit from s — 4/300 foils were
            # single-edit states, exactly accounting for the copy control's 0.993)
            a2 = actions.sample_valid(s_next, rng)
            r2 = actions.apply(s_next, a2)
            if (r2.ttype == actions.STATE_CHANGING and r2.state != s_next
                    and _SYM.distance(s, r2.state) >= 2):
                e = adapter.distance(pred, adapter.encode(r2.state, _ctx(tr, t + 1)))
                hits_two.append((err_true < e) + 0.5 * (err_true == e))
                break
    return {
        "n": len(errs_true),
        "n_alt_foils": len(hits_alt),
        "n_two_edit_foils": len(hits_two),
        "foils_dropped_alt": len(errs_true) - len(hits_alt),
        "foils_dropped_two_edit": len(errs_true) - len(hits_two),
        "err_true_median": _median(errs_true),
        "acc_vs_alt_action_foil": (sum(hits_alt) / len(hits_alt)) if hits_alt else float("nan"),
        "acc_vs_two_edit_foil_copy_passable": (sum(hits_two) / len(hits_two)) if hits_two else float("nan"),
    }


def _walk_plan(s0, plan):
    """Apply a constructive plan in the true env; returns states s0..sk or None if
    any step fails to be valid (plan_for is constructive, but guard anyway)."""
    states = [s0]
    for a in plan:
        r = actions.apply(states[-1], a)
        if r.ttype == actions.INVALID:
            return None
        states.append(r.state)
    return states


def goal_ranking_eval(adapter, trajs, predicates, rng, n_episodes=100, n_random=3):
    """Episodes are STRATIFIED round-robin over predicate kinds (finding 19: the
    unsatisfied-at-s0 filter otherwise skews ~90% of episodes to one family)."""
    by_kind_pool = {}
    for p in predicates:
        by_kind_pool.setdefault(p.to_json()["kind"], []).append(p)
    kinds = sorted(by_kind_pool)
    wins_rand, wins_inv, final_d, improved, margins = [], [], [], [], []
    kind_wins = {k: [] for k in kinds}
    made, tries = 0, 0
    while made < n_episodes and tries < n_episodes * 50:
        tries += 1
        kind = kinds[made % len(kinds)]
        pool = by_kind_pool[kind]
        tr = trajs[rng.randrange(len(trajs))]
        t = rng.randrange(len(tr["states"]))
        s0 = tr["states"][t]
        pred = pool[rng.randrange(len(pool))]
        if pred.check(s0):
            continue
        plan = plan_for(pred, s0)
        states = _walk_plan(s0, plan) if plan else None
        if not states or not pred.check(states[-1]):
            continue
        exemplars = [adapter.encode(e, None)
                     for e in make_satisfying(s0, pred, rng=rng, n_variants=3)]
        goal_d = lambda z: min(adapter.distance(z, e) for e in exemplars)
        for i in range(len(plan)):
            z = adapter.encode(states[i], _ctx(tr, min(t + i, len(tr["states"]) - 1)))
            suffix = plan[i:]
            zr = z
            for a in suffix:
                zr = adapter.predict(zr, a)
            d_opt = goal_d(zr)
            for _ in range(n_random):  # random-valid continuation, same length
                sw, zw = states[i], z
                for _ in suffix:
                    a = actions.sample_valid(sw, rng)
                    sw = actions.apply(sw, a).state
                    zw = adapter.predict(zw, a)
                d_r = goal_d(zw)
                w = (d_opt < d_r) + 0.5 * (d_opt == d_r)
                wins_rand.append(w)
                kind_wins[kind].append(w)
                margins.append(d_r - d_opt)
            zi = z  # invalid continuation, same length
            for _ in suffix:
                zi = adapter.predict(zi, actions.sample_invalid(states[i], rng)[0])
            d_i = goal_d(zi)
            wins_inv.append((d_opt < d_i) + 0.5 * (d_opt == d_i))
            if i == 0:
                final_d.append(d_opt)
        d_start = goal_d(adapter.encode(states[0], _ctx(tr, t)))
        d_end = goal_d(adapter.encode(states[-1], _ctx(tr, min(t + len(plan), len(tr["states"]) - 1))))
        improved.append(1.0 if d_end < d_start else 0.5 if d_end == d_start else 0.0)
        made += 1
    return {
        "n_episodes": made,
        "episode_stratification": "round-robin over predicate kinds",
        "exemplar_ctx": "clean (exemplars always render banner=None/noise=None; "
                        "finding 19 caveat for contaminated-input runs)",
        "rank_acc_vs_random": (sum(wins_rand) / len(wins_rand)) if wins_rand else float("nan"),
        "rank_acc_vs_random_by_kind": {
            k: {"acc": (sum(v) / len(v)) if v else float("nan"), "n": len(v)}
            for k, v in kind_wins.items()
        },
        "rank_acc_vs_invalid": (sum(wins_inv) / len(wins_inv)) if wins_inv else float("nan"),
        "rank_margin_vs_random_median": _median(margins),
        "optimal_plan_final_goal_distance_median": _median(final_d),
        "executed_plan_distance_improved_frac": (sum(improved) / len(improved)) if improved else float("nan"),
    }


# -- runner -----------------------------------------------------------------------


def run_battery(adapter, data_root, seed=0, max_trajs=200):
    rng = random.Random(f"dynamics:{seed}")
    trajs = load_transitions(pathlib.Path(data_root) / "val.jsonl", max_trajs)
    manifest = json.loads((pathlib.Path(data_root) / "manifest.json").read_text())
    predicates = [Predicate.from_json(p) for p in manifest["val_predicates"]]
    report = {
        "adapter": adapter.name,
        "data": str(data_root),
        "seed": seed,
        "eval_split": "val-heldout-layouts, val-heldout-predicates",
        "rollout": rollout_eval(adapter, trajs, rng),
        "calibration": calibration_eval(adapter, trajs, rng),
        "voe": voe_eval(adapter, trajs, rng),
        "goal_ranking": goal_ranking_eval(adapter, trajs, predicates, rng),
    }
    # Strict clause scoring (finding 19): MEANS only — a median saturates at 0 for
    # any failure rate below 50% and is blind to exactly the spurious-edit failure
    # the calibration clause exists to catch.
    report["rule_clauses"] = {
        "scoring": "means (medians never used for clause verdicts)",
        "state_changing_h1_mean": report["rollout"]["h1_err_mean_by_type"][actions.STATE_CHANGING],
        "noop_h1_mean": report["rollout"]["h1_err_mean_by_type"][actions.VALID_NO_OP],
        "invalid_h1_mean": report["rollout"]["h1_err_mean_by_type"][actions.INVALID],
        "auc_state_changing_vs_non": report["calibration"]["auc_state_changing_vs_non"],
    }
    return report


_AGG_KEYS = [
    ("rule_clauses", "state_changing_h1_mean"),
    ("rule_clauses", "noop_h1_mean"),
    ("rule_clauses", "invalid_h1_mean"),
    ("rule_clauses", "auc_state_changing_vs_non"),
    ("voe", "acc_vs_alt_action_foil"),
    ("goal_ranking", "rank_acc_vs_random"),
    ("goal_ranking", "rank_acc_vs_invalid"),
]


def aggregate_reports(reports):
    """Across-seed mean/min/max/std for the clause-relevant scalars (finding 19:
    single-seed knife-edge passes are not verdicts)."""
    agg = {}
    for section, key in _AGG_KEYS:
        vals = [r[section][key] for r in reports if r[section][key] == r[section][key]]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        agg[f"{section}.{key}"] = {"mean": mean, "std": std,
                                   "min": min(vals), "max": max(vals), "n_seeds": len(vals)}
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--adapter", required=True,
                    choices=sorted(ADAPTERS) + ["tier2", "gate2", "gate2-copy",
                                                "gate2-codebook"])
    ap.add_argument("--ckpt", default=None, help="checkpoint for learned adapters")
    ap.add_argument("--codebook", default="runs/gate2/codebook.pt",
                    help="gate2-codebook: path to the built codebook")
    ap.add_argument("--input-regime", default="clean", choices=["clean", "both"],
                    help="gate2 only: nuisance regime for battery-time inputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seed list; overrides --seed and emits a "
                         "per-seed + aggregate report (finding 19: multi-seed verdicts)")
    ap.add_argument("--max-val-trajs", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.adapter == "tier2":
        from models.tier2 import Tier2Adapter  # torch only when needed

        adapter = Tier2Adapter(args.ckpt)
    elif args.adapter in ("gate2", "gate2-copy"):
        from models.gate2 import Gate2Adapter

        adapter = Gate2Adapter(args.ckpt, input_regime=args.input_regime)
        if args.adapter == "gate2-copy":
            adapter = CopyDynamics(adapter)
    elif args.adapter == "gate2-codebook":
        from models.gate2 import Gate2CodebookAdapter

        adapter = Gate2CodebookAdapter(args.ckpt, args.codebook,
                                       input_regime=args.input_regime)
    else:
        adapter = ADAPTERS[args.adapter]()
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed])
    per_seed = []
    for s in seeds:
        r = run_battery(adapter, args.data, s, args.max_val_trajs)
        r["ckpt"] = args.ckpt
        per_seed.append(r)
        print(f"seed {s}:", json.dumps(r["rule_clauses"]), flush=True)
    if len(per_seed) == 1:
        report = per_seed[0]
    else:
        report = {"adapter": adapter.name, "ckpt": args.ckpt, "data": args.data,
                  "seeds": seeds, "aggregate": aggregate_reports(per_seed),
                  "per_seed": per_seed}
    print(json.dumps(report if len(per_seed) == 1 else report["aggregate"],
                     indent=1), flush=True)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
