"""Dataset generator (terminal-jepa.md §3).

Writes train.jsonl / val.jsonl (one trajectory per line), manifest.json (splits), and
summary.json (transition-type mix, invalid coverage by failure type, banner spread).

Trajectories store compact states + actions + stdout, not rendered observations: the
renderer is a pure function of stored fields, so the training loader picks the distractor
regime (clean / banner / dynamic / both) at load time and the renderer stays the single
source of truth.

Banner ids are sampled from an RNG stream keyed only by (seed, trajectory index) —
independent of layout, policy, and split by construction (terminal-jepa.md §3).

Usage: python -m datagen.generate --out data/v0 [--train-trajs 2000] [--val-trajs 200]
       [--steps 24] [--invalid-quota 0.15] [--epsilon 0.15] [--seed 0]
"""

import argparse
import collections
import json
import pathlib
import random

from env import actions, vocab
from datagen import layouts as L
from datagen.policies import GoalReacher


def run_trajectory(layout_state, policy_kind, pred_pool, traj_seed, steps,
                   invalid_quota, epsilon):
    rng = random.Random("traj:" + ":".join(map(str, traj_seed)))
    state = layout_state.copy()
    reacher = (
        GoalReacher(pred_pool, rng, epsilon) if policy_kind == "reacher" else None
    )
    records = []
    for _ in range(steps):
        # Dataset-level invalid quota: enforced here, per step, regardless of policy,
        # so the realized invalid rate tracks the quota (terminal-jepa.md §3).
        if rng.random() < invalid_quota:
            action, intended_failure = actions.sample_invalid(state, rng)
        else:
            intended_failure = None
            action = (
                reacher.next_action(state)
                if reacher is not None
                else actions.sample_valid(state, rng)
            )
        res = actions.apply(state, action)
        res.state.check_invariants()
        records.append(
            {
                "action": list(action),
                "stdout": res.stdout,
                "ttype": res.ttype,
                "failure": res.failure,
                "intended_failure": intended_failure,
                "cwd_before": vocab.path_to_str(state.cwd),
                "state_after": res.state.to_json(),
            }
        )
        state = res.state
    return records


def generate(out_dir, n_train, n_val, steps, invalid_quota, epsilon, seed,
             n_layouts=400, layout_val_frac=0.2, reacher_frac=0.5):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_layouts, val_layouts = L.make_layout_split(n_layouts, layout_val_frac, seed)
    train_preds, val_preds = L.predicate_split()

    manifest = {
        "seed": seed,
        "steps": steps,
        "invalid_quota": invalid_quota,
        "epsilon": epsilon,
        "n_banners": len(vocab.BANNERS),
        "train_layout_ids": [lid for lid, _ in train_layouts],
        "val_layout_ids": [lid for lid, _ in val_layouts],
        "n_train_predicates": len(train_preds),
        "n_val_predicates": len(val_preds),
        "val_predicates": [p.to_json() for p in val_preds],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    stats = {
        "ttype": collections.Counter(),
        "failure": collections.Counter(),
        "banner_layout_pairs": set(),
        "banner_counts": collections.Counter(),
        "max_dirs": 0,
        "max_files": 0,
        "paths_seen": {"train": set(), "val": set()},
    }

    def emit(path, n_trajs, layout_pool, pred_pool, split_name):
        # Banner stream keyed by (seed, split, index) only: independent of layout/policy.
        banner_rng = random.Random(f"banners:{seed}:{split_name}")
        pick_rng = random.Random(f"pick:{seed}:{split_name}")
        with open(path, "w") as fh:
            for i in range(n_trajs):
                lid, layout_state = pick_rng.choice(layout_pool)
                banner_id = banner_rng.randrange(len(vocab.BANNERS))
                policy_kind = "reacher" if pick_rng.random() < reacher_frac else "random"
                recs = run_trajectory(
                    layout_state, policy_kind, pred_pool,
                    (seed, split_name, i), steps, invalid_quota, epsilon,
                )
                fh.write(json.dumps({
                    "layout_id": lid,
                    "layout": layout_state.to_json(),
                    "banner_id": banner_id,
                    "noise_seed": i,
                    "policy": policy_kind,
                    "steps": recs,
                }) + "\n")
                for r in recs:
                    stats["ttype"][r["ttype"]] += 1
                    if r["failure"]:
                        stats["failure"][r["failure"]] += 1
                    st = r["state_after"]
                    stats["max_dirs"] = max(stats["max_dirs"], len(st["dirs"]))
                    stats["max_files"] = max(stats["max_files"], len(st["files"]))
                    stats["paths_seen"][split_name].update(st["files"])
                stats["paths_seen"][split_name].update(layout_state.to_json()["files"])
                stats["banner_layout_pairs"].add((banner_id, lid))
                stats["banner_counts"][banner_id] += 1

    emit(out / "train.jsonl", n_train, train_layouts, train_preds, "train")
    emit(out / "val.jsonl", n_val, val_layouts, val_preds, "val")

    total = sum(stats["ttype"].values())
    summary = {
        "transitions_total": total,
        "transition_mix": {k: v / total for k, v in sorted(stats["ttype"].items())},
        "invalid_coverage_by_failure_type": dict(sorted(stats["failure"].items())),
        "failure_types_missing": sorted(
            set(actions.FAILURE_TYPES) - set(stats["failure"])
        ),
        "banners_used": len(stats["banner_counts"]),
        "distinct_banner_layout_pairs": len(stats["banner_layout_pairs"]),
        "train_layouts": len(train_layouts),
        "val_layouts": len(val_layouts),
        # Runtime state envelope: layout caps (12/20) bound only initial states; growth
        # is bounded by the vocabulary (42 dirs / 258 files). Reported, not enforced.
        "max_state_dirs": stats["max_dirs"],
        "max_state_files": stats["max_files"],
        # Label-coverage audit: probe metrics are only supported on paths that ever
        # exist in the split (v0 val covered 255/258).
        "file_path_coverage": {
            k: f"{len(v)}/{len(vocab.FILE_PATHS)}"
            for k, v in stats["paths_seen"].items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-trajs", type=int, default=2000)
    ap.add_argument("--val-trajs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--invalid-quota", type=float, default=0.15)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layouts", type=int, default=400)
    args = ap.parse_args(argv)
    summary = generate(
        args.out, args.train_trajs, args.val_trajs, args.steps,
        args.invalid_quota, args.epsilon, args.seed, n_layouts=args.n_layouts,
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
