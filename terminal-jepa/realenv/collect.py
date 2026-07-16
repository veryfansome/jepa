"""Collect real shell trajectories (Phase R1).

Runs task programs in isolated sessions and writes train/val JSONL where each line is a
trajectory of steps: {cmd, stdout, stderr, exit, cwd_rel} + real fs-diff labels. The
observation the model will see is the terminal transcript (prompt + command + output);
exit code and fs diff are eval LABELS derived from the real system, not shown to the
model. The val split holds out whole TASKS/tools (default: git, python) so we can test
whether a world model transfers to tools it never trained on.

Usage:
  python3 -m realenv.collect --out data/real --n-train 400 --n-val 120 --seed 0
"""

import argparse
import json
import pathlib
import random
import tempfile

from realenv import tasks
from realenv.record import Session, fs_diff


def render_obs(step):
    """The realistic partial-obs observation: prompt + command + its output. Exit code
    and fs diff are NOT here (they are labels)."""
    lines = [f"user@host:~/{step['cwd_rel']}$ {step['cmd']}"]
    if step["stdout"]:
        lines.append(step["stdout"].rstrip("\n"))
    if step["stderr"]:
        lines.append(step["stderr"].rstrip("\n"))
    return "\n".join(lines)


def run_trajectory(sandbox_root, rng, allowed_tasks):
    sess = Session(sandbox_root)
    name, cmds, tools = tasks.sample_task(rng, allowed_tasks)
    steps = []
    try:
        before = sess.snapshot()
        for cmd in cmds:
            obs = sess.run(cmd)
            after = sess.snapshot()
            diff = fs_diff(before, after)
            steps.append({
                "cmd": obs["cmd"], "stdout": obs["stdout"], "stderr": obs["stderr"],
                "exit": obs["exit"], "cwd_rel": obs["cwd_rel"],
                "success": obs["exit"] == 0, "n_changed": diff["n_changed"],
                "created": diff["created"], "deleted": diff["deleted"],
                "modified": diff["modified"],
            })
            before = after
    finally:
        sess.close()
    return {"task": name, "tools": sorted(tools), "steps": steps}


def collect(out_dir, n_train, n_val, seed, held_out):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_tasks = set(tasks.TASKS)
    train_tasks = sorted(all_tasks - set(held_out))
    val_tasks = sorted(held_out) if held_out else train_tasks
    sandbox_root = tempfile.mkdtemp(prefix="tj-sandbox-")

    def emit(path, n, allowed, split):
        rng = random.Random(f"real:{seed}:{split}")
        mix = {}
        with open(path, "w") as fh:
            for i in range(n):
                tr = run_trajectory(sandbox_root, rng, allowed)
                fh.write(json.dumps(tr) + "\n")
                mix[tr["task"]] = mix.get(tr["task"], 0) + 1
        return mix

    train_mix = emit(out / "train.jsonl", n_train, train_tasks, "train")
    val_mix = emit(out / "val.jsonl", n_val, val_tasks, "val")

    # summary over the collected data
    def summarize(path):
        n_steps = n_ok = n_fail = n_changed = 0
        with open(path) as fh:
            for line in fh:
                for s in json.loads(line)["steps"]:
                    n_steps += 1
                    n_ok += s["success"]; n_fail += not s["success"]
                    n_changed += s["n_changed"] > 0
        return {"trajs": sum(1 for _ in open(path)), "steps": n_steps,
                "success_rate": round(n_ok / max(n_steps, 1), 3),
                "fail_rate": round(n_fail / max(n_steps, 1), 3),
                "state_changing_rate": round(n_changed / max(n_steps, 1), 3)}

    summary = {
        "seed": seed, "held_out_tasks": sorted(held_out),
        "train_tasks": train_tasks, "val_tasks": val_tasks,
        "train_task_mix": train_mix, "val_task_mix": val_mix,
        "train": summarize(out / "train.jsonl"), "val": summarize(out / "val.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held-out", default="git,python",
                    help="comma tasks held out of train and used as the transfer val set")
    args = ap.parse_args(argv)
    held = [t for t in args.held_out.split(",") if t]
    summary = collect(args.out, args.n_train, args.n_val, args.seed, held)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
