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


def run_trajectory(sandbox_root, rng, allowed_tools):
    sess = Session(sandbox_root)
    name, cmds, tools = tasks.sample_session(rng, allowed_tools)
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
    train_tools = []              # universal templates only — held-out tools UNSEEN in train
    val_tools = sorted(held_out)  # val sessions feature the held-out tools in diverse contexts
    sandbox_root = tempfile.mkdtemp(prefix="tj-sandbox-")

    def emit(path, n, allowed, split):
        rng = random.Random(f"real:{seed}:{split}")
        with open(path, "w") as fh:
            for i in range(n):
                fh.write(json.dumps(run_trajectory(sandbox_root, rng, allowed)) + "\n")

    emit(out / "train.jsonl", n_train, train_tools, "train")
    emit(out / "val.jsonl", n_val, val_tools, "val")

    def summarize(path, held):
        n_steps = n_ok = n_changed = n_held = 0
        distinct_cmds, first_tokens = set(), set()
        with open(path) as fh:
            for line in fh:
                for s in json.loads(line)["steps"]:
                    n_steps += 1
                    n_ok += s["success"]; n_changed += s["n_changed"] > 0
                    tok0 = s["cmd"].split()[0] if s["cmd"].split() else ""
                    first_tokens.add(tok0)
                    distinct_cmds.add(s["cmd"])
                    n_held += tok0 in held
        return {"trajs": sum(1 for _ in open(path)), "steps": n_steps,
                "success_rate": round(n_ok / max(n_steps, 1), 3),
                "state_changing_rate": round(n_changed / max(n_steps, 1), 3),
                "distinct_commands": len(distinct_cmds), "distinct_verbs": len(first_tokens),
                "held_out_tool_step_rate": round(n_held / max(n_steps, 1), 3)}

    summary = {
        "seed": seed, "held_out_tools": val_tools, "train_tools": "universal-only",
        "train": summarize(out / "train.jsonl", set()),
        "val": summarize(out / "val.jsonl", set(val_tools)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--n-train", type=int, default=700)
    ap.add_argument("--n-val", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held-out", default="git,sed,awk,du,diff,tar",
                    help="comma TOOLS held out of train; val sessions feature them in "
                         "diverse universal-tool contexts (transfer-to-unseen-tool test)")
    args = ap.parse_args(argv)
    held = [t for t in args.held_out.split(",") if t]
    summary = collect(args.out, args.n_train, args.n_val, args.seed, held)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
