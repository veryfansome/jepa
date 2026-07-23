"""Backend CLI the evolutionary control loop (the `evolve` skill / /loop) calls.

  python -m evolve.cli seed [--mode proxy|full]        # score + archive the R4 baseline
  python -m evolve.cli score --genome g.json [--mode]  # score + archive a proposed genome
  python -m evolve.cli leaderboard [--top N]
  python -m evolve.cli sample-parent [--seed N]        # weighted parent for the next generation
  python -m evolve.cli impls [--chunk objective]       # list registered chunk implementations

Run from the terminal-jepa/ directory.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evolve import archive as A
from evolve import genome as G
from evolve.harness import score_genome

META = ("id", "parent", "generation", "inventor", "chunk_changed", "rationale")


def _record(gen, res):
    rec = {k: gen.get(k) for k in META}
    rec["chunks"] = gen["chunks"]
    rec.update(res)
    A.append(rec)
    return rec


def cmd_seed(args):
    gen = G.baseline_genome()
    res = score_genome(gen, mode=args.mode, proxy_steps=args.proxy_steps)
    _record(gen, res)
    print(json.dumps(res, indent=1))


def cmd_score(args):
    gen = json.load(open(args.genome))
    res = score_genome(gen, mode=args.mode, proxy_steps=args.proxy_steps, split=args.split,
                       data=args.data, save_dir=args.save_dir, val_data=args.val_data,
                       stats_root=args.stats_root, subsample_seqs=args.subsample_seqs,
                       subsample_seed=args.subsample_seed)
    res["data"] = args.data
    if args.val_data:
        res["val_data"] = args.val_data
    _record(gen, res)
    print(json.dumps(res, indent=1))


def cmd_ingest(args):
    """Append an externally-scored result (e.g. a RunPod job's cli-score stdout, which may
    carry cache-note lines around the JSON) to the archive, without rescoring."""
    import re
    gen = json.load(open(args.genome))
    txt = open(args.result).read()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise SystemExit(f"no JSON object found in {args.result}")
    res = json.loads(m.group(0))
    if "fitness" not in res:
        raise SystemExit(f"result {args.result} has no fitness field")
    if args.env:
        res["env"] = args.env
    rec = _record(gen, res)
    print(json.dumps({k: rec.get(k) for k in ("id", "fitness", "guardrail", "mode", "split")}, indent=1))


def cmd_leaderboard(args):
    lb = A.leaderboard(args.top, bench=None if args.bench == "all" else args.bench)
    if not lb:
        print("(archive empty)")
        return
    for r in lb:
        print(f'{r["fitness"]:+.4f}  {r.get("mode","?"):5s}  {r["id"]:26s}  '
              f'obj={r["chunks"]["objective"]["impl"]:14s}  {(r.get("rationale") or "")[:56]}')


def cmd_sample_parent(args):
    p = A.sample_parent(seed=args.seed, bench=None if args.bench == "all" else args.bench)
    print(json.dumps(p, indent=1) if p else "null")


def cmd_impls(args):
    print(json.dumps(G.list_impls(args.chunk)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evolve.cli")
    sub = ap.add_subparsers(required=True)

    def add_mode(p):
        p.add_argument("--mode", default="proxy", choices=["proxy", "full"])
        p.add_argument("--proxy-steps", type=int, default=1000)

    s = sub.add_parser("seed"); add_mode(s); s.set_defaults(fn=cmd_seed)
    s = sub.add_parser("score"); s.add_argument("--genome", required=True)
    s.add_argument("--split", default="inner", choices=["inner", "final"])
    s.add_argument("--data", default="data/dockerfs")
    s.add_argument("--save-dir", default=None, help="checkpoint trained per-seed models here (plan-eval hook)")
    # v3 §11.5 ablate plumbing (default-inert): score against a different val root, pin the
    # standardization stats to another root, and/or seeded train-side per-image subsampling.
    s.add_argument("--val-data", default=None, help="score against a different val root (train root != val root)")
    s.add_argument("--stats-root", default=None, help="standardization stats from another root (canonical full-root stats for the ablate arm)")
    s.add_argument("--subsample-seqs", type=int, default=None, help="train-side seeded per-image subsample to N seqs/image (§11.5)")
    s.add_argument("--subsample-seed", type=int, default=0, help="seed for --subsample-seqs (feeds the §13.2 train descriptor)")
    add_mode(s); s.set_defaults(fn=cmd_score)
    s = sub.add_parser("ingest"); s.add_argument("--genome", required=True)
    s.add_argument("--result", required=True)
    s.add_argument("--env", default=None, help="environment tag recorded on the entry (e.g. 'runpod-4090')")
    s.set_defaults(fn=cmd_ingest)
    bench_kw = dict(default=A.ACTIVE_BENCH, choices=["v1", "v2", "v3", "all"],
                    help="bench version to rank within (margins aren't cross-comparable); 'all' mixes")
    s = sub.add_parser("leaderboard"); s.add_argument("--top", type=int, default=10)
    s.add_argument("--bench", **bench_kw); s.set_defaults(fn=cmd_leaderboard)
    s = sub.add_parser("sample-parent"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--bench", **bench_kw); s.set_defaults(fn=cmd_sample_parent)
    s = sub.add_parser("impls"); s.add_argument("--chunk", default="objective"); s.set_defaults(fn=cmd_impls)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
