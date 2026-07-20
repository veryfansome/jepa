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
                       data=args.data, save_dir=args.save_dir)
    res["data"] = args.data
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
    lb = A.leaderboard(args.top)
    if not lb:
        print("(archive empty)")
        return
    for r in lb:
        print(f'{r["fitness"]:+.4f}  {r.get("mode","?"):5s}  {r["id"]:26s}  '
              f'obj={r["chunks"]["objective"]["impl"]:14s}  {(r.get("rationale") or "")[:56]}')


def cmd_sample_parent(args):
    p = A.sample_parent(seed=args.seed)
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
    add_mode(s); s.set_defaults(fn=cmd_score)
    s = sub.add_parser("ingest"); s.add_argument("--genome", required=True)
    s.add_argument("--result", required=True)
    s.add_argument("--env", default=None, help="environment tag recorded on the entry (e.g. 'runpod-4090')")
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("leaderboard"); s.add_argument("--top", type=int, default=10); s.set_defaults(fn=cmd_leaderboard)
    s = sub.add_parser("sample-parent"); s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_sample_parent)
    s = sub.add_parser("impls"); s.add_argument("--chunk", default="objective"); s.set_defaults(fn=cmd_impls)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
