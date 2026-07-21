"""Committed axis-1 measurement protocol (dockerfs2-prereg Amendment 3 / review-B F10):
per-verb command-only predictability on a pilot/probe jsonl pair, in e5 space.

Protocol (frozen): encode train+val with enc_e5_base renders (max_length=256, mean-pool);
standardize with TRAIN stats; retrieve_by_cmd = nearest fit cmd by squared L2 → its obs;
within_traj = nearest strictly-earlier cmd by squared L2 within the same trajectory → its
obs, zeros (predict_mean) fallback at t=0; score per-verb same-verb-foil top-1 via
realenv.seq_worldmodel.per_verb_breakdown (63 foils, 4 rounds, seed 0, strict ties).
grep is additionally scored per mode (hit ⇔ exit==0 and non-empty output).

  uv run python -m benchmarks.axis1_measure --root <dir-with-train/val.jsonl> [--out f.json]
"""

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from realenv import seq_worldmodel as M
from evolve.chunks.perception import enc_e5_base as PERC

ALLV = ("uname", "cat", "ls", "cd", "head", "tail", "stat", "find", "grep",
        "grep-hit", "grep-miss")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from transformers import AutoModel, AutoTokenizer
    device = M.pick_device()
    tok = AutoTokenizer.from_pretrained(PERC.MODEL)
    model = AutoModel.from_pretrained(PERC.MODEL).to(device).eval()

    @torch.no_grad()
    def embed(texts, bs=96):
        out = torch.zeros(len(texts), M.D)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            e = tok([texts[j] for j in b], return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
            e = {k: v.to(device) for k, v in e.items()}
            h = model(**e).last_hidden_state
            p = PERC.pool(h, e["attention_mask"]).float().cpu()
            for k, j in enumerate(b):
                out[j] = p[k]
        return out

    def load(split):
        seqs = []
        for line in open(pathlib.Path(args.root) / f"{split}.jsonl"):
            sq = json.loads(line)
            seqs.append({"image": sq["image"],
                         "cmds": [s["cmd"] for s in sq["steps"]],
                         "hits": [s.get("exit", 0) == 0 and bool((s.get("output") or "").strip())
                                  for s in sq["steps"]],
                         "obs_t": [PERC.render_obs(s) for s in sq["steps"]],
                         "cmd_t": [PERC.render_cmd(s) for s in sq["steps"]]})
        return seqs

    train, val = load("train"), load("val")
    for seqs in (train, val):
        for sq in seqs:
            sq["z_obs"] = embed(sq["obs_t"])
            sq["z_cmd"] = embed(sq["cmd_t"])
    mo, so, mc, sc = M.standardize_stats(train)
    M.apply_stats(train, mo, so, mc, sc)
    M.apply_stats(val, mo, so, mc, sc)
    fit_cmd = torch.cat([s["z_cmd"] for s in train])
    fit_obs = torch.cat([s["z_obs"] for s in train])
    true = torch.cat([s["z_obs"] for s in val])
    verbs = []
    for s in val:
        for c, h in zip(s["cmds"], s["hits"]):
            v = M.verb_of(c)
            verbs.append(f"grep-{'hit' if h else 'miss'}" if v == "grep" else v)
    qc = torch.cat([s["z_cmd"] for s in val])
    pred = torch.zeros_like(true)
    for i in range(0, qc.shape[0], 256):
        d = torch.cdist(qc[i:i + 256], fit_cmd)
        pred[i:i + 256] = fit_obs[d.argmin(1)]
    wt = torch.zeros_like(true)
    row = 0
    for s in val:
        for t in range(s["z_obs"].shape[0]):
            if t > 0:
                d = ((s["z_cmd"][:t] - s["z_cmd"][t]) ** 2).sum(-1)
                wt[row] = s["z_obs"][int(d.argmin())]
            row += 1
    res = M.per_verb_breakdown({"rbc": pred, "wt": wt}, true, verbs, seed=0, verbset=ALLV)
    counts = collections.Counter(verbs)
    table = {v: {"retrieve_by_cmd": res.get(v, {}).get("rbc"),
                 "within_traj": res.get(v, {}).get("wt"), "n": counts[v]} for v in ALLV}
    print(json.dumps(table, indent=1))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(table, indent=1))
    return table


if __name__ == "__main__":
    main()
