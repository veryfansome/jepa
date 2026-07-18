"""R3 decisive comparison (rebuilt 2026-07-16): JEPA (latent-prediction) vs a
compute-matched GENERATIVE (token-reconstruction) world model, on held-out-tool steps.

Fixes the first version's confounds (2026-07-16 review): matched random init on both aux
heads, and — the clean design — each arm's trunk is trained on its AUXILIARY objective
ONLY (no outcome supervision), then a COMMON success probe is fit on the frozen trunk
representation h and evaluated on held-out-tool steps. So the comparison is purely "which
self-supervised world-modeling objective (predict the next obs's abstract EMBEDDING vs its
surface TOKENS) yields a representation that better predicts a real outcome and
discriminates commands on unseen tools." Multi-seed.

Usage: .venv/bin/python -m realenv.jepa_vs_gen --data data/real --out runs/real/r3-vs-gen.json
"""

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from realenv.collect import render_obs
from realenv.tasks import HELD_OUT_TOOLS
from realenv.worldmodel import _auc, cached_encode

D = 768


def load_texts(path):
    return [[render_obs(s) for s in json.loads(l)["steps"]] for l in open(path)]


def build_vocab(texts_per_traj, tok, top_v=4000):
    from collections import Counter
    c = Counter()
    for traj in texts_per_traj:
        for t in traj:
            c.update(set(tok(t, truncation=True, max_length=512)["input_ids"]))
    return {tid: i for i, (tid, _) in enumerate(c.most_common(top_v))}


def bag(text, tok, vmap):
    v = torch.zeros(len(vmap))
    for tid in set(tok(text, truncation=True, max_length=512)["input_ids"]):
        if tid in vmap:
            v[vmap[tid]] = 1.0
    return v


def build(emb_trajs, texts, tok, vmap):
    ctx, cmd, nxt, bags, suc, held, verbs = [], [], [], [], [], [], []
    for tr, txt in zip(emb_trajs, texts):
        n = tr["z_obs"].shape[0]
        for i in range(n):
            ctx.append(tr["z_obs"][i - 1] if i > 0 else torch.zeros(D))
            cmd.append(tr["z_cmd"][i]); nxt.append(tr["z_obs"][i]); bags.append(bag(txt[i], tok, vmap))
            suc.append(tr["success"][i])
            v = tr["cmds"][i].split()[0] if tr["cmds"][i].split() else ""
            held.append(1 if v in HELD_OUT_TOOLS else 0); verbs.append(v)
    return {"ctx": torch.stack(ctx), "cmd": torch.stack(cmd), "next": torch.stack(nxt),
            "bag": torch.stack(bags), "success": torch.stack(suc),
            "held": torch.tensor(held), "verbs": verbs}


class Trunk(nn.Module):
    """Matched architecture; aux head predicts either the next latent (jepa) or the
    next-obs token bag (recon). Random init for both (no zero-init asymmetry). forward
    returns (aux_pred, h) where h is the shared representation probed for outcome."""

    def __init__(self, aux, vsize, d=D, hdim=512):
        super().__init__()
        self.aux = aux
        self.body = nn.Sequential(nn.Linear(2 * d, hdim), nn.GELU(), nn.Linear(hdim, hdim), nn.GELU())
        self.head = nn.Linear(hdim, d if aux == "jepa" else vsize)

    def forward(self, zc, za):
        h = self.body(torch.cat([zc, za], -1))
        return self.head(h), h


def train_trunk(aux, tr, device, vsize, steps=3000, bs=256, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    net = Trunk(aux, vsize).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = tr["ctx"].shape[0]; g = torch.Generator().manual_seed(seed)
    bce = nn.functional.binary_cross_entropy_with_logits
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        zc, za = tr["ctx"][idx].to(device), tr["cmd"][idx].to(device)
        pred, _ = net(zc, za)
        loss = (((pred - tr["next"][idx].to(device)) ** 2).mean() if aux == "jepa"
                else bce(pred, tr["bag"][idx].to(device)))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net


@torch.no_grad()
def repr_of(net, tr, device, bs=1024):
    hs = []
    for i in range(0, tr["ctx"].shape[0], bs):
        _, h = net(tr["ctx"][i:i+bs].to(device), tr["cmd"][i:i+bs].to(device))
        hs.append(h.cpu())
    return torch.cat(hs)


def probe_success(h_tr, y_tr, h_ev, y_ev, idx, device, steps=400, lr=3e-3, seed=0):
    """Common linear success probe on the frozen trunk representation; eval on subset idx."""
    torch.manual_seed(seed)
    head = nn.Linear(h_tr.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    ce = nn.functional.cross_entropy
    for _ in range(steps):
        b = torch.randint(0, h_tr.shape[0], (2048,), generator=g)
        opt.zero_grad(set_to_none=True); ce(head(h_tr[b].to(device)), y_tr[b].to(device)).backward(); opt.step()
    with torch.no_grad():
        p = head(h_ev[idx].to(device)).softmax(-1)[:, 1].cpu()
    return _auc(p, y_ev[idx])


@torch.no_grad()
def command_rank(net, ev, idx, device, seed=0):
    """In each arm's own prediction space: true command's next-obs prediction closer than
    same-verb foils'. Reported with the cross-space caveat."""
    zc, za = ev["ctx"][idx].to(device), ev["cmd"][idx].to(device)
    tgt = (ev["next"][idx] if net.aux == "jepa" else ev["bag"][idx]).to(device)
    pred, _ = net(zc, za)
    bce = nn.functional.binary_cross_entropy_with_logits
    d = lambda p: ((p - tgt) ** 2).mean(-1) if net.aux == "jepa" else bce(p, tgt, reduction="none").mean(-1)
    d_true = d(pred)
    verbs = [ev["verbs"][i] for i in idx.tolist()]
    by = defaultdict(list)
    for j, v in enumerate(verbs):
        by[v].append(j)
    rng = random.Random(f"foil:{seed}"); wins = []
    for _ in range(4):
        foil = torch.tensor([rng.choice(by[v]) for v in verbs], device=za.device)
        wins.append((d_true < d(net(zc, za[foil])[0])).float().mean().item())
    return sum(wins) / len(wins)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--top-v", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = pick_device_local()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    seeds = [int(s) for s in args.seeds.split(",")]

    tr_txt = load_texts(pathlib.Path(args.data) / "train.jsonl")
    ev_txt = load_texts(pathlib.Path(args.data) / "val.jsonl")
    vmap = build_vocab(tr_txt, tok, args.top_v)
    tr = build(cached_encode(args.data, "train", args.model, device), tr_txt, tok, vmap)
    ev = build(cached_encode(args.data, "val", args.model, device), ev_txt, tok, vmap)
    mu = tr["next"].mean(0, keepdim=True); sd = tr["next"].std(0, keepdim=True).clamp(min=1e-6)
    for k in ("ctx", "cmd", "next"):
        tr[k] = (tr[k] - mu) / sd; ev[k] = (ev[k] - mu) / sd
    held_idx = torch.nonzero(ev["held"] == 1).squeeze(-1)
    print(f"train {tr['ctx'].shape[0]} | val {ev['ctx'].shape[0]} | held-out-tool {len(held_idx)} | vocab {len(vmap)}", flush=True)

    per_seed = {"jepa": [], "recon": []}
    for s in seeds:
        for aux in ("jepa", "recon"):
            net = train_trunk(aux, tr, device, len(vmap), steps=args.steps, seed=s)
            h_tr, h_ev = repr_of(net, tr, device), repr_of(net, ev, device)
            su = probe_success(h_tr, tr["success"], h_ev, ev["success"], held_idx, device, seed=s)
            rk = command_rank(net, ev, held_idx, device, seed=s)
            per_seed[aux].append({"success_auc": su, "command_rank_sameverb": rk})
            print(f"  seed {s} {aux}: success_auc {su:.3f} rank {rk:.3f}", flush=True)

    def agg(aux, k):
        v = [p[k] for p in per_seed[aux]]; m = sum(v) / len(v)
        return {"mean": round(m, 4), "std": round((sum((x - m) ** 2 for x in v) / len(v)) ** 0.5, 4)}
    report = {"data": args.data, "seeds": seeds, "held_out_tool_steps": len(held_idx), "vocab": len(vmap),
              "jepa": {k: agg("jepa", k) for k in ("success_auc", "command_rank_sameverb")},
              "recon": {k: agg("recon", k) for k in ("success_auc", "command_rank_sameverb")}}
    report["jepa_minus_recon"] = {k: round(report["jepa"][k]["mean"] - report["recon"][k]["mean"], 3)
                                  for k in ("success_auc", "command_rank_sameverb")}
    print("=== held-out-tool JEPA vs recon ===\n" + json.dumps(report["jepa_minus_recon"], indent=1), flush=True)
    if args.out:
        p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
    return report


def pick_device_local():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    main()
