"""R3 decisive comparison: JEPA (latent-prediction) world model vs a compute-matched
GENERATIVE (token-reconstruction) twin, on held-out git.

Both arms are architecturally identical — same trunk over [z_ctx, z_cmd], same supervised
outcome heads (success, state-change) — and differ ONLY in the auxiliary world-modeling
target:
  - jepa:  predict the next observation's frozen EMBEDDING (latent L2, abstract);
  - recon: predict the next observation's BAG-OF-TOKENS (BCE over a top-V vocab, surface).
This is the V-JEPA ablation (latent vs reconstruction) for a shell world model. finding
24 predicts JEPA should WIN here (real high-nuisance regime), unlike the synthetic
clean-serialization tie (finding 22). Ranking (planning-relevant command discrimination)
is scored in each arm's own space: does the arm predict the TRUE command's next-obs
better than FOIL commands? Copy ignores the command -> 0.5 by construction.

Usage: .venv/bin/python -m realenv.jepa_vs_gen --data data/real --out runs/real/r3-vs-gen.json
"""

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from realenv.collect import render_obs
from realenv.worldmodel import _auc, cached_encode
from train.train import pick_device

D = 768


def build_vocab(trajs, tok, top_v=2000):
    from collections import Counter
    c = Counter()
    for tr in trajs:
        # tr carries no texts in the emb cache; rebuild from cmds+... use cmd only? no —
        # bags need obs text. Caller passes texts separately (see load_texts).
        pass
    return c  # unused; see load_texts_and_vocab


def load_texts(path):
    """Per-traj list of observation texts (for bag-of-tokens targets)."""
    out = []
    for line in open(path):
        tr = json.loads(line)
        out.append([render_obs(s) for s in tr["steps"]])
    return out


def build_vocab_from_texts(texts_per_traj, tok, top_v=2000):
    from collections import Counter
    c = Counter()
    for traj in texts_per_traj:
        for t in traj:
            c.update(set(tok(t, truncation=True, max_length=512)["input_ids"]))
    vocab = [tid for tid, _ in c.most_common(top_v)]
    return {tid: i for i, tid in enumerate(vocab)}


def bag(text, tok, vmap):
    ids = set(tok(text, truncation=True, max_length=512)["input_ids"])
    v = torch.zeros(len(vmap))
    for tid in ids:
        if tid in vmap:
            v[vmap[tid]] = 1.0
    return v


def build_transitions(emb_trajs, texts_per_traj, tok, vmap):
    """(z_ctx, z_cmd, z_next, next_bag, success, change). ctx = previous obs; first
    command's ctx is zeros."""
    ctx, cmd, nxt, bags, suc, chg = [], [], [], [], [], []
    for tr, texts in zip(emb_trajs, texts_per_traj):
        n = tr["z_obs"].shape[0]
        for i in range(n):
            ctx.append(tr["z_obs"][i - 1] if i > 0 else torch.zeros(D))
            cmd.append(tr["z_cmd"][i]); nxt.append(tr["z_obs"][i])
            bags.append(bag(texts[i], tok, vmap))
            suc.append(tr["success"][i]); chg.append(tr["change"][i])
    return {"ctx": torch.stack(ctx), "cmd": torch.stack(cmd), "next": torch.stack(nxt),
            "bag": torch.stack(bags), "success": torch.stack(suc), "change": torch.stack(chg)}


class Arm(nn.Module):
    def __init__(self, aux, vsize, d=D, h=512):
        super().__init__()
        self.aux = aux
        self.trunk = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.succ = nn.Linear(h, 2)
        self.chg = nn.Linear(h, 2)
        if aux == "jepa":
            self.head = nn.Linear(h, d)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)  # copy at init
        else:
            self.head = nn.Linear(h, vsize)

    def forward(self, z_ctx, z_cmd):
        f = self.trunk(torch.cat([z_ctx, z_cmd], -1))
        aux_pred = (z_ctx + self.head(f)) if self.aux == "jepa" else self.head(f)
        return aux_pred, self.succ(f), self.chg(f)


def train_arm(aux, tr, device, vsize, steps=3000, bs=256, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    net = Arm(aux, vsize).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = tr["ctx"].shape[0]; g = torch.Generator().manual_seed(seed)
    ce, bce = nn.functional.cross_entropy, nn.functional.binary_cross_entropy_with_logits
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (bs,), generator=g)
        zc, za = tr["ctx"][idx].to(device), tr["cmd"][idx].to(device)
        aux_pred, ls, lg = net(zc, za)
        if aux == "jepa":
            aux_loss = ((aux_pred - tr["next"][idx].to(device)) ** 2).mean()
        else:
            aux_loss = bce(aux_pred, tr["bag"][idx].to(device))
        loss = aux_loss + ce(ls, tr["success"][idx].to(device)) + ce(lg, tr["change"][idx].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  [{aux}] step {step} loss {loss.item():.4f}", flush=True)
    return net


@torch.no_grad()
def eval_arm(net, ev, device, seed=0):
    zc, za = ev["ctx"].to(device), ev["cmd"].to(device)
    aux_pred, ls, lg = net(zc, za)
    out = {"aux": net.aux}
    for name, logit, y in [("success", ls, ev["success"]), ("change", lg, ev["change"])]:
        out[f"{name}_auc"] = _auc(logit.softmax(-1)[:, 1].cpu(), y)
    # ranking: true command's aux-prediction closer to truth than foils'
    rng = random.Random(f"foil:{seed}"); n = zc.shape[0]
    tgt = ev["next"].to(device) if net.aux == "jepa" else ev["bag"].to(device)
    def dist(pred):
        return ((pred - tgt) ** 2).mean(-1) if net.aux == "jepa" else \
               nn.functional.binary_cross_entropy_with_logits(pred, tgt, reduction="none").mean(-1)
    d_true = dist(aux_pred)
    wins = []
    for _ in range(4):
        perm = torch.tensor([rng.randrange(n) for _ in range(n)])
        d_foil = dist(net(zc, za[perm])[0])
        wins.append((d_true < d_foil).float())
    out["command_rank"] = torch.stack(wins).mean().item()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--top-v", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = pick_device("auto")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    print("=== encode (cached) + bags ===", flush=True)
    tr_emb = cached_encode(args.data, "train", args.model, device)
    git_emb = cached_encode(args.data, "val", args.model, device)
    tr_txt = load_texts(pathlib.Path(args.data) / "train.jsonl")
    git_txt = load_texts(pathlib.Path(args.data) / "val.jsonl")
    vmap = build_vocab_from_texts(tr_txt, tok, args.top_v)
    tr = build_transitions(tr_emb, tr_txt, tok, vmap)
    ev = build_transitions(git_emb, git_txt, tok, vmap)
    mu = tr["next"].mean(0, keepdim=True); sd = tr["next"].std(0, keepdim=True).clamp(min=1e-6)
    for k in ("ctx", "cmd", "next"):
        tr[k] = (tr[k] - mu) / sd; ev[k] = (ev[k] - mu) / sd
    print(f"train {tr['ctx'].shape[0]} | git {ev['ctx'].shape[0]} | vocab {len(vmap)}", flush=True)

    report = {"data": args.data, "vocab": len(vmap), "arms": {}}
    for aux in ("jepa", "recon"):
        print(f"=== train {aux} ===", flush=True)
        net = train_arm(aux, tr, device, len(vmap), steps=args.steps, seed=args.seed)
        report["arms"][aux] = eval_arm(net, ev, device, seed=args.seed)
        print(f"  git: {json.dumps(report['arms'][aux])}", flush=True)
    j, r = report["arms"]["jepa"], report["arms"]["recon"]
    report["jepa_minus_recon"] = {k: round(j[k] - r[k], 3)
                                  for k in ("success_auc", "change_auc", "command_rank")}
    print("JEPA - recon (held-out git):", json.dumps(report["jepa_minus_recon"]), flush=True)
    if args.out:
        p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
