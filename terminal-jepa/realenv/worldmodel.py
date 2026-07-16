"""R3: a JEPA world model over real shell trajectories.

An action(command)-conditioned predictor over FROZEN ModernBERT observation embeddings:
  f(z_ctx, z_cmd) -> z_next  (predict the next observation's embedding)
plus outcome heads (success, state-change) — predicting the CONSEQUENCE of a command
BEFORE it runs (the world-model bet; you cannot read the output because it doesn't
exist yet). Perception is frozen/borrowed; only this small predictor is learned.

Evaluated on the held-out tool (git) it never trained on:
  - latent prediction error vs the COPY baseline (predict z_ctx = no change);
  - outcome AUC (success/change) from (z_ctx, z_cmd) vs a CONTEXT-ONLY head (z_ctx,
    ignoring the command) and the marginal — does knowing the command help predict
    the consequence?
  - command-discrimination ranking: WM(z_ctx, true_cmd) must predict z_next better
    than WM(z_ctx, foil_cmd) for foil commands (copy is 0.5 by construction — it
    ignores the command). This is the planning-relevant test: can the world model
    tell different commands apart by their predicted effect?

Encodings are cached (frozen encoder is deterministic). Usage:
  .venv/bin/python -m realenv.worldmodel --data data/real --out runs/real/r3.json
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
from train.train import pick_device

D = 768


@torch.no_grad()
def encode_split(path, model, tok, device, bs=32):
    """Per-trajectory arrays: z_obs[i], z_cmd[i] (frozen pooled), + labels. z_obs[i] is
    the embedding of observation i (output of command i); z_cmd[i] of command i."""
    trajs = [json.loads(l) for l in open(path)]
    # flatten all obs + command texts, encode once, regroup
    obs_texts, cmd_texts, spans = [], [], []
    for tr in trajs:
        start = len(obs_texts)
        for s in tr["steps"]:
            obs_texts.append(render_obs(s))
            cmd_texts.append(s["cmd"])
        spans.append((start, len(obs_texts), tr["task"]))

    def enc(texts):
        out = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=512)
            e = {k: v.to(device) for k, v in e.items()}
            h = model(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1)
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())
            if (i // bs) % 50 == 0:
                print(f"  enc {i}/{len(texts)}", flush=True)
        return torch.cat(out)

    z_obs, z_cmd = enc(obs_texts), enc(cmd_texts)
    out = []
    for (a, b, task), tr in zip(spans, trajs):
        out.append({"z_obs": z_obs[a:b], "z_cmd": z_cmd[a:b], "task": task,
                    "success": torch.tensor([int(s["success"]) for s in tr["steps"]]),
                    "change": torch.tensor([int(s["n_changed"] > 0) for s in tr["steps"]]),
                    "cmds": [s["cmd"] for s in tr["steps"]]})
    return out


def cached_encode(data_root, split, model_name, device):
    cache = pathlib.Path(data_root) / f"emb-{split}.pt"
    if cache.exists():
        print(f"  using cache {cache}", flush=True)
        return torch.load(cache, weights_only=False)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    trajs = encode_split(pathlib.Path(data_root) / f"{split}.jsonl", model, tok, device)
    torch.save(trajs, cache)
    return trajs


def build_transitions(trajs):
    """(z_ctx, z_cmd, z_next, success, change, held). ctx = previous observation (state
    before the command); the first command's ctx is zeros. held=1 if the command's verb is
    a held-out tool (the transfer-to-unseen-tool subset)."""
    from realenv.tasks import HELD_OUT_TOOLS
    ctx, cmd, nxt, suc, chg, held, strs = [], [], [], [], [], [], []
    for tr in trajs:
        n = tr["z_obs"].shape[0]
        for i in range(n):
            ctx.append(tr["z_obs"][i - 1] if i > 0 else torch.zeros(D))
            cmd.append(tr["z_cmd"][i]); nxt.append(tr["z_obs"][i])
            suc.append(tr["success"][i]); chg.append(tr["change"][i]); strs.append(tr["cmds"][i])
            v = tr["cmds"][i].split()[0] if tr["cmds"][i].split() else ""
            held.append(1 if v in HELD_OUT_TOOLS else 0)
    return {"ctx": torch.stack(ctx), "cmd": torch.stack(cmd), "next": torch.stack(nxt),
            "success": torch.stack(suc), "change": torch.stack(chg),
            "held": torch.tensor(held), "cmd_str": strs}


def standardize(a, mu, sd):
    return (a - mu) / sd


class WorldModel(nn.Module):
    """f(z_ctx, z_cmd) -> residual latent delta (z_next = z_ctx + delta) + outcome heads.
    Residual/zero-init delta so the model starts at the copy baseline and must learn the
    command's effect."""

    def __init__(self, d=D, h=512):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.delta = nn.Linear(h, d)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)  # copy at init
        self.succ = nn.Linear(h, 2)
        self.chg = nn.Linear(h, 2)

    def forward(self, z_ctx, z_cmd):
        f = self.trunk(torch.cat([z_ctx, z_cmd], -1))
        return z_ctx + self.delta(f), self.succ(f), self.chg(f)


def _auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return ((pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
            + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean()).item()


def train_wm(tr, device, steps=3000, bs=256, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    net = WorldModel().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = tr["ctx"].shape[0]
    g = torch.Generator().manual_seed(seed)
    ce = nn.functional.cross_entropy
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (bs,), generator=g)
        zc, za, zn = tr["ctx"][idx].to(device), tr["cmd"][idx].to(device), tr["next"][idx].to(device)
        ys, yg = tr["success"][idx].to(device), tr["change"][idx].to(device)
        zhat, ls, lg = net(zc, za)
        loss = ((zhat - zn) ** 2).mean() + ce(ls, ys) + ce(lg, yg)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  wm step {step} loss {loss.item():.4f}", flush=True)
    return net


def train_context_only(tr, device, steps=3000, bs=256, lr=3e-4, seed=0):
    """Baseline: predict outcome from z_ctx ALONE (ignore the command). If the world
    model beats this, knowing the command helps predict the consequence."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, 4)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = tr["ctx"].shape[0]; g = torch.Generator().manual_seed(seed)
    ce = nn.functional.cross_entropy
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        o = net(tr["ctx"][idx].to(device))
        loss = ce(o[:, :2], tr["success"][idx].to(device)) + ce(o[:, 2:], tr["change"][idx].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net


@torch.no_grad()
def _metrics(net, ctx_net, ev, idx, device, seed=0):
    """All transfer metrics on the transition subset `idx` (a LongTensor of indices)."""
    zc, za, zn = ev["ctx"][idx].to(device), ev["cmd"][idx].to(device), ev["next"][idx].to(device)
    zhat, ls, lg = net(zc, za)
    err_wm = ((zhat - zn) ** 2).mean(-1); err_copy = ((zc - zn) ** 2).mean(-1)
    co = ctx_net(zc)
    ysu = ev["success"][idx]
    out = {"n": len(idx),
           "latent": {"wm_mse": err_wm.mean().item(), "copy_mse": err_copy.mean().item(),
                      "wm_beats_copy_frac": (err_wm < err_copy).float().mean().item()},
           "success_auc_wm": _auc(ls.softmax(-1)[:, 1].cpu(), ysu),
           "success_auc_context_only": _auc(co[:, :2].softmax(-1)[:, 1].cpu(), ysu),
           "success_pos_rate": ysu.float().mean().item()}
    # ranking: WM(ctx, true) predicts z_next better than WM(ctx, foil)? copy -> 0.5.
    # random foils (any command) AND hard foils (same verb, different instance).
    rng = random.Random(f"foil:{seed}")
    verbs = [ev["cmd_str"][i].split()[0] if ev["cmd_str"][i].split() else "" for i in idx.tolist()]
    from collections import defaultdict
    by_verb = defaultdict(list)
    for j, v in enumerate(verbs):
        by_verb[v].append(j)
    d_true = err_wm
    def rank_with(foil_local):
        za_f = za[torch.tensor(foil_local, device=za.device)]
        d_f = ((net(zc, za_f)[0] - zn) ** 2).mean(-1)
        return (d_true < d_f).float().mean().item()
    rnd = [rank_with([rng.randrange(len(idx)) for _ in idx]) for _ in range(4)]
    hard = [rank_with([rng.choice(by_verb[v]) for v in verbs]) for _ in range(4)]
    out["command_rank_random_foils"] = sum(rnd) / len(rnd)
    out["command_rank_sameverb_foils"] = sum(hard) / len(hard)
    return out


def evaluate(net, ctx_net, ev, device, seed=0):
    n = ev["ctx"].shape[0]
    held_idx = torch.nonzero(ev["held"] == 1).squeeze(-1)
    return {"all": _metrics(net, ctx_net, ev, torch.arange(n), device, seed),
            "held_out_tool_steps": _metrics(net, ctx_net, ev, held_idx, device, seed)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = pick_device("auto")
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=== encode (cached) ===", flush=True)
    tr = build_transitions(cached_encode(args.data, "train", args.model, device))
    ev = build_transitions(cached_encode(args.data, "val", args.model, device))
    mu = tr["next"].mean(0, keepdim=True); sd = tr["next"].std(0, keepdim=True).clamp(min=1e-6)
    for k in ("ctx", "cmd", "next"):
        tr[k] = standardize(tr[k], mu, sd); ev[k] = standardize(ev[k], mu, sd)
    print(f"train transitions {tr['ctx'].shape[0]} | val {ev['ctx'].shape[0]} "
          f"| held-out-tool steps {int((ev['held']==1).sum())}", flush=True)

    per_seed = []
    for s in seeds:
        print(f"=== seed {s}: train + eval ===", flush=True)
        net = train_wm(tr, device, steps=args.steps, seed=s)
        ctx_net = train_context_only(tr, device, steps=args.steps, seed=s)
        r = evaluate(net, ctx_net, ev, device, seed=s)
        per_seed.append(r)
        print(f"  seed {s} held-out-tool: {json.dumps(r['held_out_tool_steps'])}", flush=True)

    def agg(section, key, sub=None):
        vals = [(p[section][key][sub] if sub else p[section][key]) for p in per_seed]
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        m = sum(vals) / len(vals)
        return {"mean": round(m, 4), "std": round((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5, 4)}

    report = {"data": args.data, "seeds": seeds, "steps": args.steps,
              "train_transitions": tr["ctx"].shape[0], "val_transitions": ev["ctx"].shape[0],
              "held_out_tool_steps": int((ev["held"] == 1).sum())}
    for section in ("all", "held_out_tool_steps"):
        report[section] = {
            "wm_beats_copy_frac": agg(section, "latent", "wm_beats_copy_frac"),
            "wm_mse": agg(section, "latent", "wm_mse"),
            "copy_mse": agg(section, "latent", "copy_mse"),
            "success_auc_wm": agg(section, "success_auc_wm"),
            "success_auc_context_only": agg(section, "success_auc_context_only"),
            "command_rank_random_foils": agg(section, "command_rank_random_foils"),
            "command_rank_sameverb_foils": agg(section, "command_rank_sameverb_foils"),
        }
    print("=== AGGREGATE (held-out-tool steps) ===", flush=True)
    print(json.dumps(report["held_out_tool_steps"], indent=1), flush=True)
    if args.out:
        p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
