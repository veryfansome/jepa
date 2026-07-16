"""R2 day-zero probe: does a frozen encoder's embedding of a REAL observation linearly
predict outcomes (success, state-change) and TRANSFER to the held-out tool (git)? And
does pretraining help over a random-init encoder in the real regime (the finding-14
test, where our clean synthetic domain showed random==pretrained)?

Zero training of the encoder — only linear/MLP probe heads are fit, on train tools
(files/text/python) and evaluated on held-out git.

Usage: .venv/bin/python -m realenv.probe_outcomes --data data/real --out runs/real/r2.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from realenv.collect import render_obs
from train.train import pick_device


def load_steps(path):
    obs, succ, chg = [], [], []
    for line in open(path):
        for s in json.loads(line)["steps"]:
            obs.append(render_obs(s))
            succ.append(int(s["success"]))
            chg.append(int(s["n_changed"] > 0))
    return obs, torch.tensor(succ), torch.tensor(chg)


@torch.no_grad()
def encode(texts, model, tok, device, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())
        if (i // bs) % 40 == 0:
            print(f"  encode {i + len(texts[i:i+bs])}/{len(texts)}", flush=True)
    return torch.cat(out)


def _std(ztr, zev):
    mu = ztr.mean(0, keepdim=True)
    sd = ztr.std(0, keepdim=True).clamp(min=1e-6)
    return (ztr - mu) / sd, (zev - mu) / sd


def _balacc(pred, y):
    tpr = (pred[y == 1] == 1).float().mean().item() if (y == 1).any() else float("nan")
    tnr = (pred[y == 0] == 0).float().mean().item() if (y == 0).any() else float("nan")
    return (tpr + tnr) / 2


def _auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()
    tie = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().mean().item()
    return gt + 0.5 * tie


def fit_eval(ztr, ytr, zev, yev, device, kind, steps=400, lr=3e-3, seed=0):
    d = ztr.shape[1]
    head = (nn.Linear(d, 2) if kind == "linear"
            else nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 2)))
    torch.manual_seed(seed)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    ce = nn.functional.cross_entropy
    # class-balanced sampling weight (train success/change is imbalanced)
    for _ in range(steps):
        idx = torch.randint(0, ztr.shape[0], (min(2048, ztr.shape[0]),), generator=g)
        opt.zero_grad(set_to_none=True)
        ce(head(ztr[idx].to(device)), ytr[idx].to(device)).backward()
        opt.step()
    with torch.no_grad():
        logit = head(zev.to(device)).cpu()
    return {"balacc": _balacc(logit.argmax(-1), yev),
            "auc": _auc(logit.softmax(-1)[:, 1], yev)}


def probe_encoder(model, tok, device, tr, va, tag):
    obs_tr, succ_tr, chg_tr = tr
    obs_va, succ_va, chg_va = va
    z_tr = encode(obs_tr, model, tok, device)
    z_va = encode(obs_va, model, tok, device)
    z_tr, z_va = _std(z_tr, z_va)
    out = {}
    for target, ytr, yva in [("success", succ_tr, succ_va), ("state_change", chg_tr, chg_va)]:
        out[target] = {
            "held_out_git_pos_rate": yva.float().mean().item(),
            "linear": fit_eval(z_tr, ytr, z_va, yva, device, "linear"),
            "mlp": fit_eval(z_tr, ytr, z_va, yva, device, "mlp"),
        }
        print(f"  [{tag}] {target}: linear {out[target]['linear']} (git pos rate "
              f"{out[target]['held_out_git_pos_rate']:.3f})", flush=True)
    return out


def main(argv=None):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = pick_device("auto")
    root = pathlib.Path(args.data)
    tr = load_steps(root / "train.jsonl")
    va = load_steps(root / "val.jsonl")
    print(f"train steps {len(tr[0])} | held-out git steps {len(va[0])}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    report = {"data": args.data, "model": args.model,
              "train_steps": len(tr[0]), "git_steps": len(va[0])}
    print("=== PRETRAINED ===", flush=True)
    m = AutoModel.from_pretrained(args.model).to(device).eval()
    report["pretrained"] = probe_encoder(m, tok, device, tr, va, "pretrained")
    del m
    print("=== RANDOM-INIT (architecture-matched floor) ===", flush=True)
    torch.manual_seed(args.seed)
    mr = AutoModel.from_config(AutoConfig.from_pretrained(args.model)).float().to(device).eval()
    report["random_init"] = probe_encoder(mr, tok, device, tr, va, "random-init")

    # pretraining margin on the transfer task
    report["pretraining_margin_git"] = {
        t: {m2: round(report["pretrained"][t][m2]["auc"] - report["random_init"][t][m2]["auc"], 3)
            for m2 in ("linear", "mlp")}
        for t in ("success", "state_change")
    }
    print("pretraining AUC margin (git):", json.dumps(report["pretraining_margin_git"]), flush=True)
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
