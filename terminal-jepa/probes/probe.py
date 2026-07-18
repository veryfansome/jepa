"""Probing harness (terminal-jepa.md §4, §7).

Probes are fit on train-split (train-layout) states and evaluated on val-split
(held-out-layout) states. Linear probe is the pass bar; MLP is reported for the
linear-vs-MLP gap. Metrics follow the pre-registered protocol: cwd accuracy,
file-existence pooled balanced accuracy, content accuracy conditioned on existence,
their macro average, plus the banner-identity probe and the banner-swap sensitivity
audit for RQ2.

Usage:
  .venv/bin/python -m probes.probe --data data/v0 --ckpt runs/sigreg-s0/ckpt.pt --out report.json
  .venv/bin/python -m probes.probe --data data/v0 --ckpt untrained --seed 0 --out report.json
"""

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from env import actions, render, vocab  # noqa: E402
from models import nets
from models.data import TrajectoryData, encode_text
from train.train import pick_device

N_CWD = len(vocab.CWD_PATHS)
N_FILES = len(vocab.FILE_PATHS)
N_CLS = vocab.N_CONTENT
N_BANNERS = len(vocab.BANNERS)


def load_encoder(ckpt_path, seed, device, encoder_type="cls"):
    """Returns (encoder, legacy_keys_dropped). A nonempty legacy list means the
    checkpoint predates an architecture change and does NOT reproduce its original
    embeddings — reports must carry that provenance."""
    legacy = []
    if ckpt_path != "untrained":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        enc = nets.build_encoder(ckpt.get("encoder_type", "cls"))
        legacy = nets.load_encoder_state(enc, ckpt["modules"]["encoder"])
    else:
        torch.manual_seed(seed)
        enc = nets.build_encoder(encoder_type)  # fresh random init under the seed
    return enc.to(device).eval(), legacy


@torch.no_grad()
def extract(enc, examples, device, chunk=256):
    zs, feats, banners = [], [], []
    for i in range(0, len(examples), chunk):
        batch = examples[i : i + chunk]
        ids = torch.stack([e[0] for e in batch]).long().to(device)
        zs.append(enc(ids).cpu())
        feats.extend(e[1] for e in batch)
        banners.extend(e[2] for e in batch)
    z = torch.cat(zs)
    y = {
        "cwd": torch.tensor([f["cwd_index"] for f in feats]),
        "exists": torch.tensor([f["file_exists"] for f in feats], dtype=torch.float),
        "cls": torch.tensor([f["file_class"] for f in feats]),
        "banner": torch.tensor(banners),
    }
    return z, y


def examples_from(trajs):
    out = []
    for tr in trajs:
        for i, st in enumerate(tr["states"]):
            out.append((tr["obs"][i], st.features(), tr["banner_id"]))
    return out


def chance_floors(y):
    """Chance/majority floors for every probe metric, saved into the report so status
    docs can't desync from artifacts."""
    cls_vals = y["cls"][y["cls"] >= 0]
    return {
        "cwd_majority": (y["cwd"].bincount().max() / len(y["cwd"])).item(),
        "exists_pos_rate": y["exists"].mean().item(),
        "exists_balacc_chance": 0.5,
        "cls_majority_given_exists": (cls_vals.bincount().max() / len(cls_vals)).item(),
        "banner_chance": 1.0 / N_BANNERS,
    }


def make_head(kind, d_in, d_out):
    if kind == "linear":
        return nn.Linear(d_in, d_out)
    return nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, d_out))


PROBE_FIT = {"steps": 400, "batch": 4096, "lr": 3e-3, "standardized": True}


def fit_head(head, z, target, loss_fn, device,
             steps=PROBE_FIT["steps"], bs=PROBE_FIT["batch"], lr=PROBE_FIT["lr"]):
    head = head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    n = z.shape[0]
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(head(z[idx].to(device)), target[idx].to(device))
        loss.backward()
        opt.step()
    return head.eval()


def standardize(z_fit, z_eval, eps=1e-6):
    """Per-feature standardization using fit-split statistics: encoder arms differ in
    embedding scale by up to ~5x (z_std 0.36-1.89), and one fixed probe lr shouldn't be
    silently mistuned per arm (fidelity audit, 2026-07-10)."""
    mu = z_fit.mean(dim=0, keepdim=True)
    sd = z_fit.std(dim=0, keepdim=True).clamp(min=eps)
    return (z_fit - mu) / sd, (z_eval - mu) / sd


def masked_cls_loss(logits, y):
    logits = logits.reshape(-1, N_FILES, N_CLS)
    mask = y >= 0
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[mask], y[mask])


PATH_DEPTH = torch.tensor([len(p) for p in vocab.FILE_PATHS])  # components: 1, 2, or 3


def _balacc(pred, target):
    pos, neg = target == 1, target == 0
    tpr = (pred[pos] == 1).float().mean().item() if pos.any() else float("nan")
    tnr = (pred[neg] == 0).float().mean().item() if neg.any() else float("nan")
    return (tpr + tnr) / 2


@torch.no_grad()
def evaluate(heads, z, y, device):
    z = z.to(device)
    out = {}
    out["cwd_acc"] = (
        (heads["cwd"](z).argmax(-1).cpu() == y["cwd"]).float().mean().item()
    )
    pred = (heads["exists"](z).cpu() > 0).float()
    out["exists_balacc"] = _balacc(pred, y["exists"])
    logits = heads["cls"](z).reshape(-1, N_FILES, N_CLS).cpu()
    cls_pred = logits.argmax(-1)
    mask = y["cls"] >= 0
    out["cls_acc_given_exists"] = (
        (cls_pred[mask] == y["cls"][mask]).float().mean().item()
    )
    out["macro"] = (out["cwd_acc"] + out["exists_balacc"] + out["cls_acc_given_exists"]) / 3
    out["banner_acc"] = (
        (heads["banner"](z).argmax(-1).cpu() == y["banner"]).float().mean().item()
    )
    # Compositionality diagnostic: do deeper paths fail harder?
    out["by_depth"] = {}
    for d in (1, 2, 3):
        cols = PATH_DEPTH == d
        dmask = mask[:, cols]
        out["by_depth"][d] = {
            "exists_balacc": _balacc(pred[:, cols], y["exists"][:, cols]),
            "cls_acc_given_exists": (
                (cls_pred[:, cols][dmask] == y["cls"][:, cols][dmask]).float().mean().item()
                if dmask.any() else float("nan")
            ),
        }
    return out


@torch.no_grad()
def banner_swap_audit(enc, data_val, device, n=300, seed=0):
    """RQ2: ||dz|| when only the banner changes vs when the state genuinely changes."""
    rng = random.Random(f"audit:{seed}")
    d_banner, d_state = [], []
    for _ in range(n):
        tr = data_val.trajs[rng.randrange(len(data_val.trajs))]
        i = rng.randrange(len(tr["states"]))
        st = tr["states"][i]
        b1 = tr["banner_id"]
        b2 = rng.choice([b for b in range(N_BANNERS) if b != b1])
        for _ in range(20):
            act = actions.sample_valid(st, rng)
            res = actions.apply(st, act)
            if res.ttype == actions.STATE_CHANGING:
                break
        ids = []
        for s, b in [(st, b1), (st, b2), (res.state, b1)]:
            enc_ids, trunc = encode_text(
                render.render_full(s, b, None, step=i), data_val.obs_len
            )
            assert not trunc, "audit observation truncated — dz would be confounded"
            ids.append(enc_ids)
        z = enc(torch.tensor(ids, dtype=torch.long, device=device)).cpu()
        d_banner.append((z[0] - z[1]).norm().item())
        d_state.append((z[0] - z[2]).norm().item())
    d_banner, d_state = torch.tensor(d_banner), torch.tensor(d_state)
    return {
        "banner_swap_dz_median": d_banner.median().item(),
        "state_change_dz_median": d_state.median().item(),
        "ratio": (d_banner.median() / d_state.median().clamp(min=1e-9)).item(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v0")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--regime", default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-trajs", type=int, default=800)
    ap.add_argument("--max-val-trajs", type=int, default=200)
    ap.add_argument("--seen-layouts", action="store_true",
                    help="fit and evaluate on DISJOINT trajectories from the train "
                         "split (same layout pool) — the seen-layout diagnostic")
    ap.add_argument("--encoder", default="cls", choices=list(nets.ENCODER_TYPES),
                    help="encoder type for --ckpt untrained (checkpoints are "
                         "self-describing)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-legacy", action="store_true",
                    help="permit writing --out for a legacy (function-changed) ckpt")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    torch.manual_seed(args.seed)  # head init must be reproducible for trained ckpts too
    enc, legacy_dropped = load_encoder(args.ckpt, args.seed, device, args.encoder)
    if legacy_dropped and args.out and not args.allow_legacy:
        raise SystemExit(
            f"refusing to write {args.out}: legacy keys {legacy_dropped} were dropped, "
            f"so these numbers describe a model that never existed; pass "
            f"--allow-legacy to override"
        )
    root = pathlib.Path(args.data)
    if args.seen_layouts:
        d_all = TrajectoryData(
            root / "train.jsonl", args.regime,
            args.max_train_trajs + args.max_val_trajs,
        )
        fit_trajs = d_all.trajs[: args.max_train_trajs]
        eval_trajs = d_all.trajs[args.max_train_trajs :]
        # Stratify: layouts are drawn with replacement, so ~11% of eval trajectories
        # carry layouts absent from the fit slice — drop them so this is strictly a
        # seen-layout diagnostic, and record how many were dropped.
        fit_ids = {tr["layout_id"] for tr in fit_trajs}
        n_before = len(eval_trajs)
        eval_trajs = [tr for tr in eval_trajs if tr["layout_id"] in fit_ids]
        dropped_unseen = n_before - len(eval_trajs)
        import types

        d_val = types.SimpleNamespace(trajs=eval_trajs, obs_len=d_all.obs_len)
        z_tr, y_tr = extract(enc, examples_from(fit_trajs), device)
        z_va, y_va = extract(enc, examples_from(eval_trajs), device)
        eval_split = "seen-layouts-heldout-trajectories"
    else:
        d_train = TrajectoryData(root / "train.jsonl", args.regime, args.max_train_trajs)
        d_val = TrajectoryData(root / "val.jsonl", args.regime, args.max_val_trajs)
        z_tr, y_tr = extract(enc, d_train.probe_examples(), device)
        z_va, y_va = extract(enc, d_val.probe_examples(), device)
        eval_split = "val-heldout-layouts"
        dropped_unseen = 0
    print(f"fit states={z_tr.shape[0]} eval states={z_va.shape[0]} ({eval_split})",
          flush=True)
    z_tr, z_va = standardize(z_tr, z_va)

    report = {
        "ckpt": args.ckpt,
        "regime": args.regime,
        "eval_split": eval_split,
        "fit_states": z_tr.shape[0],
        "eval_states": z_va.shape[0],
        "eval_trajs_dropped_unseen_layout": dropped_unseen,
        "probe_fit": dict(PROBE_FIT),  # actual fit_head defaults, not a copy
        "legacy_keys_dropped": legacy_dropped,
        "chance_floors": chance_floors(y_va),
    }
    ce = nn.functional.cross_entropy
    bce = nn.functional.binary_cross_entropy_with_logits
    d_in = z_tr.shape[1]
    report["encoder_d_out"] = d_in
    for kind in ["linear", "mlp"]:
        heads = {
            "cwd": fit_head(make_head(kind, d_in, N_CWD), z_tr, y_tr["cwd"], ce, device),
            "exists": fit_head(
                make_head(kind, d_in, N_FILES), z_tr, y_tr["exists"], bce, device
            ),
            "cls": fit_head(
                make_head(kind, d_in, N_FILES * N_CLS),
                z_tr, y_tr["cls"], masked_cls_loss, device,
            ),
            "banner": fit_head(
                make_head(kind, d_in, N_BANNERS), z_tr, y_tr["banner"], ce, device
            ),
        }
        report[kind] = evaluate(heads, z_va, y_va, device)
        print(kind, json.dumps(report[kind]), flush=True)

    if args.regime in ("banner", "both"):
        report["banner_swap_audit"] = banner_swap_audit(enc, d_val, device, seed=args.seed)
        print("audit", json.dumps(report["banner_swap_audit"]), flush=True)

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
