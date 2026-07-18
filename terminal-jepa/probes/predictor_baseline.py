"""Predictor-vs-copy diagnostic: does the trained predictor beat the trivial
"copy z_t" baseline that an identity-at-init predictor starts from? Split by whether
the transition actually changed state. Guards against the vacuous-objective worry:
identity-at-init + slowly-changing states could in principle satisfy the prediction
loss without learning dynamics.

Usage:
  .venv/bin/python -m probes.predictor_baseline --ckpt runs/sigreg-v1-s0-4k/ckpt.pt \
      --data data/v1 --out runs/sigreg-v1-s0-4k/predictor-vs-copy.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from models import nets
from models.data import TrajectoryData
from train.train import pick_device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--regime", default="both")
    ap.add_argument("--max-trajs", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    enc_type = ckpt.get("encoder_type", "cls")
    m = nets.build_models(enc_type)
    for k in m:
        if k == "encoder":
            nets.load_encoder_state(m[k], ckpt["modules"][k])
        else:
            m[k].load_state_dict(ckpt["modules"][k])
        m[k].to(device).eval()

    d = TrajectoryData(
        pathlib.Path(args.data) / f"{args.split}.jsonl", args.regime, args.max_trajs
    )
    errs = {"changed": {"copy": [], "pred": []}, "noop": {"copy": [], "pred": []}}
    with torch.no_grad():
        for tr in d.trajs:
            obs = tr["obs"].long().to(device)
            z = torch.cat([m["encoder"](obs[i : i + 8]) for i in range(0, obs.shape[0], 8)])
            a = m["action_encoder"](tr["acts"].long().to(device))
            zhat = m["predictor"](z[:-1], a)
            for i in range(a.shape[0]):
                kind = "changed" if tr["states"][i + 1] != tr["states"][i] else "noop"
                errs[kind]["copy"].append(((z[i + 1] - z[i]) ** 2).mean().item())
                errs[kind]["pred"].append(((z[i + 1] - zhat[i]) ** 2).mean().item())

    report = {
        "ckpt": args.ckpt,
        "encoder_type": enc_type,
        "data": args.data,
        "split": args.split,
        "regime": args.regime,
        "max_trajs": args.max_trajs,
        "device": str(device),
    }
    for kind, e in errs.items():
        n = len(e["copy"])
        copy_mse = sum(e["copy"]) / n
        pred_mse = sum(e["pred"]) / n
        report[kind] = {
            "n": n,
            "copy_baseline_mse": copy_mse,
            "predictor_mse": pred_mse,
            "ratio_pred_over_copy": pred_mse / max(copy_mse, 1e-12),
        }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
