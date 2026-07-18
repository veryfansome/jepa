"""Post-hoc IDM per-head readout: load a sigreg+idm checkpoint, evaluate verb/arg1/arg2
accuracy on a split, and save a self-auditing JSON artifact (accuracies + label floors
computed from the same sampled batches).

Usage:
  .venv/bin/python -m probes.idm_readout --ckpt runs/sigreg-idm-s0-4k/ckpt.pt \
      --data data/v0 --split val --out runs/sigreg-idm-s0-4k/idm-readout.json
"""

import argparse
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from models import nets
from models.data import TrajectoryData
from train.train import HORIZON, encode_all, pick_device

HEADS = ["verb", "arg1", "arg2"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/v0")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--regime", default="both")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if "idm" not in ckpt["modules"]:
        raise SystemExit(f"{args.ckpt} has no IDM head (arm was {ckpt.get('arm')})")
    enc = nets.build_encoder(ckpt.get("encoder_type", "cls"))
    idm = nets.IDMHead(d=enc.d_out)
    nets.load_encoder_state(enc, ckpt["modules"]["encoder"])
    idm.load_state_dict(ckpt["modules"]["idm"])
    enc.to(device).eval()
    idm.to(device).eval()

    data = TrajectoryData(
        pathlib.Path(args.data) / f"{args.split}.jsonl", args.regime,
        keep_states=False,
    )
    rng = random.Random(args.seed)
    accs = {f"{h}_acc": 0.0 for h in HEADS}
    label_counts = {h: collections.Counter() for h in HEADS}
    n_labels = 0
    with torch.no_grad():
        for _ in range(args.batches):
            b = data.sample_windows(args.batch_size, HORIZON, rng)
            z = encode_all(enc, b["obs"].to(device))
            labels = b["act_labels"].reshape(-1, 3)
            _, m = idm.loss(
                z[:, :-1].reshape(-1, z.shape[-1]),
                z[:, 1:].reshape(-1, z.shape[-1]),
                labels.to(device),
            )
            for h in HEADS:
                accs[f"{h}_acc"] += m[f"{h}_acc"].item() / args.batches
            for i, h in enumerate(HEADS):
                label_counts[h].update(labels[:, i].tolist())
            n_labels += labels.shape[0]

    report = {
        "ckpt": args.ckpt,
        "arm": ckpt.get("arm"),
        "data": args.data,
        "split": args.split,
        "regime": args.regime,
        # Full sampling config, so the artifact reproduces without reading defaults.
        "batches": args.batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "horizon": HORIZON,
        "device": str(device),
        "transitions_evaluated": n_labels,
        "accuracy": {k: round(v, 4) for k, v in accs.items()},
        # Floors from the same sampled labels, so the artifact is self-auditing.
        "majority_floors": {
            h: round(label_counts[h].most_common(1)[0][1] / n_labels, 4) for h in HEADS
        },
        "distinct_labels": {h: len(label_counts[h]) for h in HEADS},
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
