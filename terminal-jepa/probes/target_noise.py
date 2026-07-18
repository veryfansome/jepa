"""Zero-training target-noise measurement for gate-2 decision 2 (clean vs raw
prediction targets over frozen features; status doc "Path to a working agent").

Raw-rendered targets carry nuisance the predictor cannot reduce: within a
trajectory the banner is fixed, but the dynamic-noise line is resampled every
step, and its tokens contaminate every path line's contextual vector. This
script measures, over frozen-encoder per-line features:

- nuisance: path-line movement when ONLY the noise line changes (same state,
  same banner, step t -> t+1) — the exact within-trajectory raw-target case;
- signal: movement of the one line an action actually changed (write to an
  existing file with a different content class, header held identical);
- spillover: movement of the other, textually unchanged lines under that same
  state change (predictable in principle — reported for context);
- banner-swap nuisance (cross-trajectory case) as a secondary reading.

Pre-registered criterion (2026-07-13, before measurement): raw targets sum
nuisance over all ~L path lines while signal lives in ~1 line, so if
L * median(nuisance_dz)^2 > 10% of median(signal_dz)^2, clean-rendered targets
(banner=None, noise=None) win by default and no training A/B is needed.

Usage:
  .venv/bin/python -m probes.target_noise --data data/v1 --out runs/frozen-modernbert-v1/target-noise.json
"""

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from env import actions, render, vocab  # noqa: E402
from probes.frozen_probe import encode_batch, load_trajs
from train.train import pick_device

POWER_RATIO_CRITERION = 0.10


def line_map(rec):
    return {idx: rec["file_vecs"][k].float() for k, idx in enumerate(rec["file_idx"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    trajs = load_trajs(pathlib.Path(args.data) / "val.jsonl", "both", 200)
    rng = random.Random(f"target-noise:{args.seed}")

    d_noise, d_banner, d_signal, d_spill, n_lines = [], [], [], [], []
    made = 0
    while made < args.n:
        tr = trajs[rng.randrange(len(trajs))]
        t = rng.randrange(len(tr["states"]))
        st = tr["states"][t]
        if not st.files:
            continue
        b1 = tr["banner_id"]
        noise = 0  # any stream works; what matters is the line differing across steps
        b2 = rng.choice([b for b in range(len(vocab.BANNERS)) if b != b1])
        # Signal construction: write a different content class to an existing file,
        # so the changed line exists in both renders and only its [cK] tag differs.
        p = rng.choice(sorted(st.files))
        k2 = rng.choice([k for k in range(vocab.N_CONTENT) if k != st.files[p]])
        act = ("write", vocab.path_to_str(p), f"c{k2}")
        res = actions.apply(st, act)
        if res.ttype != actions.STATE_CHANGING:
            continue
        texts = [
            render.render_full(st, b1, noise, step=t),        # R0 reference
            render.render_full(st, b1, noise, step=t + 1),    # R1 noise line only
            render.render_full(st, b2, noise, step=t),        # R2 banner swap
            render.render_full(res.state, b1, noise, step=t), # R3 one-line change
        ]
        r0, r1, r2, r3 = encode_batch(model, tok, texts, device)
        m0, m1, m2, m3 = map(line_map, (r0, r1, r2, r3))
        changed = vocab.FILE_PATH_INDEX[p]
        d_noise.extend((m0[i] - m1[i]).norm().item() for i in m0 if i in m1)
        d_banner.extend((m0[i] - m2[i]).norm().item() for i in m0 if i in m2)
        d_signal.append((m0[changed] - m3[changed]).norm().item())
        d_spill.extend((m0[i] - m3[i]).norm().item()
                       for i in m0 if i in m3 and i != changed)
        n_lines.append(len(m0))
        made += 1
        if made % 50 == 0:
            print(f"  {made}/{args.n}", flush=True)

    def med(x):
        return torch.tensor(x).median().item()

    L = med([float(n) for n in n_lines])
    nuis, sig = med(d_noise), med(d_signal)
    power_ratio = (L * nuis**2) / max(sig**2, 1e-12)
    report = {
        "model": args.model,
        "data": args.data,
        "n_samples": args.n,
        "criterion": f"clean targets if L*nuisance^2/signal^2 > {POWER_RATIO_CRITERION}",
        "file_lines_per_obs_median": L,
        "nuisance_noiseline_line_dz_median": nuis,
        "nuisance_banner_line_dz_median": med(d_banner),
        "signal_changed_line_dz_median": sig,
        "spillover_unchanged_line_dz_median": med(d_spill),
        "aggregate_power_ratio_nuisance_over_signal": power_ratio,
        "verdict": "clean-targets" if power_ratio > POWER_RATIO_CRITERION else "ambiguous-run-ab",
    }
    print(json.dumps(report, indent=1), flush=True)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
