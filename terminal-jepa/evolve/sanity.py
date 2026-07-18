"""Champion sanity arms — the skill-mandated validation that an evolved champion is a real
world-model improvement, not a retrieval-metric artifact. Run on the final-test split, under the
CHAMPION'S objective:
  (1) gen-twin: does latent prediction (the JEPA bet) still beat a compute-matched generative
      token-reconstruction twin? (fitness rewards retrieval-shaped objectives, so verify the
      champion didn't just win by mimicking the metric while losing the JEPA advantage).
  (2) history ablation: does the exploration history still drive the gain (full > matched-capacity
      self-only transformer), or did the objective let a history-free model close the gap?

  .venv/bin/python -m evolve.sanity --genome g.json --split final --seeds 0,1,2 --steps 4000
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from realenv import seq_worldmodel as M
from evolve import genome as G
from evolve.splits import split_val


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--split", default="final", choices=["inner", "final"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--data", default="data/dockerfs")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--top-v", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    gen = json.load(open(args.genome))
    loss_fn = G.load_objective(gen)
    device = M.pick_device()
    seeds = [int(x) for x in args.seeds.split(",")]

    train_seqs = M.cached_encode(args.data, "train", args.model, device)
    val_seqs = M.cached_encode(args.data, "val", args.model, device)
    mo, so, mc, sc = M.standardize_stats(train_seqs)
    M.apply_stats(train_seqs, mo, so, mc, sc)
    M.apply_stats(val_seqs, mo, so, mc, sc)
    evalset = split_val(val_seqs, args.split)

    tok, vmap = M.build_vocab(args.data, "train", args.model, args.top_v)
    M.attach_bags(train_seqs, pathlib.Path(args.data) / "train.jsonl", tok, vmap)
    M.attach_bags(val_seqs, pathlib.Path(args.data) / "val.jsonl", tok, vmap)

    print(f"=== gen-twin under objective '{gen['chunks']['objective']['impl']}' on {args.split} ===", flush=True)
    s = seeds[0]
    fit, _ = M.split_train_dev(train_seqs, seed=s)
    jnet = M.train_model("jepa", fit, device, steps=args.steps, seed=s, jepa_loss=loss_fn)
    jflat = M.flatten_predictions(jnet, evalset, device)
    jepa_ret = M.retrieval(jflat["pred"], jflat["true"], jflat["verbs"], seed=s)
    gt = M.run_gen_twin(fit, evalset, device, vmap, args.steps, s, jepa_ret)

    print(f"=== history ablation under objective '{gen['chunks']['objective']['impl']}' on {args.split} ===", flush=True)
    ha = M.run_history_ablation(train_seqs, evalset, device, args.steps, seeds, jepa_loss=loss_fn)

    report = {
        "genome": gen["id"], "objective": gen["chunks"]["objective"]["impl"], "split": args.split,
        "gen_twin_seed": s,
        "gen_twin": {"jepa_top1_sameverb": round(gt["jepa"]["top1_sameverb"], 4),
                     "recon_top1_sameverb": round(gt["recon"]["top1_sameverb"], 4),
                     "jepa_minus_recon": gt["jepa_minus_recon_top1_sameverb"]},
        "history_ablation": {
            "seeds": seeds,
            "full_content_top1": ha["full"]["content_top1_sameverb"],
            "masked_content_top1": ha["masked"]["content_top1_sameverb"],
            "history_gain_content_top1": ha["history_gain_content_top1"]},
    }
    print("=== SANITY REPORT ===\n" + json.dumps(report, indent=1), flush=True)
    if args.out:
        p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}", flush=True)
    return report


if __name__ == "__main__":
    main()
