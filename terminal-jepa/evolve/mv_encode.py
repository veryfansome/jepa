"""Build a MULTI-VECTOR data root from an existing single-vector root: copy the single-vector
caches verbatim (so z_obs/z_cmd — the target/eval space — are bit-identical to the source root)
and add per-step "z_obs_multi" [n,K,D] + "obs_valid" [n,K] encoded from the perception recipe's
render_obs_multi segments.

  .venv/bin/python -m evolve.mv_encode --perception e5_multivec --src data/dockerfs-e5 \
      --out data/dockerfs-e5mv
"""

import argparse
import importlib
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from realenv.seq_worldmodel import D, pick_device


@torch.no_grad()
def encode_texts(texts, tok, model, device, bs=96, tag=""):
    """Length-sorted batched mean-pool-free encode: returns [N, D] pooled by the recipe."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.zeros(len(texts), D)
    percep = encode_texts.percep
    for i in range(0, len(order), bs):
        bidx = order[i:i + bs]
        e = tok([texts[j] for j in bidx], return_tensors="pt", padding=True,
                truncation=True, max_length=256)
        e = {k: v.to(device) for k, v in e.items()}
        h = model(**e).last_hidden_state
        pooled = percep.pool(h, e["attention_mask"]).float().cpu()
        for k, j in enumerate(bidx):
            out[j] = pooled[k]
        if (i // bs) % 100 == 0:
            print(f"  enc {tag} {i}/{len(texts)}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", required=True)
    ap.add_argument("--src", default="data/dockerfs-e5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=96)
    args = ap.parse_args(argv)

    # The multi-vector render (render_obs_multi/K/pool/MODEL) may live in a perception impl or be
    # self-contained in a stream impl (e.g. r7_role_multivec). Try perception first, then stream.
    try:
        percep = importlib.import_module(f"evolve.chunks.perception.{args.perception}")
        if not hasattr(percep, "render_obs_multi"):
            raise ImportError
    except ImportError:
        percep = importlib.import_module(f"evolve.chunks.stream.{args.perception}")
    for fn in ("render_obs_multi", "pool"):
        if not hasattr(percep, fn):
            raise AttributeError(f"impl '{args.perception}' missing {fn}")
    K = percep.K
    encode_texts.percep = percep

    src, out = pathlib.Path(args.src), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(percep.MODEL)
    model = AutoModel.from_pretrained(percep.MODEL).to(device).eval()

    for split in ("train", "val"):
        shutil.copy(src / f"{split}.jsonl", out / f"{split}.jsonl")
        seqs = torch.load(src / f"emb-seq-{split}.pt", weights_only=False)
        raws = [json.loads(l) for l in open(src / f"{split}.jsonl")]
        assert len(raws) == len(seqs), f"{split}: raw {len(raws)} != cached {len(seqs)}"

        texts, spans = [], []   # spans[si] = list over steps of (start, count)
        for s, r in zip(seqs, raws):
            assert s["image"] == r["image"], "cache/raw order mismatch"
            seq_spans = []
            for st in r["steps"]:
                segs = percep.render_obs_multi(st)
                assert 1 <= len(segs) <= K
                seq_spans.append((len(texts), len(segs)))
                texts.extend(segs)
            spans.append(seq_spans)

        print(f"[{split}] {len(texts)} segment texts over {len(seqs)} seqs", flush=True)
        emb = encode_texts(texts, tok, model, device, bs=args.bs, tag=split)

        for s, seq_spans in zip(seqs, spans):
            n = s["z_obs"].shape[0]
            assert n == len(seq_spans)
            zm = torch.zeros(n, K, D)
            valid = torch.zeros(n, K, dtype=torch.bool)
            for i, (start, cnt) in enumerate(seq_spans):
                zm[i, :cnt] = emb[start:start + cnt]
                valid[i, :cnt] = True
            s["z_obs_multi"] = zm
            s["obs_valid"] = valid
        torch.save(seqs, out / f"emb-seq-{split}.pt")
        print(f"[{split}] wrote {out}/emb-seq-{split}.pt", flush=True)


if __name__ == "__main__":
    main()
