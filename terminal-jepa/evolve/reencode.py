"""Re-encode the raw dockerfs observations/commands with a PERCEPTION recipe (encoder model +
render + pooling) into a NEW dataset root's frozen-embedding caches, so a genome can be scored on
it via `cli score --data <root>`. This is how the `perception` (Tier-1: render/pooling) and
`encoder` (Tier-2: swap the model) chunks are evolved — each recipe is a data-side re-encode, like
the exploration chunk; the retrieval space + baselines are recomputed per root, so fitness stays
the honest content-verb margin within that space. gen-0 (`baseline` perception) reproduces the
original dockerfs cache.

  .venv/bin/python -m evolve.reencode --perception cls --src data/dockerfs --out data/dockerfs-p-cls
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


def load_perception(name):
    mod = importlib.import_module(f"evolve.chunks.perception.{name}")
    for fn in ("render_obs", "render_cmd", "pool"):
        if not hasattr(mod, fn):
            raise AttributeError(f"perception impl '{name}' missing {fn}")
    return mod


@torch.no_grad()
def encode_split(src_jsonl, percep, tok, model, device, bs=96):
    """Length-sorted batched encode (mirrors seq_worldmodel.encode_split so `baseline` is exact)."""
    seqs = [json.loads(l) for l in open(src_jsonl)]
    obs_texts, cmd_texts, spans = [], [], []
    for sq in seqs:
        start = len(obs_texts)
        for s in sq["steps"]:
            obs_texts.append(percep.render_obs(s))
            cmd_texts.append(percep.render_cmd(s))
        spans.append((start, len(obs_texts)))

    def enc(texts, tag):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.zeros(len(texts), D)
        for i in range(0, len(order), bs):
            bidx = order[i:i + bs]
            e = tok([texts[j] for j in bidx], return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
            e = {k: v.to(device) for k, v in e.items()}
            h = model(**e).last_hidden_state
            pooled = percep.pool(h, e["attention_mask"]).float().cpu()
            for k, j in enumerate(bidx):
                out[j] = pooled[k]
            if (i // bs) % 50 == 0:
                print(f"  enc {tag} {i}/{len(texts)}", flush=True)
        return out

    z_obs, z_cmd = enc(obs_texts, "obs"), enc(cmd_texts, "cmd")
    out = []
    for (a, b), sq in zip(spans, seqs):
        out.append({"z_obs": z_obs[a:b], "z_cmd": z_cmd[a:b],
                    "cmds": [s["cmd"] for s in sq["steps"]], "image": sq["image"],
                    "ok": [s.get("exit", 0) == 0 and bool((s.get("output") or "").strip())
                           for s in sq["steps"]]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", required=True)
    ap.add_argument("--src", default="data/dockerfs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    percep = load_perception(args.perception)
    model_name = getattr(percep, "MODEL", "answerdotai/ModernBERT-base")
    device = pick_device()
    print(f"perception '{args.perception}' | encoder {model_name} | device {device}", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        shutil.copy(pathlib.Path(args.src) / f"{split}.jsonl", outdir / f"{split}.jsonl")
        seqs = encode_split(pathlib.Path(args.src) / f"{split}.jsonl", percep, tok, model, device)
        torch.save(seqs, outdir / f"emb-seq-{split}.pt")
        print(f"encoded {split}: {len(seqs)} seqs -> {outdir}/emb-seq-{split}.pt", flush=True)
    print("REENCODE DONE", flush=True)


if __name__ == "__main__":
    main()
