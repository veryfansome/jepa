"""DG-7 reference-constant measurement (dockerfs3 draft, freeze order step 2).

Draft mandates (benchmarks/dockerfs3-design-draft.md):
  §2 OPEN:  "DG-7 reference constant (v2 cross-split near-dup base rate — MUST be
             measured on dockerfs2 BEFORE v3 work starts)"
  §11.2:    "DG-7 cross-split contamination (val mutation-slice obs vs ALL train
             renders, cos>0.995 or exact ≤ the v2-measured reference constant ...)"
  §14 row:  "DG-7 | cross-split contamination (val mutation slice vs train) |
             near-dup rate ≤ v2 reference constant (measure FIRST) | pilot P3 + MINT"

Implemented protocol (v2 = dockerfs2 has no mutation slice, so the base rate is
measured over ALL val observation renders, with per-verb / content-slice / split
slices so the amendment can pick the matched slice):

  unit      = one val observation step (occurrence-weighted; unique-render rate
              also reported)
  surface   = the v2 canonical observation render, realenv.seq_worldmodel.render_obs:
              "cwd={cwd} exit={exit}\n{output truncated at OBS_CAP=1600 (+marker)}"
              (the champion perception enc_e5_base prepends the constant "passage: "
              — a constant prefix, so exact-match equivalence classes are identical)
  pairs     = each val obs render vs ALL train observation renders (both splits of
              data/dockerfs2; train = 115k steps, val = 57k steps)
  reading A (exact): val render string exactly equals some train render string
  reading B (cos>0.995): max cosine over all train obs embeddings > 0.995 (strict),
              in the v2 champion eval space — the e5-base-v2 caches at
              data/dockerfs2-e5/emb-seq-{train,val}.pt (encoded from the identical
              jsonl; verified byte-identical). Exact ⇒ cos≈1, so B is the union
              reading "cos>0.995 or exact" of §11.2.

Run:  cd terminal-jepa && uv run python <this file>
"""
import json, pathlib, sys, time
from collections import Counter, defaultdict

import torch

TJ = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
sys.path.insert(0, str(TJ))
from realenv.seq_worldmodel import render_obs, pick_device  # the v2 canonical surface

DATA = TJ / "data" / "dockerfs2"
EMB = TJ / "data" / "dockerfs2-e5"
OUT = pathlib.Path(
    "/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/"
    "d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/v3/p0/dg7_reference.json"
)
COS_THRESH = 0.995  # strict > per §11.2
CONTENT = {"ls", "cat", "head", "tail", "find", "grep"}  # dockerfs2-v2.0 class table
INNER_IMAGES = {"fedora:latest", "mariadb:latest"}
FINAL_IMAGES = {"rockylinux:9", "httpd:2.4"}


def load_split(path):
    """Per-step: render, verb, image, grep-miss flag (v2.0 mode rule)."""
    steps = []
    for line in open(path):
        sq = json.loads(line)
        img = sq["image"]
        for s in sq["steps"]:
            verb = (s.get("meta") or {}).get("verb") or (s["cmd"].split() or [""])[0]
            grep_miss = verb == "grep" and (
                s.get("exit", 0) != 0 or not (s.get("output") or "").strip()
            )
            steps.append((render_obs(s), verb, img, grep_miss))
    return steps


def rates(flags, meta):
    """flags: bool per val step. Returns overall + sliced near-dup rates."""
    def rate(idx):
        n = len(idx)
        return {"n": n, "dup": sum(flags[i] for i in idx),
                "rate": (sum(flags[i] for i in idx) / n) if n else None}
    all_idx = range(len(flags))
    per_verb = {}
    by_verb = defaultdict(list)
    for i, (_, verb, img, gm) in enumerate(meta):
        by_verb[verb].append(i)
    for verb, idx in sorted(by_verb.items()):
        per_verb[verb] = rate(idx)
    content_idx = [i for i, (_, v, _, gm) in enumerate(meta)
                   if v in CONTENT and not gm]
    inner_idx = [i for i, (_, _, img, _) in enumerate(meta) if img in INNER_IMAGES]
    final_idx = [i for i, (_, _, img, _) in enumerate(meta) if img in FINAL_IMAGES]
    return {"overall": rate(all_idx), "per_verb": per_verb,
            "content_slice_v2_mode_rule": rate(content_idx),
            "inner_val_images": rate(inner_idx),
            "final_test_images": rate(final_idx)}


def main():
    t0 = time.time()
    train = load_split(DATA / "train.jsonl")
    val = load_split(DATA / "val.jsonl")
    t_load = time.time() - t0
    print(f"loaded: train={len(train)} val={len(val)} steps ({t_load:.1f}s)", flush=True)

    # ---- reading A: exact render match --------------------------------------
    t0 = time.time()
    train_renders = set(r for r, *_ in train)
    exact_flags = [r in train_renders for r, *_ in val]
    uniq_val = set(r for r, *_ in val)
    uniq_dup = sum(1 for r in uniq_val if r in train_renders)
    t_exact = time.time() - t0
    exact = rates(exact_flags, val)
    exact["unique_val_renders"] = {"n": len(uniq_val), "dup": uniq_dup,
                                   "rate": uniq_dup / len(uniq_val)}
    print(f"exact: overall {exact['overall']['rate']:.4f} ({t_exact:.1f}s)", flush=True)

    # ---- reading B: cos>0.995 in the champion e5-base space -----------------
    t0 = time.time()
    dev = pick_device()
    ztr = torch.load(EMB / "emb-seq-train.pt", weights_only=False)
    zva = torch.load(EMB / "emb-seq-val.pt", weights_only=False)
    Ztr = torch.cat([s["z_obs"] for s in ztr])
    Zva = torch.cat([s["z_obs"] for s in zva])
    assert Ztr.shape[0] == len(train) and Zva.shape[0] == len(val), \
        f"emb/jsonl misalignment: {Ztr.shape[0]} vs {len(train)}, {Zva.shape[0]} vs {len(val)}"
    # jsonl in the emb root is byte-identical to data/dockerfs2 (verified outside),
    # and encode_split preserves step order, so row i aligns with step i.
    Ztr = torch.nn.functional.normalize(Ztr, dim=1).to(dev)
    Zva = torch.nn.functional.normalize(Zva, dim=1).to(dev)
    max_cos = torch.empty(Zva.shape[0])
    VB = 1024
    for i in range(0, Zva.shape[0], VB):
        sims = Zva[i:i + VB] @ Ztr.T  # [vb, n_train]
        max_cos[i:i + VB] = sims.max(dim=1).values.cpu()
        if (i // VB) % 10 == 0:
            print(f"  cos {i}/{Zva.shape[0]}", flush=True)
    cos_flags = (max_cos > COS_THRESH).tolist()
    t_cos = time.time() - t0
    cos = rates(cos_flags, val)
    qs = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    cos["max_cos_percentiles"] = {str(q): round(torch.quantile(max_cos, q).item(), 6)
                                  for q in qs}
    # agreement: exact ⇒ cos>0.995 (float noise across encode batches permitted)
    both = sum(1 for e, c in zip(exact_flags, cos_flags) if e and c)
    n_exact = sum(exact_flags)
    cos["agreement"] = {
        "exact_and_cos": both, "exact_total": n_exact,
        "exact_without_cos": n_exact - both,
        "cos_without_exact": sum(cos_flags) - both}
    print(f"cos>{COS_THRESH}: overall {cos['overall']['rate']:.4f} ({t_cos:.1f}s)", flush=True)

    result = {
        "gate": "DG-7 reference constant (v2 cross-split near-dup base rate)",
        "measured_on": "data/dockerfs2 (bench dockerfs2-v2.0; train=8 images/115073 steps, val=4 held-out images/57630 steps)",
        "draft_quotes": {
            "s2_open": "DG-7 reference constant (v2 cross-split near-dup base rate — MUST be measured on dockerfs2 BEFORE v3 work starts)",
            "s11_2": "DG-7 cross-split contamination (val mutation-slice obs vs ALL train renders, cos>0.995 or exact ≤ the v2-measured reference constant ...)",
            "s14_row": "DG-7 | cross-split contamination (val mutation slice vs train) | near-dup rate ≤ v2 reference constant (measure FIRST) | pilot P3 + MINT | eval-integrity",
        },
        "protocol": {
            "unit": "one val observation step (occurrence-weighted); unique-render rate also reported",
            "surface": "realenv.seq_worldmodel.render_obs: 'cwd={cwd} exit={exit}\\n{output[:1600] (+trunc marker)}' (champion enc_e5_base adds only a constant 'passage: ' prefix — exact-match classes identical)",
            "pairs": "each val obs render vs ALL train observation renders (train cmd renders excluded: different surface — see ambiguities)",
            "reading_exact": "val render string in set(train render strings)",
            "reading_cos": f"max cosine vs all train obs embeddings > {COS_THRESH} (strict), v2 champion eval space: e5-base-v2 caches data/dockerfs2-e5/emb-seq-*.pt (jsonl verified byte-identical to data/dockerfs2); exact => cos~1, so this IS the union reading 'cos>0.995 or exact'",
        },
        "readings": {"exact": exact, f"cos_gt_{COS_THRESH}": cos},
        "reference_constant_candidates": {
            "exact_overall": exact["overall"]["rate"],
            "exact_content_slice": exact["content_slice_v2_mode_rule"]["rate"],
            "union_cos_or_exact_overall": cos["overall"]["rate"],
            "union_cos_or_exact_content_slice": cos["content_slice_v2_mode_rule"]["rate"],
            "note": "the frozen constant MUST be metric-matched to the v3 LHS: exact-LHS ~ exact ref; (cos>0.995 or exact)-LHS ~ union ref. Conservative (stricter ceiling) = the smaller, i.e. the exact reading; §11.2's wording favors the union.",
        },
        "ambiguities_for_amendment": [
            "metric: §11.2 says 'cos>0.995 or exact' but not which defines the frozen constant — both measured; freeze the metric AND its matched constant together",
            "val-side slice: v3 gates the val MUTATION slice; v2 has no mutations — overall, per-verb, content-slice (v2 mode rule: grep-miss excluded), and inner/final image slices all reported; the amendment picks the matched slice (content-slice is the nearest v2 ancestor of the v3 mutation slice)",
            "'ALL train renders' read as all train OBSERVATION renders (all verbs/steps); train cmd renders excluded (different text surface — obs renders carry the cwd=/exit= header, so obs-vs-cmd exact match is structurally impossible)",
            "cos space: v2's own e5-base-v2 champion space; v3 will recompute in its own encoder space — the constant is a base RATE, not a distance, so it transfers as a rate",
            "e5 encoding truncates at 256 tokens, so renders differing only beyond ~256 tokens are cosine-identical — the union reading catches these (e.g. long binary cat outputs); the exact reading does not",
        ],
        "runtime_sec": {"load_jsonl": round(t_load, 1), "exact": round(t_exact, 1),
                        "cosine": round(t_cos, 1),
                        "device": str(dev)},
        "reproduce": "cd /Users/fanzhu/PyCharmProjects/jepa/terminal-jepa && uv run python /private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/v3/p0/dg7_measure.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
