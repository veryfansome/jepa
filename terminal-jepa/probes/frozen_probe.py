"""A1 day-zero probe: frozen pretrained text-encoder features, zero training
(terminal-jepa-status.md "Path to a working agent", Track A1).

Renders full observations and probes a frozen HF encoder's token states under two
readouts:

- pooled: mean over real tokens -> one vector per observation. Protocol-v2 heads
  (same fit budget, standardization, metrics as probes/probe.py), directly comparable
  to runs/untrained-cls-v1-v2.json and the trained-arm probe-v2 reports.
- path-keyed (A1's default readout): per-line mean-pooled token states keyed by the
  path each line describes. Existence is STRUCTURAL under this readout — a path's
  line is present iff the path exists, i.e. the keying is the parser — so it is
  verified (mismatch counter), not learned. The learned probes are content-class
  from a path's line vector (one head shared across all paths and layouts — the
  cross-layout transfer test), cwd from the cwd line vector, and banner identity
  from the mean of path-line vectors (nuisance-leak diagnostic feeding A3).

Banner-swap audits run on both readouts: pooled dz ratio mirrors the v2 audit;
the line-level audit measures how much banner swaps move *path-line* vectors
relative to how much a genuine state change moves the (textually unchanged)
lines of other paths — contextual contamination that a canonicalizing objective
(A3) would have to remove.

Usage:
  .venv/bin/python -m probes.frozen_probe --data data/v1 \
      --model answerdotai/ModernBERT-base --out runs/frozen-modernbert-v1/probe.json
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
from env.state import FsState
from models.data import REGIMES
from probes.probe import (
    N_BANNERS, N_CLS, N_CWD, N_FILES, PATH_DEPTH, PROBE_FIT,
    _balacc, chance_floors, evaluate, fit_head, make_head, masked_cls_loss,
    standardize,
)
from train.train import pick_device

CONTENT_FIT_CAP = 150_000  # per-path (line vector, class) fit pairs; drawn seeded


def load_trajs(jsonl_path, regime, max_trajs, verbose=False):
    """Mirror of models.data.TrajectoryData, but keeping rendered text instead of
    custom-vocab token ids (the frozen encoder brings its own tokenizer).
    verbose=True renders the lossy JEPA-regime observations (render_full_verbose)."""
    use_banner, use_noise = REGIMES[regime]
    trajs = []
    with open(jsonl_path) as fh:
        for line in fh:
            if len(trajs) >= max_trajs:
                break
            t = json.loads(line)
            states = [FsState.from_json(t["layout"])]
            for s in t["steps"]:
                states.append(FsState.from_json(s["state_after"]))
            banner = t["banner_id"] if use_banner else None
            noise = t["noise_seed"] if use_noise else None
            if verbose:
                texts = [
                    render.render_full_verbose(st, salt=t["noise_seed"], banner_id=banner,
                                               noise_seed=noise, step=i)
                    for i, st in enumerate(states)
                ]
            else:
                texts = [
                    render.render_full(st, banner, noise, step=i)
                    for i, st in enumerate(states)
                ]
            trajs.append({
                "states": states,
                "texts": texts,
                "banner_id": t["banner_id"],
                "layout_id": t["layout_id"],
            })
    return trajs


def key_lines(text, verbose=False):
    """[(char_start, char_end, kind, index)] per line. kind: cwd|banner|noise|tree|
    dir|file; index is FILE_PATH_INDEX for files, DIR_PATH_INDEX for dirs, else -1.
    verbose=True parses render_full_verbose lines, where the path is the only "/"-token
    mid-line (after ls -la-style metadata) and content is the trailing ':: snippet'."""
    spans, pos = [], 0
    for ln in text.split("\n"):
        start, end = pos, pos + len(ln)
        pos = end + 1
        if ln.startswith("cwd: "):
            spans.append((start, end, "cwd", -1))
        elif ln.startswith("### "):
            spans.append((start, end, "banner", -1))
        elif ln.startswith("[ts "):
            spans.append((start, end, "noise", -1))
        elif ln == "tree:":
            spans.append((start, end, "tree", -1))
        elif verbose:
            # verbose dir/file line: "<meta> /path[/] [:: snippet]"; path is the sole
            # token starting with "/". Snippet words carry no "/".
            ptok = next((t for t in ln.split() if t.startswith("/")), None)
            if ptok is None:
                spans.append((start, end, "other", -1))
            elif ptok.endswith("/"):
                spans.append((start, end, "dir", vocab.DIR_PATH_INDEX[vocab.str_to_path(ptok[:-1])]))
            else:
                spans.append((start, end, "file", vocab.FILE_PATH_INDEX[vocab.str_to_path(ptok)]))
        elif ln.endswith("/"):
            spans.append((start, end, "dir", vocab.DIR_PATH_INDEX[vocab.str_to_path(ln[:-1])]))
        else:
            pstr = ln.partition(" [")[0]
            spans.append((start, end, "file", vocab.FILE_PATH_INDEX[vocab.str_to_path(pstr)]))
    return spans


@torch.no_grad()
def encode_batch(model, tok, texts, device, verbose=False):
    """Returns per-text lists of line-keyed vectors. Each token is assigned to the
    line containing its last character (BPE offsets include leading whitespace, so
    first/mid-char rules would bleed tokens across the preceding newline).
    verbose=True parses render_full_verbose lines (see key_lines)."""
    enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")
    hs = model(**{k: v.to(device) for k, v in enc.items()}).last_hidden_state
    hs = hs.float().cpu()
    out = []
    for b, text in enumerate(texts):
        real = (enc["attention_mask"][b] > 0) & (offsets[b, :, 1] > offsets[b, :, 0])
        h, off = hs[b][real], offsets[b][real]
        pooled = h.mean(0)
        spans = key_lines(text, verbose)
        starts = torch.tensor([s[0] for s in spans])
        line_id = torch.searchsorted(starts, off[:, 1] - 1, right=True) - 1
        rec = {"pooled": pooled.half(), "cwd": None, "file_idx": [], "file_vecs": [],
               "dir_idx": [], "dir_vecs": [], "path_vecs": []}
        for li, (s, e, kind, idx) in enumerate(spans):
            m = line_id == li
            if not m.any():
                continue
            v = h[m].mean(0)
            if kind == "cwd":
                rec["cwd"] = v.half()
            elif kind == "file":
                rec["file_idx"].append(idx)
                rec["file_vecs"].append(v.half())
                rec["path_vecs"].append(v.half())
            elif kind == "dir":
                rec["dir_idx"].append(idx)
                rec["dir_vecs"].append(v.half())
                rec["path_vecs"].append(v.half())
        rec["file_vecs"] = (torch.stack(rec["file_vecs"]) if rec["file_vecs"]
                            else torch.empty(0, h.shape[-1], dtype=torch.half))
        rec["dir_vecs"] = (torch.stack(rec["dir_vecs"]) if rec["dir_vecs"]
                           else torch.empty(0, h.shape[-1], dtype=torch.half))
        rec["path_mean"] = (torch.stack(rec["path_vecs"]).mean(0) if rec["path_vecs"]
                            else pooled.half())
        del rec["path_vecs"]
        out.append(rec)
    return out


@torch.no_grad()
def extract(model, tok, trajs, device, batch_size=64, tag="", verbose=False):
    """Per-state records + labels + structural-existence verification."""
    items = []
    for tr in trajs:
        for i, st in enumerate(tr["states"]):
            items.append((tr["texts"][i], st.features(), tr["banner_id"]))
    recs, feats, banners = [], [], []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        recs.extend(encode_batch(model, tok, [b[0] for b in batch], device, verbose))
        feats.extend(b[1] for b in batch)
        banners.extend(b[2] for b in batch)
        if (i // batch_size) % 50 == 0:
            print(f"  extract{tag}: {i + len(batch)}/{len(items)}", flush=True)
    mismatches = 0
    for rec, f in zip(recs, feats):
        truth = {j for j, e in enumerate(f["file_exists"]) if e}
        mismatches += set(rec["file_idx"]) != truth
    y = {
        "cwd": torch.tensor([f["cwd_index"] for f in feats]),
        "exists": torch.tensor([f["file_exists"] for f in feats], dtype=torch.float),
        "cls": torch.tensor([f["file_class"] for f in feats]),
        "banner": torch.tensor(banners),
    }
    return recs, y, mismatches


def content_pairs(recs, y, cap=None, seed=0):
    """(line_vec, class, depth) for every existing file; optionally capped, seeded."""
    vecs, labels, depths = [], [], []
    for si, rec in enumerate(recs):
        for k, idx in enumerate(rec["file_idx"]):
            vecs.append(rec["file_vecs"][k])
            labels.append(y["cls"][si, idx].item())
            depths.append(PATH_DEPTH[idx].item())
    v = torch.stack(vecs).float()
    lab = torch.tensor(labels)
    dep = torch.tensor(depths)
    if cap is not None and len(lab) > cap:
        idx = torch.randperm(len(lab), generator=torch.Generator().manual_seed(seed))[:cap]
        v, lab, dep = v[idx], lab[idx], dep[idx]
    return v, lab, dep


def fit_eval_path_keyed(recs_tr, y_tr, recs_va, y_va, device, seed):
    z_cwd_tr = torch.stack([r["cwd"] for r in recs_tr]).float()
    z_cwd_va = torch.stack([r["cwd"] for r in recs_va]).float()
    z_pm_tr = torch.stack([r["path_mean"] for r in recs_tr]).float()
    z_pm_va = torch.stack([r["path_mean"] for r in recs_va]).float()
    v_tr, lab_tr, _ = content_pairs(recs_tr, y_tr, cap=CONTENT_FIT_CAP, seed=seed)
    v_va, lab_va, dep_va = content_pairs(recs_va, y_va)
    z_cwd_tr, z_cwd_va = standardize(z_cwd_tr, z_cwd_va)
    z_pm_tr, z_pm_va = standardize(z_pm_tr, z_pm_va)
    v_tr, v_va = standardize(v_tr, v_va)
    d = v_tr.shape[1]
    ce = nn.functional.cross_entropy
    out = {"content_fit_pairs": len(lab_tr), "content_eval_pairs": len(lab_va)}
    for kind in ["linear", "mlp"]:
        torch.manual_seed(seed)
        h_cwd = fit_head(make_head(kind, d, N_CWD), z_cwd_tr, y_tr["cwd"], ce, device)
        h_cls = fit_head(make_head(kind, d, N_CLS), v_tr, lab_tr, ce, device)
        h_ban = fit_head(make_head(kind, d, N_BANNERS), z_pm_tr, y_tr["banner"], ce, device)
        with torch.no_grad():
            cwd_acc = (h_cwd(z_cwd_va.to(device)).argmax(-1).cpu() == y_va["cwd"]).float().mean().item()
            cls_pred = h_cls(v_va.to(device)).argmax(-1).cpu()
            cls_acc = (cls_pred == lab_va).float().mean().item()
            ban_acc = (h_ban(z_pm_va.to(device)).argmax(-1).cpu() == y_va["banner"]).float().mean().item()
        by_depth = {
            int(dd): (cls_pred[dep_va == dd] == lab_va[dep_va == dd]).float().mean().item()
            for dd in (1, 2, 3) if (dep_va == dd).any()
        }
        out[kind] = {
            "cwd_acc": cwd_acc,
            "cls_acc_given_exists": cls_acc,
            "banner_acc": ban_acc,
            "cls_by_depth": by_depth,
            "macro_learned": (cwd_acc + cls_acc) / 2,
            "macro_with_structural_exists": (cwd_acc + 1.0 + cls_acc) / 3,
        }
        print(f"path-keyed {kind}", json.dumps(out[kind]), flush=True)
    return out


@torch.no_grad()
def banner_swap_audit_frozen(model, tok, trajs, device, n=300, seed=0):
    """Pooled arm mirrors probes.probe.banner_swap_audit. Line arm compares path-line
    movement under a banner swap vs the movement of *unchanged* paths' lines under a
    genuine state change (contextual contamination measure)."""
    rng = random.Random(f"audit:{seed}")
    d_banner_pooled, d_state_pooled, d_banner_line, d_state_line = [], [], [], []
    for _ in range(n):
        tr = trajs[rng.randrange(len(trajs))]
        i = rng.randrange(len(tr["states"]))
        st = tr["states"][i]
        b1 = tr["banner_id"]
        b2 = rng.choice([b for b in range(N_BANNERS) if b != b1])
        for _ in range(20):
            act = actions.sample_valid(st, rng)
            res = actions.apply(st, act)
            if res.ttype == actions.STATE_CHANGING:
                break
        texts = [render.render_full(s, b, None, step=i)
                 for s, b in [(st, b1), (st, b2), (res.state, b1)]]
        r0, r1, r2 = encode_batch(model, tok, texts, device)
        d_banner_pooled.append((r0["pooled"].float() - r1["pooled"].float()).norm().item())
        d_state_pooled.append((r0["pooled"].float() - r2["pooled"].float()).norm().item())
        m0 = {idx: r0["file_vecs"][k].float() for k, idx in enumerate(r0["file_idx"])}
        m1 = {idx: r1["file_vecs"][k].float() for k, idx in enumerate(r1["file_idx"])}
        m2 = {idx: r2["file_vecs"][k].float() for k, idx in enumerate(r2["file_idx"])}
        d_banner_line.extend((m0[i2] - m1[i2]).norm().item() for i2 in m0 if i2 in m1)
        f0, f2 = st.files, res.state.files
        changed = {vocab.FILE_PATH_INDEX[p] for p in set(f0) | set(f2)
                   if f0.get(p) != f2.get(p)}
        d_state_line.extend((m0[i2] - m2[i2]).norm().item()
                            for i2 in m0 if i2 in m2 and i2 not in changed)
    def med(x):
        return torch.tensor(x).median().item() if x else float("nan")
    pooled = {
        "banner_swap_dz_median": med(d_banner_pooled),
        "state_change_dz_median": med(d_state_pooled),
        "ratio": med(d_banner_pooled) / max(med(d_state_pooled), 1e-9),
    }
    line = {
        "banner_swap_line_dz_median": med(d_banner_line),
        "state_change_unchanged_line_dz_median": med(d_state_line),
        "ratio": med(d_banner_line) / max(med(d_state_line), 1e-9),
    }
    return pooled, line


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--random-init", action="store_true",
                    help="architecture-matched RANDOM-INIT encoder (finding 19: "
                         "the gate-1 rule requires this floor, not the 256-d CLS one)")
    ap.add_argument("--verbose", action="store_true",
                    help="render lossy JEPA-regime observations (render_full_verbose): "
                         "content is a class-conditional snippet, not a [cK] token — "
                         "the decisive off-regime test (finding 24)")
    ap.add_argument("--regime", default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-trajs", type=int, default=800)
    ap.add_argument("--max-val-trajs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.random_init:
        from transformers import AutoConfig

        torch.manual_seed(args.seed)
        model = AutoModel.from_config(AutoConfig.from_pretrained(args.model))
        model = model.to(device).eval()
    else:
        model = AutoModel.from_pretrained(args.model).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params/1e6:.0f}M device={device}", flush=True)

    root = pathlib.Path(args.data)
    trajs_tr = load_trajs(root / "train.jsonl", args.regime, args.max_train_trajs, args.verbose)
    trajs_va = load_trajs(root / "val.jsonl", args.regime, args.max_val_trajs, args.verbose)
    recs_tr, y_tr, mism_tr = extract(model, tok, trajs_tr, device, args.batch_size, ":fit", args.verbose)
    recs_va, y_va, mism_va = extract(model, tok, trajs_va, device, args.batch_size, ":eval", args.verbose)
    print(f"fit states={len(recs_tr)} eval states={len(recs_va)} "
          f"structural-existence mismatches: fit={mism_tr} eval={mism_va}", flush=True)

    report = {
        "model": args.model,
        "random_init": args.random_init,
        "verbose": args.verbose,
        "model_params": n_params,
        "data": args.data,
        "regime": args.regime,
        "eval_split": "val-heldout-layouts",
        "fit_states": len(recs_tr),
        "eval_states": len(recs_va),
        "probe_fit": dict(PROBE_FIT, content_fit_cap=CONTENT_FIT_CAP),
        "chance_floors": chance_floors(y_va),
        "encoder_d_out": recs_tr[0]["pooled"].shape[0],
    }

    # Pooled arm: protocol-v2 heads, comparable to prior probe-v2 reports.
    z_tr = torch.stack([r["pooled"] for r in recs_tr]).float()
    z_va = torch.stack([r["pooled"] for r in recs_va]).float()
    z_tr, z_va = standardize(z_tr, z_va)
    ce = nn.functional.cross_entropy
    bce = nn.functional.binary_cross_entropy_with_logits
    d_in = z_tr.shape[1]
    report["pooled"] = {}
    for kind in ["linear", "mlp"]:
        torch.manual_seed(args.seed)
        heads = {
            "cwd": fit_head(make_head(kind, d_in, N_CWD), z_tr, y_tr["cwd"], ce, device),
            "exists": fit_head(make_head(kind, d_in, N_FILES), z_tr, y_tr["exists"], bce, device),
            "cls": fit_head(make_head(kind, d_in, N_FILES * N_CLS), z_tr, y_tr["cls"],
                            masked_cls_loss, device),
            "banner": fit_head(make_head(kind, d_in, N_BANNERS), z_tr, y_tr["banner"], ce, device),
        }
        report["pooled"][kind] = evaluate(heads, z_va, y_va, device)
        print("pooled", kind, json.dumps(report["pooled"][kind]), flush=True)
    del z_tr, z_va, heads

    # Path-keyed arm (A1 default readout).
    report["path_keyed"] = {
        "exists_structural": {
            "balacc": 1.0 if (mism_tr + mism_va) == 0 else None,
            "note": "existence is the line-keying itself (parser-equivalent); "
                    "verified against ground truth per state, not learned",
            "mismatch_states_fit": mism_tr,
            "mismatch_states_eval": mism_va,
        },
    }
    report["path_keyed"].update(
        fit_eval_path_keyed(recs_tr, y_tr, recs_va, y_va, device, args.seed)
    )

    if args.regime in ("banner", "both") and not args.verbose:
        pooled_audit, line_audit = banner_swap_audit_frozen(
            model, tok, trajs_va, device, seed=args.seed
        )
        report["pooled"]["banner_swap_audit"] = pooled_audit
        report["path_keyed"]["banner_line_audit"] = line_audit
        print("audit pooled", json.dumps(pooled_audit), flush=True)
        print("audit line", json.dumps(line_audit), flush=True)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
