"""evolve/precompute_baselines.py — the v3 SST / within_traj_mut baseline arms (design §10.4b).

Deterministic, seed-free, cached once per root. Reads the FULL val split (inner AND final, chosen
so final-test champion scoring also has the new arms), folds `ShellState(mode="sst")` per trajectory,
renders each predicted step-dict through `render_canon.canon` + the ROOT'S perception module
(`reencode.load_perception_for_root`, e5's "passage: " prefix included), encodes, and writes:

  <root>/sst-val.pt : list of per-seq {image, z[n,768] RAW obs embeddings, determined[n] bool}
  <root>/wtm-val.pt : list of per-seq {image, z[n,768] RAW obs embeddings, defined[n]  bool}

aligned to `_data_tensors` step order — the image-contiguous val.jsonl sequence order that
`split_val` filters, so applying the same image filter to the precomputed per-seq rows reproduces
the harness step order exactly. RAW (un-standardized) embeddings are stored; `harness._base_for`
standardizes them with the eval's obs stats (mo, so) and, for the sst arm, sets the ⊥ (undetermined)
steps to ZEROS in standardized space (§10.3, sst = ⊥→zeros). The within_traj_mut arm re-encodes the
stale within-trajectory retrieval PATCHED by the tracker's overlay (SST-determined render where the
belief state determines the step, else the stale earlier observation — "mut-vs-plain is a reported
diagnostic, not an assert", §10.3); sst_composite (SST-where-determined else wtm) is assembled in the
harness from these two tensors + the determined mask.

  python -m evolve.precompute_baselines --root data/dockerfs3-e5
"""

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from realenv import render_canon
from realenv import shell_state as S
from realenv.seq_worldmodel import D, pick_device, OBS_CAP
from evolve import bench_versions as BV
from evolve.reencode import load_perception_for_root


BOT = S.BOT


def _canon_render(step, percep):
    """render_canon.canon(step) -> percep.render_obs (the root's OBSERVATION render path, §10.4b).
    canon is applied to EVERY predicted/recorded step-dict before the perception render, exactly as
    on real steps (render parity, so the arm embeddings live in the same space as the true z_obs)."""
    try:
        c = render_canon.canon(step)
    except Exception:                      # noqa: BLE001 — canon is total; never abort a precompute
        c = step
    # cap defensively (percep.render_obs caps too, but keep belief text bounded like the SST window)
    out = c.get("output", "") or ""
    if len(out) > OBS_CAP:
        c = dict(c); c["output"] = out[:OBS_CAP]
    return percep.render_obs(c)


def _sst_fold_seq(steps, error_templates_for, image):
    """Fold one trajectory through the SST (mode='sst'); return per-step
    (determined: bool, pred_step: dict|None). predict(vt, cmd) BEFORE fold(step) (§10.1 order)."""
    tmpl = error_templates_for(image) if callable(error_templates_for) else (error_templates_for or {})
    sst = S.ShellState(mode="sst", error_templates=tmpl)
    out = []
    for step in steps:
        cmd = step.get("cmd", "")
        try:
            pred = sst.predict(sst.vt, cmd)
        except S.ParseError:
            pred = BOT
        if pred is BOT:
            out.append((False, None))
        else:
            out.append((True, {"cmd": cmd, "output": pred.get("output", ""),
                               "exit": pred.get("exit", 0), "cwd": pred.get("cwd", "/")}))
        # advance the tracker (keep the virtual clock aligned even on a parse gap)
        rec = {"cmd": cmd, "output": step.get("output", "") or "",
               "exit": step.get("exit", 0), "cwd": step.get("cwd", "/")}
        try:
            sst.fold(rec)
        except S.ParseError:
            sst.vt += 1
    return out


def _nearest_earlier(z_cmd, t):
    """within_traj retrieval: the index r<t of the nearest earlier command by squared distance
    (the ratified v2 within_traj rule, applied per trajectory). None when t==0."""
    if t == 0:
        return None
    d = ((z_cmd[:t] - z_cmd[t]) ** 2).sum(-1)
    return int(d.argmin())


def _encode_texts(texts, percep, tok, model, device, bs=96):
    """Length-sorted batched encode into [len(texts), 768] (mirrors reencode.encode_split's enc())."""
    out = torch.zeros(len(texts), D)
    if not texts:
        return out
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    with torch.no_grad():
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            e = tok([texts[j] for j in b], return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
            e = {k: v.to(device) for k, v in e.items()}
            h = model(**e).last_hidden_state
            pooled = percep.pool(h, e["attention_mask"]).float().cpu()
            for k, j in enumerate(b):
                out[j] = pooled[k]
    return out


def precompute(root, error_templates_for=None):
    """Build sst-val.pt + wtm-val.pt for a v3 root. Returns (sst_seqs, wtm_seqs) and writes them."""
    root = pathlib.Path(root)
    val_jsonl = root / "val.jsonl"
    if not val_jsonl.exists():
        raise FileNotFoundError(f"{root}: no val.jsonl (precompute reads the FULL val split, §10.4b)")
    percep = load_perception_for_root(root)          # fail-closed on a stamp-less v3 root (§10.3)
    model_name = getattr(percep, "MODEL", "answerdotai/ModernBERT-base")
    device = pick_device()
    print(f"precompute perception '{getattr(percep,'__name__',percep)}' | encoder {model_name} "
          f"| device {device}", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    seqs = [json.loads(l) for l in open(val_jsonl)]

    # Pass 1: fold SST + retrieval, gather the texts to encode (cmd texts for retrieval; sst
    # determined renders; wtm undetermined stale renders). Encode all at once, scatter back.
    cmd_texts, cmd_span = [], []
    for sq in seqs:
        a = len(cmd_texts)
        for st in sq.get("steps", []):
            cmd_texts.append(percep.render_cmd(st))
        cmd_span.append((a, len(cmd_texts)))
    z_cmd_all = _encode_texts(cmd_texts, percep, tok, model, device)

    sst_folds = []
    for sq in seqs:
        sst_folds.append(_sst_fold_seq(sq.get("steps", []), error_templates_for, sq.get("image", "?")))

    obs_texts = []                                    # combined encode pool (sst + wtm renders)
    sst_ref = []    # per (seq, step) -> index into obs_texts or -1 (undetermined => z=0)
    wtm_ref = []    # per (seq, step) -> index into obs_texts or -1 (undefined  => z=0)
    for si, sq in enumerate(seqs):
        steps = sq.get("steps", [])
        a, b = cmd_span[si]
        z_cmd = z_cmd_all[a:b]
        folds = sst_folds[si]
        srow, wrow = [], []
        for t, step in enumerate(steps):
            det, pred_step = folds[t]
            # sst arm: determined -> encode the predicted render; else -1 (=> zeros in std space)
            if det:
                sst_txt = _canon_render(pred_step, percep)
                srow.append(len(obs_texts)); obs_texts.append(sst_txt)
            else:
                srow.append(-1)
            # within_traj_mut: SST-determined render where the belief determines the step (the
            # overlay fully rewrote the target); else the stale nearest-earlier observation
            # (patcher reuses the one tracker). Undefined only at t==0 with no determination.
            if det:
                wrow.append(srow[-1])                 # reuse the sst render (same text)
            else:
                r = _nearest_earlier(z_cmd, t)
                if r is None:
                    wrow.append(-1)                   # t==0, undetermined => zeros (chance)
                else:
                    wtm_txt = _canon_render(steps[r], percep)
                    wrow.append(len(obs_texts)); obs_texts.append(wtm_txt)
        sst_ref.append(srow); wtm_ref.append(wrow)

    z_obs_all = _encode_texts(obs_texts, percep, tok, model, device)

    sst_seqs, wtm_seqs = [], []
    n_det = n_steps = n_wtm_def = 0
    for si, sq in enumerate(seqs):
        steps = sq.get("steps", [])
        n = len(steps)
        z_sst = torch.zeros(n, D); det = torch.zeros(n, dtype=torch.bool)
        z_wtm = torch.zeros(n, D); wdef = torch.zeros(n, dtype=torch.bool)
        for t in range(n):
            si_ref = sst_ref[si][t]
            if si_ref >= 0:
                z_sst[t] = z_obs_all[si_ref]; det[t] = True; n_det += 1
            wi_ref = wtm_ref[si][t]
            if wi_ref >= 0:
                z_wtm[t] = z_obs_all[wi_ref]; wdef[t] = True; n_wtm_def += 1
        n_steps += n
        img = sq.get("image", "?")
        sst_seqs.append({"image": img, "z": z_sst, "determined": det})
        wtm_seqs.append({"image": img, "z": z_wtm, "defined": wdef})

    torch.save(sst_seqs, root / "sst-val.pt")
    torch.save(wtm_seqs, root / "wtm-val.pt")
    _stamp_shas(root)
    print(f"precompute DONE: {len(seqs)} seqs, {n_steps} steps | sst-determined {n_det} "
          f"({n_det/max(1,n_steps):.1%}) | wtm-defined {n_wtm_def} "
          f"-> {root}/sst-val.pt, {root}/wtm-val.pt", flush=True)
    return sst_seqs, wtm_seqs


def _stamp_shas(root):
    """Record the .pt shas into summary.json (UD-8 regenerable-instrument provenance, §10.4b).
    Because seq_worldmodel._v3_cache_guard hashes summary.json against cache_meta's
    built_summary_sha (the re-mint staleness key), any additive edit to summary.json must
    RE-STAMP that key in lock-step, else the fail-closed guard would reject the (valid) cache."""
    root = pathlib.Path(root)
    summ_p = root / "summary.json"
    if not summ_p.exists():
        return
    summ = json.loads(summ_p.read_text())
    shas = {}
    for name in ("sst-val.pt", "wtm-val.pt"):
        p = root / name
        if p.exists():
            shas[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    summ["baseline_precompute_sha256"] = shas
    summ_p.write_text(json.dumps(summ, indent=1))
    cm_p = root / "cache_meta.json"
    if cm_p.exists():
        cm = json.loads(cm_p.read_text())
        cm["built_summary_sha"] = hashlib.sha256(summ_p.read_bytes()).hexdigest()
        cm_p.write_text(json.dumps(cm, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="a v3 derived (reencoded+stamped) root")
    args = ap.parse_args(argv)
    if not BV.is_v3_policy(args.root):
        raise SystemExit(f"{args.root}: not a v3-policy root (precompute is v3-only)")
    precompute(args.root)


if __name__ == "__main__":
    main()
