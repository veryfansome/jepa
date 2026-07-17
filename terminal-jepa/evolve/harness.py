"""score_genome — the trustworthy fitness. Assemble a genome, train the world model with the
genome's objective, HARD-FILTER on the per-genome no-leakage + calibration guards, then score
the content-verb MARGIN on the inner-val (held-out) split. Reuses the validated R4 eval from
realenv/seq_worldmodel.py so a fitness number means exactly what the R4 headline meant.

Fitness = mean over seeds of
    content_top1(WM) - max(content_top1(retrieve_by_cmd), content_top1(no_history), content_top1(copy_prev))
on ls+cat (content) verbs of the inner-val images. Never touches final-test.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # terminal-jepa root

import torch

from realenv import seq_worldmodel as M
from evolve import genome as G
from evolve.splits import split_val

D = M.D
CHANCE_SLACK = 0.05  # predict-mean top-1 must stay below this (chance ~1/64=0.016)
BASE_CACHE = pathlib.Path(__file__).resolve().parent / "archive" / "base_cache.json"


def _data_tensors(evalset):
    """Model-INDEPENDENT eval tensors (true/prev/verbs/cmd-embeddings) in flatten_predictions'
    step order — lets the objective-independent baselines be computed and cached once."""
    trues, prevs, cmds = [], [], []
    for sq in evalset:
        for t in range(sq["z_obs"].shape[0]):
            trues.append(sq["z_obs"][t])
            prevs.append(sq["z_obs"][t - 1] if t > 0 else torch.zeros(D))
            cmds.append(sq["cmds"][t])
    cmd_embs = torch.stack([sq["z_cmd"][t] for sq in evalset for t in range(sq["z_obs"].shape[0])])
    return {"true": torch.stack(trues), "prev": torch.stack(prevs),
            "verbs": [M.verb_of(c) for c in cmds], "_cmd_embs": cmd_embs}


def _base_for(split, seed, steps, fit, evaldata, device, data_root="data/dockerfs"):
    """max-baseline content-top1 (retrieve_by_cmd / no_history MLP / copy_prev) + predict-mean
    calibration for a (data_root, split, seed, steps) — objective-independent, so computed once and
    cached. Returns (base, predict_mean). Halves per-genome cost (skips retraining the baseline)."""
    key = f"{data_root}|{split}|{seed}|{steps}"
    cache = json.loads(BASE_CACHE.read_text()) if BASE_CACHE.exists() else {}
    if key in cache:
        return cache[key]["base"], cache[key]["predict_mean"]
    mlp = M.train_cmd_only(fit, device, steps=steps, seed=seed)
    with torch.no_grad():
        nohist = mlp(evaldata["_cmd_embs"].to(device)).cpu()
    ct = lambda p: M.content_retrieval(p, evaldata["true"], evaldata["verbs"], seed=seed)["top1_sameverb"]
    rbc, noh = ct(M.retrieve_by_cmd_baseline(fit, evaldata)), ct(nohist)
    cpy, mean = ct(evaldata["prev"]), ct(torch.zeros_like(evaldata["true"]))
    entry = {"base": max(rbc, noh, cpy), "predict_mean": mean,
             "retrieve_by_cmd": rbc, "no_history": noh, "copy_prev": cpy}
    cache[key] = entry
    BASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BASE_CACHE.write_text(json.dumps(cache, indent=1))
    return entry["base"], entry["predict_mean"]


def _train(genome, fit, device, loss_fn, seed, steps, target_mod, stream):
    """Train the world model with the genome's objective + arch + target + stream. Returns
    (net, ok); ok=False on NaN. The objective's loss compares the model's cmd-position prediction
    to target_mod.make_target(z_obs, z_prev) — e.g. the raw next obs (identity) or the residual
    z_obs - z_prev (delta). z_prev is the previous observation (strict-causal shift of tgt)."""
    torch.manual_seed(seed)
    build, aparams = G.load_arch(genome)
    net = build(**aparams)
    if getattr(target_mod, "LEARNED", False):
        # learned-target extension: the target impl provides an nn.Module (make_target/to_obs/reg)
        # whose params are registered on the net so the genome's optimizer trains them jointly.
        # The eval stays in the FIXED obs space (to_obs must reconstruct), which keeps a learned
        # target honest: collapsing the target space breaks reconstruction and is scored down.
        net.target_module = target_mod.make(D)
    net = net.to(device)
    make_opt, bs = G.load_optim(genome)
    opt, sched = make_opt(net.parameters(), steps)
    next_batch = G.load_batcher(genome)(fit, bs, seed)
    for step in range(1, steps + 1):
        idx = next_batch(step, steps)
        if len(idx) != bs or min(idx) < 0 or max(idx) >= len(fit):
            raise ValueError("batcher contract violation (len/bounds)")
        b = stream.collate([fit[i] for i in idx], device)
        pred_full, _ = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = stream.extract_cmd_pred(pred_full, b)           # [B, maxn, D]
        tgt_full = b["tgt"]                                        # [B, maxn, D] = z_obs per step
        prev_full = torch.cat([torch.zeros_like(tgt_full[:, :1]), tgt_full[:, :-1]], dim=1)
        m = b["cmd_mask"]
        pred, tgt, prev = cmd_pred[m], tgt_full[m], prev_full[m]
        tmod = getattr(net, "target_module", None)
        if tmod is not None:
            loss = loss_fn(pred, tmod.make_target(tgt, prev)) + tmod.reg()
        else:
            loss = loss_fn(pred, target_mod.make_target(tgt, prev))
        if not torch.isfinite(loss):
            return net, False
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
    return net, True


def score_genome(genome, mode="proxy", data="data/dockerfs",
                 model="answerdotai/ModernBERT-base", proxy_steps=1000, split="inner"):
    """Return a fitness dict. mode='proxy' -> steps=proxy_steps, seeds=[0]; 'full' -> genome
    steps, seeds=[0,1,2]. split='inner' (fedora+mariadb, the optimization target) or 'final'
    (rockylinux+httpd, the untouched held-out-of-held-out test — champion validation only). Any
    guardrail failure -> fitness=-inf with a reason."""
    G.validate(genome)
    device = M.pick_device()
    train_seqs = M.cached_encode(data, "train", model, device)
    val_seqs = M.cached_encode(data, "val", model, device)
    mo, so, mc, sc = M.standardize_stats(train_seqs)
    M.apply_stats(train_seqs, mo, so, mc, sc)
    M.apply_stats(val_seqs, mo, so, mc, sc)
    inner = split_val(val_seqs, split)

    loss_fn = G.load_objective(genome)
    target_mod = G.load_target(genome)
    stream = G.load_stream(genome)
    seeds = [0] if mode == "proxy" else [0, 1, 2]
    steps = proxy_steps if mode == "proxy" else genome["chunks"]["optim"].get("steps", 4000)
    evaldata = _data_tensors(inner)  # model-independent; base + wm both score against it

    per_seed = []
    for s in seeds:
        try:
            fit, _ = M.split_train_dev(train_seqs, seed=s)
            net, ok = _train(genome, fit, device, loss_fn, s, steps, target_mod, stream)
            if not ok:
                return _fail("train_diverged (NaN/inf loss)", mode, seeds, steps, split, per_seed)
            if not stream.leakage_ok(net, device):
                return _fail("leakage_fail (cmd_t prediction moved when obs_t corrupted)", mode, seeds, steps, split, per_seed)
            base, mean = _base_for(split, s, steps, fit, evaldata, device, data_root=data)  # cached, objective-independent
            flat = stream.flatten_predictions(net, inner, device)
            tmod = getattr(net, "target_module", None)
            with torch.no_grad():  # learned to_obs runs on cpu tensors from flatten
                if tmod is not None:
                    pred_obs = tmod.cpu().to_obs(flat["pred"], flat["prev"])
                else:
                    pred_obs = target_mod.to_obs(flat["pred"], flat["prev"])  # reconstruct next-obs for retrieval
            wm = M.content_retrieval(pred_obs, evaldata["true"], evaldata["verbs"], seed=s)["top1_sameverb"]
            per_seed.append({"seed": s, "wm": round(wm, 4), "base": round(base, 4),
                             "margin": round(wm - base, 4), "predict_mean": round(mean, 4)})
        except Exception as e:  # broken inventor code must not crash the loop
            return _fail(f"exception: {type(e).__name__}: {e}", mode, seeds, steps, split, per_seed)

    mean_cal = sum(p["predict_mean"] for p in per_seed) / len(per_seed)
    if mean_cal > CHANCE_SLACK:
        return _fail(f"calibration_fail (predict_mean top1={mean_cal:.3f} > {CHANCE_SLACK})",
                     mode, seeds, steps, split, per_seed)

    def mean(k):
        return round(sum(p[k] for p in per_seed) / len(per_seed), 4)

    return {"fitness": mean("margin"), "guardrail": "pass", "mode": mode, "seeds": seeds,
            "steps": steps, "split": split, "wm_content_top1": mean("wm"),
            "base_content_top1": mean("base"), "eval_images": sorted({s["image"] for s in inner}),
            "per_seed": per_seed}


def _fail(reason, mode, seeds, steps, split, per_seed):
    return {"fitness": float("-inf"), "guardrail": reason, "mode": mode, "seeds": seeds,
            "steps": steps, "split": split, "per_seed": per_seed}
