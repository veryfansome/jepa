"""score_genome — the trustworthy fitness. Assemble a genome, train the world model with the
genome's objective, HARD-FILTER on the per-genome no-leakage + calibration guards, then score
the content-verb MARGIN on the inner-val (held-out) split. Reuses the validated R4 eval from
realenv/seq_worldmodel.py so a fitness number means exactly what the R4 headline meant.

Fitness = mean over seeds of
    content_top1(WM) - max(content_top1(retrieve_by_cmd), content_top1(no_history), content_top1(copy_prev))
on ls+cat (content) verbs of the inner-val images. Never touches final-test.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # terminal-jepa root

import torch

from realenv import seq_worldmodel as M
from evolve import genome as G
from evolve.splits import split_val

D = M.D
CHANCE_SLACK = 0.05  # predict-mean top-1 must stay below this (chance ~1/64=0.016)


def _train(genome, fit, device, loss_fn, seed, steps):
    """Train the world model with the genome's objective. Returns (net, ok); ok=False on NaN."""
    torch.manual_seed(seed)
    a, o = genome["chunks"]["arch"], genome["chunks"]["optim"]
    net = M.SeqWorldModel("jepa", d=a["d"], layers=a["layers"], heads=a["heads"],
                          dropout=a["dropout"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=o["lr"], weight_decay=o["wd"])
    g = torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(fit), (o["bs"],), generator=g).tolist()
        b = M.collate([fit[i] for i in idx], device)
        pred, _, tgt, _ = M.cmd_hidden(net, b)
        loss = loss_fn(pred, tgt)
        if not torch.isfinite(loss):
            return net, False
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net, True


@torch.no_grad()
def _leakage_ok(net, device):
    """HARD FILTER: the cmd_t prediction must not move when obs_t (or later) is corrupted — a
    genome cannot 'win' by leaking the answer. Mirrors tests/test_seq_worldmodel.py."""
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = M.collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = M.collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=device) * 100.0  # corrupt obs_3 (odd index 2*3+1)
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]  # per cmd-position max change
    return bool((chg[:4] < 1e-4).all())  # positions 0..3 must be unaffected by obs_3


def _c_top1(pred, flat, seed):
    return M.content_retrieval(pred, flat["true"], flat["verbs"], seed=seed)["top1_sameverb"]


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
    seeds = [0] if mode == "proxy" else [0, 1, 2]
    steps = proxy_steps if mode == "proxy" else genome["chunks"]["optim"]["steps"]

    per_seed = []
    for s in seeds:
        try:
            fit, _ = M.split_train_dev(train_seqs, seed=s)
            net, ok = _train(genome, fit, device, loss_fn, s, steps)
            if not ok:
                return _fail("train_diverged (NaN/inf loss)", mode, seeds, steps, split, per_seed)
            if not _leakage_ok(net, device):
                return _fail("leakage_fail (cmd_t prediction moved when obs_t corrupted)", mode, seeds, steps, split, per_seed)
            mlp = M.train_cmd_only(fit, device, steps=steps, seed=s)
            flat = M.flatten_predictions(net, inner, device)
            flat["_cmd_embs"] = torch.stack([sq["z_cmd"][t] for sq in inner
                                             for t in range(sq["z_obs"].shape[0])])
            with torch.no_grad():
                nohist = mlp(flat["_cmd_embs"].to(device)).cpu()
            wm = _c_top1(flat["pred"], flat, s)
            rbc = _c_top1(M.retrieve_by_cmd_baseline(fit, flat), flat, s)
            noh = _c_top1(nohist, flat, s)
            cpy = _c_top1(flat["prev"], flat, s)
            mean = _c_top1(torch.zeros_like(flat["true"]), flat, s)
            base = max(rbc, noh, cpy)
            per_seed.append({"seed": s, "wm": round(wm, 4), "base": round(base, 4),
                             "margin": round(wm - base, 4), "retrieve_by_cmd": round(rbc, 4),
                             "no_history": round(noh, 4), "copy_prev": round(cpy, 4),
                             "predict_mean": round(mean, 4)})
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
