"""Arch speed bench + exact-equivalence check for speed-focused evolve rounds.

  uv run python -m evolve.bench --arch r7_path_delta_fastweights_codex            # ms/step
  uv run python -m evolve.bench --arch r9_foo --ref r7_path_delta_fastweights_codex --eq

Bench protocol (fixed — comparability across the round): B=64, L=32 interleaved stream,
fwd + MSE-at-cmd-positions bwd + AdamW step; 5 warmup, median of 30 timed steps, device
synchronized. --eq loads the REFERENCE arch's state_dict into the candidate (Tier-A exact
rewrites must keep identical param names/shapes) and reports the max abs forward diff on a
fixed seeded input; a Tier-A pass is < 1e-4 (float reassociation only).
"""

import argparse
import importlib
import json
import statistics
import time

import torch

D = 768


def _inputs(bs, length, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randn(bs, length, D, generator=g).to(device)
    types = torch.zeros(bs, length, dtype=torch.long, device=device)
    types[:, 1::2] = 1
    pad = torch.zeros(bs, length, dtype=torch.bool, device=device)
    return tok, types, pad


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def build(name, params=None):
    mod = importlib.import_module(f"evolve.chunks.arch.{name}")
    torch.manual_seed(0)
    return mod.build(**(params or {}))


def bench(net, device, bs, length, steps, warmup):
    net = net.to(device).train()
    tok, types, pad = _inputs(bs, length, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    times = []
    for i in range(warmup + steps):
        _sync(device)
        t0 = time.time()
        pred, _ = net(tok, types, pad)
        loss = pred[types == 0].pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _sync(device)
        if i >= warmup:
            times.append((time.time() - t0) * 1000)
    return {"median_ms": round(statistics.median(times), 2),
            "mean_ms": round(statistics.fmean(times), 2),
            "params": sum(p.numel() for p in net.parameters())}


def eq_check(cand, ref, bs=8, length=32):
    """Load ref weights into cand (Tier-A contract: identical state_dict) and compare
    forward outputs on a fixed input. CPU float32."""
    missing = cand.load_state_dict(ref.state_dict(), strict=True)
    cand.eval(); ref.eval()
    tok, types, pad = _inputs(bs, length, seed=7)
    with torch.no_grad():
        p1, h1 = ref(tok, types, pad)
        p2, h2 = cand(tok, types, pad)
    dp = (p1 - p2).abs().max().item()
    dh = (h1 - h2).abs().max().item() if h1.shape == h2.shape else float("nan")
    return {"max_abs_pred_diff": dp, "max_abs_h_diff": dh, "tier_a_pass": dp < 1e-4}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--params", default=None, help="JSON dict of build params")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--eq", action="store_true", help="run the Tier-A equivalence check vs --ref")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--len", type=int, default=32, dest="length")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv)
    params = json.loads(args.params) if args.params else None
    out = {"arch": args.arch, "device": args.device, "bs": args.bs, "len": args.length}
    out.update(bench(build(args.arch, params), args.device, args.bs, args.length, args.steps, args.warmup))
    if args.ref:
        out["ref"] = args.ref
        out["ref_bench"] = bench(build(args.ref), args.device, args.bs, args.length, args.steps, args.warmup)
        out["speedup"] = round(out["ref_bench"]["median_ms"] / out["median_ms"], 2)
        if args.eq:
            try:
                out["eq"] = eq_check(build(args.arch, params), build(args.ref))
            except Exception as e:  # Tier-B archs have a different state_dict — report, don't crash
                out["eq"] = {"tier_a_pass": False, "error": f"{type(e).__name__}: {e}"[:300]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
