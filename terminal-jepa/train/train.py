"""Phase 1 training (terminal-jepa.md §4).

Arms:
  sigreg          — two-term default (prediction + SIGReg), LeWorldModel recipe
  vicreg          — VICReg variance/covariance in place of SIGReg
  sigreg+idm      — SIGReg + inverse-dynamics auxiliary loss
  sigreg+tempsim  — SIGReg + temporal-similarity auxiliary loss
  recon           — generative twin: same encoder trunk, next-obs token CE (no JEPA loss)

Loss (JEPA arms): 1-step teacher-forced + 3-step rollout latent L2, + lambda * regularizer.
Gradients flow through everything — no EMA, no stop-gradient.

Usage: .venv/bin/python -m train.train --data data/v0 --arm sigreg --out runs/sigreg-s0
"""

import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from models import losses, nets
from models.data import TrajectoryData

HORIZON = 3  # window: obs t..t+3, actions t..t+2


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode_all(encoder, obs, chunk=256):
    """obs: [B, S, L] -> z: [B, S, D]; flattened batched forward."""
    b, s, l = obs.shape
    flat = obs.reshape(b * s, l)
    outs = [encoder(flat[i : i + chunk]) for i in range(0, flat.shape[0], chunk)]
    return torch.cat(outs).reshape(b, s, -1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v0")
    ap.add_argument("--arm", default="sigreg",
                    choices=["sigreg", "vicreg", "sigreg+idm", "sigreg+tempsim", "recon"])
    ap.add_argument("--encoder", default="cls", choices=list(nets.ENCODER_TYPES))
    ap.add_argument("--rollout-full-bptt", action="store_true",
                    help="restore gradient flow through the rollout TARGET (the "
                         "pre-audit behavior); default detaches it — LeWorldModel "
                         "validated grad-through-target only for 1-step teacher "
                         "forcing, and full BPTT through a 3-step composed rollout "
                         "rewards latent contraction (fidelity audit, 2026-07-10)")
    ap.add_argument("--regime", default="both")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-trajs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/ckpt.pt if present (best-effort: "
                         "restores weights, optimizer moments, and RNG streams)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(f"train:{args.seed}")
    device = pick_device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = TrajectoryData(
        pathlib.Path(args.data) / "train.jsonl", args.regime, args.max_trajs,
        keep_states=False,
    )
    print(f"loaded {len(data.trajs)} trajectories; device={device}", flush=True)
    print(json.dumps({"config": vars(args)}), flush=True)
    with open(out / "train.log", "a") as fh:
        fh.write(json.dumps({"config": vars(args)}) + "\n")

    m = nets.build_models(args.encoder)
    if args.arm == "recon":
        m["decoder"] = nets.ReconDecoder(d_z=m["encoder"].d_out)
        del m["predictor"]
    if args.arm == "sigreg+idm":
        m["idm"] = nets.IDMHead(d=m["encoder"].d_out)
    for mod in m.values():
        mod.to(device).train()
    params = [p for mod in m.values() for p in mod.parameters()]
    print(f"params: {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    # Param groups: no weight decay on 1-D params (biases, LN affines, gates — decay
    # silently pulls zero-init gates back toward closed), embeddings, or learned
    # queries/positions (fidelity audit; standard ViT practice).
    decay, no_decay = [], []
    for mod in m.values():
        for name, p in mod.named_parameters():
            # ".cond." catches AdaLN gate-producing weights (blocks.N.cond.weight) —
            # decaying them pulls zero-init gates closed, the exact effect this
            # grouping exists to prevent — without catching ReconDecoder's top-level
            # functional "cond" projection (adversarial review, 2026-07-10).
            if (p.ndim <= 1 or "emb" in name or "pos" in name
                    or "queries" in name or ".cond." in name):
                no_decay.append(p)
            else:
                decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )

    log = []
    skipped = 0
    start_step = 1
    ckpt_path = out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        prev = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert prev.get("arm") == args.arm and prev.get("encoder_type") == args.encoder, \
            "resume config mismatch"
        for k, v in m.items():
            v.load_state_dict(prev["modules"][k])
        res = prev.get("resume")
        if res:
            opt.load_state_dict(res["optimizer"])
            rng.setstate(res["py_rng"])
            torch.set_rng_state(res["torch_rng"])
            if res.get("mps_rng") is not None and torch.backends.mps.is_available():
                torch.mps.set_rng_state(res["mps_rng"])
            skipped = res.get("skipped", 0)
        start_step = prev["steps"] + 1
        print(f"resumed from step {prev['steps']}", flush=True)

    t0 = time.time()
    for step in range(start_step, args.steps + 1):
        batch = data.sample_windows(args.batch, HORIZON, rng)
        obs = batch["obs"].to(device)
        acts = batch["acts"].to(device)

        if args.arm == "recon":
            z_t = m["encoder"](obs[:, 0])
            a_emb = m["action_encoder"](acts[:, 0])
            loss = m["decoder"].loss(z_t, a_emb, obs[:, 1])
            parts = {"recon_ce": loss.item()}
        else:
            z = encode_all(m["encoder"], obs)  # [B, 4, D]
            b, s, la = acts.shape
            a = m["action_encoder"](acts.reshape(b * s, la)).reshape(b, s, -1)

            pred_losses = []
            # 1-step teacher-forced at every offset
            for t in range(HORIZON):
                zhat = m["predictor"](z[:, t], a[:, t])
                pred_losses.append(((zhat - z[:, t + 1]) ** 2).mean())
            # 3-step rollout from t=0. Target detached by default: our rollout term is
            # an extrapolation beyond both sources (LeWorldModel trains 1-step only;
            # 2512.24497 rolls out against FROZEN encoders with TBPTT), and letting
            # gradients flow into a 3-step target makes latent contraction a descent
            # direction — consistent with the observed z_std plateau below 1.0.
            zroll = z[:, 0]
            for t in range(HORIZON):
                zroll = m["predictor"](zroll, a[:, t])
            roll_target = z[:, HORIZON] if args.rollout_full_bptt else z[:, HORIZON].detach()
            pred_losses.append(((zroll - roll_target) ** 2).mean())
            pred_loss = torch.stack(pred_losses).mean()

            z_flat = z.reshape(-1, z.shape[-1])
            # Cross-fitted halves split by WINDOW (= by trajectory): the covariance
            # estimator needs independent noise between halves, and each window's 4
            # timesteps are heavily correlated (adversarial review, 2026-07-10).
            half = z.shape[0] // 2
            z_a = z[:half].reshape(-1, z.shape[-1])
            z_b = z[half:].reshape(-1, z.shape[-1])
            reg_parts = {}
            if args.arm == "vicreg":
                reg = losses.vicreg_var_cov(z_a, z_b)
            elif args.encoder == "slot":
                # Slot regularization (rebuilt after fidelity audit + adversarial
                # review): SIGReg per slot INDEX (static offsets penalized, pooled
                # evasion closed) + cross-fitted joint variance/decorrelation at
                # VICReg's true normalization AND 25:1 weighting.
                z_slots = z_flat.reshape(-1, m["encoder"].n_slots, m["encoder"].d_slot)
                reg_sig = losses.sigreg_per_index(z_slots)
                reg_vc = losses.vicreg_var_cov(z_a, z_b)
                reg = reg_sig + reg_vc
                reg_parts = {"reg_sigpi": reg_sig, "reg_varcov": reg_vc}
            else:
                reg = losses.sigreg(z_flat)
            loss = pred_loss + args.lam * reg
            parts = {"pred": pred_loss.item(), "reg": reg.item()}
            if reg_parts and (step % 100 == 0 or step == 1):
                parts.update({k: v.item() for k, v in reg_parts.items()})

            if args.arm == "sigreg+idm":
                aux, idm_metrics = m["idm"].loss(
                    z[:, :-1].reshape(-1, z.shape[-1]),
                    z[:, 1:].reshape(-1, z.shape[-1]),
                    batch["act_labels"].to(device).reshape(-1, 3),
                )
                loss = loss + args.aux_weight * aux
                parts["idm"] = aux.item()
                # .item() syncs MPS, so convert per-head metrics only at log steps.
                if step % 100 == 0 or step == 1:
                    parts.update({f"idm_{k}": v.item() for k, v in idm_metrics.items()})
            elif args.arm == "sigreg+tempsim":
                aux = losses.temporal_similarity(
                    z[:, :-1].reshape(-1, z.shape[-1]),
                    z[:, 1:].reshape(-1, z.shape[-1]),
                )
                loss = loss + args.aux_weight * aux
                parts["tempsim"] = aux.item()

        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            # MPS has produced nonfinite values under system memory pressure; skip the
            # step rather than poison the weights, and count it.
            skipped += 1
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            skipped += 1
            continue
        opt.step()

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                zs = (
                    z_flat.std(dim=0).mean().item()
                    if args.arm != "recon"
                    else float("nan")
                )
            rec = {"step": step, "loss": loss.item(), "z_std": zs,
                   "sec": round(time.time() - t0, 1), "skipped": skipped, **parts}
            log.append(rec)
            line = json.dumps(rec)
            print(line, flush=True)
            with open(out / "train.log", "a") as fh:  # tail -f friendly
                fh.write(line + "\n")
        if step % 1000 == 0:
            # Periodic checkpointing: four runs have been killed externally
            # (jetsam/user); end-only saving loses everything.
            save_ckpt(out, args, m, step, opt, rng, skipped)

    save_ckpt(out, args, m, args.steps, opt, rng, skipped)
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out / 'ckpt.pt'}", flush=True)


def save_ckpt(out, args, m, step, opt=None, rng=None, skipped=0):
    ckpt = {
        "arm": args.arm,
        "encoder_type": args.encoder,
        "regime": args.regime,
        "seed": args.seed,
        "steps": step,
        # Full training config: ablation flags and hyperparameters must be recoverable
        # from the artifact, not from run-directory names (adversarial review).
        "config": {
            "rollout_full_bptt": args.rollout_full_bptt,
            "lam": args.lam,
            "lr": args.lr,
            "batch": args.batch,
            "data": args.data,
            "aux_weight": args.aux_weight,
        },
        "modules": {k: v.state_dict() for k, v in m.items()},
    }
    if opt is not None:
        # Resume state: external kills (jetsam) are routine on this machine — four so
        # far. Resume is best-effort continuation, not bit-exact (MPS is
        # nondeterministic anyway): optimizer moments + both RNG streams + skip count.
        ckpt["resume"] = {
            "optimizer": opt.state_dict(),
            "py_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "mps_rng": (torch.mps.get_rng_state()
                        if torch.backends.mps.is_available() else None),
            "skipped": skipped,
        }
    tmp = out / "ckpt.pt.tmp"
    torch.save(ckpt, tmp)
    tmp.rename(out / "ckpt.pt")  # atomic: a mid-write kill can't corrupt the ckpt


if __name__ == "__main__":
    main()
