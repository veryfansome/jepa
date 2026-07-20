"""Phase-0 Stage 2: LIVE receding-horizon latent-MPC navigation in real containers — the
planning claim proper. Given a goal observation embedding (the `ls -la` view of a target
directory on a held-out image), the agent starts at `/` in a real container and must reach
the goal directory by choosing `cd` actions, each chosen by a horizon-2 IMAGINED rollout in
the frozen champion's latent space: imagine `cd <candidate>` -> imagined obs (write-policy
fork below) -> imagine `ls -la` -> squared-L2 distance of the imagined listing latent to
z_goal; execute the argmin for real; repeat.

Stated assumptions (pre-registered): the candidate action space is AGENT-VISIBLE — child
directories parsed from the real `ls -la` observation of the cwd (plus `cd ..` off-root) —
and every planner shares the identical candidate builder and execution loop. The imagination
WRITE-POLICY fork: 'write' (imagined obs fed back as the obs token — primary) vs 'withhold'
(obs slot zeroed) — both reported. Success = box cwd == goal dir within H = min(depth+3, 7)
cd-decisions (history stays within the trained 16-step horizon).

Planners: wm (frozen champion, Stage-1-certified checkpoints), masked (self-only twin — the
history control), lexical (cosine of z_cmd('cd <target>') to z_goal — the echo navigator;
the goal render contains cwd=<goaldir>, so this is a strong honest baseline), random.
Fidelity slice: at each executed step, distance(imagined ls latent of the chosen candidate,
real encoded ls latent) by decision index — the direct measurement of rollout compounding.

  uv run python -m realenv.plan_env harvest --out data/plangoals-v1 --images fedora:latest,mariadb:latest,rockylinux:9,httpd:2.4
  uv run python -m realenv.plan_env run --goals data/plangoals-v1 --images fedora:latest,mariadb:latest \
      --planner wm --genome g.json --ckpt-dir ckpts/ --seeds 0,1,2 --write-policy write --out runs/plan/stage2-wm.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from realenv import seq_worldmodel as M
from realenv.docker_env import DockerBox
from realenv.collect_docker import UNAME_OPTS, CONFIG_FILES
from evolve import genome as G
from evolve.chunks.perception import enc_e5_base as PERC

D = M.D
LS_PROBE = "ls -la"          # deterministic in-distribution probe (in LS_OPTS)
MAX_DECISIONS = 7            # opener(2) + 7*(cd+ls) = 16 steps = the trained horizon


class Enc:
    """The exact dockerfs-e5 encoding path: enc_e5_base render + e5 mean-pool + train stats."""

    def __init__(self, data_root, device):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(PERC.MODEL)
        self.model = AutoModel.from_pretrained(PERC.MODEL).to(device).eval()
        self.device = device
        train = M.cached_encode(data_root, "train", "answerdotai/ModernBERT-base", device)
        self.mo, self.so, self.mc, self.sc = M.standardize_stats(train)

    @torch.no_grad()
    def _embed(self, texts):
        e = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        e = {k: v.to(self.device) for k, v in e.items()}
        h = self.model(**e).last_hidden_state
        return PERC.pool(h, e["attention_mask"]).float().cpu()

    def obs(self, step):
        return (self._embed([PERC.render_obs(step)]) - self.mo) / self.so

    def cmd(self, text):
        return (self._embed([PERC.render_cmd({"cmd": text})]) - self.mc) / self.sc


def parse_child_dirs(ls_la_output, cwd):
    """Child directories from a real `ls -la` observation (agent-visible action space)."""
    out = []
    for line in ls_la_output.split("\n"):
        parts = line.split()
        if len(parts) >= 9 and parts[0].startswith("d"):
            name = parts[-1]
            if name in (".", ".."):
                continue
            out.append((cwd.rstrip("/") + "/" + name) if cwd != "/" else "/" + name)
    return out


def depth_of(path):
    return len([p for p in path.split("/") if p])


def harvest(args):
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "per_image": args.per_image, "depths": [2, 4],
               "probe": LS_PROBE, "images": {}}
    for image in args.images.split(","):
        box = DockerBox(image)
        rng = torch.Generator().manual_seed(args.seed)
        dirs, _ = box.enumerate()
        elig = [d for d in dirs if 2 <= depth_of(d) <= 4]
        order = torch.randperm(len(elig), generator=rng).tolist()
        goals = []
        for i in order:
            d = elig[i]
            r_cd = box.run(f"cd {d}")
            if r_cd["exit"] != 0:
                continue
            r_ls = box.run(LS_PROBE)
            box.run("cd /")
            if r_ls["exit"] != 0 or len(r_ls["output"].split("\n")) < 4:
                continue  # need a non-trivial listing
            goals.append({"dir": d, "depth": depth_of(d), "step": r_ls})
            if len(goals) >= args.per_image:
                break
        box.close()
        tag = image.replace(":", "_").replace("/", "_")
        (outdir / f"goals-{tag}.jsonl").write_text("\n".join(json.dumps(g) for g in goals) + "\n")
        summary["images"][image] = {"n": len(goals),
                                    "depth_hist": {str(k): sum(1 for g in goals if g["depth"] == k)
                                                   for k in (2, 3, 4)}}
        print(f"{image}: {len(goals)} goals", flush=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))


def _forward_pred_at_last_cmd(net, hist, device):
    """hist: list of (z_cmd [1,D], z_obs [1,D] or None-for-pending). Build the interleaved
    stream (pending obs -> zeros) and return the prediction at the LAST cmd position."""
    n = len(hist)
    seq = {"z_cmd": torch.cat([h[0] for h in hist]),
           "z_obs": torch.cat([h[1] if h[1] is not None else torch.zeros(1, D) for h in hist])}
    b = M.collate([seq], device)
    with torch.no_grad():
        pred, _ = net(b["tok"], b["types"], b["key_pad"])
    return pred[:, 0::2][0, n - 1].cpu().unsqueeze(0)


def imagine_candidate(net, target_mod, hist, z_cd_cmd, z_ls_cmd, write_policy, device):
    """Horizon-2 rollout: imagined cd obs and imagined ls latent, both [1,D]."""
    h1 = hist + [(z_cd_cmd, None)]
    prev_obs = hist[-1][1] if hist and hist[-1][1] is not None else torch.zeros(1, D)
    pred_cd = _forward_pred_at_last_cmd(net, h1, device)
    rec_cd = target_mod.to_obs(pred_cd, prev_obs)
    obs_slot = rec_cd if write_policy == "write" else None
    h2 = hist + [(z_cd_cmd, obs_slot), (z_ls_cmd, None)]
    pred_ls = _forward_pred_at_last_cmd(net, h2, device)
    rec_ls = target_mod.to_obs(pred_ls, rec_cd if write_policy == "write" else prev_obs)
    return rec_cd, rec_ls


def goal_dist(z, z_goal, dist):
    """Candidate-score distance. 'cos' is the calibrated choice for contrastive-era models:
    their predictions rank correctly but carry off-manifold magnitudes (measured: in-dist
    pred-vs-true sqL2 1859 > random-pair 1478), and magnitude artifacts do NOT cancel when
    comparing DIFFERENT predictions (one per action) against one goal."""
    if dist == "cos":
        return 1.0 - float((z[0] / z[0].norm().clamp_min(1e-8)) @ (z_goal[0] / z_goal[0].norm().clamp_min(1e-8)))
    return float(((z - z_goal) ** 2).sum())


def run_episode(box, enc, goal, planner, net, target_mod, rng, write_policy, device,
                score_mode="min", dist_mode="cos"):
    z_goal = enc.obs(goal["step"])
    box.run("cd /")
    hist = []
    # in-distribution opener: uname + cat config (seeded like the collection policy)
    for cmd in (f"uname {UNAME_OPTS[int(torch.randint(0, len(UNAME_OPTS), (1,), generator=rng))]}".strip(),
                f"cat {CONFIG_FILES[int(torch.randint(0, len(CONFIG_FILES), (1,), generator=rng))]}"):
        st = box.run(cmd)
        hist.append((enc.cmd(cmd), enc.obs(st)))
    st_ls = box.run(LS_PROBE)
    hist.append((enc.cmd(LS_PROBE), enc.obs(st_ls)))
    H = min(goal["depth"] + 3, MAX_DECISIONS)
    fidelity = []
    for dec in range(H):
        cands = parse_child_dirs(st_ls["output"], box.cwd)
        if box.cwd != "/":
            cands.append("..")
        if not cands:
            break
        cd_cmds = [f"cd {c}" for c in cands]
        z_cds = [enc.cmd(c) for c in cd_cmds]
        z_ls = enc.cmd(LS_PROBE)
        if planner == "random":
            pick = int(torch.randint(0, len(cands), (1,), generator=rng))
            imagined = None
        elif planner == "lexical":
            gn = z_goal[0] / z_goal[0].norm().clamp_min(1e-8)
            sims = torch.tensor([float((z[0] / z[0].norm().clamp_min(1e-8)) @ gn) for z in z_cds])
            pick = int(sims.argmax())
            imagined = None
        else:  # wm / masked: latent MPC
            outs = [imagine_candidate(net, target_mod, hist, z_cds[i], z_ls, write_policy, device)
                    for i in range(len(cands))]
            scores = []
            for rec_cd, rec_ls in outs:
                d_cd = goal_dist(rec_cd, z_goal, dist_mode)
                d_ls = goal_dist(rec_ls, z_goal, dist_mode)
                scores.append({"ls": d_ls, "cd": d_cd, "min": min(d_cd, d_ls),
                               "sum": d_cd + d_ls}[score_mode])
            pick = int(torch.tensor(scores).argmin())
            imagined = outs[pick][1]
        st_cd = box.run(cd_cmds[pick])
        hist.append((z_cds[pick], enc.obs(st_cd)))
        st_ls = box.run(LS_PROBE)
        z_real_ls = enc.obs(st_ls)
        if imagined is not None:
            fidelity.append({"decision": dec, "imag_real_dist": round(float(((imagined - z_real_ls) ** 2).sum()), 3)})
        hist.append((enc.cmd(LS_PROBE), z_real_ls))
        if box.cwd == goal["dir"]:
            return {"success": True, "decisions": dec + 1, "H": H, "depth": goal["depth"],
                    "fidelity": fidelity}
    return {"success": False, "decisions": H, "H": H, "depth": goal["depth"], "fidelity": fidelity}


def run(args):
    if args.planner in ("wm", "masked") and not (args.genome and args.ckpt_dir):
        raise SystemExit(f"planner '{args.planner}' requires --genome and --ckpt-dir")
    device = M.pick_device()
    enc = Enc(args.data, device)
    gen = json.load(open(args.genome)) if args.genome else None
    target_mod = G.load_target(gen) if gen else None
    seeds = [int(x) for x in args.seeds.split(",")]
    report = {"planner": args.planner, "write_policy": args.write_policy,
              "score_mode": args.score_mode, "dist": args.dist, "images": {},
              "goals_root": args.goals, "per_seed": []}
    for s in seeds:
        net = None
        if args.planner == "wm":
            ck = torch.load(pathlib.Path(args.ckpt_dir) / f"{gen['id']}.s{s}.pt", weights_only=False)
            build, ap_ = G.load_arch(gen)
            net = build(**ap_)
            net.load_state_dict(ck["state_dict"])
            net = net.to(device).eval()
        elif args.planner == "masked":
            mk = torch.load(pathlib.Path(args.ckpt_dir) / f"masked.s{s}.pt", weights_only=False)
            net = M.SeqWorldModel("jepa", 0, no_history=True)
            net.load_state_dict(mk["state_dict"])
            net = net.to(device).eval()
        rows = []
        for image in args.images.split(","):
            tag = image.replace(":", "_").replace("/", "_")
            goals = [json.loads(l) for l in open(pathlib.Path(args.goals) / f"goals-{tag}.jsonl")]
            box = DockerBox(image)
            rng = torch.Generator().manual_seed(1000 * s + 7)
            for gi, goal in enumerate(goals):
                r = run_episode(box, enc, goal, args.planner, net, target_mod, rng,
                                args.write_policy, device, args.score_mode, args.dist)
                r.update({"image": image, "goal": goal["dir"], "seed": s})
                rows.append(r)
                if (gi + 1) % 10 == 0:
                    sr = sum(x["success"] for x in rows) / len(rows)
                    print(f"  seed {s} {image} {gi+1}/{len(goals)} running-success={sr:.3f}", flush=True)
            box.close()
        n = len(rows)
        report["per_seed"].append({
            "seed": s, "n": n,
            "success": round(sum(r["success"] for r in rows) / n, 4),
            "mean_decisions_when_success": round(
                sum(r["decisions"] for r in rows if r["success"]) /
                max(1, sum(r["success"] for r in rows)), 2),
            "fidelity_by_decision": _fid_agg(rows),
            "episodes": rows})
        print(f"seed {s}: success={report['per_seed'][-1]['success']}", flush=True)
    report["success_mean"] = round(sum(p["success"] for p in report["per_seed"]) / len(seeds), 4)
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}", flush=True)
    print(json.dumps({"planner": args.planner, "success_mean": report["success_mean"]}))
    return report


def _fid_agg(rows):
    by = {}
    for r in rows:
        for f in r["fidelity"]:
            by.setdefault(f["decision"], []).append(f["imag_real_dist"])
    return {str(k): round(sum(v) / len(v), 2) for k, v in sorted(by.items())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--out", required=True)
    h.add_argument("--images", required=True)
    h.add_argument("--per-image", type=int, default=50)
    h.add_argument("--seed", type=int, default=0)
    h.set_defaults(fn=harvest)
    r = sub.add_parser("run")
    r.add_argument("--goals", required=True)
    r.add_argument("--images", required=True)
    r.add_argument("--planner", required=True, choices=["wm", "masked", "lexical", "random"])
    r.add_argument("--genome", default=None)
    r.add_argument("--ckpt-dir", default=None)
    r.add_argument("--data", default="data/dockerfs-e5")
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--write-policy", default="write", choices=["write", "withhold"])
    r.add_argument("--score-mode", default="min", choices=["ls", "cd", "min", "sum"])
    r.add_argument("--dist", default="cos", choices=["cos", "l2"])
    r.add_argument("--out", default=None)
    r.set_defaults(fn=run)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
