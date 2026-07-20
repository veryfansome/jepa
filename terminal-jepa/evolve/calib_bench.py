"""R10-calibration fast battery: a Docker-free, seconds-fast planning-relevance +
calibration benchmark for trained world models, distilled from the Stage-2 adversarial
review's decision-0 machinery.

  uv run python -m evolve.calib_bench build --goals data/plangoals-v1 \
      --images fedora:latest,mariadb:latest --out data/plangoals-v1/dec0-battery-v1.pt
  uv run python -m evolve.calib_bench eval --battery data/plangoals-v1/dec0-battery-v1.pt \
      --genome g.json --ckpt ckpts/champ.s0.pt --data data/dockerfs-e5

`build` (one-time, needs Docker): for every inner goal, record a FIXED in-distribution
opener (uname -a; cat /etc/os-release; ls -la at /) and the root candidate set (child dirs
parsed from the real listing), encode everything with the exact dockerfs-e5 pipeline, and
store standardized embeddings + on-path labels. The artifact is the frozen instrument; its
sha256 is recorded in the round prereg.

`eval` (no Docker): for each goal, horizon-2 imagination per candidate (write policy,
per the Stage-2 review: candidate-specific imagined content is load-bearing), scored by
SUM of cosine distances of imagined cd-obs and imagined ls-obs to the goal (the review
showed `min` degenerates to ls-only and undersells the model; `sum` was better on every
seed). Reports:
  first_move_acc — fraction of goals where the picked root child is on the goal path
                   (review-measured champion band 0.37-0.54 across seeds; random ~0.06)
  matched_sqL2 / norm_ratio / cosine_matched — calibration on cached inner-val
                   (champion: ~2020 matched vs ~1430 true-pair; ||pred||^2/||true||^2 ~4.7)
"""

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from realenv import seq_worldmodel as M
from evolve import genome as G
from evolve.splits import split_val

D = M.D
OPENER = ["uname -a", "cat /etc/os-release"]
LS_PROBE = "ls -la"


def build(args):
    from realenv.plan_env import Enc, parse_child_dirs
    from realenv.docker_env import DockerBox
    device = M.pick_device()
    enc = Enc(args.data, device)
    goals_out = []
    for image in args.images.split(","):
        tag = image.replace(":", "_").replace("/", "_")
        goals = [json.loads(l) for l in open(pathlib.Path(args.goals) / f"goals-{tag}.jsonl")]
        box = DockerBox(image)
        box.run("cd /")
        hist = []
        for cmd in OPENER:
            st = box.run(cmd)
            hist.append((enc.cmd(cmd), enc.obs(st)))
        st_ls = box.run(LS_PROBE)
        hist.append((enc.cmd(LS_PROBE), enc.obs(st_ls)))
        cands = parse_child_dirs(st_ls["output"], "/")
        z_cds = torch.cat([enc.cmd(f"cd {c}") for c in cands])
        z_ls_cmd = enc.cmd(LS_PROBE)
        box.close()
        for g in goals:
            first = "/" + g["dir"].split("/")[1]
            on_path = [c == first for c in cands]
            if not any(on_path):
                continue  # root child not visible (should not happen per the review audit)
            goals_out.append({
                "image": image, "dir": g["dir"], "depth": g["depth"],
                "hist_cmd": torch.cat([h[0] for h in hist]),
                "hist_obs": torch.cat([h[1] for h in hist]),
                "cands": cands, "z_cds": z_cds, "z_ls_cmd": z_ls_cmd,
                "z_goal": enc.obs(g["step"]), "on_path": on_path,
            })
        print(f"{image}: {sum(1 for x in goals_out if x['image'] == image)} battery goals, "
              f"{len(cands)} root candidates", flush=True)
    out = pathlib.Path(args.out)
    torch.save({"version": "dec0-battery-v1", "opener": OPENER, "probe": LS_PROBE,
                "score": "sum-cos, write policy", "goals": goals_out}, out)
    print(f"wrote {out} sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")


def eval_net(net, target_mod, battery, device):
    from realenv.plan_env import imagine_candidate
    hits, n = 0, 0
    for g in battery["goals"]:
        hist = [(g["hist_cmd"][i:i + 1], g["hist_obs"][i:i + 1]) for i in range(g["hist_cmd"].shape[0])]
        gn = g["z_goal"][0] / g["z_goal"][0].norm().clamp_min(1e-8)
        scores = []
        for i in range(len(g["cands"])):
            rec_cd, rec_ls = imagine_candidate(net, target_mod, hist, g["z_cds"][i:i + 1],
                                               g["z_ls_cmd"], "write", device)
            s = (1 - float((rec_cd[0] / rec_cd[0].norm().clamp_min(1e-8)) @ gn)) + \
                (1 - float((rec_ls[0] / rec_ls[0].norm().clamp_min(1e-8)) @ gn))
            scores.append(s)
        hits += g["on_path"][int(torch.tensor(scores).argmin())]
        n += 1
    return {"first_move_acc": round(hits / n, 4), "n_goals": n}


def calib_metrics(net, target_mod, data, device, seed=0):
    train = M.cached_encode(data, "train", "answerdotai/ModernBERT-base", device)
    val = M.cached_encode(data, "val", "answerdotai/ModernBERT-base", device)
    mo, so, mc, sc = M.standardize_stats(train)
    M.apply_stats(train, mo, so, mc, sc)
    M.apply_stats(val, mo, so, mc, sc)
    inner = split_val(val, "inner")
    flat = M.flatten_predictions(net, inner, device)
    with torch.no_grad():
        pred = target_mod.to_obs(flat["pred"], flat["prev"])
    true = flat["true"]
    matched = float(((pred - true) ** 2).sum(-1).mean())
    g = torch.Generator().manual_seed(seed)
    i1 = torch.randint(0, true.shape[0], (4000,), generator=g)
    i2 = torch.randint(0, true.shape[0], (4000,), generator=g)
    rand_pair = float(((true[i1] - true[i2]) ** 2).sum(-1).mean())
    norm_ratio = float((pred ** 2).sum(-1).mean() / (true ** 2).sum(-1).mean())
    cosine = float(torch.nn.functional.cosine_similarity(pred, true, dim=-1).mean())
    return {"matched_sqL2": round(matched, 1), "rand_pair_sqL2": round(rand_pair, 1),
            "norm_ratio": round(norm_ratio, 3), "cosine_matched": round(cosine, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("--goals", required=True)
    b.add_argument("--images", required=True)
    b.add_argument("--data", default="data/dockerfs-e5")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=build)
    e = sub.add_parser("eval")
    e.add_argument("--battery", required=True)
    e.add_argument("--genome", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--data", default="data/dockerfs-e5")
    e.add_argument("--out", default=None)

    def do_eval(args):
        device = M.pick_device()
        gen = json.load(open(args.genome))
        ck = torch.load(args.ckpt, weights_only=False)
        build_fn, ap_ = G.load_arch(gen)
        net = build_fn(**ap_)
        net.load_state_dict(ck["state_dict"])
        net = net.to(device).eval()
        tmod = G.load_target(gen)
        battery = torch.load(args.battery, weights_only=False)
        res = eval_net(net, tmod, battery, device)
        res.update(calib_metrics(net, tmod, args.data, device))
        res["ckpt"] = args.ckpt
        print(json.dumps(res, indent=1))
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
        return res

    e.set_defaults(fn=do_eval)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
