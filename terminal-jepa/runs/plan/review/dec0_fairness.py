"""Adversarial eval-fairness review: decision-0 analysis on ALL 100 inner goals.

Questions attacked:
 A. score-mode cherry-pick: would ls-only / cd-only / sum have picked differently than min?
 B. WM-fair rank/calibrated MPC not tried: score candidate by rank of the goal among foil
    observations under the imagined ls prediction (Stage-1's certified skill). Does it
    materially change decision-0 choices vs absolute cos?
 C. M1 (flat field) under calibration: do GROUND-TRUTH cd/ls obs of the correct root child
    point at the goal once calibrated by foils (rank / mean-subtracted cos), even though
    absolute cos is flat?
Decision-0 only; opener replicated from the seed-0 run RNG stream (2 draws/episode).
"""
import json
import pathlib
import sys

REPO = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
sys.path.insert(0, str(REPO))
import torch

from realenv import seq_worldmodel as M
from realenv.docker_env import DockerBox
from realenv.collect_docker import UNAME_OPTS, CONFIG_FILES
from realenv.plan_env import Enc, parse_child_dirs, imagine_candidate, LS_PROBE
from evolve import genome as G
from evolve.splits import split_val

SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad")
CKPT = SCRATCH / "r9/plan/pod/ckpts"
GENOME = SCRATCH / "r9/genomes/r9-arch-chunked-codex.json"
OUT = SCRATCH / "review/dec0_results.json"
DEVICE = M.pick_device()
IMAGES = ["fedora:latest", "mariadb:latest"]
SEED = 0


def cos(a, b):
    return float((a[0] / a[0].norm().clamp_min(1e-8)) @ (b[0] / b[0].norm().clamp_min(1e-8)))


def main():
    enc = Enc(str(REPO / "data/dockerfs-e5"), DEVICE)
    gen = json.load(open(GENOME))
    target_mod = G.load_target(gen)
    build, ap_ = G.load_arch(gen)
    net = build(**ap_)
    ck = torch.load(CKPT / f"{gen['id']}.s{SEED}.pt", weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net = net.to(DEVICE).eval()

    # foil bank: inner-val logged ls observations (standardized), seeded sample of 128
    val = M.cached_encode(str(REPO / "data/dockerfs-e5"), "val", "answerdotai/ModernBERT-base", DEVICE)
    tr = M.cached_encode(str(REPO / "data/dockerfs-e5"), "train", "answerdotai/ModernBERT-base", DEVICE)
    mo, so, mc, sc = M.standardize_stats(tr)
    M.apply_stats(val, mo, so, mc, sc)
    inner = split_val(val, "inner")
    foil_rows = []
    for sq in inner:
        for t, cmd in enumerate(sq["cmds"]):
            if M.verb_of(cmd) == "ls":
                foil_rows.append(sq["z_obs"][t])
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(foil_rows), generator=g)[:128].tolist()
    foils = torch.stack([foil_rows[i] for i in idx])          # [F, D]
    foils_n = foils / foils.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    print(f"foil bank: {foils.shape[0]} ls observations from inner-val", flush=True)

    results = []
    gt_tables = {}
    for image in IMAGES:
        tag = image.replace(":", "_").replace("/", "_")
        goals = [json.loads(l) for l in open(REPO / f"data/plangoals-v1/goals-{tag}.jsonl")]
        box = DockerBox(image)
        rng = torch.Generator().manual_seed(1000 * SEED + 7)

        # fixed per-image: root listing + candidates + ground-truth cd/ls obs per child
        box.run("cd /")
        st_ls_root = box.run(LS_PROBE)
        cands = parse_child_dirs(st_ls_root["output"], "/")
        cd_cmds = [f"cd {c}" for c in cands]
        z_cds = [enc.cmd(c) for c in cd_cmds]
        z_ls_cmd = enc.cmd(LS_PROBE)
        z_root_ls = enc.obs(st_ls_root)
        gt = {}
        for c in cands:
            r_cd = box.run(f"cd {c}")
            r_ls = box.run(LS_PROBE)
            box.run("cd /")
            gt[c] = (enc.obs(r_cd), enc.obs(r_ls))
        print(f"{image}: {len(cands)} root candidates", flush=True)

        # opener obs per episode (replicate the seed-0 RNG stream: 2 draws/episode)
        for gi, goal in enumerate(goals):
            u = f"uname {UNAME_OPTS[int(torch.randint(0, len(UNAME_OPTS), (1,), generator=rng))]}".strip()
            cf = f"cat {CONFIG_FILES[int(torch.randint(0, len(CONFIG_FILES), (1,), generator=rng))]}"
            box.run("cd /")
            hist = []
            for cmd in (u, cf):
                st = box.run(cmd)
                hist.append((enc.cmd(cmd), enc.obs(st)))
            hist.append((enc.cmd(LS_PROBE), z_root_ls))
            z_goal = enc.obs(goal["step"])
            gn = z_goal[0] / z_goal[0].norm().clamp_min(1e-8)
            correct = "/" + goal["dir"].split("/")[1]

            # imagined rollouts for every candidate
            recs = [imagine_candidate(net, target_mod, hist, z_cds[i], z_ls_cmd, "write", DEVICE)
                    for i in range(len(cands))]
            row = {"image": image, "goal": goal["dir"], "depth": goal["depth"], "correct": correct}

            d_cd = torch.tensor([1 - cos(r[0], z_goal) for r in recs])
            d_ls = torch.tensor([1 - cos(r[1], z_goal) for r in recs])
            picks = {
                "cos_min": int(torch.minimum(d_cd, d_ls).argmin()),
                "cos_ls": int(d_ls.argmin()),
                "cos_cd": int(d_cd.argmin()),
                "cos_sum": int((d_cd + d_ls).argmin()),
            }
            # l2 variants (the rejected absolute-distance mode)
            l_cd = torch.tensor([float(((r[0] - z_goal) ** 2).sum()) for r in recs])
            l_ls = torch.tensor([float(((r[1] - z_goal) ** 2).sum()) for r in recs])
            picks["l2_min"] = int(torch.minimum(l_cd, l_ls).argmin())
            # rank-calibrated MPC (per-candidate foil calibration of the imagined ls)
            ranks, zs = [], []
            for r in recs:
                p = r[1][0] / r[1][0].norm().clamp_min(1e-8)
                cg = float(p @ gn)
                cf_ = foils_n @ p                      # [F]
                ranks.append(int((cf_ > cg).sum()))     # 0 = goal beats every foil
                zs.append(cg - float(cf_.mean()))
            rk = torch.tensor(ranks, dtype=torch.float)
            zt = torch.tensor(zs)
            picks["rank_ls"] = int((rk - 1e-4 * zt).argmin())   # rank, tie-break by margin
            picks["zcal_ls"] = int(zt.argmax())
            # same calibration on the imagined cd obs
            ranks_cd, zs_cd = [], []
            for r in recs:
                p = r[0][0] / r[0][0].norm().clamp_min(1e-8)
                cg = float(p @ gn)
                cf_ = foils_n @ p
                ranks_cd.append(int((cf_ > cg).sum()))
                zs_cd.append(cg - float(cf_.mean()))
            picks["zcal_min"] = int(torch.maximum(zt, torch.tensor(zs_cd)).argmax())
            # lexical decision-0
            sims = torch.tensor([cos(z, z_goal) for z in z_cds])
            picks["lexical"] = int(sims.argmax())

            row["picks"] = {k: cands[v] for k, v in picks.items()}
            row["hit"] = {k: cands[v] == correct for k, v in picks.items()}
            row["goal_rank_of_pick_rankls"] = ranks[picks["rank_ls"]]
            # fidelity sanity: imagined-vs-real ls sqL2 for the ACTUAL protocol pick (cos_min)
            pk = picks["cos_min"]
            row["fid_cos_min_pick"] = round(float(((recs[pk][1] - gt[cands[pk]][1]) ** 2).sum()), 1)

            # C: ground-truth obs geometry, absolute vs calibrated
            gt_cos_cd = {c: cos(gt[c][0], z_goal) for c in cands}
            gt_cos_ls = {c: cos(gt[c][1], z_goal) for c in cands}
            def zcal(z):
                p = z[0] / z[0].norm().clamp_min(1e-8)
                return float(p @ gn) - float((foils_n @ p).mean())
            gt_z_cd = {c: zcal(gt[c][0]) for c in cands}
            gt_z_ls = {c: zcal(gt[c][1]) for c in cands}
            row["gt_hit"] = {
                "abs_cos_cd": max(gt_cos_cd, key=gt_cos_cd.get) == correct,
                "abs_cos_ls": max(gt_cos_ls, key=gt_cos_ls.get) == correct,
                "zcal_cd": max(gt_z_cd, key=gt_z_cd.get) == correct,
                "zcal_ls": max(gt_z_ls, key=gt_z_ls.get) == correct,
            }
            results.append(row)
            if (gi + 1) % 20 == 0:
                print(f"  {image} {gi+1}/{len(goals)}", flush=True)
        box.close()
        gt_tables[image] = {"cands": cands}

    # aggregate
    modes = list(results[0]["hit"].keys())
    agg = {m: sum(r["hit"][m] for r in results) / len(results) for m in modes}
    gt_modes = list(results[0]["gt_hit"].keys())
    gt_agg = {m: sum(r["gt_hit"][m] for r in results) / len(results) for m in gt_modes}
    # agreement matrix of picks vs cos_min
    diff = {m: sum(r["picks"][m] != r["picks"]["cos_min"] for r in results) for m in modes}
    fid0 = sum(r["fid_cos_min_pick"] for r in results) / len(results)
    print("\n=== decision-0 correct-first-move rate (n=%d goals) ===" % len(results))
    for m in modes:
        print(f"  {m:10s} {agg[m]:.3f}   (picks differing from cos_min: {diff[m]})")
    print("=== ground-truth obs of correct child ranks first ===")
    for m in gt_modes:
        print(f"  {m:12s} {gt_agg[m]:.3f}")
    print(f"fidelity sanity (decision-0, cos_min pick, mean sqL2): {fid0:.1f}  [artifact seed0: 3441]")
    OUT.write_text(json.dumps({"agg": agg, "gt_agg": gt_agg, "diff_vs_cos_min": diff,
                               "fid0": fid0, "rows": results}, indent=1, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
