"""Phase-0 planning probe (Stage 1): goal-conditioned ACTION RANKING by predicted-latent
distance — the first test of the JEPA program's third leg (plan by latent distance).

At an eval position t in a held-out sequence, the GOAL is the logged true next-observation
embedding z_goal. Build K candidates = the true command + (K-1) seeded same-verb,
same-image, ABSOLUTE-ARGUMENT distractor commands from OTHER sequences (context-dependent
commands like bare `ls`/`cd ..` have ill-posed counterfactual outcomes and are excluded from
both goals and distractors). Each planner ranks candidates by predicted distance-to-goal;
plan@1 = fraction of positions where the true command ranks strictly first.

Planners (all share the identical candidate sets and environment):
  wm        — the FROZEN champion world model: substitute z_cmd(candidate) at stream
              position t (true obs_t ZEROED out of the input — never visible), forward,
              squared-L2 of the reconstructed prediction to z_goal.
  masked    — the history-free control: the R4 arch with self-only attention trained under
              the champion's objective (the sanity-arm construction), same procedure.
  lexical   — no model, no history: rank by cosine(z_cmd(candidate), z_goal). The echo
              planner; must win on cd-goals (calibration) and should fail on content goals.
  retrieve  — retrieve-by-cmd planner: candidate's predicted outcome = the fit-split obs of
              the nearest fit command embedding; banks shared cross-distro structure.
  (random floor = 1/K, analytic.)

plan-margin = plan@1(wm) − max(plan@1 of lexical, retrieve). The history mechanism control
is plan@1(wm) − plan@1(masked). Headline on content-verb goals (ls/cat); cd-goals reported
as the calibration slice only. Checkpoints come from `evolve.cli score --save-dir` and MUST
pass the fidelity gate (archived margins reproduced) before this eval is trusted.

  uv run python -m realenv.plan_eval --genome g.json --ckpt-dir ckpts/ --split inner \
      --data data/dockerfs-e5 --out plan-inner.json
"""

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from realenv import seq_worldmodel as M
from evolve import genome as G
from evolve.splits import split_val

D = M.D
ABS_ARG = {"ls": re.compile(r"^ls(\s+-\S+)*\s+/"), "cat": re.compile(r"^cat\s+/"),
           "cd": re.compile(r"^cd\s+/")}


def eligible(cmd, verb):
    rx = ABS_ARG.get(verb)
    return bool(rx and rx.match(cmd))


def build_pools(seqs):
    """Per (image, verb) pools of absolute-argument steps: (seq_i, t, cmd, z_cmd, z_obs)."""
    pools = {}
    for si, sq in enumerate(seqs):
        for t in range(sq["z_obs"].shape[0]):
            cmd = sq["cmds"][t]
            verb = M.verb_of(cmd)
            if verb in ABS_ARG and eligible(cmd, verb):
                pools.setdefault((sq["image"], verb), []).append(
                    (si, t, cmd, sq["z_cmd"][t], sq["z_obs"][t]))
    return pools


def sample_goals(seqs, pools, verbs, n, seed, min_t=2):
    """Seeded sample of eval positions: absolute-arg true command, verb in `verbs`, t>=min_t,
    and >= 2*K pool entries from OTHER sequences of the same image."""
    cands = []
    for si, sq in enumerate(seqs):
        for t in range(min_t, sq["z_obs"].shape[0]):
            verb = M.verb_of(sq["cmds"][t])
            if verb in verbs and eligible(sq["cmds"][t], verb):
                cands.append((si, t, verb))
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(cands), generator=g).tolist()
    picked = []
    for i in order:
        si, t, verb = cands[i]
        pool = pools.get((seqs[si]["image"], verb), [])
        if sum(1 for e in pool if e[0] != si) >= 16:
            picked.append((si, t, verb))
        if len(picked) >= n:
            break
    return picked


def draw_candidates(seqs, pools, goal, k, seed, dedup_cos):
    """True command + (k-1) seeded distractors (same image+verb, other sequences, distinct cmd
    text, logged-obs cosine to goal < dedup_cos). Returns list of dicts; index 0 = true."""
    si, t, verb = goal
    sq = seqs[si]
    z_goal = sq["z_obs"][t]
    pool = [e for e in pools[(sq["image"], verb)] if e[0] != si and e[2] != sq["cmds"][t]]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pool), generator=g).tolist()
    out = [{"cmd": sq["cmds"][t], "z_cmd": sq["z_cmd"][t], "logged_obs": z_goal, "true": True}]
    seen = {sq["cmds"][t]}
    gn = z_goal / z_goal.norm().clamp_min(1e-8)
    for i in order:
        _, _, cmd, zc, zo = pool[i]
        if cmd in seen:
            continue
        cos = float((zo / zo.norm().clamp_min(1e-8)) @ gn)
        if cos >= dedup_cos:
            continue
        out.append({"cmd": cmd, "z_cmd": zc, "logged_obs": zo, "true": False, "goal_cos": cos})
        seen.add(cmd)
        if len(out) == k:
            break
    return out if len(out) == k else None


def wm_distances(net, target_mod, seqs, goal, cands, device):
    """Forward the frozen WM once per candidate on the prefix stream (cmd_t swapped, obs_t
    ZEROED) and return squared-L2 distances of reconstructed predictions to the goal."""
    si, t, _ = goal
    sq = seqs[si]
    batch = []
    for c in cands:
        zc = sq["z_cmd"][: t + 1].clone()
        zo = sq["z_obs"][: t + 1].clone()
        zc[t] = c["z_cmd"]
        zo[t] = 0.0                       # the true obs never enters the input
        batch.append({"z_cmd": zc, "z_obs": zo})
    b = M.collate(batch, device)
    with torch.no_grad():
        pred, _ = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = pred[:, 0::2][:, t]     # [K, D] prediction at position t
        prev = sq["z_obs"][t - 1].to(device).expand_as(cmd_pred) if t > 0 else torch.zeros_like(cmd_pred)
        rec = target_mod.to_obs(cmd_pred.cpu(), prev.cpu())
    z_goal = sq["z_obs"][t]
    return ((rec - z_goal) ** 2).sum(-1)


def plan_at_1(dists):
    """Strict: true (index 0) must be strictly smaller than every distractor."""
    return bool((dists[0] < dists[1:]).all())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--split", default="inner", choices=["inner", "final"])
    ap.add_argument("--data", default="data/dockerfs-e5")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-goals", type=int, default=2000)
    ap.add_argument("--n-cd", type=int, default=500)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dedup-cos", type=float, default=0.99)
    ap.add_argument("--masked-steps", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    gen = json.load(open(args.genome))
    device = M.pick_device()
    train_seqs = M.cached_encode(args.data, "train", args.model, device)
    val_seqs = M.cached_encode(args.data, "val", args.model, device)
    mo, so, mc, sc = M.standardize_stats(train_seqs)
    M.apply_stats(train_seqs, mo, so, mc, sc)
    M.apply_stats(val_seqs, mo, so, mc, sc)
    evalset = split_val(val_seqs, args.split)
    target_mod = G.load_target(gen)
    obj = G.load_objective(gen)
    seeds = [int(x) for x in args.seeds.split(",")]
    ck = pathlib.Path(args.ckpt_dir)

    pools = build_pools(evalset)
    goals = sample_goals(evalset, pools, ("ls", "cat"), args.n_goals, seed=1234)
    cd_goals = sample_goals(evalset, pools, ("cd",), args.n_cd, seed=1235)
    print(f"eval positions: {len(goals)} content, {len(cd_goals)} cd", flush=True)

    # fit-split retrieval memory for the retrieve-by-cmd planner (seed-independent)
    fit0, _ = M.split_train_dev(train_seqs, seed=0)
    fit_cmd = torch.cat([s["z_cmd"] for s in fit0])
    fit_obs = torch.cat([s["z_obs"] for s in fit0])

    def retrieve_pred(zc):
        d = ((fit_cmd - zc) ** 2).sum(-1)
        return fit_obs[int(d.argmin())]

    report = {"genome": gen["id"], "split": args.split, "k": args.k,
              "dedup_cos": args.dedup_cos, "n_content": len(goals), "n_cd": len(cd_goals),
              "device": str(device), "per_seed": []}

    for s in seeds:
        ckpt = torch.load(ck / f"{gen['id']}.s{s}.pt", weights_only=False)
        build, ap_ = G.load_arch(gen)
        net = build(**ap_)
        net.load_state_dict(ckpt["state_dict"])
        net = net.to(device).eval()
        mpath = ck / f"masked.s{s}.pt"
        if mpath.exists():
            mnet = M.SeqWorldModel("jepa", 0, no_history=True)
            mnet.load_state_dict(torch.load(mpath, weights_only=False)["state_dict"])
            mnet = mnet.to(device).eval()
        else:
            fit, _ = M.split_train_dev(train_seqs, seed=s)
            mnet = M.train_model("jepa", fit, device, steps=args.masked_steps, seed=s,
                                 no_history=True, jepa_loss=obj).eval()
            torch.save({"state_dict": mnet.cpu().state_dict(), "seed": s}, mpath)
            mnet = mnet.to(device).eval()

        def run_slice(slice_goals, tag):
            hits = {"wm": 0, "masked": 0, "lexical": 0, "retrieve": 0}
            per_verb = {}
            n = 0
            for gi, goal in enumerate(slice_goals):
                cands = draw_candidates(evalset, pools, goal, args.k,
                                        seed=s * 1_000_003 + gi, dedup_cos=args.dedup_cos)
                if cands is None:
                    continue
                n += 1
                si, t, verb = goal
                z_goal = evalset[si]["z_obs"][t]
                dw = wm_distances(net, target_mod, evalset, goal, cands, device)
                dm = wm_distances(mnet, target_mod, evalset, goal, cands, device)
                gn = z_goal / z_goal.norm().clamp_min(1e-8)
                dl = torch.tensor([-float((c["z_cmd"] / c["z_cmd"].norm().clamp_min(1e-8)) @ gn)
                                   for c in cands])
                dr = torch.stack([((retrieve_pred(c["z_cmd"]) - z_goal) ** 2).sum() for c in cands])
                for name, dist in (("wm", dw), ("masked", dm), ("lexical", dl), ("retrieve", dr)):
                    hit = plan_at_1(dist)
                    hits[name] += hit
                    per_verb.setdefault(verb, {}).setdefault(name, []).append(hit)
            row = {f"{k}_plan1": round(v / max(n, 1), 4) for k, v in hits.items()}
            row["n"] = n
            row["per_verb"] = {vb: {k: round(sum(x) / len(x), 4) for k, x in d.items()}
                               for vb, d in per_verb.items()}
            return row

        content = run_slice(goals, "content")
        cd = run_slice(cd_goals, "cd")
        margin = round(content["wm_plan1"] - max(content["lexical_plan1"], content["retrieve_plan1"]), 4)
        hist_gap = round(content["wm_plan1"] - content["masked_plan1"], 4)
        report["per_seed"].append({"seed": s, "content": content, "cd_calibration": cd,
                                   "plan_margin": margin, "history_gap": hist_gap,
                                   "ckpt_margin": ckpt.get("margin")})
        print(f"seed {s}: content wm={content['wm_plan1']} lex={content['lexical_plan1']} "
              f"retr={content['retrieve_plan1']} masked={content['masked_plan1']} "
              f"-> plan_margin={margin} history_gap={hist_gap} | cd: wm={cd['wm_plan1']} "
              f"lex={cd['lexical_plan1']}", flush=True)

    def mean(path):
        vals = [r[path] for r in report["per_seed"]]
        return round(sum(vals) / len(vals), 4)

    report["plan_margin_mean"] = mean("plan_margin")
    report["history_gap_mean"] = mean("history_gap")
    report["random_floor"] = round(1 / args.k, 4)
    print(json.dumps({k: report[k] for k in ("plan_margin_mean", "history_gap_mean", "random_floor")}, indent=1))
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}", flush=True)
    return report


if __name__ == "__main__":
    main()
