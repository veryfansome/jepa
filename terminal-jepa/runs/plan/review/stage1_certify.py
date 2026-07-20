import json, pathlib, sys
sys.path.insert(0, "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
import torch
from realenv import seq_worldmodel as M
from realenv.plan_eval import build_pools, sample_goals, draw_candidates
from evolve.splits import split_val
DATA = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa/data/dockerfs-e5")
train_seqs = torch.load(DATA / "emb-seq-train.pt", weights_only=False)
val_seqs = torch.load(DATA / "emb-seq-val.pt", weights_only=False)
mo, so, mc, sc = M.standardize_stats(train_seqs)
M.apply_stats(val_seqs, mo, so, mc, sc)
evalset = split_val(val_seqs, "inner")
pools = build_pools(evalset)
goals = sample_goals(evalset, pools, ("ls", "cat"), 2000, seed=1234)
for s in (0, 1, 2):
    hits = 0; n = 0
    for gi, goal in enumerate(goals):
        cands = draw_candidates(evalset, pools, goal, 8, seed=s*1_000_003+gi, dedup_cos=0.99)
        if cands is None: continue
        n += 1
        si, t, _ = goal
        z_goal = evalset[si]["z_obs"][t]
        gn = z_goal / z_goal.norm().clamp_min(1e-8)
        dl = torch.tensor([-float((c["z_cmd"]/c["z_cmd"].norm().clamp_min(1e-8)) @ gn) for c in cands])
        hits += bool((dl[0] < dl[1:]).all())
    print(f"seed {s}: lexical_plan1={hits/n:.4f} n={n}")
