"""Third attack: retrieve-from-HISTORY planner (history, no dynamics).
pred(cand) = z_obs[u*] of the earlier step u<t in the SAME sequence whose z_cmd is nearest
to the candidate's z_cmd; rank candidates by sqL2(pred, goal). Also a copy-prev planner
control (same pred for all candidates -> plan@1 = 0 by strictness, skip) and a hybrid:
history-retrieve if nearest-cmd cos > thresh else fit-retrieve.
"""
import json, pathlib, sys
sys.path.insert(0, "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
import torch
from realenv import seq_worldmodel as M
from realenv.plan_eval import build_pools, sample_goals, draw_candidates
from evolve.splits import split_val

ROOT = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
DATA = ROOT / "data/dockerfs-e5"
train_seqs = torch.load(DATA / "emb-seq-train.pt", weights_only=False)
val_seqs = torch.load(DATA / "emb-seq-val.pt", weights_only=False)
mo, so, mc, sc = M.standardize_stats(train_seqs)
M.apply_stats(train_seqs, mo, so, mc, sc)
M.apply_stats(val_seqs, mo, so, mc, sc)
evalset = split_val(val_seqs, "inner")
pools = build_pools(evalset)
goals = sample_goals(evalset, pools, ("ls", "cat"), 2000, seed=1234)

fit0, _ = M.split_train_dev(train_seqs, seed=0)
fit_cmd = torch.cat([s["z_cmd"] for s in fit0])
fit_obs = torch.cat([s["z_obs"] for s in fit0])

def plan1(d):
    return bool((d[0] < d[1:]).all())

results = {}
for s in (0, 1, 2):
    n = 0
    hits = {"hist_retr": 0, "hist_or_fit": 0}
    per_verb = {}
    for gi, goal in enumerate(goals):
        cands = draw_candidates(evalset, pools, goal, 8, seed=s * 1_000_003 + gi, dedup_cos=0.99)
        if cands is None:
            continue
        n += 1
        si, t, verb = goal
        sq = evalset[si]
        z_goal = sq["z_obs"][t]
        hc = sq["z_cmd"][:t]           # history commands (this sequence only)
        ho = sq["z_obs"][:t]
        dh, dhf = [], []
        for c in cands:
            zc = c["z_cmd"]
            dcmd = ((hc - zc) ** 2).sum(-1)
            u = int(dcmd.argmin())
            pred_h = ho[u]
            dh.append(((pred_h - z_goal) ** 2).sum())
            # hybrid: use history match only if very close in cmd space, else fit retrieval
            zcn = zc / zc.norm().clamp_min(1e-8)
            hcn = hc / hc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cosmax = float((hcn @ zcn).max())
            if cosmax > 0.98:
                dhf.append(((pred_h - z_goal) ** 2).sum())
            else:
                df = ((fit_cmd - zc) ** 2).sum(-1)
                pred_f = fit_obs[int(df.argmin())]
                dhf.append(((pred_f - z_goal) ** 2).sum())
        row = {"hist_retr": plan1(torch.stack(dh)), "hist_or_fit": plan1(torch.stack(dhf))}
        for k, v in row.items():
            hits[k] += v
            per_verb.setdefault(verb, {}).setdefault(k, []).append(v)
    res = {"n": n, "plan1": {k: round(v / n, 4) for k, v in hits.items()},
           "per_verb": {vb: {k: round(sum(x) / len(x), 4) for k, x in d.items()}
                        for vb, d in per_verb.items()}}
    results[s] = res
    print(f"seed {s}: {json.dumps(res)}", flush=True)

out = pathlib.Path(__file__).parent / "stage1_histretr_results.json"
out.write_text(json.dumps(results, indent=1))
print("wrote", out)
