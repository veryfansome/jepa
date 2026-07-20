"""Second attack wave: cwd-locality + path-prefix shortcuts.
Scorers (no model):
  cwd_pref   — path-component common prefix between candidate arg path and cwd at t-1
  hist_pref  — max component common prefix vs {all earlier cwds} U {all earlier arg paths}
  cwd_g3     — cwd_pref with 3-gram tiebreak
  basename   — candidate basename appears in earlier ls output text
Also: oracle check — is cwd_pref of the TRUE candidate larger than distractors' on average?
"""
import json, pathlib, sys
sys.path.insert(0, "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
import torch
from realenv import seq_worldmodel as M
from realenv.plan_eval import build_pools, sample_goals, draw_candidates
from evolve.splits import split_val

ROOT = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
DATA = ROOT / "data/dockerfs-e5"
OBS_CAP = 1600

train_seqs = torch.load(DATA / "emb-seq-train.pt", weights_only=False)
val_seqs = torch.load(DATA / "emb-seq-val.pt", weights_only=False)
mo, so, mc, sc = M.standardize_stats(train_seqs)
M.apply_stats(val_seqs, mo, so, mc, sc)
evalset = split_val(val_seqs, "inner")
raw = [json.loads(l) for l in open(DATA / "val.jsonl")]
raw_inner = [r for r in raw if ("fedora" in r["image"] or "mariadb" in r["image"])]
assert all([s["cmd"] for s in rq["steps"]] == sq["cmds"] for rq, sq in zip(raw_inner, evalset))

def render_obs(step):
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"

def arg_path(cmd):
    for tok in cmd.split()[1:]:
        if tok.startswith("/"):
            return tok
    return None

def comps(p):
    return [c for c in p.split("/") if c]

def pref_len(a, b):
    ca, cb = comps(a), comps(b)
    n = 0
    for x, y in zip(ca, cb):
        if x != y:
            break
        n += 1
    return n

def grams(text, n=3):
    return set(text[i:i+n] for i in range(len(text) - n + 1))

pools = build_pools(evalset)
goals = sample_goals(evalset, pools, ("ls", "cat"), 2000, seed=1234)

def plan1(scores):
    return all(scores[0] > s for s in scores[1:])

results = {}
for s in (0, 1, 2):
    n = 0
    hits = {"cwd_pref": 0, "hist_pref": 0, "cwd_g3": 0, "basename": 0, "hist_pref_g3": 0}
    per_verb = {}
    true_pref_sum = 0.0; distr_pref_sum = 0.0
    for gi, goal in enumerate(goals):
        cands = draw_candidates(evalset, pools, goal, 8, seed=s * 1_000_003 + gi, dedup_cos=0.99)
        if cands is None:
            continue
        n += 1
        si, t, verb = goal
        steps = raw_inner[si]["steps"]
        cwd_prev = steps[t - 1].get("cwd", "/") if t > 0 else "/"
        earlier_paths = [cwd_prev]
        ls_text = []
        for u in range(t):
            p = arg_path(steps[u]["cmd"])
            if p:
                earlier_paths.append(p)
            earlier_paths.append(steps[u].get("cwd", "/"))
            if steps[u]["cmd"].split()[0] == "ls":
                ls_text.append(steps[u].get("output", "") or "")
        ls_text = "\n".join(ls_text)
        htxt = "\n".join(c["cmd"] + "\n" + render_obs(st) for c, st in [] ) # unused
        hist_g = grams("\n".join(st["cmd"] + "\n" + render_obs(st) for st in steps[:t]))
        paths = [arg_path(c["cmd"]) or "" for c in cands]
        cwd_sc = [pref_len(p, cwd_prev) for p in paths]
        hist_sc = [max(pref_len(p, q) for q in earlier_paths) for p in paths]
        g3 = []
        for c in cands:
            cg = grams(c["cmd"])
            g3.append(len(cg & hist_g) / max(len(cg), 1))
        base_sc = [1.0 if (comps(p) and comps(p)[-1] in ls_text) else 0.0 for p in paths]
        row = {
            "cwd_pref": plan1(cwd_sc),
            "hist_pref": plan1(hist_sc),
            "cwd_g3": plan1([cwd_sc[j] + 0.001 * g3[j] for j in range(8)]),
            "hist_pref_g3": plan1([hist_sc[j] + 0.001 * g3[j] for j in range(8)]),
            "basename": plan1([base_sc[j] + 0.001 * g3[j] for j in range(8)]),
        }
        true_pref_sum += cwd_sc[0]
        distr_pref_sum += sum(cwd_sc[1:]) / 7
        for k, v in row.items():
            hits[k] += v
            per_verb.setdefault(verb, {}).setdefault(k, []).append(v)
    res = {"n": n,
           "mean_cwdpref_true": round(true_pref_sum / n, 3),
           "mean_cwdpref_distr": round(distr_pref_sum / n, 3),
           "plan1": {k: round(v / n, 4) for k, v in hits.items()},
           "per_verb": {vb: {k: round(sum(x) / len(x), 4) for k, x in d.items()}
                        for vb, d in per_verb.items()}}
    results[s] = res
    print(f"seed {s}: {json.dumps(res)}", flush=True)

out = pathlib.Path(__file__).parent / "stage1_cwd_results.json"
out.write_text(json.dumps(results, indent=1))
print("wrote", out)
