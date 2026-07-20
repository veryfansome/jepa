"""Stage-1 shortcut hunt: does a no-model HISTORY-LEXICAL baseline reproduce the WM plan@1?

Reconstructs the exact Stage-1 candidate sets (same seeds, same dedup) via realenv.plan_eval
functions, aligns them with the raw text in data/dockerfs-e5/val.jsonl, and measures:
  (a) fraction of positions where the true cmd's absolute argument path appears as a substring
      of the rendered history text (steps 0..t-1), vs the same for distractors;
  (b) plan@1 of history-lexical planners (char-3gram overlap; path-substring; combos) on the
      full 2000-goal x 3-seed inner protocol.
"""
import json, pathlib, sys, re
sys.path.insert(0, "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
import torch
from realenv import seq_worldmodel as M
from realenv.plan_eval import build_pools, sample_goals, draw_candidates
from evolve.splits import split_val

ROOT = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
DATA = ROOT / "data/dockerfs-e5"
OBS_CAP = 1600

print("loading cached embeddings ...", flush=True)
train_seqs = torch.load(DATA / "emb-seq-train.pt", weights_only=False)
val_seqs = torch.load(DATA / "emb-seq-val.pt", weights_only=False)
mo, so, mc, sc = M.standardize_stats(train_seqs)
M.apply_stats(val_seqs, mo, so, mc, sc)
evalset = split_val(val_seqs, "inner")
print(f"evalset: {len(evalset)} seqs", flush=True)

# raw text aligned with evalset (same filter, order preserved)
raw = [json.loads(l) for l in open(DATA / "val.jsonl")]
raw_inner = [r for r in raw if ("fedora" in r["image"] or "mariadb" in r["image"])]
assert len(raw_inner) == len(evalset), (len(raw_inner), len(evalset))
mismatch = 0
for rq, sq in zip(raw_inner, evalset):
    if [s["cmd"] for s in rq["steps"]] != sq["cmds"]:
        mismatch += 1
print(f"alignment: {mismatch} mismatched sequences out of {len(evalset)}", flush=True)
assert mismatch == 0

def render_obs(step):
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"

# per-sequence step texts (what the model's encoder saw, minus the constant 'passage:' prefix)
step_txt = []
for rq in raw_inner:
    step_txt.append([(s["cmd"], render_obs(s)) for s in rq["steps"]])

def arg_path(cmd):
    for tok in cmd.split()[1:]:
        if tok.startswith("/"):
            return tok
    return None

def grams(text, n=3):
    return set(text[i:i+n] for i in range(len(text) - n + 1))

pools = build_pools(evalset)
goals = sample_goals(evalset, pools, ("ls", "cat"), 2000, seed=1234)
print(f"goals: {len(goals)}", flush=True)

# cache history text + grams per (si, t)
hist_cache = {}
def history(si, t):
    key = (si, t)
    if key not in hist_cache:
        txt = "\n".join(c + "\n" + o for c, o in step_txt[si][:t])
        hist_cache[key] = (txt, grams(txt))
    return hist_cache[key]

def plan1(scores):  # higher = better; strict win for index 0
    return all(scores[0] > s for s in scores[1:])

results = {}
for s in (0, 1, 2):
    n = 0
    true_in = 0; distr_in = 0; distr_tot = 0
    any_distr_in = 0
    hits = {"g3": 0, "sub": 0, "sub_g3": 0, "g3_1k": 0}
    per_verb = {}
    exclusive = 0          # true path in history AND no distractor path in history
    for gi, goal in enumerate(goals):
        cands = draw_candidates(evalset, pools, goal, 8, seed=s * 1_000_003 + gi, dedup_cos=0.99)
        if cands is None:
            continue
        n += 1
        si, t, verb = goal
        htxt, hg = history(si, t)
        htxt1k = htxt[:] if len(htxt) <= 0 else htxt   # placeholder
        paths = [arg_path(c["cmd"]) for c in cands]
        ins = [(p is not None and p in htxt) for p in paths]
        true_in += ins[0]
        distr_in += sum(ins[1:]); distr_tot += len(ins) - 1
        any_distr_in += any(ins[1:])
        if ins[0] and not any(ins[1:]):
            exclusive += 1
        # scores
        g3 = []
        for c in cands:
            cg = grams(c["cmd"])
            g3.append(len(cg & hg) / max(len(cg), 1))
        sub = [1.0 if i else 0.0 for i in ins]
        sub_g3 = [sub[j] + 0.001 * g3[j] for j in range(len(cands))]
        row = {"g3": plan1(g3), "sub": plan1(sub), "sub_g3": plan1(sub_g3)}
        for k, v in row.items():
            hits[k] += v
            per_verb.setdefault(verb, {}).setdefault(k, []).append(v)
    res = {
        "n": n,
        "true_path_in_hist": round(true_in / n, 4),
        "distractor_path_in_hist": round(distr_in / distr_tot, 4),
        "any_distractor_in_hist": round(any_distr_in / n, 4),
        "true_exclusive_in_hist": round(exclusive / n, 4),
        "plan1": {k: round(v / n, 4) for k, v in hits.items() if k != "g3_1k"},
        "per_verb": {vb: {k: round(sum(x) / len(x), 4) for k, x in d.items()}
                     for vb, d in per_verb.items()},
    }
    results[s] = res
    print(f"seed {s}: {json.dumps(res)}", flush=True)

out = pathlib.Path(__file__).parent / "stage1_shortcut_results.json"
out.write_text(json.dumps(results, indent=1))
print("wrote", out)
