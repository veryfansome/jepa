"""M1 attack: recompute the decision-0 geometry over ALL 100 inner goals, at EVERY decision
along the oracle path. For each (goal, decision): candidates = real child dirs of cwd
(+ '..' off-root, as in plan_env); rank the on-path child by
  (a) cosine(ground-truth cd-obs embedding of candidate, z_goal)   [the M1 claim's object]
  (b) cosine(cmd embedding of 'cd <candidate>', z_goal)            [lexical, the comparator]
Ground-truth cd obs is deterministic: render 'passage: cwd=<cand> exit=0\n' (cd has no stdout).
"""
import json, pathlib, sys, collections
sys.path.insert(0, "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
import torch
from realenv.plan_env import Enc, parse_child_dirs, LS_PROBE
from realenv.docker_env import DockerBox
from realenv import seq_worldmodel as M

SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/review")
DATA = "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa/data/dockerfs-e5"
GOALS = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa/data/plangoals-v1")

device = M.pick_device()
enc = Enc(DATA, device)

def embed_obs_batch(texts):
    """standardized obs embeddings for raw rendered texts (batch)."""
    out = []
    for i in range(0, len(texts), 48):
        out.append(enc._embed(texts[i:i+48]))
    e = torch.cat(out)
    return (e - enc.mo) / enc.so

def embed_cmd_batch(texts):
    out = []
    for i in range(0, len(texts), 48):
        out.append(enc._embed(texts[i:i+48]))
    e = torch.cat(out)
    return (e - enc.mc) / enc.sc

# ---- collect candidate sets from real containers (listings cached per cwd) ----
rows = []  # per (image, goal, decision)
obs_texts = {}   # cd-obs render -> idx
cmd_texts = {}   # cmd render -> idx
goal_texts = {}  # goal render -> idx

def obs_key(path):
    t = f"passage: cwd={path} exit=0\n"
    return obs_texts.setdefault(t, len(obs_texts)), t

def cmd_key(text):
    t = "passage: " + text
    return cmd_texts.setdefault(t, len(cmd_texts)), t

for image in ("fedora:latest", "mariadb:latest"):
    tag = image.replace(":", "_").replace("/", "_")
    goals = [json.loads(l) for l in open(GOALS / f"goals-{tag}.jsonl")]
    box = DockerBox(image)
    listing = {}
    def ls_of(cwd):
        if cwd not in listing:
            box.run(f"cd {cwd}")
            r = box.run(LS_PROBE)
            listing[cwd] = r["output"]
            box.run("cd /")
        return listing[cwd]
    for goal in goals:
        gdir = goal["dir"]
        # goal embedding text (exact plan_env render of the harvested step)
        gt = f"passage: cwd={goal['step']['cwd']} exit={goal['step']['exit']}\n" + \
             (goal['step']['output'][:1600] + (f"\n...[{len(goal['step']['output'])-1600} more chars]" if len(goal['step']['output'])>1600 else ""))
        gidx = goal_texts.setdefault(gt, len(goal_texts))
        segs = [p for p in gdir.split("/") if p]
        cwd = "/"
        for dec in range(len(segs)):
            nxt = (cwd.rstrip("/") + "/" + segs[dec]) if cwd != "/" else "/" + segs[dec]
            cands = parse_child_dirs(ls_of(cwd), cwd)
            if cwd != "/":
                cands.append("..")
            cand_paths = []
            for c in cands:
                if c == "..":
                    parent = "/" + "/".join([p for p in cwd.split("/") if p][:-1])
                    cand_paths.append(parent if parent else "/")
                else:
                    cand_paths.append(c)
            if nxt not in cand_paths:
                rows.append({"image": image, "goal": gdir, "depth": goal["depth"], "dec": dec,
                             "skip": "onpath-not-listed", "n": len(cands)})
                cwd = nxt
                continue
            o_idx = [obs_key(p)[0] for p in cand_paths]
            c_idx = [cmd_key(f"cd {c}")[0] for c in cands]
            rows.append({"image": image, "goal": gdir, "depth": goal["depth"], "dec": dec,
                         "remain": len(segs) - dec, "n": len(cands),
                         "on": cand_paths.index(nxt), "o_idx": o_idx, "c_idx": c_idx,
                         "g": gidx})
            cwd = nxt
    box.close()
    print(f"{image}: rows so far {len(rows)}, unique obs {len(obs_texts)}, cmd {len(cmd_texts)}", flush=True)

# ---- encode ----
ot = [None]*len(obs_texts);  [ot.__setitem__(v, k) for k, v in obs_texts.items()]
ct = [None]*len(cmd_texts);  [ct.__setitem__(v, k) for k, v in cmd_texts.items()]
gt_ = [None]*len(goal_texts); [gt_.__setitem__(v, k) for k, v in goal_texts.items()]
print(f"encoding {len(ot)} obs, {len(ct)} cmds, {len(gt_)} goals", flush=True)
Z_o = embed_obs_batch(ot); Z_c = embed_cmd_batch(ct); Z_g = embed_obs_batch(gt_)
Zo_n = torch.nn.functional.normalize(Z_o, dim=-1)
Zc_n = torch.nn.functional.normalize(Z_c, dim=-1)
Zg_n = torch.nn.functional.normalize(Z_g, dim=-1)

# ---- rank ----
res = []
for r in rows:
    if "skip" in r:
        res.append(r); continue
    g = Zg_n[r["g"]]
    so = Zo_n[torch.tensor(r["o_idx"])] @ g
    sc = Zc_n[torch.tensor(r["c_idx"])] @ g
    on = r["on"]
    r2 = dict(r)
    r2["rank_obs"] = int((so > so[on]).sum().item()) + 1
    r2["rank_cmd"] = int((sc > sc[on]).sum().item()) + 1
    r2["cos_obs_on"] = round(float(so[on]), 4)
    r2["cos_obs_best_off"] = round(float(so[torch.arange(len(so)) != on].max()), 4)
    del r2["o_idx"], r2["c_idx"]
    res.append(r2)
(SCRATCH / "m1_rows.json").write_text(json.dumps(res))

# ---- report ----
def agg(rows, key):
    by = collections.defaultdict(list)
    for r in rows:
        if "skip" in r: continue
        by[r[key]].append(r)
    out = {}
    for k in sorted(by):
        rs = by[k]
        out[k] = {"n": len(rs),
                  "obs_top1": round(sum(r["rank_obs"] == 1 for r in rs)/len(rs), 3),
                  "obs_top3": round(sum(r["rank_obs"] <= 3 for r in rs)/len(rs), 3),
                  "obs_medrank": sorted(r["rank_obs"] for r in rs)[len(rs)//2],
                  "cmd_top1": round(sum(r["rank_cmd"] == 1 for r in rs)/len(rs), 3),
                  "cmd_top3": round(sum(r["rank_cmd"] <= 3 for r in rs)/len(rs), 3),
                  "mean_ncand": round(sum(r["n"] for r in rs)/len(rs), 1)}
    return out

ok = [r for r in res if "skip" not in r]
skips = [r for r in res if "skip" in r]
print(f"\nscored decisions: {len(ok)}  skipped: {len(skips)}")
print("\nBY REMAINING DEPTH (goal levels below cwd):")
print(json.dumps(agg(ok, "remain"), indent=1))
print("\nBY DECISION INDEX:")
print(json.dumps(agg(ok, "dec"), indent=1))
d0deep = [r for r in ok if r["dec"] == 0 and r["depth"] >= 3]
print(f"\nDECISION-0, deep goals (depth>=3): n={len(d0deep)} "
      f"obs_top1={sum(r['rank_obs']==1 for r in d0deep)/len(d0deep):.3f} "
      f"obs_top3={sum(r['rank_obs']<=3 for r in d0deep)/len(d0deep):.3f} "
      f"cmd_top1={sum(r['rank_cmd']==1 for r in d0deep)/len(d0deep):.3f} "
      f"cmd_top3={sum(r['rank_cmd']<=3 for r in d0deep)/len(d0deep):.3f}")
