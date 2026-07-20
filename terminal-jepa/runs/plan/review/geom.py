"""M1 attack: multi-goal, multi-decision quantification of the 'flat field' claim.
At every decision point on every inner goal's true path: rank the TRUE next component
among all real candidates (incl. '..' off-root) by
  (a) cosine(ground-truth cd-obs embedding, z_goal)   [obs space]
  (b) cosine(cmd-text embedding 'cd <path>', z_goal)  [what lexical uses]
Report top-1 rate + MRR stratified by remaining depth to goal."""
import json, sys, pathlib
sys.path.insert(0, ".")
import torch
from realenv import seq_worldmodel as M
from evolve.chunks.perception import enc_e5_base as PERC
from transformers import AutoModel, AutoTokenizer

SCRATCH = "/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/review"
device = M.pick_device()
tok = AutoTokenizer.from_pretrained(PERC.MODEL)
model = AutoModel.from_pretrained(PERC.MODEL).to(device).eval()
train = M.cached_encode("data/dockerfs-e5", "train", "answerdotai/ModernBERT-base", device)
mo, so, mc, sc = M.standardize_stats(train)

@torch.no_grad()
def embed(texts, bs=64):
    outs = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=256)
        e = {k: v.to(device) for k, v in e.items()}
        h = model(**e).last_hidden_state
        outs.append(PERC.pool(h, e["attention_mask"]).float().cpu())
    return torch.cat(outs)

def obs_txt(cwd):  # ground-truth cd obs render (cd output empty, exit 0)
    return PERC.render_obs({"cmd": "x", "output": "", "exit": 0, "cwd": cwd})

def cmd_txt(target):
    return PERC.render_cmd({"cmd": f"cd {target}"})

reach = json.load(open(f"{SCRATCH}/reach.json"))
rows_out = []
for image, tag in [("fedora:latest", "fedora_latest"), ("mariadb:latest", "mariadb_latest")]:
    goals = {g["dir"]: g for g in (json.loads(l) for l in open(f"data/plangoals-v1/goals-{tag}.jsonl"))}
    # collect unique texts
    texts = {}
    def add(t):
        if t not in texts: texts[t] = len(texts)
    for row in reach["images"][image]["rows"]:
        add(PERC.render_obs(goals[row["goal"]]["step"]))
        for ch in row["chain"]:
            cwd = ch["cwd"]
            parent = "/".join(cwd.split("/")[:-1]) or "/"
            cands = [((cwd.rstrip("/") + "/" + n) if cwd != "/" else "/" + n) for n in ch["cands"]]
            for p in cands:
                add(obs_txt(p)); add(cmd_txt(p))
            if cwd != "/":
                add(obs_txt(parent)); add(cmd_txt(".."))
    order = sorted(texts, key=texts.get)
    Z = embed(order)
    zo = (Z - mo) / so   # obs-standardized view
    zc = (Z - mc) / sc   # cmd-standardized view
    def zn(v): return v / v.norm().clamp_min(1e-8)
    for row in reach["images"][image]["rows"]:
        g = goals[row["goal"]]
        zg = zn(zo[texts[PERC.render_obs(g["step"])]])
        depth = len([c for c in row["goal"].split("/") if c])
        for i, ch in enumerate(row["chain"]):
            cwd = ch["cwd"]
            parent = "/".join(cwd.split("/")[:-1]) or "/"
            cands = [((cwd.rstrip("/") + "/" + n) if cwd != "/" else "/" + n) for n in ch["cands"]]
            names = list(ch["cands"])
            if cwd != "/":
                cands.append(parent); names.append("..")
            true_i = names.index(ch["true_next"]) if ch["true_next"] in names else None
            if true_i is None: continue
            s_obs = torch.tensor([float(zn(zo[texts[obs_txt(p)]]) @ zg) for p in cands])
            s_cmd = torch.tensor([float(zn(zc[texts[cmd_txt(p if n != '..' else '..')]]) @ zg) for p, n in zip(cands, names)])
            # note: '..' cmd is 'cd ..' as in the live episode
            r_obs = int((s_obs > s_obs[true_i]).sum()) + 1
            r_cmd = int((s_cmd > s_cmd[true_i]).sum()) + 1
            rows_out.append({"image": image, "goal": row["goal"], "depth": depth, "dec": i,
                             "remaining": depth - i, "n": len(cands),
                             "r_obs": r_obs, "r_cmd": r_cmd})
    print(image, "done", flush=True)

pathlib.Path(f"{SCRATCH}/geom.json").write_text(json.dumps(rows_out))
from collections import defaultdict
agg = defaultdict(lambda: {"n": 0, "t_obs": 0, "t_cmd": 0, "rr_obs": 0.0, "rr_cmd": 0.0, "exp_rand": 0.0})
for r in rows_out:
    a = agg[r["remaining"]]
    a["n"] += 1; a["t_obs"] += r["r_obs"] == 1; a["t_cmd"] += r["r_cmd"] == 1
    a["rr_obs"] += 1 / r["r_obs"]; a["rr_cmd"] += 1 / r["r_cmd"]; a["exp_rand"] += 1 / r["n"]
print(f"{'remain':>6} {'n':>4} {'top1_obs':>9} {'top1_cmd':>9} {'mrr_obs':>8} {'mrr_cmd':>8} {'rand_top1':>9}")
for k in sorted(agg):
    a = agg[k]
    print(f"{k:>6} {a['n']:>4} {a['t_obs']/a['n']:>9.3f} {a['t_cmd']/a['n']:>9.3f} "
          f"{a['rr_obs']/a['n']:>8.3f} {a['rr_cmd']/a['n']:>8.3f} {a['exp_rand']/a['n']:>9.3f}")
