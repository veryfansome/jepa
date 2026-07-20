"""M3 attack: is the write-mode advantage due to INFORMATIVE imagined cd-obs, or merely
because withhold's zero token is more off-distribution? Decision-0 ranking control over all
100 inner goals x 3 certified seeds. Obs-slot variants for the imagined-ls step:
  write    = model's own imagined cd-obs for that candidate (the claimed mechanism)
  withhold = zeros (the Stage-2 comparison arm)
  shuf     = imagined cd-obs of a DIFFERENT candidate (plausible token, wrong content)
  gt       = ground-truth cd-obs embedding for that candidate
  randobs  = one fixed real train obs embedding (in-distribution, irrelevant content)
Metric: rank of the on-path root child under min(cos_cd, cos_ls) and under ls-only cosine.
Also: cos-vs-L2 pick agreement for write mode (M2 support)."""
import json, pathlib, sys, collections
sys.path.insert(0, "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
import torch
from realenv.plan_env import Enc, parse_child_dirs, LS_PROBE, goal_dist
from realenv.docker_env import DockerBox
from realenv.collect_docker import UNAME_OPTS, CONFIG_FILES
from realenv import seq_worldmodel as M
from evolve import genome as G

SCR = pathlib.Path("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/review")
CK = pathlib.Path("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/plan/pod/ckpts")
GEN = json.load(open("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/genomes/r9-arch-chunked-codex.json"))
DATA = "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa/data/dockerfs-e5"
GOALS = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa/data/plangoals-v1")
D = M.D

device = M.pick_device()
enc = Enc(DATA, device)
# one fixed in-distribution train obs (standardized), for the randobs variant
train = M.cached_encode(DATA, "train", "x", device)
z_randobs = ((train[0]["z_obs"][2] - enc.mo[0]) / enc.so[0]).unsqueeze(0)

build, ap_ = G.load_arch(GEN)
nets = {}
for s in (0, 1, 2):
    ck = torch.load(CK / f"r9-arch-chunked-codex.s{s}.pt", weights_only=False)
    net = build(**ap_); net.load_state_dict(ck["state_dict"]); nets[s] = net.to(device).eval()

@torch.no_grad()
def batch_pred_last_cmd(net, seqs):
    """seqs: list of dicts with z_cmd [n,D], z_obs [n,D] (same n). pred at last cmd, [B,D]."""
    b = M.collate(seqs, device)
    pred, _ = net(b["tok"], b["types"], b["key_pad"])
    n = seqs[0]["z_cmd"].shape[0]
    return pred[:, 0::2][:, n - 1].cpu()

def mk(hist_pairs):
    return {"z_cmd": torch.cat([p[0] for p in hist_pairs]),
            "z_obs": torch.cat([p[1] if p[1] is not None else torch.zeros(1, D) for p in hist_pairs])}

VARIANTS = ("write", "withhold", "shuf", "gt", "randobs")
rows = []
gen0 = torch.Generator().manual_seed(1007)
for image in ("fedora:latest", "mariadb:latest"):
    tag = image.replace(":", "_").replace("/", "_")
    goals = [json.loads(l) for l in open(GOALS / f"goals-{tag}.jsonl")]
    box = DockerBox(image)
    # root candidates (fixed per image)
    st_ls0 = box.run(LS_PROBE)
    cands = parse_child_dirs(st_ls0["output"], "/")
    z_cds = torch.cat([enc.cmd(f"cd {c}") for c in cands])          # [K,D]
    z_gt = torch.cat([enc.obs({"cwd": c, "exit": 0, "output": ""}) for c in cands])  # [K,D]
    z_lscmd = enc.cmd(LS_PROBE)
    K = len(cands)
    for gi, goal in enumerate(goals):
        z_goal = enc.obs(goal["step"])
        segs = [p for p in goal["dir"].split("/") if p]
        on = cands.index("/" + segs[0]) if "/" + segs[0] in cands else None
        if on is None:
            continue
        # opener (fresh draw per goal, as episodes do)
        u = f"uname {UNAME_OPTS[int(torch.randint(0, len(UNAME_OPTS), (1,), generator=gen0))]}".strip()
        cf = f"cat {CONFIG_FILES[int(torch.randint(0, len(CONFIG_FILES), (1,), generator=gen0))]}"
        hist = []
        for cmd in (u, cf):
            st = box.run(cmd)
            hist.append((enc.cmd(cmd), enc.obs(st)))
        hist.append((enc.cmd(LS_PROBE), enc.obs(st_ls0)))
        for s, net in nets.items():
            # step 1: imagined cd obs per candidate (shared by all variants)
            h1 = [mk(hist + [(z_cds[i:i+1], None)]) for i in range(K)]
            rec_cd = batch_pred_last_cmd(net, h1)                    # [K,D]
            slots = {"write": rec_cd,
                     "withhold": torch.zeros(K, D),
                     "shuf": torch.roll(rec_cd, 1, dims=0),
                     "gt": z_gt,
                     "randobs": z_randobs.expand(K, D)}
            row = {"image": image, "goal": goal["dir"], "depth": goal["depth"], "seed": s, "K": K, "on": on}
            d_cd = torch.tensor([goal_dist(rec_cd[i:i+1], z_goal, "cos") for i in range(K)])
            for v in VARIANTS:
                h2 = [mk(hist + [(z_cds[i:i+1], slots[v][i:i+1]), (z_lscmd, None)]) for i in range(K)]
                rec_ls = batch_pred_last_cmd(net, h2)                # [K,D]
                d_ls = torch.tensor([goal_dist(rec_ls[i:i+1], z_goal, "cos") for i in range(K)])
                sc_min = torch.minimum(d_cd, d_ls)
                row[f"{v}_rank_min"] = int((sc_min < sc_min[on]).sum().item()) + 1
                row[f"{v}_rank_ls"] = int((d_ls < d_ls[on]).sum().item()) + 1
                if v == "write":
                    d_ls_l2 = torch.tensor([goal_dist(rec_ls[i:i+1], z_goal, "l2") for i in range(K)])
                    d_cd_l2 = torch.tensor([goal_dist(rec_cd[i:i+1], z_goal, "l2") for i in range(K)])
                    sc_l2 = torch.minimum(d_cd_l2, d_ls_l2)
                    row["write_pick_cos"] = int(sc_min.argmin())
                    row["write_pick_l2"] = int(sc_l2.argmin())
                    row["write_rank_min_l2"] = int((sc_l2 < sc_l2[on]).sum().item()) + 1
                    row["lsnorm_std_over_mean"] = round(float(rec_ls.norm(dim=1).std() / rec_ls.norm(dim=1).mean()), 3)
            rows.append(row)
        if (gi + 1) % 20 == 0:
            print(f"{image} {gi+1}/{len(goals)}", flush=True)
    box.close()

(SCR / "m3_rows.json").write_text(json.dumps(rows))
n = len(rows)
print(f"\nrows (goal x seed): {n}")
for v in VARIANTS:
    t1m = sum(r[f"{v}_rank_min"] == 1 for r in rows) / n
    t3m = sum(r[f"{v}_rank_min"] <= 3 for r in rows) / n
    t1l = sum(r[f"{v}_rank_ls"] == 1 for r in rows) / n
    t3l = sum(r[f"{v}_rank_ls"] <= 3 for r in rows) / n
    print(f"{v:9s} min-mode top1={t1m:.3f} top3={t3m:.3f} | ls-only top1={t1l:.3f} top3={t3l:.3f}")
agree = sum(r["write_pick_cos"] == r["write_pick_l2"] for r in rows) / n
t1_l2 = sum(r["write_rank_min_l2"] == 1 for r in rows) / n
print(f"\nwrite: cos-pick == l2-pick {agree:.3f}; l2 min-mode top1={t1_l2:.3f}")
print(f"mean std/mean of ||rec_ls|| across candidates: {sum(r['lsnorm_std_over_mean'] for r in rows)/n:.3f}")
