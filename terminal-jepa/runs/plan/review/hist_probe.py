"""M4 attack: does history actually change the wm's decision-0 choice?
Replicate the planner's decision-0 scoring (imagine_candidate + cos + score-mode min)
for every inner goal under three histories:
  matched  = real opener (uname -a, cat /etc/os-release, ls -la at /) from the goal's image
  swapped  = the OTHER image's opener (wrong system identity + wrong root listing)
  zeroed   = same cmds, all obs slots zeroed
Measure pick agreement across conditions and first-move accuracy (pick == true first
component). If picks barely change, 'history drives navigation' is unsupported at the
decision level."""
import json, sys
sys.path.insert(0, ".")
import torch
from realenv import seq_worldmodel as M
from realenv.plan_env import Enc, imagine_candidate, goal_dist, LS_PROBE, D
from realenv.docker_env import DockerBox
from evolve import genome as G

CK = "/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/plan/pod/ckpts"
SCR = "/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/review"
device = M.pick_device()
gen = json.load(open("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/genomes/r9-arch-chunked-codex.json"))
build, ap = G.load_arch(gen)
net = build(**ap)
net.load_state_dict(torch.load(f"{CK}/r9-arch-chunked-codex.s0.pt", weights_only=False)["state_dict"])
net = net.to(device).eval()
target_mod = G.load_target(gen)
enc = Enc("data/dockerfs-e5", device)

# real openers from both images
openers = {}
for image in ("fedora:latest", "mariadb:latest"):
    box = DockerBox(image)
    steps = []
    for cmd in ("uname -a", "cat /etc/os-release", LS_PROBE):
        st = box.run(cmd)
        steps.append((cmd, st))
    box.close()
    openers[image] = steps

reach = json.load(open(f"{SCR}/reach.json"))
enc_cache = {}
def ecmd(t):
    if ("c", t) not in enc_cache: enc_cache[("c", t)] = enc.cmd(t)
    return enc_cache[("c", t)]

def hist_from(steps, zero_obs=False):
    h = []
    for cmd, st in steps:
        zo = torch.zeros(1, D) if zero_obs else enc.obs(st)
        h.append((ecmd(cmd), zo))
    return h

z_ls = ecmd(LS_PROBE)
res = {"matched": [], "swapped": [], "zeroed": []}
agree_ms, agree_mz = [], []
for image, other in (("fedora:latest", "mariadb:latest"), ("mariadb:latest", "fedora:latest")):
    tag = image.replace(":", "_").replace("/", "_")
    goals = {g["dir"]: g for g in (json.loads(l) for l in open(f"data/plangoals-v1/goals-{tag}.jsonl"))}
    hists = {"matched": hist_from(openers[image]),
             "swapped": hist_from(openers[other]),
             "zeroed": hist_from(openers[image], zero_obs=True)}
    for row in reach["images"][image]["rows"]:
        ch0 = row["chain"][0]
        assert ch0["cwd"] == "/"
        cands = ["/" + n for n in ch0["cands"]]
        z_cds = [ecmd(f"cd {c}") for c in cands]
        z_goal = enc.obs(goals[row["goal"]]["step"])
        true0 = "/" + ch0["true_next"]
        picks = {}
        for cond, h in hists.items():
            scores = []
            for zc in z_cds:
                rec_cd, rec_ls = imagine_candidate(net, target_mod, h, zc, z_ls, "write", device)
                scores.append(min(goal_dist(rec_cd, z_goal, "cos"), goal_dist(rec_ls, z_goal, "cos")))
            picks[cond] = int(torch.tensor(scores).argmin())
            res[cond].append(cands[picks[cond]] == true0)
        agree_ms.append(picks["matched"] == picks["swapped"])
        agree_mz.append(picks["matched"] == picks["zeroed"])
    print(image, "done", flush=True)

n = len(res["matched"])
print(f"n={n} decision-0 cases")
for cond in ("matched", "swapped", "zeroed"):
    print(f"  first-move top1 {cond}: {sum(res[cond])/n:.3f}")
print(f"  pick agreement matched-vs-swapped: {sum(agree_ms)/n:.3f}")
print(f"  pick agreement matched-vs-zeroed:  {sum(agree_mz)/n:.3f}")
