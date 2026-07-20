"""M4 attack part 2: for the goals wm-write actually solved, rebuild the REAL on-path
history (opener + ls at / + cd/ls down the true path) and, at the FINAL decision
(cwd = goal's parent), compare the wm pick with matched vs zeroed history."""
import json, sys
sys.path.insert(0, ".")
import torch
from realenv import seq_worldmodel as M
from realenv.plan_env import Enc, imagine_candidate, goal_dist, parse_child_dirs, LS_PROBE, D
from realenv.docker_env import DockerBox
from evolve import genome as G

CK = "/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/plan/pod/ckpts"
device = M.pick_device()
gen = json.load(open("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/genomes/r9-arch-chunked-codex.json"))
build, ap = G.load_arch(gen)
net = build(**ap)
net.load_state_dict(torch.load(f"{CK}/r9-arch-chunked-codex.s0.pt", weights_only=False)["state_dict"])
net = net.to(device).eval()
target_mod = G.load_target(gen)
enc = Enc("data/dockerfs-e5", device)

SOLVED = {"fedora:latest": ["/etc/ssl", "/usr/share/locale/zam", "/usr/share/licenses/libtirpc",
                            "/usr/share/licenses/zstd"],
          "mariadb:latest": ["/var/cache/apt", "/etc/iproute2/rt_tables.d", "/etc/ssl/certs",
                             "/usr/share/apport", "/usr/share/locale/ga"]}
cc = {}
def ecmd(t):
    if t not in cc: cc[t] = enc.cmd(t)
    return cc[t]
z_ls = ecmd(LS_PROBE)

tot = m_top1 = z_top1 = agree = 0
for image, dirs in SOLVED.items():
    tag = image.replace(":", "_").replace("/", "_")
    goals = {g["dir"]: g for g in (json.loads(l) for l in open(f"data/plangoals-v1/goals-{tag}.jsonl"))}
    box = DockerBox(image)
    for gd in dirs:
        # real history down to parent of goal
        box.run("cd /")
        steps = [("uname -a", box.run("uname -a")), ("cat /etc/os-release", box.run("cat /etc/os-release")),
                 (LS_PROBE, box.run(LS_PROBE))]
        comps = [c for c in gd.split("/") if c]
        for c in comps[:-1]:
            st_cd = box.run(f"cd {c}")
            steps.append((f"cd {box.cwd}", st_cd))  # planner uses full-path cd cmds
            steps.append((LS_PROBE, box.run(LS_PROBE)))
        st_ls = steps[-1][1]
        cands = parse_child_dirs(st_ls["output"], box.cwd)
        if box.cwd != "/":
            cands.append("..")
        cd_cmds = [f"cd {c}" for c in cands]
        z_goal = enc.obs(goals[gd]["step"])
        hist_m = [(ecmd(c), enc.obs(st)) for c, st in steps]
        hist_z = [(ecmd(c), torch.zeros(1, D)) for c, st in steps]
        picks = {}
        for cond, h in (("m", hist_m), ("z", hist_z)):
            sc = []
            for cmd in cd_cmds:
                rec_cd, rec_ls = imagine_candidate(net, target_mod, h, ecmd(cmd), z_ls, "write", device)
                sc.append(min(goal_dist(rec_cd, z_goal, "cos"), goal_dist(rec_ls, z_goal, "cos")))
            picks[cond] = cands[int(torch.tensor(sc).argmin())]
        tot += 1
        m_top1 += picks["m"] == gd
        z_top1 += picks["z"] == gd
        agree += picks["m"] == picks["z"]
        print(f"{image} {gd} ({len(cands)} cands, hist {len(steps)} steps): "
              f"matched->{picks['m']}  zeroed->{picks['z']}", flush=True)
    box.close()
print(f"\nfinal-decision on solved goals: n={tot} matched-top1 {m_top1}/{tot}  "
      f"zeroed-top1 {z_top1}/{tot}  pick-agreement {agree}/{tot}")
