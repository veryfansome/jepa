"""3-seed robustness for the decision-0 score-mode comparison (each seed uses its own
checkpoint AND its own replicated opener RNG stream, as in the actual runs)."""
import json
import pathlib
import sys

REPO = pathlib.Path("/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
sys.path.insert(0, str(REPO))
import torch

from realenv import seq_worldmodel as M
from realenv.docker_env import DockerBox
from realenv.collect_docker import UNAME_OPTS, CONFIG_FILES
from realenv.plan_env import Enc, parse_child_dirs, imagine_candidate, LS_PROBE
from evolve import genome as G

SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad")
DEVICE = M.pick_device()


def cos(a, b):
    return float((a[0] / a[0].norm().clamp_min(1e-8)) @ (b[0] / b[0].norm().clamp_min(1e-8)))


def main():
    enc = Enc(str(REPO / "data/dockerfs-e5"), DEVICE)
    gen = json.load(open(SCRATCH / "r9/genomes/r9-arch-chunked-codex.json"))
    target_mod = G.load_target(gen)
    build, ap_ = G.load_arch(gen)
    nets = {}
    for s in (0, 1, 2):
        net = build(**ap_)
        ck = torch.load(SCRATCH / f"r9/plan/pod/ckpts/{gen['id']}.s{s}.pt", weights_only=False)
        net.load_state_dict(ck["state_dict"])
        nets[s] = net.to(DEVICE).eval()

    hits = {s: {m: 0 for m in ("cos_min", "cos_ls", "cos_cd", "cos_sum", "lexical")} for s in (0, 1, 2)}
    ntot = 0
    for image in ("fedora:latest", "mariadb:latest"):
        tag = image.replace(":", "_").replace("/", "_")
        goals = [json.loads(l) for l in open(REPO / f"data/plangoals-v1/goals-{tag}.jsonl")]
        box = DockerBox(image)
        box.run("cd /")
        st_ls_root = box.run(LS_PROBE)
        cands = parse_child_dirs(st_ls_root["output"], "/")
        z_cds = [enc.cmd(f"cd {c}") for c in cands]
        z_ls_cmd = enc.cmd(LS_PROBE)
        z_root_ls = enc.obs(st_ls_root)
        # per-seed opener streams
        rngs = {s: torch.Generator().manual_seed(1000 * s + 7) for s in (0, 1, 2)}
        opener_cache = {}
        for gi, goal in enumerate(goals):
            z_goal = enc.obs(goal["step"])
            gn = z_goal[0] / z_goal[0].norm().clamp_min(1e-8)
            correct = "/" + goal["dir"].split("/")[1]
            sims = torch.tensor([cos(z, z_goal) for z in z_cds])
            for s in (0, 1, 2):
                rng = rngs[s]
                u = f"uname {UNAME_OPTS[int(torch.randint(0, len(UNAME_OPTS), (1,), generator=rng))]}".strip()
                cf = f"cat {CONFIG_FILES[int(torch.randint(0, len(CONFIG_FILES), (1,), generator=rng))]}"
                hist = []
                for cmd in (u, cf):
                    if cmd not in opener_cache:
                        st = box.run(cmd)
                        box.run("cd /")
                        opener_cache[cmd] = (enc.cmd(cmd), enc.obs(st))
                    hist.append(opener_cache[cmd])
                hist.append((enc.cmd(LS_PROBE), z_root_ls))
                recs = [imagine_candidate(nets[s], target_mod, hist, z_cds[i], z_ls_cmd, "write", DEVICE)
                        for i in range(len(cands))]
                d_cd = torch.tensor([1 - cos(r[0], z_goal) for r in recs])
                d_ls = torch.tensor([1 - cos(r[1], z_goal) for r in recs])
                picks = {"cos_min": int(torch.minimum(d_cd, d_ls).argmin()),
                         "cos_ls": int(d_ls.argmin()),
                         "cos_cd": int(d_cd.argmin()),
                         "cos_sum": int((d_cd + d_ls).argmin()),
                         "lexical": int(sims.argmax())}
                for m, p in picks.items():
                    hits[s][m] += (cands[p] == correct)
            ntot += 1
            if (gi + 1) % 25 == 0:
                print(f"{image} {gi+1}/{len(goals)}", flush=True)
        box.close()
    n = ntot  # per-seed goal count = 100
    print("\nper-seed decision-0 correct-first-move rate (n=100 each):")
    for m in ("cos_min", "cos_ls", "cos_cd", "cos_sum", "lexical"):
        vals = [hits[s][m] / 100 for s in (0, 1, 2)]
        print(f"  {m:8s} {vals} mean {sum(vals)/3:.3f}")


if __name__ == "__main__":
    main()
