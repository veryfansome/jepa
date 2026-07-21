"""R11 path battery: Docker-free MULTI-STEP planning instrument. Battery-v1 (calib_bench)
measures only decision-0; the Stage-2 adversarial review located the planning losses at
LATER decisions. This instrument walks each goal's oracle path and measures on-path
accuracy at every depth, in two modes:

  path_acc_real — teacher-forced: real history up to level k, imagine horizon-2 per
                  candidate, rank by sum-of-cosine to goal. Isolates per-decision choice.
  path_acc_imag — self-imagined: the history's cd/ls observations along the oracle path
                  are the model's OWN sequential imaginations (write policy). The direct
                  measure of rollout compounding — the literature's exposure-bias axis.

The battery stores TEXT (opener steps, per-directory ls output, candidate names), so it is
re-encodable under any perception recipe (--percep) and any data root's stats — perception
candidates are scorable in their own obs space. Encodings are cached per (root, percep).

  uv run python -m evolve.path_battery build --goals data/plangoals-v1 \
      --images fedora:latest,mariadb:latest --out data/plangoals-v1/path-battery-v1.json
  uv run python -m evolve.path_battery eval --battery data/plangoals-v1/path-battery-v1.json \
      --genome g.json --ckpt ckpts/champ.s0.pt --data data/dockerfs-e5
"""

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from realenv import seq_worldmodel as M
from evolve import genome as G

D = M.D
OPENER = ["uname -a", "cat /etc/os-release"]
LS_PROBE = "ls -la"


def _prefixes(path):
    """'/usr/share/locale/lt' -> ['/', '/usr', '/usr/share', '/usr/share/locale', ...full]."""
    parts = [p for p in path.split("/") if p]
    out = ["/"]
    for i in range(len(parts)):
        out.append("/" + "/".join(parts[: i + 1]))
    return out


def build(args):
    from realenv.docker_env import DockerBox
    from realenv.plan_env import parse_child_dirs
    images = {}
    for image in args.images.split(","):
        tag = image.replace(":", "_").replace("/", "_")
        goals = [json.loads(l) for l in open(pathlib.Path(args.goals) / f"goals-{tag}.jsonl")]
        box = DockerBox(image)
        box.run("cd /")
        opener_steps = [box.run(c) for c in OPENER]
        need = set()
        for g in goals:
            need.update(_prefixes(g["dir"])[:-1])   # every cwd a decision is made at
        nodes = {}
        for cwd in sorted(need):
            r_cd = box.run(f"cd {cwd}")
            if r_cd["exit"] != 0:
                continue
            r_ls = box.run(LS_PROBE)
            box.run("cd /")
            names = [c.rsplit("/", 1)[1] for c in parse_child_dirs(r_ls["output"], cwd)]
            nodes[cwd] = {"ls_step": r_ls, "cands": names}
        box.close()
        # keep only goals whose full decision chain is intact and on-path child visible
        kept = []
        for g in goals:
            pref = _prefixes(g["dir"])
            ok = all(p in nodes for p in pref[:-1])
            ok = ok and all(pref[i + 1].rsplit("/", 1)[1] in nodes[pref[i]]["cands"]
                            for i in range(len(pref) - 1))
            if ok:
                kept.append({"dir": g["dir"], "depth": g["depth"], "path": pref})
        images[image] = {"opener": opener_steps, "nodes": nodes, "goals": kept}
        print(f"{image}: {len(kept)}/{len(goals)} goals intact, {len(nodes)} nodes", flush=True)
    out = pathlib.Path(args.out)
    out.write_text(json.dumps({"version": "path-battery-v1", "opener_cmds": OPENER,
                               "probe": LS_PROBE, "images": images}, indent=1))
    print(f"wrote {out} sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")


class TextEnc:
    """Encode battery text under a perception recipe + a data root's standardization stats,
    with a per-(root,percep) disk cache so repeated evals are cheap."""

    def __init__(self, data_root, percep_name, device):
        import importlib
        self.percep = importlib.import_module(f"evolve.chunks.perception.{percep_name}")
        self.device = device
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.percep.MODEL)
        self.model = AutoModel.from_pretrained(self.percep.MODEL).to(device).eval()
        train = M.cached_encode(data_root, "train", "answerdotai/ModernBERT-base", device)
        self.mo, self.so, self.mc, self.sc = M.standardize_stats(train)

    @torch.no_grad()
    def _embed(self, texts, bs=96):
        out = torch.zeros(len(texts), D)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for i in range(0, len(order), bs):
            bidx = order[i:i + bs]
            e = self.tok([texts[j] for j in bidx], return_tensors="pt", padding=True,
                         truncation=True, max_length=256)
            e = {k: v.to(self.device) for k, v in e.items()}
            h = self.model(**e).last_hidden_state
            pooled = self.percep.pool(h, e["attention_mask"]).float().cpu()
            for k, j in enumerate(bidx):
                out[j] = pooled[k]
        return out

    def obs(self, steps):
        z = self._embed([self.percep.render_obs(s) for s in steps])
        return (z - self.mo) / self.so

    def cmd(self, texts):
        z = self._embed([self.percep.render_cmd({"cmd": t}) for t in texts])
        return (z - self.mc) / self.sc


def _cd_step(target):
    return {"cmd": f"cd {target}", "output": "", "exit": 0, "cwd": target}


def encode_battery(battery, data_root, percep_name, device, cache_dir="data/plangoals-v1"):
    """One encoded bundle per (root, percep): all obs/cmd embeddings the eval needs."""
    key = f"pbenc-{pathlib.Path(data_root).name}-{percep_name}.pt"
    cache = pathlib.Path(cache_dir) / key
    if cache.exists():
        return torch.load(cache, weights_only=False)
    enc = TextEnc(data_root, percep_name, device)
    bundle = {"images": {}}
    for image, im in battery["images"].items():
        nodes = im["nodes"]
        cwds = sorted(nodes)
        z_ls_obs = enc.obs([nodes[c]["ls_step"] for c in cwds])
        z_opener_obs = enc.obs(im["opener"])
        cmd_texts, cmd_index = [], {}
        for t in battery["opener_cmds"] + [battery["probe"]]:
            cmd_index[t] = len(cmd_texts); cmd_texts.append(t)
        for c in cwds:
            for name in nodes[c]["cands"]:
                t = f"cd {(c.rstrip('/') + '/' + name) if c != '/' else '/' + name}"
                if t not in cmd_index:
                    cmd_index[t] = len(cmd_texts); cmd_texts.append(t)
        z_cmds = enc.cmd(cmd_texts)
        cd_obs_texts, cd_obs_index = [], {}
        for c in cwds:
            for name in nodes[c]["cands"]:
                tgt = (c.rstrip('/') + '/' + name) if c != '/' else '/' + name
                if tgt not in cd_obs_index:
                    cd_obs_index[tgt] = len(cd_obs_texts); cd_obs_texts.append(tgt)
        z_cd_obs = enc.obs([_cd_step(t) for t in cd_obs_texts])
        bundle["images"][image] = {"cwds": cwds, "z_ls_obs": z_ls_obs,
                                   "z_opener_obs": z_opener_obs, "cmd_index": cmd_index,
                                   "z_cmds": z_cmds, "cd_obs_index": cd_obs_index,
                                   "z_cd_obs": z_cd_obs}
        # goal embedding = the ls view AT the goal dir (the plangoals oracle view is the same
        # probe; the battery's own node ls at the goal dir is not stored — use plangoals)
    torch.save(bundle, cache)
    return bundle


def imagine_candidates_batched(net, target_mod, hist, z_cds, z_ls_cmd, device):
    """Vectorized horizon-2 write-policy imagination for K candidates sharing one history:
    2 batched forwards instead of 2K sequential ones. Identical math to plan_env.
    imagine_candidate (all K streams have equal length — no padding asymmetry).
    Returns (rec_cd [K,D], rec_ls [K,D])."""
    K = z_cds.shape[0]
    n = len(hist)
    hc = torch.cat([h[0] for h in hist])                      # [n,D]
    ho = torch.cat([h[1] if h[1] is not None else torch.zeros(1, D) for h in hist])
    prev_obs = hist[-1][1] if hist and hist[-1][1] is not None else torch.zeros(1, D)
    # pass 1: history + candidate cd (pending obs)
    seqs = [{"z_cmd": torch.cat([hc, z_cds[k:k + 1]]),
             "z_obs": torch.cat([ho, torch.zeros(1, D)])} for k in range(K)]
    b = M.collate(seqs, device)
    with torch.no_grad():
        pred, _ = net(b["tok"], b["types"], b["key_pad"])
    rec_cd = target_mod.to_obs(pred[:, 0::2][:, n].cpu(), prev_obs.expand(K, D))
    # pass 2: write-imagined cd obs + ls probe (pending)
    seqs2 = [{"z_cmd": torch.cat([hc, z_cds[k:k + 1], z_ls_cmd]),
              "z_obs": torch.cat([ho, rec_cd[k:k + 1], torch.zeros(1, D)])} for k in range(K)]
    b2 = M.collate(seqs2, device)
    with torch.no_grad():
        pred2, _ = net(b2["tok"], b2["types"], b2["key_pad"])
    rec_ls = target_mod.to_obs(pred2[:, 0::2][:, n + 1].cpu(), rec_cd)
    return rec_cd, rec_ls


def eval_battery(net, target_mod, battery, bundle, goal_embs, device, mode="real"):
    """Returns on-path top-1 aggregated over all (goal, level) decisions."""
    hits, n, by_rem = 0, 0, {}
    for image, im in battery["images"].items():
        b = bundle["images"][image]
        nodes = im["nodes"]
        cwd_i = {c: i for i, c in enumerate(b["cwds"])}
        for g in im["goals"]:
            if (image, g["dir"]) not in goal_embs:
                continue
            pref = g["path"]
            depth = len(pref) - 1
            # history: opener + ls@/
            hist = [(b["z_cmds"][b["cmd_index"][battery["opener_cmds"][0]]].unsqueeze(0), b["z_opener_obs"][0:1]),
                    (b["z_cmds"][b["cmd_index"][battery["opener_cmds"][1]]].unsqueeze(0), b["z_opener_obs"][1:2]),
                    (b["z_cmds"][b["cmd_index"][battery["probe"]]].unsqueeze(0), b["z_ls_obs"][cwd_i["/"]:cwd_i["/"] + 1])]
            z_goal = goal_embs[(image, g["dir"])]
            gn = z_goal[0] / z_goal[0].norm().clamp_min(1e-8)
            for k in range(depth):
                cwd = pref[k]
                nxt = pref[k + 1].rsplit("/", 1)[1]
                cands = nodes[cwd]["cands"]
                on_path_idx = cands.index(nxt)
                z_ls_cmd = b["z_cmds"][b["cmd_index"][battery["probe"]]].unsqueeze(0)
                tgts = [(cwd.rstrip('/') + '/' + name) if cwd != '/' else '/' + name
                        for name in cands]
                z_cds = torch.stack([b["z_cmds"][b["cmd_index"][f"cd {t}"]] for t in tgts])
                rec_cd, rec_ls = imagine_candidates_batched(net, target_mod, hist, z_cds,
                                                            z_ls_cmd, device)
                d_cd = 1 - (rec_cd / rec_cd.norm(dim=1, keepdim=True).clamp_min(1e-8)) @ gn
                d_ls = 1 - (rec_ls / rec_ls.norm(dim=1, keepdim=True).clamp_min(1e-8)) @ gn
                pick = int((d_cd + d_ls).argmin())
                hit = int(pick == on_path_idx)
                hits += hit; n += 1
                rem = depth - k
                by_rem.setdefault(rem, [0, 0])
                by_rem[rem][0] += hit; by_rem[rem][1] += 1
                # advance history ALONG THE ORACLE PATH (teacher forcing of the walk)
                tgt_next = pref[k + 1]
                z_cd_next = b["z_cmds"][b["cmd_index"][f"cd {tgt_next}"]].unsqueeze(0)
                if mode == "real":
                    cd_obs = b["z_cd_obs"][b["cd_obs_index"][tgt_next]].unsqueeze(0)
                    ls_obs = b["z_ls_obs"][cwd_i[tgt_next]].unsqueeze(0) if tgt_next in cwd_i else None
                else:  # imagined: the model's own write-policy imagination becomes history
                    i_cd, i_ls = imagine_candidates_batched(net, target_mod, hist,
                                                            z_cd_next, z_ls_cmd, device)
                    cd_obs, ls_obs = i_cd[0:1], i_ls[0:1]
                hist = hist + [(z_cd_next, cd_obs)]
                if ls_obs is not None:
                    hist = hist + [(z_ls_cmd, ls_obs)]
    return {"path_acc": round(hits / max(n, 1), 4), "n_decisions": n,
            "by_remaining_depth": {str(k): round(v[0] / v[1], 3)
                                   for k, v in sorted(by_rem.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    bld = sub.add_parser("build")
    bld.add_argument("--goals", required=True)
    bld.add_argument("--images", required=True)
    bld.add_argument("--out", required=True)
    bld.set_defaults(fn=build)
    ev = sub.add_parser("eval")
    ev.add_argument("--battery", required=True)
    ev.add_argument("--goals", default="data/plangoals-v1")
    ev.add_argument("--genome", required=True)
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--data", default="data/dockerfs-e5")
    ev.add_argument("--percep", default="enc_e5_base")
    ev.add_argument("--out", default=None)

    def do_eval(args):
        device = M.pick_device()
        battery = json.loads(pathlib.Path(args.battery).read_text())
        bundle = encode_battery(battery, args.data, args.percep, device)
        # goal embeddings: the plangoals oracle ls views, encoded under the same recipe/stats
        enc = TextEnc(args.data, args.percep, device)
        goal_embs = {}
        for image, im in battery["images"].items():
            tag = image.replace(":", "_").replace("/", "_")
            for line in open(pathlib.Path(args.goals) / f"goals-{tag}.jsonl"):
                gr = json.loads(line)
                goal_embs[(image, gr["dir"])] = enc.obs([gr["step"]])
        gen = json.load(open(args.genome))
        ck = torch.load(args.ckpt, weights_only=False)
        build_fn, ap_ = G.load_arch(gen)
        net = build_fn(**ap_)
        net.load_state_dict(ck["state_dict"])
        net = net.to(device).eval()
        tmod = G.load_target(gen)
        res = {"ckpt": args.ckpt}
        for mode in ("real", "imag"):
            res[mode] = eval_battery(net, tmod, battery, bundle, goal_embs, device, mode)
        res["compounding_gap"] = round(res["real"]["path_acc"] - res["imag"]["path_acc"], 4)
        print(json.dumps(res, indent=1))
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
        return res

    ev.set_defaults(fn=do_eval)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
