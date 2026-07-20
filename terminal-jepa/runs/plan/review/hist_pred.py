"""Within-net history test (M4/M5 reconciliation): champion prediction quality at cmd
positions under (a) real history vs (b) same cmds with ALL past obs zeroed.
If (a)==(b), the masked-twin 'history gap' is an arch artifact; if (a)>>(b), history
matters for prediction and the stage-2 finding is specific to goal-distance ranking."""
import json, sys
sys.path.insert(0, ".")
import torch
from realenv import seq_worldmodel as M
from evolve import genome as G

CK = "/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/plan/pod/ckpts"
device = M.pick_device()
gen = json.load(open("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/genomes/r9-arch-chunked-codex.json"))
build, ap = G.load_arch(gen)
net = build(**ap)
net.load_state_dict(torch.load(f"{CK}/r9-arch-chunked-codex.s0.pt", weights_only=False)["state_dict"])
net = net.to(device).eval()
train = M.cached_encode("data/dockerfs-e5", "train", "x", device)
val = M.cached_encode("data/dockerfs-e5", "val", "x", device)
mo, so, mc, sc = M.standardize_stats(train)
M.apply_stats(val, mo, so, mc, sc)
inner = [s for s in val if s["image"] in ("fedora:latest", "mariadb:latest")][:120]

cos_a, cos_b, r_a, r_b = [], [], [], []
with torch.no_grad():
    for s in inner:
        n = s["z_obs"].shape[0]
        b = M.collate([s], device)
        pa, _ = net(b["tok"], b["types"], b["key_pad"])
        s0 = {"z_cmd": s["z_cmd"], "z_obs": torch.zeros_like(s["z_obs"])}
        b0 = M.collate([s0], device)
        pb, _ = net(b0["tok"], b0["types"], b0["key_pad"])
        Pa = pa[:, 0::2][0, 2:n].cpu(); Pb = pb[:, 0::2][0, 2:n].cpu()
        T = s["z_obs"][2:n]
        cos_a += torch.nn.functional.cosine_similarity(Pa, T, 1).tolist()
        cos_b += torch.nn.functional.cosine_similarity(Pb, T, 1).tolist()
        # in-seq retrieval: rank of true obs among this seq's obs
        for i in range(T.shape[0]):
            da = ((Pa[i:i+1] - s["z_obs"]) ** 2).sum(1)
            db = ((Pb[i:i+1] - s["z_obs"]) ** 2).sum(1)
            r_a.append(int(da.argmin()) == i + 2); r_b.append(int(db.argmin()) == i + 2)
import statistics as st
print(f"n positions {len(cos_a)}")
print(f"cos(pred,true): real-history {st.mean(cos_a):.3f}  zeroed-obs-history {st.mean(cos_b):.3f}")
print(f"in-seq top1:    real-history {st.mean(r_a):.3f}  zeroed-obs-history {st.mean(r_b):.3f}")
