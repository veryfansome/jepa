"""Checkpoint-based checks:
(A) M2 re-measurement: in-dist pred-vs-true sqL2 vs random-pair sqL2, on inner-val images,
    champion s0; also cosine variants and whether cos vs L2 changes candidate ranking.
(B) _forward_pred_at_last_cmd indexing: causality probes (pending-obs zeros must not move
    the last-cmd pred; the last cmd must; appending a step must not change earlier preds).
(C) 34-token (17-step) sequence through champion + masked twin: finite? crash?
(D) masked twin: pred at last cmd invariant to entire history (self-only attention).
"""
import json, sys
sys.path.insert(0, ".")
import torch
from realenv import seq_worldmodel as M
from realenv.plan_env import _forward_pred_at_last_cmd, D
from evolve import genome as G

CK = "/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/plan/pod/ckpts"
device = M.pick_device()
gen = json.load(open("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/genomes/r9-arch-chunked-codex.json"))
build, ap = G.load_arch(gen)
net = build(**ap)
ck = torch.load(f"{CK}/r9-arch-chunked-codex.s0.pt", weights_only=False)
net.load_state_dict(ck["state_dict"]); net = net.to(device).eval()
mnet = M.SeqWorldModel("jepa", 0, no_history=True)
mnet.load_state_dict(torch.load(f"{CK}/masked.s0.pt", weights_only=False)["state_dict"])
mnet = mnet.to(device).eval()

train = M.cached_encode("data/dockerfs-e5", "train", "x", device)
val = M.cached_encode("data/dockerfs-e5", "val", "x", device)
mo, so, mc, sc = M.standardize_stats(train)
M.apply_stats(val, mo, so, mc, sc)
inner = [s for s in val if s["image"] in ("fedora:latest", "mariadb:latest")]
print("inner seqs:", len(inner), "steps:", inner[0]["z_obs"].shape)

# ---- (A) M2 ----
torch.manual_seed(0)
sub = inner[:120]
preds, trues = [], []
with torch.no_grad():
    for s in sub:
        b = M.collate([s], device)
        p, _ = net(b["tok"], b["types"], b["key_pad"])
        preds.append(p[:, 0::2][0].cpu())
        trues.append(s["z_obs"])
P = torch.cat(preds); T = torch.cat(trues)
sq_true = ((P - T) ** 2).sum(1)
perm = torch.randperm(T.shape[0])
sq_rand = ((P - T[perm]) ** 2).sum(1)
cos = torch.nn.functional.cosine_similarity(P, T, 1)
cos_rand = torch.nn.functional.cosine_similarity(P, T[perm], 1)
print(f"(A) n={P.shape[0]}  pred-vs-true sqL2 mean {sq_true.mean():.0f}  random-pair {sq_rand.mean():.0f}")
print(f"    cos true {cos.mean():.3f}  cos rand {cos_rand.mean():.3f}")
print(f"    ||pred|| mean {P.norm(dim=1).mean():.1f}  ||true|| mean {T.norm(dim=1).mean():.1f}")
# does L2 vs cos flip rankings? compare candidate-style: for each pred, rank 32 candidate trues
# true-true random pair distance (the likely referent of 'random-pair ~1478')
perm2 = torch.randperm(T.shape[0])
sq_tt = ((T - T[perm2]) ** 2).sum(1)
print(f"    true-vs-true random-pair sqL2 mean {sq_tt.mean():.0f}")
# seq-length histogram, train + val
from collections import Counter
print("    val step-counts:", dict(Counter(s['z_obs'].shape[0] for s in val)))
print("    train step-counts:", dict(Counter(s['z_obs'].shape[0] for s in train)))
flips = 0; tot = 0
for i in range(0, P.shape[0] - 1, 40):
    cands = T[perm[:32]]
    dl2 = ((P[i:i+1] - cands) ** 2).sum(1)
    dcs = 1 - torch.nn.functional.cosine_similarity(P[i:i+1], cands, 1)
    flips += int(dl2.argmin() != dcs.argmin()); tot += 1
print(f"    argmin(L2) != argmin(cos) on 32-candidate sets: {flips}/{tot}")

# ---- (B) indexing / causality ----
s = inner[0]
n = 8
hist = [(s["z_cmd"][i:i+1], s["z_obs"][i:i+1]) for i in range(n - 1)] + [(s["z_cmd"][n-1:n], None)]
p1 = _forward_pred_at_last_cmd(net, hist, device)
# pending obs slot filled with junk instead of zeros: must not change last-cmd pred (causal)
hist_junk = hist[:-1] + [(s["z_cmd"][n-1:n], torch.randn(1, D) * 5)]
p2 = _forward_pred_at_last_cmd(net, hist_junk, device)
# different last cmd: must change
hist_cmd = hist[:-1] + [(s["z_cmd"][0:1], None)]
p3 = _forward_pred_at_last_cmd(net, hist_cmd, device)
# append one more step: earlier pred must be unchanged (causal) -> recompute at pos n-1
hist_app = hist[:-1] + [(s["z_cmd"][n-1:n], s["z_obs"][n-1:n]), (s["z_cmd"][n:n+1], None)]
seq = {"z_cmd": torch.cat([h[0] for h in hist_app]),
       "z_obs": torch.cat([h[1] if h[1] is not None else torch.zeros(1, D) for h in hist_app])}
b = M.collate([seq], device)
with torch.no_grad():
    pf, _ = net(b["tok"], b["types"], b["key_pad"])
p_at_n1 = pf[:, 0::2][0, n - 1].cpu().unsqueeze(0)
print(f"(B) pending-obs junk delta {(p1-p2).abs().max():.2e} (want ~0)  "
      f"cmd-swap delta {(p1-p3).abs().max():.2e} (want >0)  "
      f"append-step delta {(p1-p_at_n1).abs().max():.2e} (want ~0)")

# ---- (C) 17-step / 34-token ----
s2 = inner[1]
hist17 = [(s["z_cmd"][i:i+1], s["z_obs"][i:i+1]) for i in range(16)] + [(s2["z_cmd"][0:1], None)]
p17 = _forward_pred_at_last_cmd(net, hist17, device)
pm17 = _forward_pred_at_last_cmd(mnet, hist17, device)
print(f"(C) 17-step champion pred finite: {bool(torch.isfinite(p17).all())} norm {p17.norm():.1f}; "
      f"masked finite: {bool(torch.isfinite(pm17).all())} norm {pm17.norm():.1f}")
# error at last cmd for 16-step (in-horizon) across sub, vs a 17-step variant (same last step)
# pred-norm spread across steps within a sequence (the 'does not cancel' driver)
with torch.no_grad():
    norm_stds = []
    for s in sub[:40]:
        b = M.collate([s], device)
        p, _ = net(b["tok"], b["types"], b["key_pad"])
        norm_stds.append(float(p[:, 0::2][0].norm(dim=1).std()))
import statistics as st
print(f"    within-seq std of ||pred|| across steps: mean {st.mean(norm_stds):.1f} (mean ||pred|| 57)")
errs16, errs17 = [], []
sub16 = [s for s in sub if s["z_obs"].shape[0] >= 16]
with torch.no_grad():
    for s in sub16[:40]:
        h16 = [(s["z_cmd"][i:i+1], s["z_obs"][i:i+1]) for i in range(15)] + [(s["z_cmd"][15:16], None)]
        e16 = ((_forward_pred_at_last_cmd(net, h16, device) - s["z_obs"][15:16]) ** 2).sum()
        pre = [(sub[3]["z_cmd"][0:1], sub[3]["z_obs"][0:1])]  # prepend 1 foreign step -> 17 steps
        e17 = ((_forward_pred_at_last_cmd(net, pre + h16, device) - s["z_obs"][15:16]) ** 2).sum()
        errs16.append(float(e16)); errs17.append(float(e17))
import statistics
print(f"    last-step sqL2: 16-step {statistics.mean(errs16):.0f}  17-step(prepended) {statistics.mean(errs17):.0f}")

# ---- (D) masked twin history-invariance ----
hist_a = [(s["z_cmd"][i:i+1], s["z_obs"][i:i+1]) for i in range(5)] + [(s["z_cmd"][5:6], None)]
hist_b = [(torch.randn(1, D), torch.randn(1, D)) for _ in range(5)] + [(s["z_cmd"][5:6], None)]
da = _forward_pred_at_last_cmd(mnet, hist_a, device)
db = _forward_pred_at_last_cmd(mnet, hist_b, device)
print(f"(D) masked twin pred delta under total history swap: {(da-db).abs().max():.2e} (want ~0)")
