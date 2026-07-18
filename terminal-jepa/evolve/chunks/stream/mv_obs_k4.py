"""stream chunk: STRUCTURED MULTI-VECTOR observations. Each step contributes 1 cmd token plus up
to K=4 observation tokens — a structured readout of the SAME observation (segment 0 = the
cwd/exit status header; segments 1..3 = contiguous strips of the output lines) instead of one
mean-pooled vector. The step layout is [cmd_i, obs_i^0, obs_i^1, obs_i^2, obs_i^3], stride 1+K.

The earlier readout findings (structured/path-keyed readout: 100% vs mean-pool 44.4% content
decode) motivated this axis; single-token poolings (cls/last) were NOT that idea and scored below
mean-pool. This impl finally tests the multi-vector version inside the evolve harness.

FIXED across streams (so fitness stays the honest content-verb margin): the per-step TARGET is
still the single standardized mean-pooled z_obs, the eval space, the foils, and all baselines are
unchanged — only what the predictor READS changes. Empty/absent segments are padded out via
key_pad (a cd with no output contributes only its header segment).

Requires a data root whose caches carry per-step "z_obs_multi" [n,K,D] and "obs_valid" [n,K]
(built by evolve/mv_encode.py). z_obs_multi is standardized with the OBS stats (see
seq_worldmodel.apply_stats), keeping the target space untouched.

Causality: identical stream order (cmd_i before its obs segments; steps in order), so a causal
arch mask gives cmd_t access only to segments of obs_<t. leakage_ok corrupts ALL K segments of
obs_3 and requires cmd predictions at steps <= 3 to be bit-stable.
"""

import torch

from realenv import seq_worldmodel as M

NAME = "mv_obs_k4"
DESCRIPTION = ("Multi-vector observation stream: [cmd, obs_seg0..obs_seg3] per step (stride 5); "
               "targets/eval/baselines unchanged (single-vector space).")

K = 4
STRIDE = 1 + K


def collate(batch, device):
    maxn = max(s["z_obs"].shape[0] for s in batch)
    L = STRIDE * maxn
    B = len(batch)
    tok = torch.zeros(B, L, M.D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)          # True = pad
    tgt = torch.zeros(B, maxn, M.D)
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
    for bi, s in enumerate(batch):
        n = s["z_obs"].shape[0]
        zm, valid = s["z_obs_multi"], s["obs_valid"]
        for i in range(n):
            base = STRIDE * i
            tok[bi, base] = s["z_cmd"][i]
            types[bi, base] = 0
            key_pad[bi, base] = False
            for k in range(K):
                if valid[i, k]:
                    tok[bi, base + 1 + k] = zm[i, k]
                    key_pad[bi, base + 1 + k] = False
                types[bi, base + 1 + k] = 1
            tgt[bi, i] = s["z_obs"][i]
            cmd_mask[bi, i] = True
    return {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
            "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::STRIDE]


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    """Mirrors seq_worldmodel.flatten_predictions (step order, prev/true from single z_obs)."""
    net.eval()
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, h = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = pred[:, 0::STRIDE].cpu()
        cmd_h = h[:, 0::STRIDE].cpu()
        for bi, s in enumerate(chunk):
            n = s["z_obs"].shape[0]
            for t in range(n):
                preds.append(cmd_pred[bi, t]); hids.append(cmd_h[bi, t])
                trues.append(s["z_obs"][t])
                prevs.append(s["z_obs"][t - 1] if t > 0 else torch.zeros(M.D))
                cmds.append(s["cmds"][t]); imgs.append(s["image"])
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs,
            "verbs": [M.verb_of(c) for c in cmds]}


@torch.no_grad()
def leakage_ok(net, device):
    """Corrupt ALL K segments of obs_3; cmd predictions at steps 0..3 must not move."""
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, M.D), "z_cmd": torch.randn(6, M.D),
            "z_obs_multi": torch.randn(6, K, M.D), "obs_valid": torch.ones(6, K, dtype=torch.bool),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::STRIDE].clone().cpu()
    b1 = collate(seq, device)
    base = STRIDE * 3
    b1["tok"][0, base + 1:base + 1 + K] = torch.randn(K, M.D, device=device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::STRIDE].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
