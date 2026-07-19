"""stream chunk (R8): PATH-KEYED ADDITIVE multi-vector observations — the untried idea from the
original findings (synthetic-phase path-keyed readout: 100% vs 44% mean-pool content decode),
designed around WHY both earlier multi-vector streams lost to their single-vector controls:

  * mv_obs_k4 (0.5361 vs control 0.5386): segments were arbitrary line-INDEX strips — no stable
    identity, no key a later command could address; and it REPLACED the aggregate obs vector.
  * r7_role_multivec (0.5880 vs control 0.5885): role-canonical slots fixed alignment but the key
    was the ROLE (state/names/body/sig), still not the LOCATION a command queries; the aggregate
    was again replaced, so the arch had to re-learn its whole information path for ~0 headroom.

Two fixes, both structural:
  1. ADDITIVE, not substitutive: the step layout is [cmd_i, obs_i, p_i^1..p_i^K] (stride 2+K).
     Token obs_i is the untouched standardized single z_obs — so the baseline information path is
     preserved EXACTLY, and under the champion fastweights arch the delta-rule write semantics are
     unchanged (pending (cmd,obs) pairs are written and cleared at the FIRST obs-typed token,
     whose value is precisely the baseline z_obs; the following path tokens trigger no write).
     The path tokens are pure extra context flowing through the recurrent state / attention.
  2. PATH-KEYED: each p_i^k is the frozen-e5 embedding of a segment whose text BEGINS with the
     absolute path(s) it describes (per-directory-entry paths for ls, the file path for cat; see
     the companion perception recipe r8_pathkey_multivec). Commands are encoded as "passage: cat
     /etc/os-release", so segment keys live in the SAME lexical/embedding space commands query —
     the frozen retrieval-tuned encoder itself binds content to location.

FIXED across streams: the per-step TARGET is still the single standardized z_obs; eval space,
foils and all baselines unchanged — only what the predictor READS changes.

Causality: within a step the order is cmd_i, obs_i, then obs_i's path segments — every
obs-derived token sits strictly after its own command and strictly before cmd_{i+1}, so a causal
arch gives cmd_t access only to tokens of steps < t. leakage_ok corrupts the aggregate AND all K
path tokens of obs_3 and requires cmd predictions at steps <= 3 to be bit-stable.

Requires a data root whose caches carry per-step "z_obs_multi" [n,K,D] + "obs_valid" [n,K],
built by: uv run python -m evolve.mv_encode --perception r8_pathkey_multivec \
              --src data/dockerfs-e5 --out data/dockerfs-e5pk
(z_obs/z_cmd are copied verbatim from the source root; z_obs_multi is standardized with the OBS
stats by seq_worldmodel.apply_stats, keeping the target space untouched.)
"""

import torch

from realenv import seq_worldmodel as M

NAME = "r8_pathkey_entries"
DESCRIPTION = ("Additive path-keyed multi-vector stream: [cmd, obs, path-seg^1..K] per step "
               "(stride 2+K); aggregate obs token and target/eval space untouched; segments are "
               "absolute-path-keyed so command embeddings can address them.")

K = 4
STRIDE = 2 + K


def collate(batch, device):
    maxn = max(s["z_obs"].shape[0] for s in batch)
    L = STRIDE * maxn
    B = len(batch)
    tok = torch.zeros(B, L, M.D)
    types = torch.zeros(B, L, dtype=torch.long)   # {0=cmd, 1=obs} — archs have Embedding(2, d)
    key_pad = torch.ones(B, L, dtype=torch.bool)  # True = pad
    tgt = torch.zeros(B, maxn, M.D)               # FIXED single-vector target
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
    for bi, s in enumerate(batch):
        n = s["z_obs"].shape[0]
        zm, valid = s["z_obs_multi"], s["obs_valid"]
        assert zm.shape[1] == K, f"data root has K={zm.shape[1]}, stream expects K={K}"
        for i in range(n):
            base = STRIDE * i
            tok[bi, base] = s["z_cmd"][i]
            types[bi, base] = 0
            key_pad[bi, base] = False
            tok[bi, base + 1] = s["z_obs"][i]      # untouched aggregate obs (baseline info path)
            types[bi, base + 1] = 1
            key_pad[bi, base + 1] = False
            for k in range(K):                     # additive path-keyed segments
                if valid[i, k]:
                    tok[bi, base + 2 + k] = zm[i, k]
                    key_pad[bi, base + 2 + k] = False
                types[bi, base + 2 + k] = 1
            tgt[bi, i] = s["z_obs"][i]
            cmd_mask[bi, i] = True
    return {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
            "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::STRIDE]


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    """Mirrors seq_worldmodel.flatten_predictions (step order; prev/true from the single z_obs)."""
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
    """Corrupt the aggregate AND all K path segments of obs_3; cmd predictions at steps 0..3
    must not move (bit-stable causality, stream-aware)."""
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, M.D), "z_cmd": torch.randn(6, M.D),
            "z_obs_multi": torch.randn(6, K, M.D), "obs_valid": torch.ones(6, K, dtype=torch.bool),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::STRIDE].clone().cpu()
    b1 = collate(seq, device)
    base = STRIDE * 3
    b1["tok"][0, base + 1:base + 2 + K] = torch.randn(1 + K, M.D, device=device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::STRIDE].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
