"""head chunk: MULTI-STEP ROLLOUT CHAIN consistency + imagined-context InfoNCE (rollout-v2).

Evolves r10_rollout_write_consistency_aux (proxy fitness 0.6124, dec0 first-move 0.23 vs
0.03) into the form the R11 path battery actually measures. The battery (path_battery.py)
imagines a HORIZON-2 write-policy chain per candidate and ranks candidates by SUM OF
COSINE to the goal; the Stage-2 review located the binding constraint at PER-DECISION
DISCRIMINATION (worst at remaining-depth 1, fan-outs up to 237), not raw fidelity — the
ground-truth e5 obs field already ranks the on-path child first on 64/100 goals, so what
is missing is imagined observations that land close enough IN COSINE to inherit that
field's discrimination. Three upgrades over R10, each aimed at one measured gap:

  1. CHAINED 2-STEP IMAGINATION (multi-step rollout training, arXiv:2512.24497;
     scheduled sampling, arXiv:1506.03099): instead of independent scattered one-step
     writes, each sampled row rolls a length-H chain (H schedules 1 -> 2): pass 0
     predicts obs_s under real context and WRITES it; pass j (1..H) predicts obs_{s+j}
     under the accumulated imagined context, is penalized against the TRUE obs_{s+j},
     and (while j < H) writes its own prediction for the next pass. Gradients flow
     through every write (backprop-through-rollout): the producer is trained so its
     step-1 output still supports a correct step-2 prediction — exactly the battery's
     horizon-2 operating condition, which R10's one-step writes never exercised.
  2. COSINE PENALTY alongside MSE: the battery ranks by cosine, and MSE alone spends
     gradient on norm calibration that cosine ranking ignores. L_cos = 1 - cos(pred',
     true) trains the DIRECTION the instrument scores; the MSE term is kept (smaller)
     because Stage-2 showed the row-softmax objective leaves predictions grossly
     off-manifold in norm and the on-manifold anchor demonstrably does not hurt margin.
  3. IMAGINED-CONTEXT InfoNCE: at each supervised chain position, the prediction made
     UNDER IMAGINED CONTEXT must rank its true next-obs above every other valid true
     obs in the row-subsample (cosine logits / tau, cross-entropy to the true index).
     With the sysblock batcher the in-batch pool contains same-system sibling
     listings — precisely the foils the depth-1 decision must separate. This supplies
     the per-decision discrimination gradient under the planning distribution that
     neither the (teacher-forced) main contrastive loss nor R10's pure-MSE aux gives.

Schedules (DAgger-style, state carried in head_state): aux weight warms up linearly
over `warmup` calls (teacher forcing dominates early, so the chain feeds back garbage
only briefly), and the chain deepens 1 -> 2 after `h2_after` calls (curriculum on
rollout depth). Defaults are sized for the 1000-step proxy (10%/10%).

Why fitness (G1) survives: forward untouched; main loss untouched on the full batch
every step; the aux touches a row-subsample only, its total weight (0.15+0.15+0.1 after
warmup) is in the regime the champion already tolerates (it carries a 0.05 MSE anchor
harmlessly and R10's 0.25 aux GAINED margin at proxy); MSE/cosine to varying true
targets and InfoNCE over varying pools are all collapse-averse.

Contract / safety (kept from R10):
  * wrap() adds NO modules and does NOT re-point net.forward (no module cycle, no
    forward recursion; the leakage probe runs the arch's own forward). head_state is a
    plain mutable dict (config + call counter).
  * Strictly causal: every written token is the net's causal prediction at cmd position
    2p (a function of tokens <= 2p — earlier writes in the chain are themselves causal)
    placed at obs position 2p+1 > 2p. True future obs enter only as loss LABELS /
    InfoNCE pool rows, never as inputs.
  * Hard-zero fallbacks: non-interleave layout, degenerate batch, no eligible chain
    start, non-finite loss -> 0.0. Guarded gathers; no in-place ops on graph tensors.
"""

import math

import torch
import torch.nn.functional as F

NAME = "r11_rollout_chain_cosnce_aux"
DESCRIPTION = (
    "Rollout-consistency v2: chained horizon-2 write-imagined rollouts (depth curriculum "
    "1->2, weight warmup) trained with backprop-through-rollout; penalties in cosine (the "
    "battery's metric) + MSE, plus an imagined-context InfoNCE row that makes each "
    "post-feedback prediction rank its true next-obs above in-batch foils — per-decision "
    "discrimination under the planning distribution. Forward untouched, aux train-only."
)

_ROW_FRAC = 0.5     # fraction of batch rows the aux touches
_W_MSE = 0.15       # on-manifold anchor (norm calibration)
_W_COS = 0.15       # the battery's ranking geometry
_W_NCE = 0.10       # imagined-context discrimination
_TAU = 0.2          # InfoNCE temperature (cosine logits)
_WARMUP = 100       # calls to ramp the aux weight 0 -> 1 (10% of the 1000-step proxy)
_H2_AFTER = 100     # calls before the chain deepens from 1 to 2 imagined steps
_MAX_H = 2          # matches the battery's horizon-2 imagination


def wrap(net, D, row_frac=_ROW_FRAC, w_mse=_W_MSE, w_cos=_W_COS, w_nce=_W_NCE,
         tau=_TAU, warmup=_WARMUP, h2_after=_H2_AFTER, max_h=_MAX_H):
    """No modules, no forward re-pointing: the arch's readout is untouched (composes
    with any arch). Returns a mutable config dict (with a call counter for the
    schedules) as head_state."""
    return {"frac": float(row_frac), "w_mse": float(w_mse), "w_cos": float(w_cos),
            "w_nce": float(w_nce), "tau": float(tau), "warmup": int(warmup),
            "h2_after": int(h2_after), "max_h": int(max_h), "D": int(D), "calls": 0}


def _interleave_layout_ok(b):
    """The write-back indexing assumes the single-vector interleave: tok[2t]=cmd_t,
    tok[2t+1]=obs_t, tgt[:,t]=z_obs_t. Verify cheaply; other streams -> aux off."""
    tok, types, tgt = b["tok"], b["types"], b["tgt"]
    if tok.dim() != 3 or tgt.dim() != 3 or tok.shape[1] != 2 * tgt.shape[1]:
        return False
    live = ~b["key_pad"]
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    return bool((even == 0).all()) and bool((odd == 1).all())


def _gather_step(x, p):
    """x [n, m, D], p [n] long -> x[i, p[i]] as [n, D] (guarded gather)."""
    return x.gather(1, p.view(-1, 1, 1).expand(-1, 1, x.shape[-1])).squeeze(1)


def aux_loss(head_state, batch, net, device):
    """Chained write-imagined consistency + imagined-context InfoNCE (train only).
    Strictly causal: every fed-back token derives from history <= its own position via
    the net's causal passes; true future obs appear only as LOSS LABELS / negative-pool
    rows, never as inputs."""
    hs = head_state
    if hs is None:
        return 0.0
    hs["calls"] += 1
    w_scale = min(1.0, hs["calls"] / max(1, hs["warmup"]))
    if w_scale == 0.0 or (hs["w_mse"] == 0.0 and hs["w_cos"] == 0.0 and hs["w_nce"] == 0.0):
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0
    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    tgt, cmd_mask = batch["tgt"], batch["cmd_mask"]
    B, maxn, D = tgt.shape
    if maxn < 2:
        return 0.0

    # Depth curriculum: 1 imagined step early, then the battery's horizon-2.
    H = 1 if hs["calls"] <= hs["h2_after"] else max(1, int(hs["max_h"]))
    H = min(H, maxn - 1)

    # Row subsample: aux compute ~= (H+1) forwards on frac rows.
    nrows = max(1, int(math.ceil(B * hs["frac"])))
    rows = torch.randperm(B, device=device)[:nrows]
    types_s, pad_s = types[rows], key_pad[rows]
    tgt_s, cm_s = tgt[rows], cmd_mask[rows]
    cmd_tok = tok[rows][:, 0::2]                    # [n, maxn, D] real cmd tokens
    obs_tok = tok[rows][:, 1::2]                    # [n, maxn, D] real obs tokens

    # Chain start eligibility: steps s..s+H all real (writes at s..s+H-1, losses at
    # s+1..s+H). Sample one start per row, uniform over eligible; dead rows mask out.
    S = maxn - H
    elig = cm_s[:, :S].clone()
    for j in range(1, H + 1):
        elig &= cm_s[:, j:j + S]
    alive = elig.any(dim=1)
    if not alive.any():
        return 0.0
    scores = torch.where(elig, torch.rand(nrows, S, device=device),
                         torch.full((nrows, S), -1.0, device=device))
    start = scores.argmax(dim=1)                    # [n]; arbitrary (masked) on dead rows

    # InfoNCE pool: every valid TRUE obs in the subsample (labels only, never inputs).
    pool = tgt_s[cm_s]                              # [M, D]
    idx_map = torch.full((nrows, maxn), -1, dtype=torch.long, device=device)
    idx_map[cm_s] = torch.arange(pool.shape[0], device=device)
    pool_n = F.normalize(pool, dim=1, eps=1e-8)
    use_nce = hs["w_nce"] > 0.0 and pool.shape[0] >= 4

    tmod = getattr(net, "target_module", None)
    # prev obs for to_obs at the chain start: real obs at start-1 (zeros at start=0),
    # mirroring plan_env.imagine_candidate / collate's zero left-pad.
    prev = torch.where((start > 0).view(-1, 1),
                       _gather_step(obs_tok, (start - 1).clamp_min(0)),
                       torch.zeros(nrows, D, device=device))

    mixed = obs_tok                                 # functional updates only (grad flows)
    step_pos = start
    arange_m = torch.arange(maxn, device=device)
    n_alive = alive.float().sum().clamp_min(1.0)
    total = tgt_s.new_zeros(())
    for j in range(H + 1):
        tok_mod = torch.stack([cmd_tok, mixed], dim=2).reshape(nrows, 2 * maxn, D)
        pred_full, _ = net(tok_mod, types_s, pad_s)
        cmd_pred = pred_full[:, 0::2][:, :maxn]     # [n, maxn, D]
        pred_p = _gather_step(cmd_pred, step_pos)   # pred of obs at step_pos
        obs_pred = tmod.to_obs(pred_p, prev) if tmod is not None else pred_p
        if j >= 1:
            true_p = _gather_step(tgt_s, step_pos)  # LABEL: true z_obs at step_pos
            mse_row = ((obs_pred - true_p) ** 2).mean(dim=1)
            cos_row = 1.0 - F.cosine_similarity(obs_pred, true_p, dim=1, eps=1e-8)
            term = (hs["w_mse"] * (mse_row * alive).sum() / n_alive
                    + hs["w_cos"] * (cos_row * alive).sum() / n_alive)
            if use_nce and bool(alive.any()):
                logits = (F.normalize(obs_pred[alive], dim=1, eps=1e-8)
                          @ pool_n.t()) / hs["tau"]
                pos_idx = idx_map.gather(1, step_pos.view(-1, 1)).squeeze(1)[alive]
                term = term + hs["w_nce"] * F.cross_entropy(logits, pos_idx)
            total = total + term / H                # average over the H supervised passes
        if j < H:
            # WRITE the imagined obs at step_pos (gradient flows through the write).
            wmask = (arange_m.view(1, -1) == step_pos.view(-1, 1)) & alive.view(-1, 1)
            mixed = torch.where(wmask.unsqueeze(-1), obs_pred.unsqueeze(1), mixed)
            prev = obs_pred
            step_pos = (step_pos + 1).clamp_max(maxn - 1)
    if not torch.isfinite(total):
        return 0.0
    return w_scale * total


def leak_safe(mod, params):
    """Certify no-future-leakage:
      - forward is UNTOUCHED (wrap adds nothing), so the stream's leakage_ok probe
        runs the arch's own causal forward unchanged;
      - every written token is the net's causal output at cmd position 2p (earlier
        chain writes are themselves causal) placed at obs position 2p+1 > 2p;
      - true future observations enter only as loss labels / InfoNCE pool rows inside
        aux_loss, which is absent from forward() and thus dropped at eval.
    Validate params: fractions/temps/weights finite and in range."""
    f = float(params.get("row_frac", _ROW_FRAC))
    ws = [float(params.get(k, d)) for k, d in
          (("w_mse", _W_MSE), ("w_cos", _W_COS), ("w_nce", _W_NCE))]
    tau = float(params.get("tau", _TAU))
    wu = int(params.get("warmup", _WARMUP))
    h2 = int(params.get("h2_after", _H2_AFTER))
    mh = int(params.get("max_h", _MAX_H))
    return ((0.0 < f <= 1.0) and all(w >= 0.0 and math.isfinite(w) for w in ws)
            and tau > 0.0 and math.isfinite(tau) and wu >= 1 and h2 >= 0 and 1 <= mh <= 4)
