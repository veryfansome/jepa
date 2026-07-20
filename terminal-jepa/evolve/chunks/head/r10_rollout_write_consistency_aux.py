"""head chunk: ROLLOUT WRITE-CONSISTENCY auxiliary (scheduled-sampling for latent JEPA rollouts).

Cross-domain principle (sequence-model exposure bias): a model trained only under
teacher forcing is never exposed to its own outputs, so at rollout time it consumes
inputs from a distribution it has never seen and errors compound (Bengio et al. 2015,
scheduled sampling, arXiv:1506.03099; Ross & Bagnell DAgger, arXiv:1011.0686). The
R10 Stage-2 probe measured EXACTLY this failure in the champion: predictions are
grossly off-manifold (||pred||^2/||true||^2 ~ 4.7, matched sqL2 ~2020 > random true
pair ~1430) because the contrastive row-softmax is invariant to each prediction's own
squared norm, and write-policy imagination (feeding predictions back as observations,
as latent-MPC planning must) runs the trunk on inputs it never trained on.

Mechanism (train-only, forward untouched):
  1. First pass on a row-subsample of the batch: pred_t = net(tok)[cmd pos 2t].
  2. WRITE-IMAGINED feedback, exactly the Stage-2 `write` policy: with prob p per
     valid step t (t+1 also valid), replace the obs token at position 2t+1 with the
     model's OWN obs-space prediction pred_t (via net.target_module.to_obs when a
     learned target is attached; identity otherwise -- the champion's target). The
     fed-back token is a function of tokens <= 2t only (the net is causal), so
     causality is preserved by construction. Gradients FLOW through the fed-back
     prediction (no detach): the producer is trained to emit tokens that work as
     observations.
  3. Second pass on the modified tokens; at every cmd position t+1 whose preceding
     obs was imagined, penalize mean squared error of the new prediction against the
     TRUE obs_{t+1} (a label, never an input):
        L_aux = w * mean_{(b,t+1): replaced(b,t)} ||pred'_{t+1} - z_obs_{t+1}||^2 / D

Why calibration improves without losing the ranking margin: the aux MSE is an
absolute-placement (norm-sensitive) loss measured UNDER IMAGINED CONTEXT -- the
exact operating condition of planning -- so it supplies the norm-calibration
gradient the row-softmax objective provably lacks, while the main contrastive loss
(untouched, full batch, every step) keeps shaping the ranking geometry. The weight
is modest and the aux touches only a row-subsample and only post-imagination
positions, so it perturbs rather than replaces the champion's training signal (the
champion already carries a 0.05 MSE anchor without margin damage).

Contract / safety:
  * wrap() adds NO modules and does NOT re-point net.forward (no parent<->child
    module cycle, no forward-recursion hazard; the leakage probe runs the original
    forward). It returns a plain config dict as head_state.
  * aux_loss returns 0.0 (hard zero) when the stream is not the single-vector
    interleave layout (guarded by shape+types check), when no step is sampled, or
    when the batch is degenerate. NaN-safe: masked means guarded by .any().
  * Anti-collapse: the aux is an MSE to varying true targets -- a constant
    prediction cannot minimize it; the main objective is untouched.
"""

import math

import torch

NAME = "r10_rollout_write_consistency_aux"
DESCRIPTION = (
    "Train-time write-imagined rollout consistency: on a row-subsample, replace obs "
    "tokens with the model's own causal predictions (Stage-2 `write` policy) and MSE-"
    "penalize the next-step prediction against the true next obs. Scheduled-sampling / "
    "DAgger exposure-bias fix supplying the norm-calibration gradient the contrastive "
    "row-softmax lacks; forward untouched, aux dropped at eval."
)

_P_REPLACE = 0.5   # per-step probability of writing the imagined obs
_ROW_FRAC = 0.5    # fraction of batch rows the aux touches (compute ~= 1 extra fwd/step)
_AUX_W = 0.25      # weight on the consistency MSE (champion tolerates a 0.05 anchor)


def wrap(net, D, p_replace=_P_REPLACE, row_frac=_ROW_FRAC, aux_weight=_AUX_W):
    """No modules, no forward re-pointing: the arch's readout is untouched (so the
    aux composes with ANY arch, including non-per-position-pred mechanisms). Returns
    a config dict as head_state for aux_loss."""
    return {"p": float(p_replace), "frac": float(row_frac), "w": float(aux_weight), "D": int(D)}


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


def aux_loss(head_state, batch, net, device):
    """Write-imagined one-step consistency term (train only). Strictly causal: every
    fed-back token derives from history <= its own position via the net's first
    causal pass; true future obs appear only as LOSS LABELS, never as inputs."""
    hs = head_state
    if hs is None or hs["w"] == 0.0 or hs["p"] == 0.0:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0
    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    tgt, cmd_mask = batch["tgt"], batch["cmd_mask"]
    B, maxn, D = tgt.shape
    if maxn < 2:
        return 0.0

    # Row subsample: aux compute ~= 2 forwards on frac of rows ~= 1 extra forward/step.
    nrows = max(1, int(math.ceil(B * hs["frac"])))
    rows = torch.randperm(B, device=device)[:nrows]
    tok_s, types_s, pad_s = tok[rows], types[rows], key_pad[rows]
    tgt_s, cm_s = tgt[rows], cmd_mask[rows]

    # Steps eligible for imagination: t and t+1 both real; sample with prob p.
    elig = cm_s[:, :-1] & cm_s[:, 1:]                       # [n, maxn-1] -> step t
    rep = elig & (torch.rand(elig.shape, device=device) < hs["p"])
    if not rep.any():
        return 0.0

    # Pass 1 (grad ON): the model's own causal predictions at each cmd position.
    pred_full, _ = net(tok_s, types_s, pad_s)               # [n, L, D]
    cmd_pred = pred_full[:, 0::2][:, :maxn]                 # [n, maxn, D] pred of obs_t at cmd_t

    # Map prediction -> obs space (identity for the champion; exact inverse for a
    # LEARNED target, mirroring plan_env.imagine_candidate's to_obs call).
    tmod = getattr(net, "target_module", None)
    if tmod is not None:
        prev = torch.cat([torch.zeros_like(tgt_s[:, :1]), tgt_s[:, :-1]], dim=1)
        n_, m_, d_ = cmd_pred.shape
        obs_pred = tmod.to_obs(cmd_pred.reshape(-1, d_), prev.reshape(-1, d_)).reshape(n_, m_, d_)
    else:
        obs_pred = cmd_pred

    # WRITE the imagined obs at positions 2t+1 (gradient flows through the write).
    # Rebuild the interleave functionally (no in-place on a no-grad clone).
    rep_full = torch.zeros(nrows, maxn, dtype=torch.bool, device=device)
    rep_full[:, :-1] = rep
    obs_tok = tok_s[:, 1::2]                                # true obs tokens [n, maxn, D]
    mixed_obs = torch.where(rep_full.unsqueeze(-1), obs_pred, obs_tok)
    tok_mod = torch.stack([tok_s[:, 0::2], mixed_obs], dim=2).reshape(nrows, 2 * maxn, D)

    # Pass 2: predict under imagined context; supervise at cmd_{t+1} for replaced t.
    pred2_full, _ = net(tok_mod, types_s, pad_s)
    cmd_pred2 = pred2_full[:, 0::2][:, :maxn]               # [n, maxn, D]
    next_mask = torch.zeros_like(rep_full)
    next_mask[:, 1:] = rep                                  # position t+1 after imagined obs_t
    if not next_mask.any():
        return 0.0
    pred_next = cmd_pred2[next_mask]                        # [k, D]
    if tmod is not None:
        prev_next = mixed_obs[:, :-1][next_mask[:, 1:]]     # imagined prev, as at rollout
        pred_next = tmod.to_obs(pred_next, prev_next)
    true_next = tgt_s[next_mask]                            # LABELS: true z_obs_{t+1}
    err = ((pred_next - true_next) ** 2).mean()
    if not torch.isfinite(err):
        return 0.0
    return hs["w"] * err


def leak_safe(mod, params):
    """Certify no-future-leakage:
      - forward is UNTOUCHED (wrap adds nothing), so the stream's leakage_ok probe
        runs the arch's own causal forward unchanged;
      - every written token is the net's causal output at position 2t (a function of
        tokens <= 2t) placed at position 2t+1 >= its information horizon;
      - true future observations enter only as loss labels in aux_loss, which is
        absent from forward() and thus dropped at eval.
    Validate params: probabilities/fractions in [0,1], finite non-negative weight."""
    p = float(params.get("p_replace", _P_REPLACE))
    f = float(params.get("row_frac", _ROW_FRAC))
    w = float(params.get("aux_weight", _AUX_W))
    return (0.0 <= p <= 1.0) and (0.0 < f <= 1.0) and (w >= 0.0) and math.isfinite(w)
