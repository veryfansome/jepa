"""head chunk: INVERSE-DYNAMICS ACTION-WITNESS auxiliary — EB-JEPA's planning-critical
aux, applied to the model's OWN predictions.

Cross-domain principle (inverse dynamics as the planning-enabling feature): EB-JEPA's
ablation measured planning success collapsing 97% -> 1% when the inverse-dynamics
auxiliary is removed (JEPA.md §5); the same mechanism is the backbone of ICM (Pathak
et al., arXiv:1705.05363) and goal-directed manipulation from inverse models (Agrawal
et al., arXiv:1606.07419). The reason it is planning-critical: forward-prediction
losses alone let the predictor under-use the ACTION — predictions drift toward the
context's mean continuation, so imagined outcomes of different actions from the same
state are barely distinguishable, and argmin-over-actions planning becomes noise.

That failure is exactly what the R11 battery measured in the champion: the binding
constraint is per-decision accuracy at remaining-depth 1 (0.42-0.51), where up to 237
sibling `cd` candidates share ONE history and the imagined next-ls must discriminate
the goal's own listing from its siblings' (the real obs field ranks the on-path child
first on 64/100 goals; the model's imagined ls lags it, ~3446 sqL2 vs random-pair
1436). All K candidate rollouts differ ONLY in the cmd token — if the prediction
under-uses the command, the K imagined listings crowd together and cosine ranking
cannot pick the goal's child.

Mechanism (train-only, forward untouched): an action-witness InfoNCE. One causal pass
on a row-subsample; at each transition t (t-1, t valid) take the model's own
obs-space prediction pred_t and the REAL previous observation z_obs_{t-1}, and train
a small private decoder g to identify WHICH command produced the transition among the
in-batch commands:

    x_t   = [z_obs_{t-1} ; pred_t]                       (2D)
    s_ij  = <g(x_i)/|g(x_i)|,  z_cmd_j/|z_cmd_j|> / tau  (K x K logits)
    L_aux = w * CE(s, diag)   with near-duplicate foil commands masked
            (cos(z_cmd_i, z_cmd_j) > 0.98 -> logit -1e4; `ls -la` repeats are not
            false negatives, so the gradient concentrates on path-specific cd/cat
            commands — the planning-relevant ones)

z_obs_{t-1} and z_cmd_t come from the batch (no grad); the ONLY gradient path into
the trunk is through pred_t, so the trunk is pressured to emit predictions from which
the action is decodable — an injectivity-in-the-action force on exactly the
imagination the battery ranks. Anti-collapse by construction: a constant prediction
makes the command undecodable beyond the prior in z_obs_{t-1}, leaving L_aux near
log K; only action-sensitive predictions minimize it.

Why fitness (G1) survives: the fitness metric discriminates the true next obs against
SAME-VERB foils — listings/files addressed by different commands. Decodability of the
command from the prediction is the same direction of specificity, not a competing
one; the main loss, forward, and readout are untouched, the aux touches a row
subsample at modest weight (0.1, vs the 0.25 MSE aux the champion lineage tolerated).

Contract / safety:
  * wrap() adds NO modules to net and does NOT re-point net.forward. The decoder g is
    PRIVATE state in head_state (built lazily on the batch device, trained by its own
    Adam inside aux_loss via a deferred step on the grads the harness's backward left
    behind). net.state_dict() therefore stays arch-pure, so the FROZEN planning
    instruments (path_battery / plan_env), which rebuild the arch WITHOUT head wraps
    and load checkpoints strictly, keep working — the hazard that forces r10's
    no-module design, respected here with learnable capacity anyway.
  * Causal: pred_t is the net's causal output at cmd position 2t (function of tokens
    <= 2t); z_obs_{t-1} is the past; cmd_t is used only as the contrastive LABEL of a
    train-only loss absent from forward() — dropped at eval, leakage probe unaffected.
  * NaN-safe: clamped norms, finiteness gate, hard-zero fallbacks on degenerate
    batches/layouts; masked logits use -1e4, never -inf.
"""

import math

import torch
import torch.nn.functional as F

NAME = "r11_invdyn_action_witness_aux"
DESCRIPTION = (
    "Train-only inverse-dynamics InfoNCE on the model's OWN predictions: a private "
    "decoder must identify WHICH command produced [prev_obs; pred_t] among in-batch "
    "commands (near-duplicate foils masked). Gradient reaches the trunk only through "
    "pred_t, forcing imagined outcomes of different actions apart — the EB-JEPA "
    "planning-critical aux (97%->1% ablation), aimed at the battery's measured "
    "remaining-depth-1 sibling-discrimination failure. Forward untouched; decoder "
    "params live in head_state (own Adam), so checkpoints stay arch-pure for the "
    "frozen battery."
)

_AUX_W = 0.1       # modest: champion lineage tolerated a 0.25 aux without margin damage
_ROW_FRAC = 0.5    # fraction of batch rows the aux touches (~0.5 extra fwd per step)
_TAU = 0.1         # InfoNCE temperature on unit vectors
_MAX_POS = 256     # cap on contrastive pool size (logits are cap x cap)
_DUP_COS = 0.98    # foil commands this similar to the positive are masked (repeats)
_HEAD_LR = 1e-3    # private Adam lr for the decoder
_NEG = -1e4        # masked-logit value (softmax-safe, no -inf arithmetic)


def wrap(net, D, aux_weight=_AUX_W, row_frac=_ROW_FRAC, tau=_TAU,
         max_pos=_MAX_POS, dup_cos=_DUP_COS, head_lr=_HEAD_LR):
    """No modules on net, no forward re-pointing (composes with ANY arch, and keeps
    net.state_dict() loadable by the frozen battery/plan_env which never apply head
    wraps). Returns the mutable head_state dict that owns the private decoder."""
    return {"w": float(aux_weight), "frac": float(row_frac), "tau": float(tau),
            "cap": int(max_pos), "dup": float(dup_cos), "lr": float(head_lr),
            "D": int(D), "mod": None, "opt": None}


def _interleave_layout_ok(b):
    """The indexing assumes the single-vector interleave: tok[2t]=cmd_t, tok[2t+1]=obs_t,
    tgt[:,t]=z_obs_t. Verify cheaply; other streams -> aux off (hard zero)."""
    tok, types, tgt = b["tok"], b["types"], b["tgt"]
    if tok.dim() != 3 or tgt.dim() != 3 or tok.shape[1] != 2 * tgt.shape[1]:
        return False
    live = ~b["key_pad"]
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    return bool((even == 0).all()) and bool((odd == 1).all())


def _ensure_decoder(hs, device):
    """Lazily build (or migrate) the private inverse-dynamics decoder + its Adam."""
    if hs["mod"] is None:
        D = hs["D"]
        mod = torch.nn.Sequential(
            torch.nn.Linear(2 * D, D), torch.nn.GELU(), torch.nn.Linear(D, D)
        ).to(device)
        hs["mod"] = mod
        hs["opt"] = torch.optim.Adam(mod.parameters(), lr=hs["lr"])
    else:
        p0 = next(hs["mod"].parameters())
        if p0.device != device:  # rare device migration: move + fresh optimizer state
            hs["mod"] = hs["mod"].to(device)
            hs["opt"] = torch.optim.Adam(hs["mod"].parameters(), lr=hs["lr"])
    return hs["mod"], hs["opt"]


def aux_loss(head_state, batch, net, device):
    """Action-witness InfoNCE (train only). Strictly causal: pred_t is the net's own
    causal output; the future never enters forward(); cmd_t appears only as the
    contrastive label of this train-only term."""
    hs = head_state
    if hs is None or hs["w"] == 0.0:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0
    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    tgt, cmd_mask = batch["tgt"], batch["cmd_mask"]
    B, maxn, D = tgt.shape
    if maxn < 2 or D != hs["D"]:
        return 0.0
    dev = tok.device

    dec, opt = _ensure_decoder(hs, dev)
    # Deferred private step: the harness's backward AFTER the previous aux_loss call
    # populated grads on the decoder (it is not in the main optimizer, whose zero_grad
    # and clip touch only net.parameters()); apply them now, then clear.
    if any(p.grad is not None for p in dec.parameters()):
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Row subsample: aux compute ~= 1 forward on frac of rows.
    nrows = max(1, int(math.ceil(B * hs["frac"])))
    rows = torch.randperm(B, device=dev)[:nrows]
    tok_s, types_s, pad_s = tok[rows], types[rows], key_pad[rows]
    tgt_s, cm_s = tgt[rows], cmd_mask[rows]

    # The model's own causal predictions at each cmd position (grad ON).
    pred_full, _ = net(tok_s, types_s, pad_s)               # [n, L, D]
    cmd_pred = pred_full[:, 0::2][:, :maxn]                 # [n, maxn, D]

    # Map prediction -> obs space (identity for the champion; exact inverse for a
    # LEARNED target, mirroring plan_env.imagine_candidate's to_obs call).
    tmod = getattr(net, "target_module", None)
    if tmod is not None:
        prev = torch.cat([torch.zeros_like(tgt_s[:, :1]), tgt_s[:, :-1]], dim=1)
        n_, m_, d_ = cmd_pred.shape
        obs_pred = tmod.to_obs(cmd_pred.reshape(-1, d_), prev.reshape(-1, d_)).reshape(n_, m_, d_)
    else:
        obs_pred = cmd_pred

    # Transitions with a REAL previous obs: steps t>=1 where t-1 and t are both valid.
    valid = cm_s[:, 1:] & cm_s[:, :-1]                      # [n, maxn-1] -> step t=idx+1
    if not valid.any():
        return 0.0
    prev_obs = tgt_s[:, :-1][valid]                         # z_obs_{t-1}: the PAST (no grad)
    pred_t = obs_pred[:, 1:][valid]                         # causal prediction of obs_t (grad)
    cmds = tok_s[:, 0::2][:, :maxn][:, 1:][valid]           # cmd_t token: the LABEL (no grad)
    K = pred_t.shape[0]
    if K < 2:
        return 0.0
    if K > hs["cap"]:
        sel = torch.randperm(K, device=dev)[: hs["cap"]]
        prev_obs, pred_t, cmds = prev_obs[sel], pred_t[sel], cmds[sel]
        K = hs["cap"]

    z = dec(torch.cat([prev_obs, pred_t], dim=-1))
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = cmds / cmds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    logits = (z @ c.t()) / hs["tau"]                        # [K, K], positives on diag
    # False-negative guard: foils whose command is (near-)identical to the positive
    # (`ls -la` repeats dominate shell histories) are masked out of the softmax.
    csim = c @ c.t()
    dup = (csim > hs["dup"]) & ~torch.eye(K, dtype=torch.bool, device=dev)
    logits = logits.masked_fill(dup, _NEG)
    loss = F.cross_entropy(logits, torch.arange(K, device=dev))
    if not torch.isfinite(loss):
        return 0.0
    return hs["w"] * loss


def leak_safe(mod, params):
    """Certify no-future-leakage:
      - forward is UNTOUCHED (wrap adds nothing to net), so the stream's leakage_ok
        probe runs the arch's own causal forward unchanged;
      - the aux consumes only past inputs (z_obs_{t-1}) and the net's causal output
        pred_t; the command at t is a contrastive LABEL inside a train-only term that
        is absent from forward() and thus dropped at eval;
      - the private decoder never enters net.state_dict(), so eval-time checkpoints
        are bit-identical to a head-free net's.
    Validate params: fractions/probabilities in range, finite non-negative weight,
    positive temperature/lr, pool cap >= 2, dup threshold in (0, 1]."""
    w = float(params.get("aux_weight", _AUX_W))
    f = float(params.get("row_frac", _ROW_FRAC))
    t = float(params.get("tau", _TAU))
    cap = int(params.get("max_pos", _MAX_POS))
    d = float(params.get("dup_cos", _DUP_COS))
    lr = float(params.get("head_lr", _HEAD_LR))
    return (w >= 0.0 and math.isfinite(w) and 0.0 < f <= 1.0 and t > 0.0
            and math.isfinite(t) and cap >= 2 and 0.0 < d <= 1.0 and lr > 0.0)
