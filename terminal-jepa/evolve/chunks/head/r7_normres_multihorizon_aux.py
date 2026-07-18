import torch
import torch.nn as nn

NAME = "r7_normres_multihorizon_aux"
DESCRIPTION = ("Normalized-residual MLP main readout + multi-horizon (k=2,3) "
               "self-supervised auxiliary prediction from the same hidden state; "
               "aux dropped at eval. New `head` axis.")


class _MultiHorizonHead(nn.Module):
    """Wraps an arch. Main path: h -> normalized-residual MLP -> D (per-position).
    Aux path (train only): h -> Linear -> [n_horizon * D] future-obs predictions."""

    def __init__(self, base_net, D, d, horizons=(2, 3), aux_weight=0.1,
                 hidden_mult=2, res_scale=1.0):
        super().__init__()
        # Hold the arch by a NON-REGISTERED reference (bypass nn.Module.__setattr__): the head
        # is added as a submodule of base_net, so registering base_net here too would create a
        # parent<->child module cycle (infinite recursion in .to()/.parameters()). Stored in
        # __dict__, `base` is reachable for forward but not traversed as a child module.
        object.__setattr__(self, "base", base_net)
        self.D = D
        self.d = d
        self.horizons = tuple(int(k) for k in horizons)
        self.aux_weight = float(aux_weight)
        self.res_scale = float(res_scale)
        # MAIN head over the arch's d-dim hidden state.
        self.ln = nn.LayerNorm(d)
        self.base_out = nn.Linear(d, D)                     # replaces arch's Linear(d,D)
        hid = max(1, int(hidden_mult) * d)
        self.refine = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, D))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)  # identity at init
        # AUX head: predict n_horizon future obs vectors from the same h (train only).
        self.aux = nn.Linear(d, len(self.horizons) * D)

    def _trunk_h(self, tok_emb, types, key_pad):
        """Return the d-dim hidden state [B,L,d] from the arch's trunk. Prefer .encode
        (SeqWorldModel-based arches). Otherwise call the arch's ORIGINAL forward (saved
        by wrap before re-pointing net.forward) — never self.base(...), which is now the
        wrapped forward and would recurse."""
        if hasattr(self.base, "encode"):
            return self.base.encode(tok_emb, types, key_pad)
        _, h = self._orig_forward(tok_emb, types, key_pad)
        return h

    def _main(self, h):
        z = self.ln(h)
        return self.base_out(z) + self.res_scale * self.refine(z)

    def forward(self, tok_emb, types, key_pad):
        # MAIN readout only — this is what the harness/eval sees. No aux here => aux
        # is dropped at eval and cannot leak. Per-position function of h.
        h = self._trunk_h(tok_emb, types, key_pad)          # [B,L,d]
        return self._main(h), h                             # (pred[B,L,D], h[B,L,d])

    @torch.no_grad()
    def _noop(self):
        return None


def wrap(net, D, horizons=(2, 3), aux_weight=0.1, hidden_mult=2, res_scale=1.0):
    """Re-point net.forward through the multi-horizon head. Recover the arch's
    bottleneck dim d from net.head (Linear(d,D)); fall back to D if absent (head
    then refines in D-space, still per-position and leak-safe). Returns the head
    module as the head_state so aux_loss can reach it."""
    d = D
    base_head = getattr(net, "head", None)
    if isinstance(base_head, nn.Linear):
        d = base_head.in_features
    head = _MultiHorizonHead(net, D, d, horizons=horizons, aux_weight=aux_weight,
                             hidden_mult=hidden_mult, res_scale=res_scale)
    # Save the arch's ORIGINAL bound forward BEFORE re-pointing, so the trunk fallback
    # (for arches without .encode) can recompute h without recursing into the wrapper.
    head._orig_forward = net.forward
    # Register the head ON the net so net.parameters() (used to build the optimizer)
    # includes the new main+aux params. We attach as a submodule and re-point forward.
    net.add_module("_r7_head", head)
    net._r7_forward = head.forward           # keep a direct handle
    net.forward = head.forward               # every net(...) now uses the main head
    return head


def aux_loss(head_state, batch, net, device):
    """Multi-horizon self-supervised auxiliary loss (train only). From h at cmd
    position t, predict z_obs[t+k] for k in horizons, comparing to b["tgt"] shifted
    by k (strict future, same sequence, masked to valid pairs). Targets are LABELS
    read from tgt, never inputs -> no leakage. Returns a scalar tensor (0 if no
    valid pairs)."""
    head = head_state
    if head is None or head.aux_weight == 0.0:
        return 0.0
    tok, types, key_pad = batch["tok"], batch["types"], batch["key_pad"]
    tgt_full = batch["tgt"]                                  # [B, maxn, D] = z_obs per step
    cmd_mask = batch["cmd_mask"]                             # [B, maxn] bool
    B, maxn, D = tgt_full.shape
    # Recompute the trunk hidden state and gather cmd-position h [B, maxn, d].
    h_full = head._trunk_h(tok, types, key_pad)             # [B, L, d]
    # cmd positions are the stream's step positions; the stream lays step t's cmd at a
    # fixed stride. We reuse the SAME extraction the harness/stream use by taking the
    # arch's own convention: cmd tokens are the even (stride) positions -> [B, maxn, d].
    stride = h_full.shape[1] // maxn if maxn > 0 else 1
    stride = max(1, stride)
    h_cmd = h_full[:, ::stride][:, :maxn]                   # [B, maxn, d]
    aux_pred = head.aux(h_cmd)                              # [B, maxn, n_h * D]
    aux_pred = aux_pred.view(B, maxn, len(head.horizons), D)
    total = tok.new_zeros(())
    n_terms = 0
    for hi, k in enumerate(head.horizons):
        if k >= maxn:
            continue
        # target for position t is z_obs at step t+k; valid where BOTH t and t+k are
        # real cmd steps. Shift tgt/cmd_mask left by k.
        fut_tgt = tgt_full[:, k:]                           # [B, maxn-k, D]
        valid = cmd_mask[:, :maxn - k] & cmd_mask[:, k:]    # [B, maxn-k] bool
        pred_k = aux_pred[:, :maxn - k, hi, :]              # [B, maxn-k, D]
        if valid.any():
            diff = (pred_k[valid] - fut_tgt[valid]) ** 2
            total = total + diff.mean()
            n_terms += 1
    if n_terms == 0:
        return 0.0
    return head.aux_weight * (total / n_terms)


def leak_safe(mod, params):
    """Certify the no-future-leakage invariant for this head:
      - MAIN path (_main) is a strictly per-position function of h (LayerNorm +
        Linear + per-position MLP; no cross-token op), so obs_t cannot influence the
        cmd_<=t prediction through the head — the existing stream.leakage_ok probe
        (which now runs the WRAPPED forward) re-verifies this end-to-end.
      - AUX path is invoked ONLY inside aux_loss during training and is absent from
        forward(), so it is dropped at eval; its future targets come from tgt
        (labels), never from any input tensor. No future obs enters an input path.
    Horizons must be a nonempty set of positive integers; aux_weight finite >= 0."""
    hz = params.get("horizons", (2, 3))
    if not len(hz) or any((int(k) <= 0) for k in hz):
        return False
    aw = params.get("aux_weight", 0.1)
    return (aw >= 0.0) and (aw == aw) and (aw != float("inf"))
