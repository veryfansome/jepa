"""Theta-sweep forward-rollout predictor: a causal transformer ("slow neocortex") whose
per-step context is rolled FORWARD by a compressed hippocampal "theta sweep" (a small
CA3-style recurrent transition operator) before it content-addresses the sequence's
strictly-past observation embeddings. The swept-ahead place code retrieves an on-manifold
convex combination of REAL past observations as the prediction, confidence-gated into the
parametric head.

CROSS-DOMAIN MOTIVATION (neuroscience -> architecture)
------------------------------------------------------
A shell exploration is literal navigation: `cd` moves between places (directories); `ls`/`cat`
are observations AT a place. During each theta cycle the hippocampus does not merely represent
the current place -- CA3 recurrent dynamics generate a temporally COMPRESSED forward "sweep"
that re-simulates the trajectory AHEAD of the animal (look-ahead theta sequences; goal-directed
theta sweeps). Sequential-predictive-learning theory (Recanatesi et al., bioRxiv 2024) frames
these sweeps as compressed forward PREDICTIONS produced by the same recurrent operator that
encodes place; the CA3 circuit model (bioRxiv 2023.05.24.542204) shows recurrent connectivity
compresses and replays learned place sequences within one theta cycle.

Translation. The current best arch (hippo_episodic_place_read) reads episodic memory keyed by
the CURRENT place: "a past place like where I am now". This module instead ROLLS the place code
forward through a learned recurrent transition cell (the theta sweep / CA3 recurrence) for a
learned, gated number of compressed micro-steps, and keys the read by the SWEPT-AHEAD place:
"a past place like where this command is taking me". The value it returns is still a convex
combination of ACTUAL strictly-past standardized observation embeddings, so it is guaranteed to
lie on the manifold the retrieval eval scores, and requires no cross-system transfer.

WHY THIS SHOULD RAISE THE HELD-OUT CONTENT-VERB MARGIN
------------------------------------------------------
The eval ranks the true next obs by squared-L2 against same-verb foils, so on-manifold convex
reads win (this is what made the backward hippo read work). The forward dual should recruit MORE
of the retrievable structure: per the ledger only ~4.7% of content-verb eval steps repeat an
exact command earlier in the sequence, but ~45% share a same-verb-at-same-cwd antecedent. A
current-place read matches "where I am"; a swept-ahead read matches "where the command lands",
which is exactly how an `ls` earlier in a directory, or a `cat` of a file whose name a prior
`ls` revealed, becomes pattern-completable even when the literal command never repeated. A
learned soft gate over the K micro-steps lets the sweep length adapt per step (0 steps recovers
the current-place read; the parametric confidence gate recovers the baseline transformer), so
the module is a strict, learnable generalization of both the baseline and the current-place read.

DISTINCTION FROM EXISTING ARCH IMPLS
------------------------------------
- vs hippo_episodic_place_read: hippo keys the read by the current place; here a learned
  recurrent transition operator sweeps the place code FORWARD (K gated micro-steps) before
  keying -- the read is addressed by the predicted destination, not the current location. The
  sweep operator, the per-step sweep-length gate, and the forward key are all new.
- vs typed_recency_sysid / recency_alibi: those are priors on the transformer's OWN attention;
  this adds a separate forward-rollout read pathway whose value is the raw obs embedding and
  whose output directly forms the prediction (on-manifold), not a hidden feature a head decodes.
- vs retrieve_by_cmd baseline (nearest TRAIN command, cross-episode): this reads over the CURRENT
  sequence's strictly-past observations, keyed by a swept-ahead learned place code.
- vs copy_prev (fixed t-1): content-addressed over the whole strictly-past sweep, not step t-1.

CAUSALITY (leak-free by construction)
-------------------------------------
The transformer uses the standard upper-triangular causal mask, so the command hidden at
position 2t depends only on tokens 0..2t. The theta sweep is a function of h_cmd_t and the raw
command embedding (both available at position <= 2t) only -- it uses NO observation of step t or
later. The read at step t attends ONLY over entries j<t (strict lower triangular) whose value is
obs_j at token position 2j+1 <= 2t-1 < 2t. Disallowed/pad entries get a true -inf logit (exact
zero weight); all-masked rows are nan-to-num'd to a zero read with the gate forced to zero via a
has-memory mask. Corrupting obs_t cannot move any command prediction at or before t: the sweep
inputs and every readable value are strictly earlier. Passes the per-genome no-leakage guard.

NUMERICAL STABILITY
-------------------
No exponentials over sequence position; no division by accumulated mass. Softmaxes are the
standard causal transformer attention and the read over a finite [n,n] logit matrix with a
bounded learned temperature (softplus + 0.5). The sweep gate is a bounded sigmoid; the recurrent
cell is a GRU-style update whose state is LayerNorm'd each micro-step and whose reach is a fixed
small K, so the rollout cannot blow up. Entropy uses a clamped log.

I/O CONTRACT
------------
forward(tok_emb [B,L,768], types [B,L] in {0,1}, key_pad [B,L] bool True=pad)
    -> (pred [B,L,768], h [B,L,d]); prediction at EVERY position (harness reads pred[:,0::2]).
~2.5M params at d=192, layers=4, heads=4, K=3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "theta_sweep_forward_rollout"
DESCRIPTION = (
    "Causal transformer + hippocampal theta-sweep forward rollout: at each command position a "
    "small CA3-style recurrent transition operator sweeps the place code forward a learned, "
    "gated number of compressed micro-steps, and the swept-ahead place content-addresses a "
    "convex combination of strictly-past raw observation embeddings, confidence-gated into the "
    "parametric head. Strictly causal, on-manifold read, numerically bounded."
)


class ThetaSweepWorldModel(nn.Module):
    """Baseline causal transformer front-end + a per-step forward theta sweep that keys an
    episodic read over strictly-past observation embeddings.

    Per command step t:
        place_t = LayerNorm( h_cmd_t + cmd_proj(cmd_emb_t) )            # current place code
        # compressed forward sweep (CA3 recurrence), K gated micro-steps:
        s <- place_t
        for m in 1..K:
            a_m = sigmoid(sweep_gate([s, place_t]))                     # per-step, per-position reach
            s   = LayerNorm( s + a_m * (GRUCell(place_t, s) - s) )      # advance toward destination
        q_t = Wq(s)                                                     # swept-ahead read key
        k_j = Wk(place_j)                                               # keys = past current-place codes
        logit_{t,j} = (q_t . k_j)/sqrt(kq)/temp,  temp = softplus(log_temp)+0.5
        e_t = sum_{j<t, j valid} softmax_j(logit_{t,j}) * obs_j         # value = raw std obs (on-manifold)
        g_t = sigmoid(gate([h_cmd_t, maxw, entropy])) * has_memory
        pred_cmd_t = (1 - g_t) * p_t + g_t * e_t
    obs positions keep the parametric prediction (unused by the harness).
    """

    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, max_len=64, kq=128, sweep_k=3,
                 no_history=False):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.kq = kq
        self.sweep_k = sweep_k
        # no_history=True is the matched-capacity ablation control: self-only attention AND the
        # sweep/read disabled (the read IS history by construction). Same modules/params.
        self.no_history = no_history

        # ---- slow neocortex: baseline-matched causal transformer front-end ----
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)          # 0=cmd, 1=obs
        self.pos_emb = nn.Embedding(max_len, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, D)

        # ---- place code (structural context + sensory-command location) ----
        self.cmd_proj = nn.Linear(D, d)
        self.place_norm = nn.LayerNorm(d)

        # ---- theta sweep: compressed CA3-style recurrent forward rollout ----
        self.sweep_cell = nn.GRUCell(d, d)          # transition operator (input = current place)
        self.sweep_gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.sweep_norm = nn.LayerNorm(d)

        # ---- forward-keyed episodic read ----
        self.Wq = nn.Linear(d, kq)                  # query from the SWEPT-AHEAD state
        self.Wk = nn.Linear(d, kq)                  # keys from past current-place codes
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.gate = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 1))

    def _sweep(self, place):
        """Roll the place code forward K compressed micro-steps via the recurrent transition
        operator, with a learned per-step per-position soft reach. Input `place` [.,d]; the GRU
        input is held fixed at `place` (the command's intended move), the recurrent state is what
        advances -- so K=0 reach recovers the current place and larger reach sweeps ahead."""
        B, n, d = place.shape
        p = place.reshape(B * n, d)
        s = p
        for _ in range(self.sweep_k):
            a = torch.sigmoid(self.sweep_gate(torch.cat([s, p], dim=-1)))   # [B*n,1] reach
            cand = self.sweep_cell(p, s)                                    # GRU transition
            s = self.sweep_norm(s + a * (cand - s))                        # gated advance, bounded
        return s.reshape(B, n, d)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        dev = tok_emb.device
        idx = torch.arange(L, device=dev).clamp(max=self.max_len - 1)
        x = self.proj(tok_emb) + self.type_emb(types.clamp(0, 1)) + self.pos_emb(idx)[None]

        if self.no_history:
            mask = ~torch.eye(L, device=dev, dtype=torch.bool)
            h = self.tf(x, mask=mask, src_key_padding_mask=None)
        else:
            mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1)
            h = self.tf(x, mask=mask, src_key_padding_mask=key_pad)        # [B,L,d]
        pred = self.head(h)                                                # [B,L,768] parametric
        if self.no_history:
            return pred, h                                                 # sweep/read disabled

        # ---- assemble per-step command/observation views (aligned by step) ----
        n = L // 2
        if n == 0:
            return pred, h
        hc = h[:, 0:2 * n:2]                     # [B,n,d]   command hidden (causal context)
        cmd_raw = tok_emb[:, 0:2 * n:2]          # [B,n,768] raw command embedding (location)
        obs_val = tok_emb[:, 1:2 * n:2]          # [B,n,768] raw standardized obs = episodic value
        valid_cmd = (~key_pad)[:, 0:2 * n:2]     # [B,n] bool valid command steps

        # ---- current-place code (structural context + sensory-command location) ----
        place = self.place_norm(hc + self.cmd_proj(cmd_raw))               # [B,n,d]

        # ---- theta sweep: roll the place code forward to the predicted destination ----
        swept = self._sweep(place)                                         # [B,n,d]

        # ---- forward-keyed read: query = swept-ahead place, keys = past current places ----
        q = self.Wq(swept)                                                 # [B,n,kq]
        k = self.Wk(place)                                                 # [B,n,kq]
        temp = F.softplus(self.log_temp) + 0.5
        logits = (q @ k.transpose(-2, -1)) / (self.kq ** 0.5) / temp       # [B,n,n]

        # strict causal read: query t may attend only to entries j < t, and only valid keys.
        strict = torch.tril(torch.ones(n, n, device=dev, dtype=torch.bool), -1)  # j<t
        allowed = strict[None] & valid_cmd[:, None, :]                     # [B,n,n]
        logits = logits.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)                            # all-masked rows -> 0
        e = attn @ obs_val                                                # [B,n,768] convex combo

        # ---- confidence-gated blend ----
        maxw = attn.max(dim=-1, keepdim=True).values                      # [B,n,1] match sharpness
        ent = -(attn * torch.log(attn.clamp(min=1e-9))).sum(-1, keepdim=True)  # [B,n,1]
        has_mem = allowed.any(dim=-1, keepdim=True).to(x.dtype)           # 0 when no valid past entry
        g = torch.sigmoid(self.gate(torch.cat([hc, maxw, ent], dim=-1))) * has_mem  # [B,n,1]

        p_cmd = pred[:, 0:2 * n:2]                                         # parametric cmd preds
        blended = (1.0 - g) * p_cmd + g * e                               # [B,n,768]
        pred = pred.clone()
        pred[:, 0:2 * n:2] = blended
        return pred, h


def build(d=192, layers=4, heads=4, dropout=0.1, max_len=64, kq=128, sweep_k=3,
          no_history=False):
    return ThetaSweepWorldModel(d=d, layers=layers, heads=heads, dropout=dropout,
                                max_len=max_len, kq=kq, sweep_k=sweep_k, no_history=no_history)

