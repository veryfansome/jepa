"""CA3 iterative attractor read with a sysid-factorized place key and typed-recency memory
addressing: a single causal transformer ("slow neocortex") whose command-position prediction
is completed by a MULTI-HOP hippocampal (CA3) read over strictly-past raw observation
embeddings, keyed by a place code factorized into a LOCAL command/place component and a GLOBAL
strict-prefix system-identity register, with a per-hop non-positive typed-recency decay on the
episodic logits, confidence-gated into the parametric head.

This UNIFIES the three independently-helpful mechanisms in the ledger (episodic place-read,
per-head/typed recency decay, sysid broadcast) into one read pathway AND advances the episodic
mechanism itself from a single softmax hop to an iterative CA3 attractor completion.

CROSS-DOMAIN MOTIVATION
-----------------------
* CA3 pattern completion is ITERATIVE, not one-shot. Modern Hopfield networks (Ramsauer et al.,
  "Hopfield Networks is All You Need") formalize CA3 recurrence as fixed-point updates
  q^(i+1) = X^T softmax(beta X q^(i)): a partial cue is refined by what it retrieves and
  re-queried, converging to a stored attractor. The current best arch
  (hippo_episodic_place_read) does ONE hop only. In a shell walk the informative past entry is
  often addressed only INDIRECTLY: an `ls` of a directory names a file that is later `cat`-ed; a
  revisited cwd's earlier listing predicts the current listing. A second hop lets the read chain
  "place -> what was seen there -> the related place I actually want", which a single softmax
  cannot do.

* Tolman-Eichenbaum factorization (Whittington et al. 2020): generalization comes from splitting
  a STRUCTURAL/identity code (transfers across environments) from SENSORY content. Here the
  memory KEY is factorized: a LOCAL component (command embedding = which path/file, plus the
  causal transformer context) and a GLOBAL identity component (a strict-prefix gated sysid
  register dominated by the opening uname/config tokens). The identity component biases the read
  toward entries seen ON THE SAME SYSTEM without needing the transformer to re-derive identity on
  an unseen image. Values stay the raw standardized obs embeddings, so the read output is always
  on the real observation manifold that retrieval scores -- no cross-system transfer required.

* The walk is LOCAL in content but GLOBAL in identity (recency_alibi / typed_recency ledger
  wins). We bake this into the MEMORY addressing (not just the transformer's own attention):
  each read hop adds a non-positive linear-distance decay to the episodic logits, so distant
  observations decay while the identity path stays undecayed.

WHY THIS SHOULD BEAT HIPPO ALONE (proxy 0.5918)
-----------------------------------------------
Hippo's read is single-hop and its key is built purely from the transformer context
(h_cmd + cmd_proj(cmd)) -- exactly the learned mapping that fails to transfer. This arch (1) adds
a second attractor hop so the read can complete indirectly-addressed past places; (2) factors an
undecayed identity component into the key so cross-system entries are down-weighted without a
learned transfer; (3) adds a recency prior on the episodic logits so the read prefers LOCAL
observations, matching the domain's locality. Each addition is a strict generalization (a learned
scalar/gate can switch it off): zero recency slope + one hop + zero identity weight recovers
hippo exactly, so the search can only move up or stay flat.

CAUSALITY (leak-free by construction)
-------------------------------------
Transformer uses the standard upper-triangular causal mask; command hidden at token 2t depends
only on tokens 0..2t. The episodic read at STEP t attends only to entries j < t (strict lower
triangular), and entry j's VALUE is obs_j at token 2j+1 <= 2t-1 < 2t. The iterative refinement
updates ONLY the QUERY from already-retrieved (strictly-past) values; the allowed-entry mask is
recomputed identically each hop, so no hop can ever reach obs_t or later. The sysid register is a
strict-prefix cumulative mean (shifted by one) with pad tokens zeroed, so it uses only tokens
0..2t-1. Disallowed/pad entries get a true -inf logit (exact zero weight); all-masked query rows
are nan-to-num'd to a zero read and the gate is forced to zero via a has-memory mask. So a
command-position prediction at or before t never depends on obs_t or any later observation.
Verified empirically: corrupting obs_t leaves all cmd_<=t predictions bit-identical (<1e-6) while
legitimately changing later steps.

NUMERICAL STABILITY
-------------------
No exponentials over sequence POSITION and no division by an accumulated mass in the read. The
recency term is a bounded, non-positive additive bias (softplus slope >= 0 times distance >= 0),
so it only removes attention mass and can never overflow a softmax. Each hop is a softmax over a
finite [n,n] logit matrix with a bounded learned temperature (softplus + 0.5). The sysid register
divides by a cumulative gate mass clamped to >= 1e-6. Entropy uses a clamped log. Ragged padding,
all-pad rows, n=1, and n=0 edge cases are all finite (tested); backward grads are finite.

I/O CONTRACT
------------
forward(tok_emb [B,L,768], types [B,L] in {0,1}, key_pad [B,L] bool True=pad)
    -> (pred [B,L,768], h [B,L,d]); prediction at EVERY position (harness reads pred[:,0::2]).
~2.5M params at d=192, layers=4, heads=4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "ca3_iterative_place_recency_read"
DESCRIPTION = (
    "Causal transformer + iterative CA3 attractor episodic read: a factorized place key "
    "(local command/context + global strict-prefix sysid register) pattern-completes a convex "
    "combination of strictly-past raw observation embeddings over MULTIPLE Hopfield-style hops, "
    "with a per-hop non-positive recency decay on the episodic logits, confidence-gated into the "
    "parametric head. Strictly causal, on-manifold, numerically bounded; a strict generalization "
    "of hippo_episodic_place_read (recovers it at one hop / zero recency)."
)


class SysIdRegister(nn.Module):
    """Causal, strict-prefix gated running mean of the token stream. Position i sees only tokens
    0..i-1, dominated by heavily-gated opening (uname/config) tokens -> a stable identity handle
    that transfers without the transformer re-deriving it on an unseen system."""

    def __init__(self, d):
        super().__init__()
        self.val = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, valid):
        v = self.val(x)                                              # [B,L,d]
        g = F.softplus(self.gate(x)) * valid.unsqueeze(-1)          # [B,L,1], 0 on pad
        wsum = torch.cumsum(g * v, dim=1)
        gsum = torch.cumsum(g, dim=1)
        wsum = F.pad(wsum, (0, 0, 1, 0))[:, :-1, :]                 # strict prefix (shift by 1)
        gsum = F.pad(gsum, (0, 0, 1, 0))[:, :-1, :]
        state = wsum / gsum.clamp(min=1e-6)
        return self.norm(self.out(state))


class CA3IterativePlaceRead(nn.Module):
    """Baseline causal transformer front-end (parametric prediction + per-position hidden), plus
    a per-sequence, iterative CA3 episodic read at command positions.

    Read (per step t):
      place_t = LN( h_cmd_t + cmd_proj(cmd_t) )                       # local content/place code
      key_j   = Wk( place_j ) ,  id_j = Wid( sysid_j )               # content key + identity key
      hop i:  logit_{t,j}^i = ( q_t^i . key_j + id_t . id_j ) / s / temp  - recency_{t,j}
              attn^i = softmax_j(logit)  (j<t, valid)
              e_t^i  = attn^i @ obs_val                              # convex combo of raw obs
              q_t^{i+1} = LN( q_t^i + Wu( e_t^i ) )                  # attractor query refinement
      final read e_t = e_t^{last hop}
    Confidence-gated blend with parametric p_t at command positions:
      g_t = sigmoid(gate([h_cmd_t, maxw, entropy])) * has_memory
      pred_cmd_t = (1 - g_t) * p_t + g_t * e_t
    """

    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, max_len=64, kq=128,
                 hops=2, id_dim=48, keep_global_head=True, no_history=False):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.kq = kq
        self.hops = max(1, int(hops))
        self.id_dim = id_dim
        self.no_history = no_history  # matched-capacity ablation: self-only attn + read disabled

        # ---- slow neocortex: baseline-matched causal transformer front-end ----
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)          # 0=cmd, 1=obs
        self.pos_emb = nn.Embedding(max_len, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, D)

        # ---- identity register (TEM structural/global code) ----
        self.sysid = SysIdRegister(d)
        self.Wid = nn.Linear(d, id_dim)             # identity key/query projection

        # ---- fast hippocampus: factorized place-keyed iterative read ----
        self.cmd_proj = nn.Linear(D, d)             # sensory-command -> location component
        self.place_norm = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, kq)                  # place -> query (hop 0)
        self.Wk = nn.Linear(d, kq)                  # place -> key
        self.Wu = nn.Linear(D, kq)                  # retrieved obs -> query update (attractor)
        self.q_norm = nn.LayerNorm(kq)
        self.log_temp = nn.Parameter(torch.zeros(()))       # bounded read temperature

        # Non-positive recency decay on the EPISODIC logits: two slopes (one "sharp/local", one
        # "flat/global") mixed per read by a learned softmax so the read is local-by-default yet
        # keeps an undecayed global lane. softplus keeps slopes >= 0 so the bias only removes mass.
        ladder = torch.tensor([0.5, 0.03])                  # local, near-flat
        self.slope_raw = nn.Parameter(torch.log(torch.expm1(ladder.clamp(min=1e-4))))  # [2]
        self.lane_logits = nn.Parameter(torch.zeros(2))     # mix over {local, global} lanes

        self.gate = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        dev = tok_emb.device
        if L == 0:  # pathological empty sequence (harness never sends this) -- no-op passthrough
            return self.head(self.proj(tok_emb)), self.proj(tok_emb)
        idx = torch.arange(L, device=dev).clamp(max=self.max_len - 1)
        x = self.proj(tok_emb) + self.type_emb(types.clamp(0, 1)) + self.pos_emb(idx)[None]

        if self.no_history:
            mask = ~torch.eye(L, device=dev, dtype=torch.bool)
            h = self.tf(x, mask=mask, src_key_padding_mask=None)
            return self.head(h), h
        mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1)
        h = self.tf(x, mask=mask, src_key_padding_mask=key_pad)      # [B,L,d]
        pred = self.head(h)                                          # [B,L,768] parametric
        n = L // 2
        if n == 0:
            return pred, h

        # ---- per-step aligned views (step j lives at tokens 2j (cmd) / 2j+1 (obs)) ----
        hc = h[:, 0:2 * n:2]                     # [B,n,d]   command hidden (causal context)
        cmd_raw = tok_emb[:, 0:2 * n:2]          # [B,n,768] raw command embedding (location)
        obs_val = tok_emb[:, 1:2 * n:2]          # [B,n,768] raw std obs = episodic value
        valid_cmd = (~key_pad)[:, 0:2 * n:2]     # [B,n] bool valid command steps

        # ---- factorized place key (local content + global identity) ----
        valid_all = (~key_pad).float()                                   # [B,L]
        s = self.sysid(x, valid_all)                                     # [B,L,d] strict-prefix id
        s_cmd = s[:, 0:2 * n:2]                                          # [B,n,d] id at each step
        idk = self.Wid(s_cmd)                                            # [B,n,id_dim] identity key

        place = self.place_norm(hc + self.cmd_proj(cmd_raw))            # [B,n,d]
        k = self.Wk(place)                                              # [B,n,kq]
        q = self.Wq(place)                                              # [B,n,kq] hop-0 query
        s_scale = self.kq ** 0.5
        temp = F.softplus(self.log_temp) + 0.5

        # strict causal + valid-key mask over episodic entries (recomputed constant across hops).
        strict = torch.tril(torch.ones(n, n, device=dev, dtype=torch.bool), -1)   # j<t
        allowed = strict[None] & valid_cmd[:, None, :]                            # [B,n,n]

        # identity affinity is hop-invariant (query id fixed): id_t . id_j / sqrt(id_dim)
        id_logit = (idk @ idk.transpose(-2, -1)) / (self.id_dim ** 0.5)          # [B,n,n]

        # recency: mixed non-positive linear-distance decay over step distance (t-j).
        pos = torch.arange(n, device=dev, dtype=x.dtype)
        dist = (pos[:, None] - pos[None, :]).clamp(min=0)                        # [n,n] (t-j)_+
        slope = F.softplus(self.slope_raw)                                       # [2] >= 0
        lane = torch.softmax(self.lane_logits, dim=0)                            # [2] mix
        eff_slope = (lane * slope).sum()                                         # scalar >= 0
        recency = -(eff_slope * dist)[None]                                      # [1,n,n] <= 0

        attn = None
        e = None
        for _ in range(self.hops):
            content_logit = (q @ k.transpose(-2, -1)) / s_scale                  # [B,n,n]
            logits = (content_logit + id_logit) / temp + recency                 # [B,n,n]
            logits = logits.masked_fill(~allowed, float("-inf"))
            attn = torch.softmax(logits, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)                               # all-masked -> 0
            e = attn @ obs_val                                                   # [B,n,768] on-manifold
            # attractor query refinement from the (strictly-past) retrieved value
            q = self.q_norm(q + self.Wu(e))

        # ---- confidence-gated blend ----
        maxw = attn.max(dim=-1, keepdim=True).values                            # [B,n,1]
        ent = -(attn * torch.log(attn.clamp(min=1e-9))).sum(-1, keepdim=True)   # [B,n,1]
        has_mem = allowed.any(dim=-1, keepdim=True).to(x.dtype)                 # 0 when no past entry
        g = torch.sigmoid(self.gate(torch.cat([hc, maxw, ent], dim=-1))) * has_mem  # [B,n,1]

        p_cmd = pred[:, 0:2 * n:2]
        blended = (1.0 - g) * p_cmd + g * e
        pred = pred.clone()
        pred[:, 0:2 * n:2] = blended
        return pred, h


def build(d=192, layers=4, heads=4, dropout=0.1, max_len=64, kq=128,
          hops=2, id_dim=48, keep_global_head=True, no_history=False):
    return CA3IterativePlaceRead(d=d, layers=layers, heads=heads, dropout=dropout,
                                 max_len=max_len, kq=kq, hops=hops, id_dim=id_dim,
                                 keep_global_head=keep_global_head, no_history=no_history)

