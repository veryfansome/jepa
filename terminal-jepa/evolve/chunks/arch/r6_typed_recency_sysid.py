"""Typed-recency ALiBi transformer with an explicit system-identity register.

A single arch that UNIFIES the three independently-helpful mechanisms recorded in the
ledger — per-head ALiBi recency decay (recency_alibi_perhead), a system-identity
broadcast pathway (sysid), and typed cmd/obs routing (stateroute) — by fusing the first
and third into ONE new primitive, TYPED-RECENCY attention, and pairing it with an
explicit strict-prefix identity register.

WHY (mechanism)
---------------
A shell exploration is a walk whose next observation depends on TWO very different
timescales that split cleanly along token TYPE:
  * OBSERVATION tokens (listings / file contents / cwd) are LOCAL — a distant `ls`
    output is irrelevant to the current command's result; their relevance decays fast.
  * COMMAND tokens carry IDENTITY and CONTEXT — the opening `uname`/config-`cat`
    commands fix "which system am I on" and should stay reachable UNDECAYED across the
    whole walk.
Plain ALiBi gives each head ONE slope over ALL keys, so a head must choose to be either
local or global; it cannot be "local over observations but global over commands", which
is exactly the routing this domain wants. This arch generalizes ALiBi to a per-head,
per-KEY-TYPE slope softplus(slope_h[type_j]) and adds a per-head, per-key-type additive
LEVEL bias beta_h[type_j]. In one mechanism this subsumes: recency (per-head decay),
typed routing (decay + level depend on cmd vs obs), and an identity pathway (a head can
learn slope_h[cmd] -> 0, keeping opening command tokens on an undecayed lane). It is a
strict generalization: equal per-type slopes recover per-head ALiBi, zero slopes recover
the baseline, so the data can switch any part off if it does not transfer.

On top of the attention, an explicit SYSTEM-IDENTITY REGISTER (a strict-prefix gated
cumulative mean of the token stream, dominated by the heavily-gated opening tokens) is
injected additively at the input and re-fused via a gate after the encoder. This gives a
low-variance, attention-independent handle on system identity that does not rely on the
transformer learning the right slope on an UNSEEN image — the biology motivation is the
entorhinal split between a fast local-trajectory code (MEC grid/place) and a slow
context/identity code (LEC), feeding one predictive map.

NUMERICAL STABILITY / CAUSALITY
-------------------------------
No exponentials over sequence position anywhere (unlike the diverged EMA-bank arch). The
recency term is a bounded, NON-POSITIVE additive bias: slopes are softplus (>=0) times
the non-negative distance clamp(i-j, min=0), so it only removes attention mass. The level
term beta_h[type_j] is a finite learned scalar added to logits; softmax over finite
logits cannot overflow. Attention uses the standard upper-triangular causal mask (future
keys = -inf); the typed bias depends only on key j's TYPE ID (known for j<=i and
independent of any observation value), so a command position 2t attends to tokens 0..2t
only — never its own observation 2t+1. The identity register is a strict-prefix cumulative
mean (shifted by one) with pad tokens zeroed, so position i sees only tokens 0..i-1.
Fully-padded query rows are guarded to a uniform row so softmax never returns NaN. The
per-genome no-leakage guard therefore passes by construction.

I/O CONTRACT
------------
forward(tok_emb [B,L,768], types [B,L] in {0,1}, key_pad [B,L] bool True=pad)
    -> (pred [B,L,768], h [B,L,d]); prediction at EVERY position (harness reads pred[:,0::2]).
~2.3M params at d=192, layers=4, heads=4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D_IN = 768

NAME = "typed_recency_sysid"
DESCRIPTION = (
    "Causal transformer unifying per-head ALiBi recency, typed cmd/obs routing, and a "
    "system-identity register: TYPED-RECENCY attention (per-head, per-key-type slope + "
    "level bias) plus an explicit strict-prefix gated identity state injected and gated-fused."
)


class SysIdRegister(nn.Module):
    """Causal, strict-prefix gated running mean of the token stream. For each position i,
    produces a state built ONLY from tokens 0..i-1, dominated by heavily-gated opening
    (uname/config) tokens -> a stable identity handle."""

    def __init__(self, d):
        super().__init__()
        self.val = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, valid):
        v = self.val(x)                                                # [B,L,d]
        g = F.softplus(self.gate(x)) * valid.unsqueeze(-1)             # [B,L,1], 0 on pad
        wsum = torch.cumsum(g * v, dim=1)
        gsum = torch.cumsum(g, dim=1)
        wsum = F.pad(wsum, (0, 0, 1, 0))[:, :-1, :]                    # strict prefix
        gsum = F.pad(gsum, (0, 0, 1, 0))[:, :-1, :]
        state = wsum / gsum.clamp(min=1e-6)
        return self.norm(self.out(state))


class TypedRecencyLayer(nn.Module):
    """Pre-norm encoder layer whose self-attention adds, per head, a per-KEY-TYPE ALiBi
    recency decay -softplus(slope_h[type_j])*(i-j) plus a per-key-type level bias
    beta_h[type_j]. Explicit SDPA (not nn.MultiheadAttention) to inject the typed bias."""

    def __init__(self, d, heads, dropout, mlp_mult=4, keep_global_head=True):
        super().__init__()
        assert d % heads == 0, f"d ({d}) must be divisible by heads ({heads})"
        self.d, self.heads, self.dh = d, heads, d // heads

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d), nn.GELU(),
            nn.Linear(mlp_mult * d, d), nn.Dropout(dropout),
        )

        # Per-head, per-key-type recency slope: effective slope = softplus(raw) >= 0.
        # Ladder spread across heads (ALiBi-style), duplicated over the two key types so
        # each type starts at diverse timescales; the two columns then diverge in training.
        ladder = torch.logspace(start=-3.0, end=0.0, steps=heads, base=2.0)   # 1/8 .. 1
        raw0 = torch.log(torch.expm1(ladder.clamp(min=1e-4)))                 # softplus^{-1}
        self.slope_raw = nn.Parameter(raw0[:, None].repeat(1, 2))             # [heads,2]

        # Per-head, per-key-type additive LEVEL bias (routing: prefer cmd vs obs keys).
        self.type_bias = nn.Parameter(torch.zeros(heads, 2))                  # [heads,2]

        # Optionally reserve head 0 as an undecayed GLOBAL head (both type slopes forced 0)
        # so an identity path always exists even if every learned slope goes positive.
        gmask = torch.ones(heads, 1)
        if keep_global_head and heads >= 2:
            gmask[0, 0] = 0.0
        self.register_buffer("slope_keep", gmask)                            # [heads,1]

    def _typed_bias(self, types, L, dtype):
        """[B,heads,L,L] additive bias; entry (b,h,i,j) = beta_h[t_j] - slope_h[t_j]*(i-j)_+
        with t_j = types[b,j]. slope >= 0 so the decay term is <= 0."""
        dev = types.device
        slope = F.softplus(self.slope_raw) * self.slope_keep                 # [heads,2] >=0
        level = self.type_bias                                               # [heads,2]
        t = types.clamp(0, 1)                                                # [B,L]
        slope_key = slope[:, t].permute(1, 0, 2)                             # [B,heads,L]
        level_key = level[:, t].permute(1, 0, 2)                             # [B,heads,L]
        pos = torch.arange(L, device=dev, dtype=dtype)
        dist = (pos[:, None] - pos[None, :]).clamp(min=0)                    # [L,L] (i-j)_+
        bias = level_key[:, :, None, :] - slope_key[:, :, None, :] * dist[None, None]
        return bias.to(dtype)                                               # [B,heads,L,L]

    def _sdpa(self, x, types, key_pad):
        B, L, _ = x.shape
        H, dh = self.heads, self.dh
        q = self.q(x).view(B, L, H, dh).transpose(1, 2)
        k = self.k(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v(x).view(B, L, H, dh).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / (dh ** 0.5)                     # [B,H,L,L]
        logits = logits + self._typed_bias(types, L, logits.dtype)

        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(causal[None, None], float("-inf"))
        if key_pad is not None:
            logits = logits.masked_fill(key_pad[:, None, None, :], float("-inf"))
            dead = torch.isneginf(logits).all(dim=-1, keepdim=True)
            logits = logits.masked_fill(dead, 0.0)                          # guard pad rows

        attn = self.attn_drop(torch.softmax(logits, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.d)
        return self.o(out)

    def forward(self, x, types, key_pad):
        x = x + self.resid_drop(self._sdpa(self.norm1(x), types, key_pad))
        x = x + self.mlp(self.norm2(x))
        return x


class TypedRecencySysId(nn.Module):
    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, max_len=64,
                 keep_global_head=True, use_sysid=True):
        super().__init__()
        self.d, self.max_len, self.use_sysid = d, max_len, use_sysid
        self.proj = nn.Linear(D_IN, d)
        self.type_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Embedding(max_len, d)
        if use_sysid:
            self.sysid = SysIdRegister(d)
            self.fuse_gate = nn.Linear(2 * d, d)
            self.fuse_val = nn.Linear(d, d)
        self.layers = nn.ModuleList([
            TypedRecencyLayer(d, heads, dropout, keep_global_head=keep_global_head)
            for _ in range(layers)
        ])
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, D_IN)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        dev = tok_emb.device
        types = types.clamp(0, 1)
        idx = torch.arange(L, device=dev).clamp(max=self.max_len - 1)
        x = self.proj(tok_emb) + self.type_emb(types) + self.pos_emb(idx)[None]

        s = None
        if self.use_sysid:
            valid = (~key_pad).float() if key_pad is not None else torch.ones(B, L, device=dev)
            s = self.sysid(x, valid)                                        # strict-prefix state
            x = x + s

        for layer in self.layers:
            x = layer(x, types, key_pad)

        h = self.norm_f(x)
        if self.use_sysid:
            gate = torch.sigmoid(self.fuse_gate(torch.cat([h, s], dim=-1)))
            h = h + gate * self.fuse_val(s)
        if key_pad is not None:
            h = h * (~key_pad).unsqueeze(-1).to(h.dtype)
        return self.head(h), h


def build(d=192, layers=4, heads=4, dropout=0.1, max_len=64,
          keep_global_head=True, use_sysid=True):
    return TypedRecencySysId(d=d, layers=layers, heads=heads, dropout=dropout,
                             max_len=max_len, keep_global_head=keep_global_head,
                             use_sysid=use_sysid)

