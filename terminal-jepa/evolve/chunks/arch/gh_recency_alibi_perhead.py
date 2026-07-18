"""Recency-biased causal transformer: per-head learned linear-distance (ALiBi-style)
attention decay, so nearer history is weighted more when previewing the next observation.

WHY recency should help on UNSEEN systems
------------------------------------------
A shell exploration is a walk: `cd A; ls; cd B; ls; cat f`. The observation a command
produces depends far more on the *local* state (the current working directory, the file
just cd'd into, the directory just listed) than on distant early tokens — with ONE
exception: the system identity, fixed by the opening `uname`/config tokens. Vanilla
self-attention has a flat prior over distance: every past token is a-priori equally
reachable, and the model must *learn* from data which distances matter. On unseen images
that learned, content-specific distance routing is exactly what fails to transfer.

This arch bakes the walk's locality into the ARCHITECTURE as a distance prior, the way
ALiBi bakes recency into language models: each head adds a bias `-m_h * (i - j)` to the
attention logit for a query at position i attending to key at position j (j <= i, strictly
causal). `m_h >= 0` is a LEARNED per-head slope, so different heads adopt different
timescales — some sharp/local (large slope, "what's my cwd right now"), some nearly flat
(slope ~ 0, "what system am I on"). The slopes are the only added recency machinery; the
flat-slope heads recover the baseline, so this is a strict generalization that the data can
switch off if recency does not help. No summary state, no EMA, no recurrence — just a bias
added to the existing attention, which is why it stays cheap and numerically trivial.

An OPTIONAL, off-by-default global lane leaves one head unbiased (slope forced to 0) so the
system-identity signal always has an undecayed path even if training drives every learned
slope positive.

NUMERICAL STABILITY (contrast with the diverged EMA-bank arch)
--------------------------------------------------------------
The prior NaN arch renormalized log-space cumulative EMAs with paired exp(+.)/exp(-.)
branches over sequence position; in float those leave inf*0 / inf/inf residue that a
min-clamp cannot catch. This arch introduces NO exponentials over position. The recency
term is a bounded, NON-POSITIVE additive bias on the attention logits: slopes are made
non-negative with softplus and multiplied by the non-negative distance (i - j), so the
bias is <= 0 and only ever *removes* attention mass. Adding a <= 0 bias to softmax logits
can never overflow; combined with the causal mask (future = -inf), each query row always
retains at least its own valid key, so softmax is well-defined at every position. There is
no division by an accumulated quantity anywhere.

CAUSALITY
---------
Attention uses the standard upper-triangular causal mask (future keys = -inf). The recency
bias is only defined and applied for j <= i (it is added under the same mask), so a command
position 2t attends to tokens 0..2t only — never its own observation 2t+1 or later. Padding
is removed via key_padding_mask. The per-genome no-leakage guard (corrupt obs_t, check
cmd_<=t predictions do not move) therefore passes by construction.

I/O CONTRACT
------------
forward(tok_emb [B,L,768], types [B,L] in {0,1}, key_pad [B,L] bool True=pad)
    -> (pred [B,L,768], h [B,L,d]); prediction at EVERY position (harness reads pred[:,0::2]).
~2-3M params at d=192, layers=4, heads=4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D_IN = 768

NAME = "recency_alibi_perhead"
DESCRIPTION = (
    "Causal transformer with a learned per-head ALiBi-style linear-distance attention "
    "decay (nearer history weighted more), an optional undecayed global head for the "
    "system-identity path; strictly causal, numerically trivial (bounded non-positive bias)."
)


class RecencyBiasedEncoderLayer(nn.Module):
    """A pre-norm Transformer encoder layer whose self-attention adds a per-head, learned,
    non-positive linear-distance bias -softplus(slope_h) * (i - j) to the attention logits.

    Implemented with an explicit scaled-dot-product attention (not nn.MultiheadAttention)
    because we need to inject a per-head [L,L] additive bias, which the fused module does not
    expose cleanly. The math is the standard SDPA; only the additive mask carries the bias.
    """

    def __init__(self, d, heads, dropout, mlp_mult=4, keep_global_head=True):
        super().__init__()
        assert d % heads == 0, f"d ({d}) must be divisible by heads ({heads})"
        self.d = d
        self.heads = heads
        self.dh = d // heads
        self.keep_global_head = keep_global_head

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

        # Per-head recency slope, parameterized so the effective slope = softplus(raw) >= 0.
        # Init spread across an ALiBi-like geometric ladder (converted to raw-space) so heads
        # start at diverse timescales: some near-flat (~global), some sharply local. softplus
        # is ~monotone so ordering of the ladder is preserved; exact values are learnable.
        ladder = torch.logspace(start=-3.0, end=0.0, steps=heads, base=2.0)  # 1/8 .. 1
        raw0 = torch.log(torch.expm1(ladder.clamp(min=1e-4)))                # softplus^{-1}
        self.slope_raw = nn.Parameter(raw0)

        # Optionally force head 0 to be a permanently-undecayed "global identity" lane by
        # zeroing its slope (registered as a buffer mask, not a parameter).
        gmask = torch.ones(heads)
        if keep_global_head and heads >= 2:
            gmask[0] = 0.0
        self.register_buffer("slope_keep", gmask)

    def _recency_bias(self, L, device, dtype):
        """[heads, L, L] additive bias: entry (h,i,j) = -slope_h * (i - j) for j <= i, else 0
        (the 0 there is irrelevant: the causal mask sets those positions to -inf). slope_h >= 0
        so the bias is <= 0 for valid (past) positions."""
        pos = torch.arange(L, device=device, dtype=dtype)
        dist = pos[:, None] - pos[None, :]          # [L,L]; i - j (>=0 on/below diagonal)
        dist = dist.clamp(min=0)                     # only past distances carry bias
        slope = F.softplus(self.slope_raw) * self.slope_keep  # [heads] >= 0, head0 maybe 0
        return -(slope[:, None, None] * dist[None, :, :]).to(dtype)   # [heads, L, L] <= 0

    def _sdpa(self, x, key_pad):
        B, L, _ = x.shape
        H, dh = self.heads, self.dh
        q = self.q(x).view(B, L, H, dh).transpose(1, 2)  # [B,H,L,dh]
        k = self.k(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v(x).view(B, L, H, dh).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / (dh ** 0.5)  # [B,H,L,L]

        # Recency bias (per head), broadcast over batch.
        logits = logits + self._recency_bias(L, x.device, logits.dtype)[None]

        # Causal mask: forbid attending to future (j > i).
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(causal[None, None], float("-inf"))

        # Padding mask: forbid attending to pad keys. Guard fully-pad rows so a padded QUERY
        # (whose output is discarded downstream) never yields a NaN softmax over an all-inf row.
        if key_pad is not None:
            kp = key_pad[:, None, None, :]                       # [B,1,1,L] True=pad key
            logits = logits.masked_fill(kp, float("-inf"))
            all_masked = torch.isneginf(logits).all(dim=-1, keepdim=True)
            logits = logits.masked_fill(all_masked, 0.0)         # uniform over that dead row

        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                           # [B,H,L,dh]
        out = out.transpose(1, 2).reshape(B, L, self.d)
        return self.o(out)

    def forward(self, x, key_pad):
        x = x + self.resid_drop(self._sdpa(self.norm1(x), key_pad))
        x = x + self.mlp(self.norm2(x))
        return x


class RecencyTransformer(nn.Module):
    """Baseline projection/type/pos embedding front-end (matching SeqWorldModel), then a stack
    of recency-biased causal encoder layers, then a linear head back to 768-d."""

    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, max_len=64,
                 keep_global_head=True):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.proj = nn.Linear(D_IN, d)
        self.type_emb = nn.Embedding(2, d)          # 0=cmd, 1=obs
        self.pos_emb = nn.Embedding(max_len, d)      # absolute pos kept: recency is RELATIVE,
        #                                              absolute helps mark the opening tokens.
        self.layers = nn.ModuleList([
            RecencyBiasedEncoderLayer(d, heads, dropout, keep_global_head=keep_global_head)
            for _ in range(layers)
        ])
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, D_IN)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        dev = tok_emb.device
        idx = torch.arange(L, device=dev).clamp(max=self.max_len - 1)
        x = self.proj(tok_emb) + self.type_emb(types.clamp(0, 1)) + self.pos_emb(idx)[None]

        for layer in self.layers:
            x = layer(x, key_pad)

        h = self.norm_f(x)
        # Zero pad positions so a downstream reduction never mixes in garbage (harness only
        # reads valid cmd positions, but this keeps the output clean and finite).
        if key_pad is not None:
            h = h * (~key_pad).unsqueeze(-1).to(h.dtype)
        return self.head(h), h


def build(d=192, layers=4, heads=4, dropout=0.1, max_len=64, keep_global_head=True):
    return RecencyTransformer(d=d, layers=layers, heads=heads, dropout=dropout,
                              max_len=max_len, keep_global_head=keep_global_head)
