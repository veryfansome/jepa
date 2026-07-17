"""Hierarchical bounded-context predictor: a compact strict-prefix system-summary
token (persistent identity) + band-limited local causal attention (local cmd->obs
dynamics), fused by a gated residual.

Motivation (HISTORY/CONTEXT dimension). A shell exploration factorizes into two very
different pieces of information:
  (1) a PERSISTENT system identity fixed by the opening tokens (uname + a cat of a
      config file) — low-variance, needed at every later step, and the thing that must
      TRANSFER to unseen images; and
  (2) LOCAL command->observation dynamics — how *this* ls/cat/cd, given the current cwd
      and the last couple of steps, maps to its observation.
Vanilla full causal self-attention entangles these: it must re-derive identity through
soft attention over a growing, mostly-irrelevant history at every layer/position, and its
attention budget for the local dynamics is diluted by the long prefix. On UNSEEN systems
that re-derivation is exactly what fails to transfer.

This arch structures the history explicitly and hierarchically:
  * A compact GLOBAL SUMMARY s_i is a strict-prefix, softmax-normalized attention pool of
    all tokens 0..i-1 toward a learned identity query, with a learned position prior that
    favors the opening tokens. It is a running "which system am I on" handle, broadcast to
    every position. Being a SOFTMAX-normalized convex combination, it is intrinsically
    bounded (weights sum to 1) — no cumsum/mass ratios that can explode.
  * The token mixer is a stack of BAND-LIMITED causal attention blocks: each position may
    attend only to itself and a bounded look-back window of the previous `window` STEPS
    (2*window tokens). This concentrates capacity on local cmd->obs dynamics and makes the
    per-step computation independent of how long the exploration has run so far.
  * A gated residual re-injects the global summary after the local blocks, so identity
    conditions the local dynamics without the local blocks having to re-derive it.

Causality. The summary at position i is a STRICT-prefix cumulative softmax (numerator and
denominator both inclusive-cumsum then shifted by one), so command position 2t sees only
tokens 0..2t-1 — never its own observation (2t+1) or anything later. The local attention
uses a banded upper-triangular mask (a subset of the causal mask). Padding is removed from
the summary weights (masked to -inf before the running softmax) and from attention (key_pad).

Numerical stability (explicitly avoiding the g3_recomb NaN mode). g3_recomb blew up because
its multi-timescale EMA divided a rescaled cumulative sum by a `mass` that could be tiny for
slow-decay lanes early in the sequence, then re-inflated by exp(+m) — an unbounded ratio that
InfoNCE amplified to inf. Here EVERY history aggregation is a bounded convex combination:
  - the global summary uses a running softmax (weights in [0,1], sum to 1) computed with a
    running (inclusive cummax) max-subtraction, so the exponentials never overflow and the
    denominator is >= the largest single weight at every valid position;
  - the local blocks are ordinary softmax attention with a finite banded mask.
There is no raw-cumsum-over-tiny-mass division anywhere, and no learned decay in log-space.
Everything is standard pre-norm residual arithmetic.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "hier_window_sysid"
DESCRIPTION = (
    "Hierarchical bounded context: strict-prefix softmax-pooled system-summary token "
    "(persistent identity, position-biased to the opening tokens) + band-limited local "
    "causal attention (local cmd->obs dynamics), fused by a gated residual. Strictly "
    "causal and numerically bounded (all history aggregations are convex combinations)."
)


class PrefixSummary(nn.Module):
    """Strict-prefix, softmax-normalized attention pool toward a learned identity query.

    For each position i, produces s_i = sum_{j<i} w_ij v_j where w_ij is a softmax over the
    strict prefix of a per-token score (learned query . key) plus a position prior that biases
    toward EARLY tokens (the uname/config opener that fixes system identity). Computed in
    O(L) with a running (online-softmax) cumulative sum — no L*L attention matrix — and with a
    running max subtraction so the exponentials are bounded. Being a convex combination, the
    output magnitude is bounded by the token magnitudes; there is no dividing-by-tiny-mass path.
    """

    def __init__(self, d):
        super().__init__()
        self.key = nn.Linear(d, d)
        self.val = nn.Linear(d, d)
        self.q = nn.Parameter(torch.zeros(d))       # learned identity query
        self.scale = 1.0 / math.sqrt(d)
        # position prior: score bonus decaying with position -> favors opening tokens.
        self.log_lam = nn.Parameter(torch.tensor(0.0))   # softplus -> nonneg decay rate
        self.prior_gain = nn.Parameter(torch.tensor(1.0))
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, valid):
        # x: [B,L,d]; valid: [B,L] float (1=real, 0=pad)
        B, L, d = x.shape
        dev = x.device
        k = self.key(x)                                     # [B,L,d]
        v = self.val(x)                                     # [B,L,d]
        score = (k * self.q[None, None, :]).sum(-1) * self.scale   # [B,L] q.k per token
        pos = torch.arange(L, device=dev, dtype=x.dtype)
        lam = F.softplus(self.log_lam)
        prior = self.prior_gain * torch.exp(-lam * pos)[None, :]   # [1,L] early-token bonus
        logit = score + prior                                      # [B,L]
        # mask pad positions out of the pool (contribute zero weight)
        logit = logit.masked_fill(valid < 0.5, float("-inf"))
        # numerically-stable running softmax over the INCLUSIVE prefix, then shift for strict.
        # running max via cummax keeps every exponential in (0,1].
        run_max, _ = torch.cummax(logit, dim=1)                    # [B,L] max over 0..i
        # tokens after a pad-only prefix have run_max=-inf; guard so exp is finite (weight 0).
        safe_max = torch.where(torch.isfinite(run_max), run_max, torch.zeros_like(run_max))
        w = torch.exp(logit - safe_max)                            # [B,L] in [0,1], 0 on pad
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        num = torch.cumsum(w[..., None] * v, dim=1)                # [B,L,d] inclusive
        den = torch.cumsum(w, dim=1).clamp(min=1e-6)[..., None]    # [B,L,1] inclusive
        pooled = num / den                                        # [B,L,d] convex combo (bounded)
        # strict prefix: position i must see only 0..i-1
        pooled = F.pad(pooled, (0, 0, 1, 0))[:, :-1, :]
        return self.norm(self.out(pooled))


class BandedCausalBlock(nn.Module):
    """Pre-norm self-attention restricted to a bounded look-back window + SwiGLU FFN.

    Each position attends to itself and the previous `span` tokens (span = 2*window steps),
    a strict SUBSET of the causal mask, so it remains leak-free while concentrating capacity
    on local cmd->obs dynamics. Standard softmax attention -> bounded weights.
    """

    def __init__(self, d, heads, span, dropout=0.1):
        super().__init__()
        if d % heads != 0:
            heads = next((h for h in range(min(heads, d), 0, -1) if d % h == 0), 1)
        self.span = span
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        inner = 4 * d
        self.ff1 = nn.Linear(d, 2 * inner)
        self.ff2 = nn.Linear(inner, d)
        self.drop = nn.Dropout(dropout)

    def _banded_mask(self, L, dev):
        idx = torch.arange(L, device=dev)
        # allow j in [i-span, i]: causal (j<=i) AND within window (i-j<=span)
        future = idx[None, :] > idx[:, None]
        too_far = (idx[:, None] - idx[None, :]) > self.span
        return future | too_far                              # True = blocked

    def forward(self, x, key_pad):
        B, L, d = x.shape
        mask = self._banded_mask(L, x.device)
        xn = self.n1(x)
        y, _ = self.attn(xn, xn, xn, attn_mask=mask,
                         key_padding_mask=key_pad, need_weights=False)
        # a fully-masked row (all keys pad) can yield NaN; zero it (padded positions are dropped
        # downstream by the valid mask anyway).
        y = torch.nan_to_num(y, nan=0.0)
        x = x + self.drop(y)
        h = self.n2(x)
        a, b = self.ff1(h).chunk(2, dim=-1)
        x = x + self.drop(self.ff2(a * F.silu(b)))
        return x


class HierWindowWorldModel(nn.Module):
    """Bounded/hierarchical history world model matching the SeqWorldModel forward contract:
    forward(tok_emb[B,L,768], types[B,L], key_pad[B,L]) -> (pred[B,L,768], h[B,L,d])."""

    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, window=3, max_len=64, **_):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)                   # 0=cmd, 1=obs
        self.pos_emb = nn.Embedding(max_len, d)
        self.max_len = max_len
        self.summary = PrefixSummary(d)
        span = 2 * int(window)                               # window steps -> tokens
        self.blocks = nn.ModuleList(
            [BandedCausalBlock(d, heads, span, dropout=dropout) for _ in range(layers)])
        self.fuse_gate = nn.Linear(2 * d, d)
        self.fuse_val = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, D)
        self.drop = nn.Dropout(dropout)

    def _pos(self, L, dev):
        if L <= self.max_len:
            return self.pos_emb(torch.arange(L, device=dev))
        base = self.pos_emb(torch.arange(self.max_len, device=dev))
        return torch.cat([base, base[-1:].expand(L - self.max_len, -1)], dim=0)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        dev = tok_emb.device
        types = types.clamp(0, 1)
        valid = (~key_pad).to(tok_emb.dtype)                 # [B,L]

        x = self.proj(tok_emb) + self.type_emb(types) + self._pos(L, dev)[None]
        x = self.drop(x)

        s = self.summary(x, valid)                           # [B,L,d] strict-prefix identity
        h = x + s                                            # condition local mixing on identity
        for blk in self.blocks:
            h = blk(h, key_pad)

        gate = torch.sigmoid(self.fuse_gate(torch.cat([h, s], dim=-1)))
        fused = h + gate * self.fuse_val(s)                  # re-inject identity, gated
        fused = self.norm(fused) * valid.unsqueeze(-1)
        return self.head(fused), fused


def build(d=192, layers=4, heads=4, dropout=0.1, window=3, max_len=64):
    return HierWindowWorldModel(d=d, layers=layers, heads=heads, dropout=dropout,
                                window=window, max_len=max_len)

