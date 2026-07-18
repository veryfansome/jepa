"""System-identity broadcast predictor: a causal transformer augmented with an
explicit, causally-accumulated "system-identity" state.

Idea: a shell exploration always opens by identifying the machine (uname, then a
cat of a config file), so the identity of the system is knowable from the earliest
tokens. Vanilla causal self-attention must re-derive this identity through soft
attention at every layer and every position; on UNSEEN systems that re-derivation is
exactly what fails to transfer. This arch instead maintains an explicit prefix state
s_t = gated_cumulative_summary(tokens strictly before t) injected additively into every
position. Because the state is a running summary dominated by the (heavily-weighted)
opening uname/config tokens, it gives each later command position a stable, low-variance
handle on "which system am I on" that does not need to be re-attended. The transformer
then only has to model the local command->observation dynamics conditioned on that identity.

Causality: the state at position i is a STRICT-prefix cumulative sum (shifted by one),
so command position 2t sees only tokens 0..2t-1 — never its own observation (2t+1) or
anything later. The transformer uses the standard upper-triangular causal mask. Padding
is removed from the cumulative aggregation (zeroed) and from attention (key_pad).
"""

import torch
import torch.nn as nn

D = 768

NAME = "sysid_prefix_state"
DESCRIPTION = ("Causal transformer + explicit causally-accumulated system-identity "
               "prefix state, gate-pooled from opening tokens and broadcast to every "
               "position to condition local cmd->obs dynamics.")


class SysIdState(nn.Module):
    """Causal, strict-prefix gated summary of the token stream. For each position i,
    produces a state vector built ONLY from tokens 0..i-1 (strict prefix)."""

    def __init__(self, d):
        super().__init__()
        self.val = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, valid):
        # x: [B,L,d] projected+typed tokens; valid: [B,L] float (1=real, 0=pad)
        v = self.val(x)                                    # [B,L,d]
        g = nn.functional.softplus(self.gate(x)) * valid.unsqueeze(-1)  # [B,L,1], 0 on pad
        wsum = torch.cumsum(g * v, dim=1)                  # inclusive cumulative
        gsum = torch.cumsum(g, dim=1)                      # inclusive cumulative gate mass
        # shift by one -> STRICT prefix (position i sees only 0..i-1)
        wsum = nn.functional.pad(wsum, (0, 0, 1, 0))[:, :-1, :]
        gsum = nn.functional.pad(gsum, (0, 0, 1, 0))[:, :-1, :]
        state = wsum / gsum.clamp(min=1e-6)                # [B,L,d] running gated mean
        return self.norm(self.out(state))


class SysIdWorldModel(nn.Module):
    """Baseline causal TransformerEncoder over interleaved cmd/obs frozen tokens, with an
    additive system-identity prefix state injected at the input (and re-injected after the
    encoder via a gated residual). Matches the SeqWorldModel forward contract exactly."""

    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, max_len=64):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)      # 0=cmd, 1=obs
        self.pos_emb = nn.Embedding(max_len, d)
        self.max_len = max_len
        self.sysid = SysIdState(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.fuse_gate = nn.Linear(2 * d, d)
        self.fuse_val = nn.Linear(d, d)
        self.head = nn.Linear(d, D)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        dev = tok_emb.device
        pos = torch.arange(min(L, self.max_len), device=dev)
        pe = self.pos_emb(pos)
        if L > self.max_len:  # safety: tile/clamp positions beyond table
            pe = torch.cat([pe, pe[-1:].expand(L - self.max_len, -1)], dim=0)
        x = self.proj(tok_emb) + self.type_emb(types) + pe[None]      # [B,L,d]

        valid = (~key_pad).float()                                    # [B,L]
        s = self.sysid(x, valid)                                      # [B,L,d] strict-prefix state
        x = x + s                                                     # inject identity at input

        mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1)  # causal
        h = self.tf(x, mask=mask, src_key_padding_mask=key_pad)       # [B,L,d]

        gate = torch.sigmoid(self.fuse_gate(torch.cat([h, s], dim=-1)))
        fused = h + gate * self.fuse_val(s)
        return self.head(fused), fused


def build(d=192, layers=4, heads=4, dropout=0.1, max_len=64):
    return SysIdWorldModel(d=d, layers=layers, heads=heads, dropout=dropout, max_len=max_len)
