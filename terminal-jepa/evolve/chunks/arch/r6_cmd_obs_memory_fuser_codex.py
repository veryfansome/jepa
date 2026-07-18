"""
Command-conditioned causal observation-memory predictor.

The model keeps a normal causal token stream, but adds a separate retrieval path over
completed observation tokens only. At each position, the current token queries summaries of
previous observations with a learned recency bias, then a gated fusion combines current-command
features, local causal context, and retrieved environment-state evidence. This may help JEPA shell
prediction because many next observations are determined by command semantics plus the evolving
filesystem/session state, which is more naturally stored in prior observations than in every token.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r6_cmd_obs_memory_fuser"
DESCRIPTION = "Causal transformer with command-conditioned retrieval over prior observation memory and gated fusion."


D_IN = 768


class SwiGLU(nn.Module):
    def __init__(self, d, hidden_mult=4, dropout=0.1):
        super().__init__()
        h = int(hidden_mult * d)
        self.w12 = nn.Linear(d, 2 * h)
        self.out = nn.Linear(h, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.out(self.drop(F.silu(a) * b))


class CausalBlock(nn.Module):
    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = SwiGLU(d, hidden_mult=3, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.g_attn = nn.Linear(2 * d, d)
        self.g_ffn = nn.Linear(2 * d, d)

    def forward(self, x, causal_mask, key_pad):
        xn = self.attn_norm(x)
        a, _ = self.attn(
            xn, xn, xn,
            attn_mask=causal_mask,
            key_padding_mask=key_pad,
            need_weights=False,
        )
        ga = torch.sigmoid(self.g_attn(torch.cat([x, a], dim=-1)))
        x = x + self.drop(ga * a)

        f = self.ffn(self.ffn_norm(x))
        gf = torch.sigmoid(self.g_ffn(torch.cat([x, f], dim=-1)))
        x = x + self.drop(gf * f)
        return x


class ObsMemoryFusion(nn.Module):
    def __init__(self, d, heads, dropout=0.1, max_len=128):
        super().__init__()
        self.d = d
        self.heads = heads
        self.max_len = max_len
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.norm_q = nn.LayerNorm(d)
        self.norm_m = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.fuse = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.Sigmoid(),
        )
        self.recency = nn.Parameter(torch.linspace(0.15, 1.25, heads).view(1, heads, 1, 1))

    def forward(self, x, types, key_pad):
        B, L, d = x.shape
        H = self.heads
        dh = d // H
        device = x.device

        q = self.q(self.norm_q(x)).view(B, L, H, dh).transpose(1, 2)
        mem = self.norm_m(x)
        k = self.k(mem).view(B, L, H, dh).transpose(1, 2)
        v = self.v(mem).view(B, L, H, dh).transpose(1, 2)

        pos = torch.arange(L, device=device)
        query_pos = pos.view(L, 1)
        key_pos = pos.view(1, L)

        obs_key = types.eq(1) & (~key_pad)
        allowed = (key_pos < query_pos).unsqueeze(0) & obs_key.unsqueeze(1)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
        distance = (query_pos - key_pos).clamp_min(0).to(x.dtype)
        logits = logits - self.recency.to(x.dtype) * torch.log1p(distance).view(1, 1, L, L)

        logits = logits.masked_fill(~allowed.unsqueeze(1), -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * allowed.unsqueeze(1).to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        r = torch.matmul(self.drop(weights), v).transpose(1, 2).contiguous().view(B, L, d)
        r = self.out(r)

        g = self.fuse(torch.cat([x, r, x * r], dim=-1))
        return x + self.drop(g * r)


class CmdObsMemoryFuser(nn.Module):
    def __init__(self, d=192, layers=3, heads=4, dropout=0.1, max_len=128):
        super().__init__()
        if d % heads != 0:
            d = heads * max(1, d // heads)
        self.d = d
        self.max_len = max_len

        self.proj = nn.Linear(D_IN, d)
        self.type_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Embedding(max_len, d)

        self.blocks = nn.ModuleList([
            CausalBlock(d, heads, dropout=dropout) for _ in range(layers)
        ])
        self.obs_memory = ObsMemoryFusion(d, heads, dropout=dropout, max_len=max_len)
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, D_IN),
        )

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device

        pos = torch.arange(L, device=device).clamp_max(self.max_len - 1)
        x = self.proj(tok_emb) + self.type_emb(types.clamp(0, 1)) + self.pos_emb(pos).unsqueeze(0)
        x = x.masked_fill(key_pad.unsqueeze(-1), 0.0)

        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), 1)

        for block in self.blocks:
            x = block(x, causal_mask, key_pad)
            x = x.masked_fill(key_pad.unsqueeze(-1), 0.0)

        x = self.obs_memory(x, types, key_pad)
        h = self.final_norm(x).masked_fill(key_pad.unsqueeze(-1), 0.0)
        pred = self.head(h).masked_fill(key_pad.unsqueeze(-1), 0.0)
        return pred, h


def build(**params):
    return CmdObsMemoryFuser(**params)
