"""Causal slot-conv prefix memory.

A short left-padded causal convolution captures local cmd/obs pair structure, while
learned compressed slots store longer history. Each position reads slots before the
current token is written, so cmd_t can use prior observations plus cmd_t, never obs_t.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "g3_slotconv_prefix_memory"
DESCRIPTION = "Read-before-write compressed history slots plus causal local convolution."
D_IN = 768


class RoutedFFN(nn.Module):
    def __init__(self, d, experts=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.router = nn.Linear(d, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
            for _ in range(experts)
        ])
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x)
        w = torch.softmax(self.router(y), dim=-1)
        vals = torch.stack([e(y) for e in self.experts], dim=1)
        return x + self.drop(torch.sigmoid(self.gate(y)) * (w.unsqueeze(-1) * vals).sum(1))


class SlotConvPrefixMemory(nn.Module):
    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, slots=8, kernel=3,
                 experts=4, max_period=10000.0, **_):
        super().__init__()
        del heads
        self.d, self.slots, self.kernel, self.max_period = d, slots, kernel, float(max_period)
        self.in_proj = nn.Linear(D_IN, d)
        self.type_emb = nn.Embedding(2, d)
        self.drop = nn.Dropout(dropout)
        self.local_dw = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        self.local_pw = nn.Linear(2 * d, d)
        self.slot_seed = nn.Parameter(torch.randn(slots, d) / math.sqrt(d))
        self.slot_norm = nn.LayerNorm(d)
        self.read_norm = nn.LayerNorm(d)
        self.read_q, self.read_k, self.read_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.read_out = nn.Linear(d, d)
        self.write_norm = nn.LayerNorm(d)
        self.write_slot, self.write_key = nn.Linear(d, d), nn.Linear(d, d)
        self.write_val, self.write_keep = nn.Linear(d, d), nn.Linear(d, d)
        self.type_slot_bias = nn.Embedding(2, slots)
        self.write_gate = nn.Linear(3 * d, d)
        self.slot_update_norm = nn.LayerNorm(d)
        self.fuse = nn.Linear(4 * d, d)
        self.fuse_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([RoutedFFN(d, experts, dropout) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, D_IN)

    def _pos_features(self, L, device, dtype):
        half = max(self.d // 2, 1)
        pos = torch.arange(L, device=device, dtype=dtype)
        freq = torch.exp(-math.log(self.max_period) * torch.arange(half, device=device, dtype=dtype) / half)
        pe = torch.cat([torch.sin(pos[:, None] * freq[None]), torch.cos(pos[:, None] * freq[None])], -1)
        return F.pad(pe, (0, max(0, self.d - pe.shape[-1])))[:, :self.d]

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        valid_seq = (~key_pad).to(tok_emb.dtype).unsqueeze(-1)
        x = self.in_proj(tok_emb) + self.type_emb(types.clamp(0, 1))
        x = self.drop(x + self._pos_features(L, tok_emb.device, x.dtype).unsqueeze(0))
        x_valid = x * valid_seq
        local = self.local_dw(F.pad(x_valid.transpose(1, 2), (self.kernel - 1, 0))).transpose(1, 2)
        local = self.local_pw(torch.cat([x_valid, local], -1))

        slots = self.slot_seed.unsqueeze(0).expand(B, -1, -1)
        outs, scale = [], 1.0 / math.sqrt(self.d)
        for t in range(L):
            xt, lt = x[:, t], local[:, t]
            valid = (~key_pad[:, t]).to(xt.dtype).unsqueeze(-1)
            sn = self.slot_norm(slots)
            q = self.read_q(self.read_norm(xt + lt))
            attn = torch.softmax(torch.einsum("bd,bsd->bs", q, self.read_k(sn)) * scale, -1)
            ctx = self.read_out(torch.einsum("bs,bsd->bd", attn, self.read_v(sn)))
            h = self.fuse_norm(xt + self.drop(torch.tanh(self.fuse(torch.cat([xt, lt, ctx, xt - ctx], -1)))))
            for block in self.blocks:
                h = block(h)
            outs.append(self.out_norm(h) * valid)

            tn = self.write_norm(xt)
            logits = torch.einsum("bsd,bd->bs", self.write_slot(sn), self.write_key(tn)) * scale
            assign = torch.softmax(logits + self.type_slot_bias(types[:, t].clamp(0, 1)), -1).unsqueeze(-1)
            te = tn.unsqueeze(1).expand(-1, self.slots, -1)
            cand = torch.tanh(self.write_val(tn).unsqueeze(1) + self.write_keep(sn))
            gate = torch.sigmoid(self.write_gate(torch.cat([sn, te, sn * te], -1)))
            slots = self.slot_update_norm(slots + valid.unsqueeze(1) * assign * gate * (cand - slots))

        h = torch.stack(outs, 1)
        return self.head(h), h


def build(**params):
    return SlotConvPrefixMemory(**params)
