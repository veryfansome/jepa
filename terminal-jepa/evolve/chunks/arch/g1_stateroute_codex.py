"""
Type-routed causal state memory.

Instead of full self-attention, this model maintains two recurrent memories: one
for command evidence and one for observation evidence. Each token updates only
the matching stream through a learned gate, then a small stack of gated mixing
blocks combines current-token features with the causal memories. This may help
the JEPA objective by preserving distinct roles for actions and observations
while giving command-position predictions direct access to a compact history
summary without any future-token path.
"""

import math
import torch
import torch.nn as nn


NAME = "type_routed_state_memory"
DESCRIPTION = "Two-stream causal recurrent memory with type-routed cmd/obs state updates and gated residual mixing."


D_IN = 768


class SwiGLU(nn.Module):
    def __init__(self, d, mult=3, dropout=0.1):
        super().__init__()
        inner = int(mult * d)
        self.fc = nn.Linear(d, 2 * inner)
        self.out = nn.Linear(inner, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.out(self.drop(a * torch.nn.functional.silu(b)))


class GatedMixBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ff = SwiGLU(d, mult=3, dropout=dropout)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ff(self.norm(x))
        g = torch.sigmoid(self.gate(x))
        return x + self.drop(g * y)


class TypeRoutedStateMemory(nn.Module):
    def __init__(self, d=192, layers=4, dropout=0.1, max_period=10000.0, **_):
        super().__init__()
        self.d = d
        self.max_period = float(max_period)

        self.in_proj = nn.Linear(D_IN, d)
        self.type_emb = nn.Embedding(2, d)

        self.cmd_update = nn.Linear(3 * d, 2 * d)
        self.obs_update = nn.Linear(3 * d, 2 * d)

        self.mix_in = nn.Linear(4 * d, d)
        self.blocks = nn.ModuleList([GatedMixBlock(d, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, D_IN)
        self.drop = nn.Dropout(dropout)

    def _pos_features(self, L, device, dtype):
        half = self.d // 2
        pos = torch.arange(L, device=device, dtype=dtype)
        if half == 0:
            return torch.zeros(L, self.d, device=device, dtype=dtype)
        freq = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=device, dtype=dtype)
            / max(half, 1)
        )
        ang = pos[:, None] * freq[None, :]
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if pe.shape[-1] < self.d:
            pe = torch.nn.functional.pad(pe, (0, self.d - pe.shape[-1]))
        return pe[:, : self.d]

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device

        x = self.in_proj(tok_emb)
        x = x + self.type_emb(types.clamp(0, 1))
        x = x + self._pos_features(L, device, x.dtype).unsqueeze(0)
        x = self.drop(x)

        cmd_state = x.new_zeros(B, self.d)
        obs_state = x.new_zeros(B, self.d)
        outs = []

        for t in range(L):
            xt = x[:, t]
            typ = types[:, t]
            valid_bool = ~key_pad[:, t]
            valid = valid_bool.to(dtype=xt.dtype).unsqueeze(-1)

            shared = torch.cat([xt, cmd_state, obs_state], dim=-1)

            cmd_gate, cmd_cand = self.cmd_update(shared).chunk(2, dim=-1)
            obs_gate, obs_cand = self.obs_update(shared).chunk(2, dim=-1)

            cmd_gate = torch.sigmoid(cmd_gate)
            obs_gate = torch.sigmoid(obs_gate)
            cmd_cand = torch.tanh(cmd_cand)
            obs_cand = torch.tanh(obs_cand)

            is_cmd = (typ == 0).unsqueeze(-1).to(dtype=xt.dtype) * valid
            is_obs = (typ == 1).unsqueeze(-1).to(dtype=xt.dtype) * valid

            cmd_state = cmd_state * (1.0 - is_cmd * cmd_gate) + cmd_cand * (is_cmd * cmd_gate)
            obs_state = obs_state * (1.0 - is_obs * obs_gate) + obs_cand * (is_obs * obs_gate)

            mixed = self.mix_in(torch.cat([xt, cmd_state, obs_state, cmd_state - obs_state], dim=-1))
            for block in self.blocks:
                mixed = block(mixed)
            ht = self.norm(mixed)

            ht = ht * valid
            outs.append(ht)

        h = torch.stack(outs, dim=1)
        pred = self.head(h)
        return pred, h


def build(**params):
    return TypeRoutedStateMemory(**params)
