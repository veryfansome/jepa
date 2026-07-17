"""Type-routed multi-timescale state with a system-identity anchor and one gated
causal look-back (recombination of the two arch leaders).

Fuses the two leaders' engines into one strictly-causal predictor: the two-stream
cmd/obs routing generalized into a BANK of parallel multi-timescale causal EMA prefix
memories per stream (fast->slow decay lanes); plus an explicit strict-prefix gated
system-identity slot with an early-token-favoring position-decay prior; plus one gated
strict-causal look-back attention block for selective retrieval on top of the compressed
state. All history ops are strict prefixes (inclusive cumsum shifted by one) or an
upper-triangular attention mask, so command position 2t never sees obs 2t+1 or later.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


NAME = "routed_emabank_sysid_lookback"
DESCRIPTION = (
    "Two-stream type-routed multi-timescale causal EMA memory bank + explicit "
    "gated strict-prefix system-identity slot + one gated causal look-back block; "
    "a transferable compressed state with selective retrieval, all strictly causal."
)

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
        return self.out(self.drop(a * F.silu(b)))


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


def _strict_prefix(cum):
    return F.pad(cum, (0, 0, 1, 0))[:, :-1, :]


class RoutedEMABank(nn.Module):
    def __init__(self, d, n_ema=4):
        super().__init__()
        self.d = d
        self.n_ema = n_ema
        self.cmd_val = nn.Linear(d, d)
        self.obs_val = nn.Linear(d, d)
        self.cmd_gate = nn.Linear(d, n_ema)
        self.obs_gate = nn.Linear(d, n_ema)
        init = torch.linspace(-2.0, 2.5, n_ema)
        self.cmd_decay_raw = nn.Parameter(init.clone())
        self.obs_decay_raw = nn.Parameter(init.clone())
        self.out = nn.Linear(2 * n_ema * d, d)
        self.norm = nn.LayerNorm(d)

    def _stream_ema(self, v, gate_logits, is_stream, decay_raw):
        B, L, d = v.shape
        K = self.n_ema
        dev = v.device
        alpha = torch.sigmoid(decay_raw).clamp(1e-3, 1 - 1e-3)
        log_a = torch.log(alpha)
        g = F.softplus(gate_logits) * is_stream
        pos = torch.arange(L, device=dev, dtype=v.dtype)
        neg_j_loga = (-pos)[None, :, None] * log_a[None, None, :]
        m = neg_j_loga.amax(dim=1, keepdim=True)
        scale_up = torch.exp(neg_j_loga - m)
        gscaled = g * scale_up
        gv = gscaled[..., None] * v[:, :, None, :]
        csum = torch.cumsum(gv, dim=1)
        gsum = torch.cumsum(gscaled, dim=1)
        j_loga = pos[None, :, None] * log_a[None, None, :]
        down = torch.exp(j_loga + m)
        ema = csum * down[..., None]
        mass = gsum * down
        ema = ema / mass.clamp(min=1e-6)[..., None]
        ema = _strict_prefix(ema.reshape(B, L, K * d)).reshape(B, L, K, d)
        return ema

    def forward(self, x, types, valid):
        B, L, d = x.shape
        is_cmd = ((types == 0).to(x.dtype) * valid).unsqueeze(-1)
        is_obs = ((types == 1).to(x.dtype) * valid).unsqueeze(-1)
        cmd = self._stream_ema(self.cmd_val(x), self.cmd_gate(x), is_cmd, self.cmd_decay_raw)
        obs = self._stream_ema(self.obs_val(x), self.obs_gate(x), is_obs, self.obs_decay_raw)
        cat = torch.cat([cmd.reshape(B, L, -1), obs.reshape(B, L, -1)], dim=-1)
        return self.norm(self.out(cat))


class SysIdSlot(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.val = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.log_lam = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, valid):
        B, L, d = x.shape
        dev = x.device
        v = self.val(x)
        pos = torch.arange(L, device=dev, dtype=x.dtype)
        lam = F.softplus(self.log_lam)
        prior = torch.exp(-lam * pos)[None, :, None]
        g = F.softplus(self.gate(x)) * valid.unsqueeze(-1) * prior
        wsum = torch.cumsum(g * v, dim=1)
        gsum = torch.cumsum(g, dim=1)
        wsum = _strict_prefix(wsum)
        gsum = _strict_prefix(gsum)
        state = wsum / gsum.clamp(min=1e-6)
        return self.norm(self.out(state))


class GatedCausalAttn(nn.Module):
    def __init__(self, d, heads=3, dropout=0.1):
        super().__init__()
        if d % heads != 0:
            heads = next((h for h in range(min(heads, d), 0, -1) if d % h == 0), 1)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(d, d)

    def forward(self, x, key_pad):
        B, L, d = x.shape
        dev = x.device
        xn = self.norm(x)
        mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1)
        y, _ = self.attn(xn, xn, xn, attn_mask=mask,
                         key_padding_mask=key_pad, need_weights=False)
        y = torch.nan_to_num(y, nan=0.0)
        g = torch.sigmoid(self.gate(x))
        return x + g * y


class RoutedEmaSysidLookback(nn.Module):
    def __init__(self, d=180, layers=3, n_ema=4, heads=3, dropout=0.1,
                 max_period=10000.0, **_):
        super().__init__()
        self.d = d
        self.max_period = float(max_period)
        self.in_proj = nn.Linear(D_IN, d)
        self.type_emb = nn.Embedding(2, d)
        self.drop = nn.Dropout(dropout)

        self.ema = RoutedEMABank(d, n_ema=n_ema)
        self.sysid = SysIdSlot(d)
        self.mix_in = nn.Linear(4 * d, d)
        self.blocks = nn.ModuleList([GatedMixBlock(d, dropout=dropout) for _ in range(layers)])
        self.lookback = GatedCausalAttn(d, heads=heads, dropout=dropout)
        self.fuse_gate = nn.Linear(2 * d, d)
        self.fuse_val = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, D_IN)

    def _pos_features(self, L, device, dtype):
        half = self.d // 2
        pos = torch.arange(L, device=device, dtype=dtype)
        if half == 0:
            return torch.zeros(L, self.d, device=device, dtype=dtype)
        freq = torch.exp(-math.log(self.max_period)
                         * torch.arange(half, device=device, dtype=dtype) / max(half, 1))
        ang = pos[:, None] * freq[None, :]
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if pe.shape[-1] < self.d:
            pe = F.pad(pe, (0, self.d - pe.shape[-1]))
        return pe[:, : self.d]

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        types = types.clamp(0, 1)
        valid = (~key_pad).to(tok_emb.dtype)

        x = self.in_proj(tok_emb) + self.type_emb(types)
        x = x + self._pos_features(L, device, x.dtype).unsqueeze(0)
        x = self.drop(x)

        ema = self.ema(x, types, valid)
        sysid = self.sysid(x, valid)

        mixed = self.mix_in(torch.cat([x, ema, sysid, ema - sysid], dim=-1))
        for block in self.blocks:
            mixed = block(mixed)
        mixed = self.lookback(mixed, key_pad)

        gate = torch.sigmoid(self.fuse_gate(torch.cat([mixed, sysid], dim=-1)))
        fused = mixed + gate * self.fuse_val(sysid)
        h = self.norm(fused) * valid.unsqueeze(-1)
        pred = self.head(h)
        return pred, h


def build(**params):
    return RoutedEmaSysidLookback(**params)
