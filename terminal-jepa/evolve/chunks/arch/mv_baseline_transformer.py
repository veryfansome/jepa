"""Layout-agnostic baseline transformer for non-interleave streams: identical to the baseline
causal transformer (pre-LN, learned positions, type embedding, linear head) except the position
table length is a parameter (the historical SeqWorldModel hardcodes 64, which a multi-vector
stream of stride 5 exceeds: n<=20 steps -> L<=100). Makes NO assumption about the token layout —
it reads types/positions only — so it is valid under any stream chunk. Note: a different pos-table
size changes the init RNG draw, so even under the baseline stream this is a distinct genome from
baseline_transformer (compare within-protocol, not across)."""

import torch
import torch.nn as nn

D = 768

NAME = "mv_baseline_transformer"
DESCRIPTION = ("Baseline causal transformer with parameterized max_len position table; "
               "layout-agnostic (safe under any stream chunk).")


class MvSeqModel(nn.Module):
    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, max_len=160):
        super().__init__()
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.max_len = max_len
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, D)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        pos = torch.arange(L, device=tok_emb.device).clamp(max=self.max_len - 1)
        x = self.proj(tok_emb) + self.type_emb(types.clamp(0, 1)) + self.pos_emb(pos)[None]
        mask = torch.triu(torch.ones(L, L, device=tok_emb.device, dtype=torch.bool), 1)
        h = self.tf(x, mask=mask, src_key_padding_mask=key_pad)
        return self.head(h), h


def build(d=192, layers=4, heads=4, dropout=0.1, max_len=160):
    return MvSeqModel(d=d, layers=layers, heads=heads, dropout=dropout, max_len=max_len)
