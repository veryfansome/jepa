"""
Path-delta fastweight predictor.

The model keeps a causal recurrent route state, then writes each completed
(command, observation) pair into two online delta-rule associative memories:
one keyed by command content and one keyed by the path/context state at the
command. Later command positions read those target-space memories before their
own observation exists, giving an explicit causal interpolation mechanism over
past outcomes instead of relying only on transformer attention.
"""

import math
import torch
import torch.nn as nn


NAME = "r7_path_delta_fastweights"
DESCRIPTION = (
    "Typed recurrent shell-state mixer with causal command/path delta-rule "
    "fastweight memories that read past observations directly in target space."
)


class PathDeltaFastweights(nn.Module):
    def __init__(self, d=176, layers=2, key_d=64, ffn_mult=2, **unused):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = 768
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        ffn_h = max(self.d, int(ffn_mult * self.d))

        self.cmd_proj = nn.Linear(self.D, self.d)
        self.obs_proj = nn.Linear(self.D, self.d)
        self.type_emb = nn.Embedding(2, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        self.cmd_cells = nn.ModuleList()
        self.obs_cells = nn.ModuleList()
        self.cell_norms = nn.ModuleList()
        self.ff_norms = nn.ModuleList()
        self.ff = nn.ModuleList()
        self.ff_gates = nn.ModuleList()
        for _ in range(self.layers):
            self.cmd_cells.append(nn.GRUCell(self.d, self.d))
            self.obs_cells.append(nn.GRUCell(self.d, self.d))
            self.cell_norms.append(nn.LayerNorm(self.d))
            self.ff_norms.append(nn.LayerNorm(self.d))
            self.ff.append(nn.Sequential(nn.Linear(self.d, ffn_h), nn.SiLU(), nn.Linear(ffn_h, self.d)))
            self.ff_gates.append(nn.Linear(self.d, self.d))

        self.content_read = nn.Linear(self.d, self.key_d, bias=False)
        self.content_write = nn.Linear(self.d, self.key_d, bias=False)
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.read_mix = nn.Linear(self.d, 2)
        self.read_to_h = nn.Linear(self.D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d, self.d)
        self.out_norm = nn.LayerNorm(self.d)

        self.write_gate = nn.Linear(2 * self.d, 1)
        self.direct_gate = nn.Linear(2 * self.d, 1)
        self.head = nn.Linear(self.d, self.D)

        self.logit_decay = nn.Parameter(torch.tensor(math.log(0.985 / 0.015)))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)

    def _positional(self, L, device, dtype):
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d // 2])
        return pe

    @staticmethod
    def _unit(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _read(mem, key):
        return torch.bmm(key.unsqueeze(1), mem).squeeze(1)

    def _delta_write(self, mem, key, value, amount, active, decay):
        old = self._read(mem, key)
        delta = value - old
        update = key.unsqueeze(2) * delta.unsqueeze(1) * amount.view(-1, 1, 1)
        active = active.float().view(-1, 1, 1)
        candidate = decay * mem + update
        return mem * (1.0 - active) + candidate * active

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        t = types.long().clamp(0, 1)
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        states = [torch.zeros(B, self.d, device=device, dtype=x.dtype) for _ in range(self.layers)]
        mem_content = torch.zeros(B, self.key_d, self.D, device=device, dtype=dtype)
        mem_path = torch.zeros(B, self.key_d, self.D, device=device, dtype=dtype)

        pending_c = torch.zeros(B, self.key_d, device=device, dtype=x.dtype)
        pending_p = torch.zeros(B, self.key_d, device=device, dtype=x.dtype)
        pending_h = torch.zeros(B, self.d, device=device, dtype=x.dtype)
        pending_valid = torch.zeros(B, device=device, dtype=torch.bool)

        preds = []
        hs = []
        decay = torch.sigmoid(self.logit_decay)

        for i in range(L):
            vi = valid[:, i]
            is_cmd = (t[:, i] == 0) & vi
            is_obs = (t[:, i] == 1) & vi

            z = x[:, i, :]
            new_states = []
            for layer in range(self.layers):
                prev = states[layer]
                cmd_next = self.cmd_cells[layer](z, prev)
                obs_next = self.obs_cells[layer](z, prev)
                cand = torch.where((t[:, i] == 0).unsqueeze(-1), cmd_next, obs_next)
                cand = self.cell_norms[layer](cand)

                n = self.ff_norms[layer](cand)
                cand = cand + torch.sigmoid(self.ff_gates[layer](n)) * self.ff[layer](n)
                cand = torch.where(vi.unsqueeze(-1), cand, prev)

                new_states.append(cand)
                z = cand
            states = new_states
            core = states[-1]

            q_content = self._unit(self.content_read(x[:, i, :]))
            q_path = self._unit(self.path_read(core))
            read_content = self._read(mem_content, q_content)
            read_path = self._read(mem_path, q_path)

            mix = torch.softmax(self.read_mix(core), dim=-1)
            target_read = mix[:, 0:1] * read_content + mix[:, 1:2] * read_path
            mem_h = self.read_to_h(target_read.to(x.dtype))

            fuse_in = torch.cat([core, mem_h], dim=-1)
            h_i = self.out_norm(core + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)
            pred_i = self.head(h_i) + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read

            pred_i = torch.where(vi.unsqueeze(-1), pred_i, torch.zeros_like(pred_i))
            h_i = torch.where(vi.unsqueeze(-1), h_i, torch.zeros_like(h_i))
            preds.append(pred_i)
            hs.append(h_i)

            new_c = self._unit(self.content_write(x[:, i, :]))
            new_p = self._unit(self.path_write(core))
            pending_c = torch.where(is_cmd.unsqueeze(-1), new_c, pending_c)
            pending_p = torch.where(is_cmd.unsqueeze(-1), new_p, pending_p)
            pending_h = torch.where(is_cmd.unsqueeze(-1), core, pending_h)
            pending_valid = torch.where(is_cmd, torch.ones_like(pending_valid), pending_valid)

            pair_active = is_obs & pending_valid
            if bool(pair_active.any()):
                amount = torch.sigmoid(self.write_gate(torch.cat([pending_h, core], dim=-1))).squeeze(-1)
                value = tok_emb[:, i, :]
                mem_content = self._delta_write(mem_content, pending_c.to(dtype), value, amount.to(dtype), pair_active, decay.to(dtype))
                mem_path = self._delta_write(mem_path, pending_p.to(dtype), value, amount.to(dtype), pair_active, decay.to(dtype))

            pending_valid = torch.where(is_obs, torch.zeros_like(pending_valid), pending_valid)

        return torch.stack(preds, dim=1), torch.stack(hs, dim=1)


def build(**params):
    return PathDeltaFastweights(**params)
