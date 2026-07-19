"""
Tier-A exact rewrite of the R7 path-delta fastweight champion.

Same parameter names, shapes, and update equations as r7_path_delta_fastweights_codex,
but the computation is factored into a recurrent-core pass and a fastweight-memory
scan. Token-local projections, read-mix logits, key maps, target readout, fusion,
and output heads are hoisted into batched kernels instead of being launched once
per position. This follows the same hardware lesson as chunkwise DeltaNet /
linear-recurrence work: keep the recurrence causal, but move token-local algebra
out of the scan (see arxiv.org/abs/2604.21100 and arxiv.org/abs/2605.13473).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


NAME = "r9_path_delta_fastweights_exact_codex"
DESCRIPTION = (
    "Tier-A exact state_dict-compatible rewrite of the R7 path-delta fastweight "
    "architecture: hoisted token-local maps, layer-wise recurrent scan, and "
    "batched two-bank target-space memory reads/writes."
)


class R9PathDeltaFastweightsExact(nn.Module):
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
    def _gru_from_gates(input_gates, hidden_gates, h):
        i_r, i_z, i_n = input_gates.chunk(3, dim=-1)
        h_r, h_z, h_n = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        new = torch.tanh(i_n + reset * h_n)
        return new + update * (h - new)

    @staticmethod
    def _read_bank(mem, key):
        B, banks, K, D = mem.shape
        out = torch.bmm(
            key.reshape(B * banks, 1, K),
            mem.reshape(B * banks, K, D),
        )
        return out.reshape(B, banks, D)

    def _delta_write_bank(self, mem, key, value, amount, active, decay):
        key = key.to(mem.dtype)
        old = self._read_bank(mem, key)
        delta = value.unsqueeze(1) - old
        update = key.unsqueeze(-1) * delta.unsqueeze(-2) * amount.to(mem.dtype).view(-1, 1, 1, 1)
        active = active.float().view(-1, 1, 1, 1)
        candidate = decay * mem + update
        return mem * (1.0 - active) + candidate * active

    @staticmethod
    def _type_layout(t):
        B, L = t.shape
        if B == 0:
            return False, []
        tc = t.detach().cpu()
        same = B == 1 or bool(torch.equal(tc, tc[:1].expand_as(tc)))
        return same, [int(v) for v in tc[0].tolist()]

    def _core_sequence(self, x, t, valid, same_type_rows, type_seq):
        B, L, _ = x.shape
        valid_cols = valid.unbind(1)
        type_cmd_cols = None if same_type_rows else (t == 0).unbind(1)
        seq = x

        for layer in range(self.layers):
            cmd_cell = self.cmd_cells[layer]
            obs_cell = self.obs_cells[layer]

            cmd_input = F.linear(seq, cmd_cell.weight_ih, cmd_cell.bias_ih)
            obs_input = F.linear(seq, obs_cell.weight_ih, obs_cell.bias_ih)
            cmd_cols = cmd_input.unbind(1)
            obs_cols = obs_input.unbind(1)

            ff0 = self.ff[layer][0]
            ff2 = self.ff[layer][2]
            gate = self.ff_gates[layer]
            ff0_w = torch.cat((ff0.weight, gate.weight), dim=0)
            ff0_b = torch.cat((ff0.bias, gate.bias), dim=0)
            ff_h = ff0.out_features

            h = seq.new_zeros(B, self.d)
            outs = []
            for i in range(L):
                if same_type_rows:
                    if type_seq[i] == 0:
                        gh = F.linear(h, cmd_cell.weight_hh, cmd_cell.bias_hh)
                        cand = self._gru_from_gates(cmd_cols[i], gh, h)
                    else:
                        gh = F.linear(h, obs_cell.weight_hh, obs_cell.bias_hh)
                        cand = self._gru_from_gates(obs_cols[i], gh, h)
                else:
                    gh_cmd = F.linear(h, cmd_cell.weight_hh, cmd_cell.bias_hh)
                    gh_obs = F.linear(h, obs_cell.weight_hh, obs_cell.bias_hh)
                    cand_cmd = self._gru_from_gates(cmd_cols[i], gh_cmd, h)
                    cand_obs = self._gru_from_gates(obs_cols[i], gh_obs, h)
                    cand = torch.where(type_cmd_cols[i].unsqueeze(-1), cand_cmd, cand_obs)

                cand = self.cell_norms[layer](cand)
                n = self.ff_norms[layer](cand)
                both = F.linear(n, ff0_w, ff0_b)
                ff_pre, gate_logits = both.split((ff_h, self.d), dim=-1)
                cand = cand + torch.sigmoid(gate_logits) * F.linear(F.silu(ff_pre), ff2.weight, ff2.bias)
                h = torch.where(valid_cols[i].unsqueeze(-1), cand, h)
                outs.append(h)

            seq = torch.stack(outs, dim=1)

        return seq

    def _memory_reads(self, tok_emb, t, valid, x, core, same_type_rows, type_seq):
        B, L, _ = tok_emb.shape
        dtype = tok_emb.dtype
        decay = torch.sigmoid(self.logit_decay).to(dtype)

        q_content = self._unit(self.content_read(x))
        q_path = self._unit(self.path_read(core))
        w_content = self._unit(self.content_write(x))
        w_path = self._unit(self.path_write(core))

        q_bank = torch.stack((q_content, q_path), dim=2)
        w_bank = torch.stack((w_content, w_path), dim=2)
        mix = torch.softmax(self.read_mix(core), dim=-1)

        q_cols = q_bank.unbind(1)
        w_cols = w_bank.unbind(1)
        mix_cols = mix.unbind(1)
        core_cols = core.unbind(1)
        tok_cols = tok_emb.unbind(1)
        valid_cols = valid.unbind(1)

        mem = tok_emb.new_zeros(B, 2, self.key_d, self.D)
        pending_k = x.new_zeros(B, 2, self.key_d)
        pending_h = x.new_zeros(B, self.d)
        pending_valid = torch.zeros(B, device=tok_emb.device, dtype=torch.bool)

        target_reads = []

        if same_type_rows:
            for i in range(L):
                reads = self._read_bank(mem, q_cols[i])
                m = mix_cols[i]
                target_reads.append(m[:, 0:1] * reads[:, 0, :] + m[:, 1:2] * reads[:, 1, :])

                vi = valid_cols[i]
                if type_seq[i] == 0:
                    pending_k = torch.where(vi.view(B, 1, 1), w_cols[i], pending_k)
                    pending_h = torch.where(vi.unsqueeze(-1), core_cols[i], pending_h)
                    pending_valid = pending_valid | vi
                else:
                    pair_active = vi & pending_valid
                    amount = torch.sigmoid(self.write_gate(torch.cat((pending_h, core_cols[i]), dim=-1))).squeeze(-1)
                    mem = self._delta_write_bank(mem, pending_k, tok_cols[i], amount, pair_active, decay)
                    pending_valid = pending_valid & ~vi
        else:
            cmd_cols = (t == 0).unbind(1)
            obs_cols = (t == 1).unbind(1)
            one = torch.ones_like(pending_valid)
            zero = torch.zeros_like(pending_valid)

            for i in range(L):
                reads = self._read_bank(mem, q_cols[i])
                m = mix_cols[i]
                target_reads.append(m[:, 0:1] * reads[:, 0, :] + m[:, 1:2] * reads[:, 1, :])

                vi = valid_cols[i]
                is_cmd = cmd_cols[i] & vi
                is_obs = obs_cols[i] & vi

                pending_k = torch.where(is_cmd.view(B, 1, 1), w_cols[i], pending_k)
                pending_h = torch.where(is_cmd.unsqueeze(-1), core_cols[i], pending_h)
                pending_valid = torch.where(is_cmd, one, pending_valid)

                pair_active = is_obs & pending_valid
                amount = torch.sigmoid(self.write_gate(torch.cat((pending_h, core_cols[i]), dim=-1))).squeeze(-1)
                mem = self._delta_write_bank(mem, pending_k, tok_cols[i], amount, pair_active, decay)
                pending_valid = torch.where(is_obs, zero, pending_valid)

        return torch.stack(target_reads, dim=1)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device

        if L == 0:
            h = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, self.D), h

        t = types.long().clamp(0, 1)
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(B, L, device=device, dtype=torch.bool)
        same_type_rows, type_seq = self._type_layout(t)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        core = self._core_sequence(x, t, valid, same_type_rows, type_seq)
        target_read = self._memory_reads(tok_emb, t, valid, x, core, same_type_rows, type_seq)

        mem_h = self.read_to_h(target_read.to(core.dtype))
        fuse_in = torch.cat((core, mem_h), dim=-1)
        h_raw = self.out_norm(core + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)
        pred_raw = self.head(h_raw) + torch.sigmoid(self.direct_gate(fuse_in)).to(tok_emb.dtype) * target_read

        mask = valid.unsqueeze(-1)
        pred = torch.where(mask, pred_raw, torch.zeros_like(pred_raw))
        h = torch.where(mask, h_raw, torch.zeros_like(h_raw))
        return pred, h


def build(**params):
    return R9PathDeltaFastweightsExact(**params)
