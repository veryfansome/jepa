"""
R8 architecture: RLS-gated DeltaProduct fastweights.

This keeps the champion's typed recurrent shell-state mixer and target-space
fastweight reads, but replaces scalar Widrow-Hoff writes with an adaptive-filter
write rule: a per-sequence recursive-least-squares/Kaczmarz precision matrix
chooses directional learning rates, while Gated DeltaNet-2 style erase and write
gates are decoupled. Each completed (cmd, obs) pair performs two sequential
DeltaProduct-like micro-edits per memory, using a normalized base key and an
orthogonal companion key, to reduce key collisions in the compressed associative
state.

Grounding:
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- DeltaProduct / Householder products: https://arxiv.org/abs/2502.10297
- Gated DeltaNet-2 decoupled erase/write: https://arxiv.org/abs/2605.22791
- RLS with forgetting from adaptive filtering/control supplies the precision
  matrix update used here.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


D = 768

NAME = "r8_rls_gated_deltaproduct_fastweights_codex"
DESCRIPTION = (
    "Typed recurrent shell-state mixer with target-space associative memories. "
    "Writes use RLS/Kaczmarz precision gains, decoupled erase/write gates, and "
    "two orthogonal DeltaProduct-style micro-keys per completed command/obs pair."
)


class RLSGatedDeltaProductFastweights(nn.Module):
    def __init__(self, d=176, layers=2, key_d=64, ffn_mult=2, **unused):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        ffn_h = max(self.d, int(ffn_mult * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
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
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)

        self.content_write0 = nn.Linear(self.d, self.key_d, bias=False)
        self.content_write1 = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write0 = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write1 = nn.Linear(self.d, self.key_d, bias=False)

        self.read_mix = nn.Linear(self.d + 4, 2)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 4, self.d)
        self.out_norm = nn.LayerNorm(self.d)

        self.write_gate = nn.Linear(2 * self.d, 8)
        self.forget_gate = nn.Linear(2 * self.d, 2)
        self.direct_gate = nn.Linear(2 * self.d + 4, 1)
        self.head = nn.Linear(self.d, D)

        self.log_p0 = nn.Parameter(torch.tensor(math.log(math.expm1(8.0))))

        nn.init.constant_(self.write_gate.bias, 0.75)
        nn.init.zeros_(self.forget_gate.weight)
        nn.init.constant_(self.forget_gate.bias, math.log(0.8585858586 / (1.0 - 0.8585858586)))
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
        # rsqrt(sum+eps), not norm().clamp_min(): norm's backward is NaN at exactly-zero
        # inputs (the t=0 path keys are bias-free projections of a zero state), and the
        # clamp multiplies that NaN by 0 instead of removing it.
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    def _orthogonal_unit(self, x, base):
        x = x - (x * base).sum(dim=-1, keepdim=True) * base
        return self._unit(x)

    @staticmethod
    def _read(mem, key):
        return torch.bmm(key.unsqueeze(1), mem).squeeze(1)

    def _certainty(self, prec, key):
        pk = torch.bmm(prec, key.unsqueeze(-1)).squeeze(-1)
        quad = (key * pk).sum(dim=-1, keepdim=True).clamp_min(0.0)
        return 1.0 / (1.0 + quad)

    def _rls_write(self, mem, prec, key, value, erase, write, active, lam):
        k = key.to(mem.dtype)
        v = value.to(mem.dtype)
        erase = erase.to(mem.dtype).clamp(0.0, 1.0)
        write = write.to(mem.dtype).clamp(0.0, 1.0)
        lam = lam.to(mem.dtype).clamp(0.90, 0.999)

        pk = torch.bmm(prec, k.unsqueeze(-1)).squeeze(-1)
        kp = torch.bmm(k.unsqueeze(1), prec).squeeze(1)
        quad = (k * pk).sum(dim=-1).clamp_min(0.0)
        denom = (lam + quad).clamp_min(1e-4)
        gain = pk / denom.unsqueeze(-1)

        old = self._read(mem, k)
        delta = write.unsqueeze(-1) * v - erase.unsqueeze(-1) * old
        cand_mem = mem + gain.unsqueeze(2) * delta.unsqueeze(1)

        beta = (0.5 * (erase + write)).unsqueeze(-1).unsqueeze(-1)
        cand_prec = (prec - beta * gain.unsqueeze(2) * kp.unsqueeze(1)) / lam.view(-1, 1, 1)
        cand_prec = 0.5 * (cand_prec + cand_prec.transpose(1, 2))

        cand_mem = torch.nan_to_num(cand_mem, nan=0.0, posinf=1e4, neginf=-1e4)
        cand_prec = torch.nan_to_num(cand_prec, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        a = active.to(mem.dtype).view(-1, 1, 1)
        return mem * (1.0 - a) + cand_mem * a, prec * (1.0 - a) + cand_prec * a

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h

        t = types.long().clamp(0, 1)
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        states = [torch.zeros(B, self.d, device=device, dtype=x.dtype) for _ in range(self.layers)]

        eye = torch.eye(self.key_d, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        p_init = (F.softplus(self.log_p0) + 1e-3).to(dtype)

        mem_content = torch.zeros(B, self.key_d, D, device=device, dtype=dtype)
        mem_path = torch.zeros(B, self.key_d, D, device=device, dtype=dtype)
        prec_content = p_init * eye.clone()
        prec_path = p_init * eye.clone()

        zkey = torch.zeros(B, self.key_d, device=device, dtype=x.dtype)
        pending_c0 = zkey.clone()
        pending_c1 = zkey.clone()
        pending_p0 = zkey.clone()
        pending_p1 = zkey.clone()
        pending_h = torch.zeros(B, self.d, device=device, dtype=x.dtype)
        pending_valid = torch.zeros(B, device=device, dtype=torch.bool)

        preds = []
        hs = []

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

            cert_c = self._certainty(prec_content.to(x.dtype), q_content).to(x.dtype)
            cert_p = self._certainty(prec_path.to(x.dtype), q_path).to(x.dtype)
            norm_c = (read_content.to(x.dtype).pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()  # eps: sqrt backward is NaN at the exact-zero reads of an empty memory
            norm_p = (read_path.to(x.dtype).pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            read_feat = torch.cat([norm_c, norm_p, cert_c, cert_p], dim=-1)

            mix = torch.softmax(self.read_mix(torch.cat([core, read_feat], dim=-1)), dim=-1)
            target_read = mix[:, 0:1].to(dtype) * read_content + mix[:, 1:2].to(dtype) * read_path
            mem_h = self.read_to_h(target_read.to(x.dtype))

            fuse_in = torch.cat([core, mem_h, read_feat], dim=-1)
            h_i = self.out_norm(core + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)
            pred_i = self.head(h_i) + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read

            pred_i = torch.where(vi.unsqueeze(-1), pred_i, torch.zeros_like(pred_i))
            h_i = torch.where(vi.unsqueeze(-1), h_i, torch.zeros_like(h_i))
            preds.append(pred_i)
            hs.append(h_i)

            c0 = self._unit(self.content_write0(x[:, i, :]))
            c1 = self._orthogonal_unit(self.content_write1(x[:, i, :]), c0)
            p0 = self._unit(self.path_write0(core))
            p1 = self._orthogonal_unit(self.path_write1(core), p0)

            pending_c0 = torch.where(is_cmd.unsqueeze(-1), c0, pending_c0)
            pending_c1 = torch.where(is_cmd.unsqueeze(-1), c1, pending_c1)
            pending_p0 = torch.where(is_cmd.unsqueeze(-1), p0, pending_p0)
            pending_p1 = torch.where(is_cmd.unsqueeze(-1), p1, pending_p1)
            pending_h = torch.where(is_cmd.unsqueeze(-1), core, pending_h)
            pending_valid = torch.where(is_cmd, torch.ones_like(pending_valid), pending_valid)

            pair_active = is_obs & pending_valid
            pair_feat = torch.cat([pending_h, core], dim=-1)
            gates = torch.sigmoid(self.write_gate(pair_feat))
            lam = 0.90 + 0.099 * torch.sigmoid(self.forget_gate(pair_feat))
            value = tok_emb[:, i, :]

            mem_content, prec_content = self._rls_write(
                mem_content, prec_content, pending_c0.to(dtype), value,
                gates[:, 0], gates[:, 1], pair_active, lam[:, 0]
            )
            mem_content, prec_content = self._rls_write(
                mem_content, prec_content, pending_c1.to(dtype), value,
                gates[:, 2], gates[:, 3], pair_active, torch.ones_like(lam[:, 0]) * 0.995
            )
            mem_path, prec_path = self._rls_write(
                mem_path, prec_path, pending_p0.to(dtype), value,
                gates[:, 4], gates[:, 5], pair_active, lam[:, 1]
            )
            mem_path, prec_path = self._rls_write(
                mem_path, prec_path, pending_p1.to(dtype), value,
                gates[:, 6], gates[:, 7], pair_active, torch.ones_like(lam[:, 1]) * 0.995
            )

            pending_valid = torch.where(is_obs, torch.zeros_like(pending_valid), pending_valid)

        return torch.stack(preds, dim=1), torch.stack(hs, dim=1)


def build(**params):
    return RLSGatedDeltaProductFastweights(**params)
