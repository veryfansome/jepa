"""
Mesa/RLS fastweight predictor — second-order upgrade of the champion's delta-rule memories.

The champion (r7_path_delta_fastweights) writes each completed (command, observation)
pair into two associative memories with the Widrow-Hoff delta rule: one gradient step
on ||k^T M - v||^2 with a scalar learned step size and one constant global decay. That
rule under-corrects interference between CORRELATED keys — and shell exploration keys
are highly correlated (repeated `ls` of sibling directories, `cat`s sharing a path
prefix, revisited cwds). This module keeps the champion's trunk, read path, gating and
causal write schedule EXACTLY, and replaces only the write rule with exponentially
weighted recursive least squares (the "mesa layer" view of fast weights): each memory
also carries an inverse-covariance state P and writes with the Kalman/RLS gain

    g_t   = P k_t / (lam_t / beta_t + k_t^T P k_t)
    M_t   = rho M_{t-1} + g_t (v_t - k_t^T rho M_{t-1})
    P_t   = (P_{t-1} - g_t (P_{t-1} k_t)^T) / lam_t        (symmetrized, trace-capped)

so M is the closed-form ridge solution of min_M sum_j w_j ||k_j^T M - v_j||^2 over all
past pairs: novel key directions get a high write gain, saturated directions a low one,
and reads at a correlated (never exactly repeated) key return the least-squares
interpolation of past outcomes instead of a decayed correlation sum. lam_t is a
DATA-DEPENDENT forgetting gate (Gated-DeltaNet-style), beta_t the champion's write
gate reused as an RLS sample weight, rho the champion's mild value decay. At init
(P = I, lam ~ 0.98, beta ~ 0.73) the first writes reproduce the champion's step size,
so the module starts as a near-exact champion and learns second-order behavior.
"""

import math
import torch
import torch.nn as nn


NAME = "r8_mesa_rls_fastweights"
DESCRIPTION = (
    "Champion path-delta fastweights with the Widrow-Hoff write upgraded to exponentially "
    "weighted recursive least squares (mesa-layer): per-memory inverse-covariance state, "
    "Kalman/RLS write gain, data-dependent forgetting gate — decorrelates interference "
    "between the correlated command/path keys of shell exploration."
)


class MesaRLSFastweights(nn.Module):
    def __init__(self, d=176, layers=2, key_d=64, ffn_mult=2, lam_min=0.75, p_cap=100.0,
                 **unused):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = 768
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.lam_min = float(lam_min)
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
            self.ff.append(nn.Sequential(nn.Linear(self.d, ffn_h), nn.SiLU(),
                                         nn.Linear(ffn_h, self.d)))
            self.ff_gates.append(nn.Linear(self.d, self.d))

        self.content_read = nn.Linear(self.d, self.key_d, bias=False)
        self.content_write = nn.Linear(self.d, self.key_d, bias=False)
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.read_mix = nn.Linear(self.d, 2)
        self.read_to_h = nn.Linear(self.D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d, self.d)
        self.out_norm = nn.LayerNorm(self.d)

        self.write_gate = nn.Linear(2 * self.d, 1)     # beta: RLS sample weight
        self.forget_gate = nn.Linear(2 * self.d, 1)    # lam:  data-dependent forgetting
        self.direct_gate = nn.Linear(2 * self.d, 1)
        self.head = nn.Linear(self.d, self.D)

        self.logit_decay = nn.Parameter(torch.tensor(math.log(0.985 / 0.015)))  # rho (value)
        self.p0_raw = nn.Parameter(torch.tensor(0.5413))  # softplus -> P0 scale ~= 1.0
        self.register_buffer("p_cap", torch.tensor(float(p_cap)))

        nn.init.constant_(self.write_gate.bias, 1.0)
        # lam = lam_min + (1-lam_min)*sigmoid(.); bias 2.5 -> lam ~= 0.98 at init
        nn.init.constant_(self.forget_gate.bias, 2.5)
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

    def _rls_write(self, mem, P, key, value, beta, lam, active, vdecay):
        """Exponentially weighted RLS write with sample weight beta and forgetting lam.
        mem [B,kd,D]; P [B,kd,kd] (inverse covariance); key [B,kd] unit; value [B,D];
        beta, lam [B]; active [B] bool. Returns (mem', P')."""
        mem_d = vdecay * mem                                        # value forgetting (rho)
        err = value - self._read(mem_d, key)                        # [B,D] innovation
        Pk = torch.bmm(P, key.unsqueeze(2)).squeeze(2)              # [B,kd]
        kPk = (key * Pk).sum(-1)                                    # [B] >= 0 (P PSD)
        denom = (lam / beta.clamp_min(1e-4) + kPk).clamp_min(1e-3)  # [B]
        g = Pk / denom.unsqueeze(1)                                 # [B,kd] Kalman gain
        mem_new = mem_d + g.unsqueeze(2) * err.unsqueeze(1)         # rank-1 LS correction
        P_new = (P - g.unsqueeze(2) * Pk.unsqueeze(1)) / lam.view(-1, 1, 1)
        P_new = 0.5 * (P_new + P_new.transpose(1, 2))               # keep symmetric
        diag = torch.diagonal(P_new, dim1=1, dim2=2).mean(-1).clamp_min(1e-6)
        P_new = P_new * (self.p_cap / diag).clamp(max=1.0).view(-1, 1, 1)  # bound growth
        a = active.to(mem.dtype).view(-1, 1, 1)
        return mem * (1.0 - a) + mem_new * a, P * (1.0 - a) + P_new * a

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        t = types.long().clamp(0, 1)
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        states = [torch.zeros(B, self.d, device=device, dtype=x.dtype)
                  for _ in range(self.layers)]
        mem_content = torch.zeros(B, self.key_d, self.D, device=device, dtype=dtype)
        mem_path = torch.zeros(B, self.key_d, self.D, device=device, dtype=dtype)
        p0 = torch.nn.functional.softplus(self.p0_raw).to(dtype)
        eye = torch.eye(self.key_d, device=device, dtype=dtype)
        P_content = (p0 * eye).expand(B, -1, -1).clone()
        P_path = (p0 * eye).expand(B, -1, -1).clone()

        pending_c = torch.zeros(B, self.key_d, device=device, dtype=x.dtype)
        pending_p = torch.zeros(B, self.key_d, device=device, dtype=x.dtype)
        pending_h = torch.zeros(B, self.d, device=device, dtype=x.dtype)
        pending_valid = torch.zeros(B, device=device, dtype=torch.bool)

        preds = []
        hs = []
        vdecay = torch.sigmoid(self.logit_decay).to(dtype)

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
            read_content = self._read(mem_content, q_content.to(dtype))
            read_path = self._read(mem_path, q_path.to(dtype))

            mix = torch.softmax(self.read_mix(core), dim=-1)
            target_read = mix[:, 0:1].to(dtype) * read_content + mix[:, 1:2].to(dtype) * read_path
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
                gate_in = torch.cat([pending_h, core], dim=-1)
                beta = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1).to(dtype)
                lam = (self.lam_min + (1.0 - self.lam_min)
                       * torch.sigmoid(self.forget_gate(gate_in)).squeeze(-1)).to(dtype)
                value = tok_emb[:, i, :]
                mem_content, P_content = self._rls_write(
                    mem_content, P_content, pending_c.to(dtype), value, beta, lam,
                    pair_active, vdecay)
                mem_path, P_path = self._rls_write(
                    mem_path, P_path, pending_p.to(dtype), value, beta, lam,
                    pair_active, vdecay)

            pending_valid = torch.where(is_obs, torch.zeros_like(pending_valid), pending_valid)

        return torch.stack(preds, dim=1), torch.stack(hs, dim=1)


def build(**params):
    return MesaRLSFastweights(**params)
