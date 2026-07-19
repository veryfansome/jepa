"""
Tier-A exact rewrite of r7_path_delta_fastweights_codex: chunk-scan fastweights.

Same parameters, same mathematics, restructured for kernel-level parallelism.
Three structural observations about the champion make this possible:

1. The trunk GRU recurrence never consumes the memory reads (reads only feed the
   per-position outputs h_i / pred_i), so the trunk can run first and the ENTIRE
   memory system + readout can be computed afterwards in batched form.
2. The delta-rule write is a linear recurrence in the memory matrix,
   M_i = (g_i I - a_i k_i k_i^T) M_{i-1} + a_i k_i v_i^T  (g_i = decay at write
   steps, 1 elsewhere; a_i = 0 at non-write steps), i.e. exactly the gated
   DeltaNet form. All L reads and writes therefore collapse into a handful of
   bmm's via the chunkwise WY/UT representation (Yang et al., "Parallelizing
   Linear Transformers with the Delta Rule over Sequence Length", NeurIPS 2024;
   Gated DeltaNet, ICLR 2025) — exact linear algebra, float reassociation only.
3. The champion's pending (cmd -> obs) pair matching is a closed-form
   scatter/gather: cummax over valid-cmd/valid-obs position indices yields, for
   every obs position, the index of its pending command (or none). Garbage keys
   at inactive steps get a_i = 0, which provably zeroes their pseudo-values in
   the triangular solve, so they contribute nothing — no python-level branching
   on tensor values remains (torch.compile-friendly).

The remaining serial part is the trunk: per position, the two typed GRU cells
are fused into one pair of addmm's (concatenated cmd|obs weights; the gate
nonlinearities commute with the type-select torch.where), and layer-0's
input-side matmul is hoisted out of the loop for all positions at once.

state_dict keys/shapes are IDENTICAL to the champion; forward matches to float
reassociation (verified < 1e-5 max abs diff at init weights, plus fuzz over
random types/padding patterns, trained-weights and decay-drift checks; measured
3.6x on the fixed MPS bench protocol vs the champion in the same run).
"""

import math
import torch
import torch.nn as nn


NAME = "r9_chunkscan_fastweights_exact"
DESCRIPTION = (
    "Tier-A exact rewrite of the path-delta fastweights champion: fused dual-GRU "
    "trunk + chunkwise-parallel (WY/UT) delta-rule memories + fully batched "
    "readout; identical state_dict and forward up to float reassociation."
)


class ChunkScanFastweights(nn.Module):
    def __init__(self, d=176, layers=2, key_d=64, ffn_mult=2, chunk_c=16, **unused):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = 768
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.chunk_c = max(1, int(chunk_c))
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

    def _chunked_delta(self, K, V, Q, a, gamma):
        """All-positions reads of the gated delta-rule memory, chunkwise-exact.

        Recurrence (per position i):  M_i = (gamma_i I - a_i k_i k_i^T) M_{i-1}
                                            + a_i k_i v_i^T,   M_{-1} = 0
        Read at position i:           r_i = q_i^T M_{i-1}      (strictly causal)

        Within a chunk, scale S_p = M_p / Lam_p with Lam_p = prod_{q<=p} gamma_q,
        giving S_p = S_{p-1} + k_p w_p^T with pseudo-values w solving the unit
        lower-triangular system (I + tril_strict(diag(a/gamma) K K^T)) W =
        diag(a/Lam) V - diag(a/gamma) K M_in.  The inverse of I + N (N strictly
        lower triangular, nilpotent) is the finite Neumann series, computed by
        squaring in log2(chunk) bmm's.  Everything is exact linear algebra.
        K,Q: [B*,L,k]  V: [B*,L,Dv]  a,gamma: [B*,L]  ->  reads [B*,L,Dv].
        """
        Bx, L, kd = K.shape
        Dv = V.shape[-1]
        C = self.chunk_c
        M = K.new_zeros(Bx, kd, Dv)
        outs = []
        for s in range(0, L, C):
            e = min(s + C, L)
            c = e - s
            Kc = K[:, s:e]
            Qc = Q[:, s:e]
            Vc = V[:, s:e]
            ac = a[:, s:e]
            gc = gamma[:, s:e]

            lam = torch.cumprod(gc, dim=1)                       # [Bx,c]
            b = ac / gc                                          # beta_p
            KK = torch.bmm(Kc, Kc.transpose(1, 2))               # [Bx,c,c]
            A = -torch.tril(KK * b.unsqueeze(-1), -1)            # A = -N, nilpotent
            # Tinv = sum_m A^m  (finite: A^c = 0), by repeated squaring
            Tinv = torch.eye(c, device=K.device, dtype=K.dtype).unsqueeze(0) + A
            Apow = A
            m = 2
            while m < c:
                Apow = torch.bmm(Apow, Apow)
                Tinv = Tinv + torch.bmm(Tinv, Apow)
                m *= 2

            rhs = (ac / lam).unsqueeze(-1) * Vc - b.unsqueeze(-1) * torch.bmm(Kc, M)
            W = torch.bmm(Tinv, rhs)                             # pseudo-values [Bx,c,Dv]

            lam_shift = torch.cat([lam.new_ones(Bx, 1), lam[:, :-1]], dim=1)
            QK = torch.tril(torch.bmm(Qc, Kc.transpose(1, 2)), -1)
            outs.append(lam_shift.unsqueeze(-1) * (torch.bmm(Qc, M) + torch.bmm(QK, W)))

            M = lam[:, -1].view(Bx, 1, 1) * (M + torch.bmm(Kc.transpose(1, 2), W))
        return torch.cat(outs, dim=1)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype
        if L == 0:
            return tok_emb.new_zeros(B, 0, self.D), tok_emb.new_zeros(B, 0, self.d)

        t = types.long().clamp(0, 1)
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(B, L, device=device, dtype=torch.bool)
        tc = t == 0                                             # cmd-type mask [B,L]

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where(tc.unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        # ---- trunk: serial over positions, dual GRU cells fused per layer ----
        d = self.d
        Wi, bi, Wh, bh = [], [], [], []
        for l in range(self.layers):
            cc, oc = self.cmd_cells[l], self.obs_cells[l]
            Wi.append(torch.cat([cc.weight_ih, oc.weight_ih], 0))    # [6d, d]
            bi.append(torch.cat([cc.bias_ih, oc.bias_ih], 0))
            Wh.append(torch.cat([cc.weight_hh, oc.weight_hh], 0))
            bh.append(torch.cat([cc.bias_hh, oc.bias_hh], 0))

        # layer-0 input-side gates for ALL positions in one matmul, type-selected
        gi0 = torch.addmm(bi[0], x.reshape(B * L, d), Wi[0].t()).view(B, L, 6 * d)
        gi0 = torch.where(tc.unsqueeze(-1), gi0[..., : 3 * d], gi0[..., 3 * d:])
        gi0_cols = gi0.unbind(1)
        x_cols = x.unbind(1)
        tcm = tc.unsqueeze(-1).unbind(1)                             # [B,1] per position
        vm = valid.unsqueeze(-1).unbind(1)

        states = [x.new_zeros(B, d) for _ in range(self.layers)]
        cores = []
        for i in range(L):
            z = x_cols[i]
            m = tcm[i]
            for l in range(self.layers):
                prev = states[l]
                if l == 0:
                    gi = gi0_cols[i]
                else:
                    gi = torch.addmm(bi[l], z, Wi[l].t())
                    gi = torch.where(m, gi[:, : 3 * d], gi[:, 3 * d:])
                gh = torch.addmm(bh[l], prev, Wh[l].t())
                gh = torch.where(m, gh[:, : 3 * d], gh[:, 3 * d:])
                # GRU gate math (elementwise -> commutes with the type-select)
                rz = torch.sigmoid(gi[:, : 2 * d] + gh[:, : 2 * d])
                r = rz[:, :d]
                zg = rz[:, d:]
                n = torch.tanh(gi[:, 2 * d:] + r * gh[:, 2 * d:])
                cand = torch.lerp(n, prev, zg)                       # (1-z)*n + z*prev
                cand = self.cell_norms[l](cand)
                nrm = self.ff_norms[l](cand)
                cand = cand + torch.sigmoid(self.ff_gates[l](nrm)) * self.ff[l](nrm)
                cand = torch.where(vm[i], cand, prev)
                states[l] = cand
                z = cand
            cores.append(states[-1])
        core = torch.stack(cores, dim=1)                             # [B,L,d]

        # ---- pending (cmd -> obs) pair matching, closed form ----
        idx = torch.arange(L, device=device)
        is_cmd = tc & valid
        is_obs = (~tc) & valid
        neg1 = idx.new_full((), -1)
        last_cmd = torch.where(is_cmd, idx, neg1).cummax(dim=1).values          # [B,L]
        last_obs = torch.where(is_obs, idx, neg1).cummax(dim=1).values
        prev_last_obs = torch.cat([last_obs.new_full((B, 1), -1), last_obs[:, :-1]], dim=1)
        write_active = is_obs & (last_cmd > prev_last_obs)
        j = last_cmd.clamp_min(0).unsqueeze(-1)                                 # gather index

        cw = self._unit(self.content_write(x))                                  # write keys, all positions
        pw = self._unit(self.path_write(core))
        Kc = torch.gather(cw, 1, j.expand(B, L, self.key_d))                    # pending content key
        Kp = torch.gather(pw, 1, j.expand(B, L, self.key_d))                    # pending path key
        pending_h = torch.gather(core, 1, j.expand(B, L, d))
        amount = torch.sigmoid(self.write_gate(torch.cat([pending_h, core], dim=-1))).squeeze(-1)

        wa = write_active
        decay = torch.sigmoid(self.logit_decay).to(dtype)
        one = torch.ones((), device=device, dtype=dtype)
        a = amount.to(dtype) * wa.to(dtype)                                     # 0 at non-write steps
        gamma = torch.where(wa, decay, one)                                     # decay only at write steps

        q_c = self._unit(self.content_read(x))
        q_p = self._unit(self.path_read(core))

        # both memories share (a, gamma, values); stack them on the batch dim
        K2 = torch.cat([Kc.to(dtype), Kp.to(dtype)], dim=0)
        Q2 = torch.cat([q_c.to(dtype), q_p.to(dtype)], dim=0)
        V2 = torch.cat([tok_emb, tok_emb], dim=0)
        a2 = torch.cat([a, a], dim=0)
        g2 = torch.cat([gamma, gamma], dim=0)
        R2 = self._chunked_delta(K2, V2, Q2, a2, g2)
        read_content, read_path = R2[:B], R2[B:]

        # ---- readout: identical per-position ops, batched over [B,L,...] ----
        mix = torch.softmax(self.read_mix(core), dim=-1)
        target_read = mix[..., 0:1] * read_content + mix[..., 1:2] * read_path
        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([core, mem_h], dim=-1)
        h = self.out_norm(core + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)
        pred = self.head(h) + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read

        vmask = valid.unsqueeze(-1)
        pred = torch.where(vmask, pred, torch.zeros_like(pred))
        h = torch.where(vmask, h, torch.zeros_like(h))
        return pred, h


def build(**params):
    return ChunkScanFastweights(**params)
