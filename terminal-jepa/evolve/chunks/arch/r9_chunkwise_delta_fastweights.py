"""
Chunkwise-parallel exact rewrite of the champion path-delta fastweights arch (TIER A).

Key observation: in r7_path_delta_fastweights_codex the GRU trunk never reads the
associative memories — the memory reads feed only the output head. So the per-position
python loop over BOTH delta-rule memories is unnecessary: after one sequential trunk
pass, every write key, value, write-gate amount and decay is known for all positions,
and the memory recurrence

    M_j = gamma_j * M_{j-1} + b_j * k_j (v_j - M_{j-1}^T k_j)^T ,
    gamma_j = decay if a (cmd,obs) pair completes at j else 1,  b_j = amount_j * active_j

is the (gated) DeltaNet linear recurrence in the memory matrix. It is solved in closed
form with the WY/UT-transform: the pseudo-values U satisfy a unit-lower-triangular
system (I + T) U = diag(b) V with T[j,i] = b_j * (G_{j-1}/G_i) * (k_i . k_j) (G =
cumulative decay), one batched triangular solve; all reads at every position are then
one masked bmm. Both memories are stacked into a single 2B-batched solve/bmm. The GRU
trunk stays sequential but is dispatch-minimized (typed cmd/obs cells fused into one
matmul pair per layer, layer-0 input projections hoisted, gate+FFN-in fused). Exact
same parameters, same equations, same causality; float reassociation only.

References: Yang et al., "Parallelizing Linear Transformers with the Delta Rule over
Sequence Length" (arXiv:2406.06484); Yang et al., "Gated Delta Networks"
(arXiv:2412.06464); Schlag et al., "Linear Transformers Are Secretly Fast Weight
Programmers" (arXiv:2102.11174).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


NAME = "r9_chunkwise_delta_fastweights"
DESCRIPTION = (
    "Tier-A exact rewrite of r7_path_delta_fastweights: sequential dispatch-minimized "
    "GRU trunk + both delta-rule fastweight memories computed chunkwise-parallel via a "
    "DeltaNet-style WY/UT triangular solve (one batched solve + two bmms replaces the "
    "per-position memory loop)."
)


class ChunkwiseDeltaFastweights(nn.Module):
    # __init__ is intentionally verbatim-identical to r7_path_delta_fastweights_codex:
    # identical state_dict keys/shapes AND identical RNG draw order at build time.
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

    def _trunk(self, x, t, valid):
        """Sequential typed-GRU trunk, dispatch-minimized. Exact same math as the
        champion's per-position GRUCell/FFN stack; returns last-layer state at every
        position [B, L, d]."""
        B, L, d = x.shape
        is_cmd = (t == 0)  # champion selects the typed cell on raw type, not validity

        # Fuse the cmd/obs cell weights: one input matmul + one hidden matmul per
        # layer per position computes BOTH typed candidates (gate order r,z,n).
        Wih, bih, Whh, bhh, Wg, bg = [], [], [], [], [], []
        for l in range(self.layers):
            c, o = self.cmd_cells[l], self.obs_cells[l]
            Wih.append(torch.cat([c.weight_ih, o.weight_ih], 0))
            bih.append(torch.cat([c.bias_ih, o.bias_ih], 0))
            Whh.append(torch.cat([c.weight_hh, o.weight_hh], 0))
            bhh.append(torch.cat([c.bias_hh, o.bias_hh], 0))
            # fuse the FFN gate and FFN first linear into one matmul
            Wg.append(torch.cat([self.ff_gates[l].weight, self.ff[l][0].weight], 0))
            bg.append(torch.cat([self.ff_gates[l].bias, self.ff[l][0].bias], 0))

        # hoist ALL layer-0 input-side projections (position-independent)
        gi0 = torch.matmul(x, Wih[0].t()) + bih[0]  # [B, L, 6d]

        states = [x.new_zeros(B, d) for _ in range(self.layers)]
        cores = []
        for i in range(L):
            vi = valid[:, i].unsqueeze(-1)
            ci = is_cmd[:, i].unsqueeze(-1)  # [B, 1]
            z = None
            for l in range(self.layers):
                prev = states[l]
                gi = gi0[:, i] if l == 0 else torch.addmm(bih[l], z, Wih[l].t())
                gh = torch.addmm(bhh[l], prev, Whh[l].t())
                gi4 = gi.view(B, 2, 3, d)
                gh4 = gh.view(B, 2, 3, d)
                rz = torch.sigmoid(gi4[:, :, :2] + gh4[:, :, :2])
                r, zz = rz[:, :, 0], rz[:, :, 1]
                n = torch.tanh(gi4[:, :, 2] + r * gh4[:, :, 2])
                both = (1.0 - zz) * n + zz * prev.unsqueeze(1)  # [B, 2, d]
                cand = torch.where(ci, both[:, 0], both[:, 1])
                cand = self.cell_norms[l](cand)

                nrm = self.ff_norms[l](cand)
                gg = torch.addmm(bg[l], nrm, Wg[l].t())
                gate = torch.sigmoid(gg[:, :d])
                mid = F.silu(gg[:, d:])
                cand = cand + gate * torch.addmm(self.ff[l][2].bias, mid, self.ff[l][2].weight.t())
                cand = torch.where(vi, cand, prev)

                states[l] = cand
                z = cand
            cores.append(states[-1])
        return torch.stack(cores, dim=1)  # [B, L, d]

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype
        if L == 0:
            return tok_emb.new_zeros(B, 0, self.D), tok_emb.new_zeros(B, 0, self.d)

        t = types.long().clamp(0, 1)
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        cores = self._trunk(x, t, valid)  # [B, L, d]

        # ---- all keys/queries/gates, fully batched ----
        kd = self.key_d
        qc = self._unit(self.content_read(x))     # [B, L, kd]
        qp = self._unit(self.path_read(cores))    # [B, L, kd]
        wc = self._unit(self.content_write(x))    # write key as computed at each position
        wp = self._unit(self.path_write(cores))

        # Pending-pair bookkeeping, vectorized. A write fires at a VALID obs position
        # whose most recent valid cmd is more recent than the most recent valid obs
        # strictly before it (i.e. pending_valid not yet cleared) — exactly the
        # champion's pending_c/p/h + pending_valid state machine.
        is_cmd_v = (t == 0) & valid
        is_obs_v = (t == 1) & valid
        idx = torch.arange(L, device=device)
        neg1 = torch.full((1,), -1, device=device, dtype=torch.long)
        cmd_idx = torch.where(is_cmd_v, idx.expand(B, L), neg1)
        last_cmd = torch.cummax(cmd_idx, dim=1).values                  # [B, L]
        obs_idx = torch.where(is_obs_v, idx.expand(B, L), neg1)
        last_obs = torch.cummax(obs_idx, dim=1).values
        last_obs_prev = torch.cat([last_obs.new_full((B, 1), -1), last_obs[:, :-1]], dim=1)
        pair = is_obs_v & (last_cmd >= 0) & (last_cmd > last_obs_prev)  # [B, L]

        gidx = last_cmd.clamp_min(0)
        pend_h = torch.gather(cores, 1, gidx.unsqueeze(-1).expand(B, L, self.d))
        kc = torch.gather(wc, 1, gidx.unsqueeze(-1).expand(B, L, kd))
        kp = torch.gather(wp, 1, gidx.unsqueeze(-1).expand(B, L, kd))

        amount = torch.sigmoid(self.write_gate(torch.cat([pend_h, cores], dim=-1))).squeeze(-1)
        pf = pair.to(dtype)
        b = amount * pf                                                  # [B, L]
        decay = torch.sigmoid(self.logit_decay).to(dtype)
        gamma = 1.0 + pf * (decay - 1.0)                                 # decay iff a write fires
        G = torch.cumprod(gamma, dim=1)                                  # cumulative decay
        Gs = G.clamp_min(1e-30)                                          # NaN/inf-safe divisor
        Gprev = torch.cat([G.new_ones(B, 1), G[:, :-1]], dim=1)
        ratio = Gprev.unsqueeze(2) / Gs.unsqueeze(1)                     # [B, Lp, Li] = G_{p-1}/G_i

        # ---- both memories in ONE 2B-batched chunkwise solve (DeltaNet WY/UT) ----
        K2 = torch.cat([kc, kp], dim=0)                                  # [2B, L, kd]
        Q2 = torch.cat([qc, qp], dim=0)
        b2 = b.repeat(2, 1)
        ratio2 = ratio.repeat(2, 1, 1)
        V2 = tok_emb.repeat(2, 1, 1)                                     # raw obs embeddings

        KK = torch.bmm(K2, K2.transpose(1, 2))                           # k_i . k_j
        Tm = torch.tril(b2.unsqueeze(2) * ratio2 * KK, diagonal=-1)      # strictly lower
        RHS = b2.unsqueeze(-1) * V2                                      # [2B, L, D]
        # (I + Tm) U = RHS  — unit-lower-triangular; U are the delta-rule pseudo-values
        U = torch.linalg.solve_triangular(Tm, RHS, upper=False, unitriangular=True)

        QK = torch.bmm(Q2, K2.transpose(1, 2))                           # q_p . k_i
        Ar = torch.tril(ratio2 * QK, diagonal=-1)                        # reads see writes < p only
        reads = torch.bmm(Ar, U)                                         # [2B, L, D]
        read_content, read_path = reads[:B], reads[B:]

        # ---- output head, fully batched (identical equations) ----
        mix = torch.softmax(self.read_mix(cores), dim=-1)
        target_read = mix[..., 0:1] * read_content + mix[..., 1:2] * read_path
        mem_h = self.read_to_h(target_read.to(x.dtype))

        fuse_in = torch.cat([cores, mem_h], dim=-1)
        h = self.out_norm(cores + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)
        pred = self.head(h) + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read

        vm = valid.unsqueeze(-1)
        pred = torch.where(vm, pred, torch.zeros_like(pred))
        h = torch.where(vm, h, torch.zeros_like(h))
        return pred, h


def build(**params):
    return ChunkwiseDeltaFastweights(**params)
