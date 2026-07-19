"""
R9 Tier-A exact rewrite of the champion path-delta fastweight predictor.

Same parameters (identical state_dict keys/shapes) and same update equations as
r7_path_delta_fastweights_codex; only the compute is restructured to kill dispatch
overhead (the champion issues ~212k tiny aten ops per training step, 53k of them
bare aten::select from python indexing in the position loop):

  1. The read/fuse/output path (read_mix, read_to_h, fuse_gate, direct_gate,
     out_norm, head) never feeds back into the recurrence -> it is hoisted out of
     the loop and computed ONCE, batched over all L positions (the d->768 head
     matmul disappears from the loop entirely).
  2. All x-derived per-position quantities (layer-0 GRU input gates for BOTH typed
     cells, content read/write keys) are computed upfront in ONE fused matmul.
  3. The pending (cmd -> obs) pairing state is a pure function of types/key_pad:
     the whole schedule (last-cmd index j(i), pair_active) is precomputed on CPU
     with two cummax ops — no per-step `.any()` device sync — and the pending
     key/value/gate lookups become batched `gather`s over position-parallel
     projections (write_gate is split linearly across its cat input).
  4. The two associative memories are stacked into one [2B, key_d, 768] tensor;
     each step's reads (and the write's old-value read) are ONE bmm, and the
     delta-rule write is ONE baddbmm.
  5. GRUCell is decomposed into a precomputed input-gate addmm + one hidden addmm
     per (position, layer); with per-position type uniformity flags (CPU-known)
     only the selected typed cell is evaluated — value- and gradient-exact, since
     torch.where routes zero gradient to the unselected branch.

Float reassociation only (fused/cat matmuls, baddbmm, n + z*(h-n) form of the GRU
blend); passes the Tier-A eq gate (< 1e-4 max abs forward diff vs the champion).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NAME = "r9_path_delta_fastweights_exact_fused"
DESCRIPTION = (
    "Tier-A exact rewrite of r7_path_delta_fastweights_codex: identical params/"
    "equations, restructured compute — hoisted batched output path, fused x-side "
    "projections, CPU-precomputed pair schedule (no per-step sync), stacked dual-"
    "memory bmm/baddbmm reads+writes, and per-type GRU cell decomposition."
)


class PathDeltaFastweightsExactFused(nn.Module):
    def __init__(self, d=176, layers=2, key_d=64, ffn_mult=2, **unused):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = 768
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        ffn_h = max(self.d, int(ffn_mult * self.d))
        self.ffn_h = ffn_h

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

        self._pe_cache = {}  # plain attr (not a buffer): keeps state_dict identical

    def _positional(self, L, device, dtype):
        key = (L, str(device), str(dtype))
        pe = self._pe_cache.get(key)
        if pe is not None:
            return pe
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d // 2])
        self._pe_cache[key] = pe
        return pe

    @staticmethod
    def _unit(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _gru(ig, hg, h, d):
        # exact GRUCell math: r=sig(ir+hr), z=sig(iz+hz), n=tanh(in + r*hn),
        # h' = (1-z)*n + z*h  ==  n + z*(h-n)   (float reassociation only)
        rz = torch.sigmoid(ig[:, : 2 * d] + hg[:, : 2 * d])
        r, zg = rz[:, :d], rz[:, d:]
        n = torch.tanh(ig[:, 2 * d :] + r * hg[:, 2 * d :])
        return n + zg * (h - n)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype
        d, k, D = self.d, self.key_d, self.D
        if L == 0:
            return tok_emb.new_zeros(B, 0, D), tok_emb.new_zeros(B, 0, d)

        t = types.long().clamp(0, 1)
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        # ---- schedule (pure function of types/key_pad; ints/bools only, no float data,
        # so this is causal-safe) — computed on CPU once, BEFORE any device compute,
        # to avoid every per-step `.any()` sync of the reference loop ----
        t_cpu = t.detach().cpu()
        valid_cpu = valid.detach().cpu()
        is_cmd_cpu = (t_cpu == 0) & valid_cpu
        is_obs_cpu = (t_cpu == 1) & valid_cpu
        idx = torch.arange(L)
        neg1 = idx.new_full((), -1)
        jc = torch.cummax(torch.where(is_cmd_cpu, idx, neg1), dim=1).values  # last valid cmd <= i
        jo = torch.cummax(torch.where(is_obs_cpu, idx, neg1), dim=1).values  # last valid obs <= i
        jo_prev = torch.cat([jo.new_full((B, 1), -1), jo[:, :-1]], dim=1)    # last valid obs <  i
        # pending_valid at obs i  <=>  a valid cmd exists <= i, newer than the last valid obs < i
        pair_cpu = is_obs_cpu & (jc >= 0) & (jc > jo_prev)
        write_any = pair_cpu.any(dim=0).tolist()
        write_all = pair_cpu.all(dim=0).tolist()
        cmd_col = t_cpu == 0
        all_cmd = cmd_col.all(dim=0).tolist()
        all_obs = (~cmd_col).all(dim=0).tolist()
        no_pad = bool(valid_cpu.all())
        any_write = any(write_any)
        any_mixed = not all(all_cmd[i] or all_obs[i] for i in range(L))

        jp1 = (jc + 1).clamp_min(0).to(device) if any_write else None  # 0 -> zero-pad row
        pair_dev = pair_cpu.to(device) if any_write else None

        # ---- input stage (identical to reference) ----
        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        tm = (t == 0).unsqueeze(-1)
        x = torch.where(tm, cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        # ---- one fused matmul for every x-derived per-position quantity ----
        Wx = torch.cat(
            [self.cmd_cells[0].weight_ih, self.obs_cells[0].weight_ih,
             self.content_read.weight, self.content_write.weight], dim=0)
        bx = torch.cat(
            [self.cmd_cells[0].bias_ih, self.obs_cells[0].bias_ih, x.new_zeros(2 * k)], dim=0)
        xf = F.linear(x, Wx, bx)                       # [B, L, 6d + 2k]
        ig0c = xf[..., : 3 * d]
        ig0o = xf[..., 3 * d : 6 * d]
        qc_all = self._unit(xf[..., 6 * d : 6 * d + k])   # content read keys (all positions)
        kc_all = self._unit(xf[..., 6 * d + k :])         # content write keys (all positions)

        # ---- recurrent trunk (the only truly sequential float compute) ----
        ig0c_u = ig0c.unbind(1)
        ig0o_u = ig0o.unbind(1)
        vexp_u = valid.unsqueeze(-1).unbind(1) if not no_pad else None
        tm_u = tm.unbind(1) if any_mixed else None

        w_hh = [(self.cmd_cells[l].weight_hh, self.obs_cells[l].weight_hh) for l in range(self.layers)]
        b_hh = [(self.cmd_cells[l].bias_hh, self.obs_cells[l].bias_hh) for l in range(self.layers)]
        w_ih = [(self.cmd_cells[l].weight_ih, self.obs_cells[l].weight_ih) for l in range(self.layers)]
        b_ih = [(self.cmd_cells[l].bias_ih, self.obs_cells[l].bias_ih) for l in range(self.layers)]
        ff_w1 = [torch.cat([self.ff[l][0].weight, self.ff_gates[l].weight], dim=0) for l in range(self.layers)]
        ff_b1 = [torch.cat([self.ff[l][0].bias, self.ff_gates[l].bias], dim=0) for l in range(self.layers)]
        ff_w2 = [self.ff[l][2].weight for l in range(self.layers)]
        ff_b2 = [self.ff[l][2].bias for l in range(self.layers)]
        cn = list(self.cell_norms)
        fn = list(self.ff_norms)
        ffn_h = self.ffn_h
        gru = self._gru

        states = [x.new_zeros(B, d) for _ in range(self.layers)]
        cores = []
        for i in range(L):
            z = None
            for l in range(self.layers):
                prev = states[l]
                if all_cmd[i]:
                    ig = ig0c_u[i] if l == 0 else F.linear(z, w_ih[l][0], b_ih[l][0])
                    hg = F.linear(prev, w_hh[l][0], b_hh[l][0])
                    cand = gru(ig, hg, prev, d)
                elif all_obs[i]:
                    ig = ig0o_u[i] if l == 0 else F.linear(z, w_ih[l][1], b_ih[l][1])
                    hg = F.linear(prev, w_hh[l][1], b_hh[l][1])
                    cand = gru(ig, hg, prev, d)
                else:
                    igc = ig0c_u[i] if l == 0 else F.linear(z, w_ih[l][0], b_ih[l][0])
                    igo = ig0o_u[i] if l == 0 else F.linear(z, w_ih[l][1], b_ih[l][1])
                    hgc = F.linear(prev, w_hh[l][0], b_hh[l][0])
                    hgo = F.linear(prev, w_hh[l][1], b_hh[l][1])
                    cand = torch.where(tm_u[i], gru(igc, hgc, prev, d), gru(igo, hgo, prev, d))
                cand = cn[l](cand)
                n = fn[l](cand)
                fgf = F.linear(n, ff_w1[l], ff_b1[l])                   # [B, ffn_h + d]
                ffo = F.linear(F.silu(fgf[:, :ffn_h]), ff_w2[l], ff_b2[l])
                cand = cand + torch.sigmoid(fgf[:, ffn_h:]) * ffo
                if not no_pad:
                    cand = torch.where(vexp_u[i], cand, prev)
                states[l] = cand
                z = cand
            cores.append(states[-1])
        core_seq = torch.stack(cores, dim=1)                             # [B, L, d]

        # ---- one fused matmul for every core-derived per-position quantity ----
        # write_gate(cat[pending_h, core]) splits linearly: W[:, :d] @ pending_h + W[:, d:] @ core + b;
        # the pending part is the same projection evaluated at the pending cmd's core -> gatherable.
        Wc = torch.cat(
            [self.path_read.weight, self.path_write.weight, self.read_mix.weight,
             self.write_gate.weight[:, d:], self.write_gate.weight[:, :d]], dim=0)
        bc = torch.cat([x.new_zeros(2 * k), self.read_mix.bias, x.new_zeros(2)], dim=0)
        cf = F.linear(core_seq, Wc, bc)                                  # [B, L, 2k + 4]
        qp_all = self._unit(cf[..., :k])                                 # path read keys
        kp_all = self._unit(cf[..., k : 2 * k])                          # path write keys
        mix = torch.softmax(cf[..., 2 * k : 2 * k + 2], dim=-1)
        wg_core = cf[..., 2 * k + 2]
        wg_pend_src = cf[..., 2 * k + 3]

        # ---- memory recurrence: both memories stacked -> one bmm per step ----
        q2 = torch.cat([qc_all, qp_all], dim=0)                          # [2B, L, k]
        decay = torch.sigmoid(self.logit_decay).to(dtype)
        memcat = torch.zeros(2 * B, k, D, device=device, dtype=dtype)
        if any_write:
            pad_row = lambda tns: torch.cat([tns.new_zeros(B, 1, tns.shape[-1]), tns], dim=1)
            gk = jp1.unsqueeze(-1).expand(B, L, k)
            pend_c = torch.gather(pad_row(kc_all), 1, gk)
            pend_p = torch.gather(pad_row(kp_all), 1, gk)
            wg_pend = torch.gather(torch.cat([wg_pend_src.new_zeros(B, 1), wg_pend_src], dim=1), 1, jp1)
            amount = torch.sigmoid(wg_pend + wg_core + self.write_gate.bias)     # [B, L]
            k2 = torch.cat([pend_c, pend_p], dim=0)                              # [2B, L, k]
            a_amt = torch.cat([amount, amount], dim=0).to(dtype)
            ka = (k2.to(dtype) * a_amt.unsqueeze(-1))
            qk = torch.stack([q2, k2], dim=2)                                    # [2B, L, 2, k]
            qk_u = qk.unbind(1)                                                  # per-pos [2B, 2, k]
            ka_u = ka.unsqueeze(-1).unbind(1)                                    # per-pos [2B, k, 1]
            v2_u = torch.cat([tok_emb, tok_emb], dim=0).unsqueeze(2).unbind(1)   # per-pos [2B, 1, D]
            act = pair_dev.to(dtype)
            act2_u = torch.cat([act, act], dim=0).view(2 * B, L, 1, 1).unbind(1) # per-pos [2B, 1, 1]
        q2_u = q2.unsqueeze(2).unbind(1)                                         # per-pos [2B, 1, k]

        reads = []
        for i in range(L):
            if any_write and write_any[i]:
                out = torch.bmm(qk_u[i], memcat)                # rows: [read | old] in one bmm
                reads.append(out[:, 0:1])
                old = out[:, 1:2]
                delta = v2_u[i] - old
                cand = torch.baddbmm(decay * memcat, ka_u[i], delta)
                if write_all[i]:
                    memcat = cand
                else:
                    a = act2_u[i]
                    memcat = memcat * (1.0 - a) + cand * a
            else:
                reads.append(torch.bmm(q2_u[i], memcat))
        reads = torch.cat(reads, dim=1)                                  # [2B, L, D]
        read_content, read_path = reads[:B], reads[B:]

        # ---- hoisted output path: batched over all positions at once ----
        target_read = mix[..., 0:1] * read_content + mix[..., 1:2] * read_path
        mem_h = self.read_to_h(target_read.to(x.dtype))
        Wfd = torch.cat([self.fuse_gate.weight, self.direct_gate.weight], dim=0)
        bfd = torch.cat([self.fuse_gate.bias, self.direct_gate.bias], dim=0)
        fi = F.linear(torch.cat([core_seq, mem_h], dim=-1), Wfd, bfd)    # [B, L, d + 1]
        h_all = self.out_norm(core_seq + torch.sigmoid(fi[..., :d]) * mem_h)
        pred = self.head(h_all) + torch.sigmoid(fi[..., d:]).to(dtype) * target_read
        if not no_pad:
            vm = valid.unsqueeze(-1)
            pred = torch.where(vm, pred, torch.zeros_like(pred))
            h_all = torch.where(vm, h_all, torch.zeros_like(h_all))
        return pred, h_all


def build(**params):
    return PathDeltaFastweightsExactFused(**params)
