"""
R9-SPEED Tier-B arch: parallel-scan trunk + chunkwise (WY) delta-rule fastweights.

The champion (r7_path_delta_fastweights_codex) serializes a python loop over L=32
positions, dispatching ~212k tiny kernels per training step. Two structural facts
make it parallelizable end-to-end:
  1. The trunk state never consumes the memory reads (reads only feed the output
     head), so trunk and memory decouple into two batched stages.
  2. The write-gate amount depends only on trunk states, never on the memory —
     so the delta-rule memory update is a LINEAR recurrence in the memory matrix
     with all inputs precomputable.

This module therefore:
  * replaces the typed GRUCell trunk with a typed minGRU recurrence ("Were RNNs
    All We Needed?", Feng et al. 2024, arXiv:2410.01201): update gate and
    candidate depend only on the input, giving a diagonal affine recurrence
    h_t = (1-z_t)*h_{t-1} + z_t*htilde_t computed in ceil(log2 L) Hillis-Steele
    sweeps (5 for L=32). A Mamba-style causal depthwise conv (kernel 4) in front
    of the gates restores local state context lost with the GRU's
    state-dependent gating; cmd/obs typed routing is kept via per-type gate and
    candidate projections selected by torch.where.
  * computes the two delta-rule associative memories (content-keyed and
    path-keyed, decay + Widrow-Hoff write, target-space read) EXACTLY, in
    parallel, via the WY / UT-transform used by chunkwise DeltaNet (Yang et al.,
    "Parallelizing Linear Transformers with the Delta Rule over Sequence
    Length", arXiv:2406.06484; Gated DeltaNet, arXiv:2412.06464): the pseudo-
    value system (I + A) U = diag(a) V with A[p,q] = a_p * beta^(c_p-c_q-1) *
    (k_q . k_p) on write pairs q<p is one unit-lower-triangular solve; reads are
    one masked, decay-weighted (Q K^T) U matmul. Given identical keys/values/
    amounts this is the champion's memory recurrence up to float reassociation
    (verified 1e-6 vs a sequential reference sharing the same parameters).
  * keeps the champion's cmd->obs pending-pair semantics without a loop: the
    "most recent valid position, and it is a cmd" pairing is a log-depth running
    max over position indices + one gather.

Mechanism preserved (typed routing, path/content fastweights, target-space
direct read, decay, gates, init biases); only the trunk cell is reformulated —
Tier B. Everything is O(log L) sweeps + a handful of large kernels
(~10^2 dispatches vs ~2*10^5), no data-dependent python branching on tensor
values, so it also compiles cleanly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "r9_pscan_mingru_wy_fastweights"
DESCRIPTION = (
    "Parallel-scan typed minGRU trunk (log-depth affine scan + causal depthwise "
    "conv) with the champion's dual delta-rule fastweight memories computed "
    "exactly in parallel via the chunkwise WY/UT-transform (one triangular "
    "solve + masked decay-weighted matmul reads)."
)


class PScanMinGRUWYFastweights(nn.Module):
    def __init__(self, d=176, layers=2, key_d=64, ffn_mult=2, conv_k=4, **unused):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.conv_k = max(1, int(conv_k))
        ffn_h = max(self.d, int(ffn_mult * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(2, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        self.convs = nn.ModuleList()
        self.z_cmd = nn.ModuleList()
        self.z_obs = nn.ModuleList()
        self.h_cmd = nn.ModuleList()
        self.h_obs = nn.ModuleList()
        self.cell_norms = nn.ModuleList()
        self.ff_norms = nn.ModuleList()
        self.ff = nn.ModuleList()
        self.ff_gates = nn.ModuleList()
        for _ in range(self.layers):
            self.convs.append(nn.Conv1d(self.d, self.d, self.conv_k,
                                        groups=self.d, padding=self.conv_k - 1))
            zc = nn.Linear(self.d, self.d)
            zo = nn.Linear(self.d, self.d)
            # retention-biased update gates: z ~ sigmoid(-1) = 0.27 at init so the
            # scan state carries context over ~ a handful of tokens from step 0.
            nn.init.constant_(zc.bias, -1.0)
            nn.init.constant_(zo.bias, -1.0)
            self.z_cmd.append(zc)
            self.z_obs.append(zo)
            self.h_cmd.append(nn.Linear(self.d, self.d))
            self.h_obs.append(nn.Linear(self.d, self.d))
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
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d, self.d)
        self.out_norm = nn.LayerNorm(self.d)

        self.write_gate = nn.Linear(2 * self.d, 1)
        self.direct_gate = nn.Linear(2 * self.d, 1)
        self.head = nn.Linear(self.d, D)

        self.logit_decay = nn.Parameter(torch.tensor(math.log(0.985 / 0.015)))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)

    # ---- helpers -------------------------------------------------------------

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
        # rsqrt(sum + eps): norm().clamp_min() has a NaN backward at exactly-zero
        # rows (bias-free projections of zeroed pad rows land there).
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _affine_scan(alpha, b):
        """Inclusive scan of h_t = alpha_t * h_{t-1} + b_t with h_0 = 0, along
        dim 1, in ceil(log2 L) Hillis-Steele sweeps. alpha, b: [B, L, d]."""
        L = alpha.shape[1]
        o = 1
        while o < L:
            a_sh = torch.cat([alpha.new_ones(alpha[:, :o].shape), alpha[:, :-o]], dim=1)
            b_sh = torch.cat([b.new_zeros(b[:, :o].shape), b[:, :-o]], dim=1)
            b = b + alpha * b_sh
            alpha = alpha * a_sh
            o <<= 1
        return b

    def _dw_conv(self, l, y):
        """Causal depthwise conv, computed as k shifted elementwise mul-adds
        (a grouped Conv1d kernel decomposes into per-channel convs on some
        backends and resists fusion; this form is k fused-friendly elementwise
        ops on [B,L,d]). Parameters live in a standard nn.Conv1d container."""
        w = self.convs[l].weight[:, 0, :]                        # [d, k]
        out = y * w[:, -1] + self.convs[l].bias
        for s in range(1, self.conv_k):
            sh = torch.cat([y.new_zeros(y[:, :s].shape), y[:, :-s]], dim=1)
            out = out + sh * w[:, self.conv_k - 1 - s]
        return out

    @staticmethod
    def _running_max(x, fill):
        """Inclusive running max along dim 1 (log-depth; avoids relying on a
        backend cummax kernel). x: [B, L] integer tensor."""
        L = x.shape[1]
        o = 1
        while o < L:
            sh = torch.cat([x.new_full(x[:, :o].shape, fill), x[:, :-o]], dim=1)
            x = torch.maximum(x, sh)
            o <<= 1
        return x

    # ---- forward -------------------------------------------------------------

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            return tok_emb.new_zeros(B, 0, D), tok_emb.new_zeros(B, 0, self.d)

        t = types.long().clamp(0, 1)
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, device=device, dtype=torch.bool)
        vf = valid.to(dtype).unsqueeze(-1)                       # [B,L,1]
        is_cmd_tok = (t == 0).unsqueeze(-1)                      # [B,L,1] bool

        # -- embed (identical to champion), then zero pad rows so the conv and
        #    the scan see exactly no signal from padding.
        x = torch.where(is_cmd_tok, self.cmd_proj(tok_emb), self.obs_proj(tok_emb))
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, dtype).unsqueeze(0)
        x = self.in_norm(x) * vf

        # -- trunk: typed minGRU layers, each one parallel scan + pointwise FFN.
        y = x
        for l in range(self.layers):
            u = y + F.silu(self._dw_conv(l, y))
            z = torch.sigmoid(torch.where(is_cmd_tok, self.z_cmd[l](u), self.z_obs[l](u)))
            hcand = torch.where(is_cmd_tok, self.h_cmd[l](u), self.h_obs[l](u))
            alpha = 1.0 - z * vf                                  # pad -> hold (alpha=1)
            bvec = z * hcand * vf                                 # pad -> b=0
            s = self._affine_scan(alpha, bvec)
            c = self.cell_norms[l](s)
            n = self.ff_norms[l](c)
            y = (c + torch.sigmoid(self.ff_gates[l](n)) * self.ff[l](n)) * vf
        core = y                                                  # [B,L,d]

        # -- pending-pair detection (champion semantics, loop-free): an obs at
        #    position p writes iff the most recent valid position j < p exists
        #    and is a cmd; the write uses keys/state from j.
        pos = torch.arange(L, device=device)
        idx = torch.where(valid, pos.unsqueeze(0).expand(B, L), pos.new_full((), -1))
        prev_idx = self._running_max(idx, -1)
        prev_idx = torch.cat([prev_idx.new_full((B, 1), -1), prev_idx[:, :-1]], dim=1)
        has_prev = prev_idx >= 0
        j = prev_idx.clamp_min(0)                                 # [B,L]
        tj = torch.gather(t, 1, j)
        pair = valid & (t == 1) & has_prev & (tj == 0)            # write indicator
        w = pair.to(dtype)                                        # [B,L]

        jd = j.unsqueeze(-1).expand(B, L, self.d)
        core_j = torch.gather(core, 1, jd)                        # pending cmd core
        x_j = torch.gather(x, 1, jd)                              # pending cmd input

        # -- keys / queries / amounts (all positions at once)
        q_c = self._unit(self.content_read(x))                    # [B,L,kd]
        q_p = self._unit(self.path_read(core))
        k_c = self._unit(self.content_write(x_j))
        k_p = self._unit(self.path_write(core_j))
        amount = torch.sigmoid(self.write_gate(torch.cat([core_j, core], dim=-1))).squeeze(-1)
        a_eff = (amount * w).unsqueeze(-1)                        # [B,L,1]

        beta = torch.sigmoid(self.logit_decay).to(dtype)
        cnt = torch.cumsum(w, dim=1)                              # inclusive write count
        m = cnt - w                                               # writes strictly before i

        # -- WY / UT transform: (I + A) U = diag(a) V, A strictly lower over
        #    write pairs, with decay powers beta^(c_p - c_q - 1). Exact.
        lower = (pos.unsqueeze(1) > pos.unsqueeze(0)).to(dtype)   # [L,L] p>q
        e_a = (cnt.unsqueeze(2) - cnt.unsqueeze(1) - 1.0).clamp_min(0.0)
        mask_a = lower.unsqueeze(0) * w.unsqueeze(2) * w.unsqueeze(1)
        pow_a = torch.pow(beta, e_a) * mask_a                     # [B,L,L]
        gram_c = torch.bmm(k_c, k_c.transpose(1, 2))
        gram_p = torch.bmm(k_p, k_p.transpose(1, 2))
        A2 = torch.cat([gram_c, gram_p], dim=0) * (a_eff * pow_a).repeat(2, 1, 1)
        eye = torch.eye(L, device=device, dtype=dtype).unsqueeze(0)
        rhs = (a_eff * tok_emb).repeat(2, 1, 1)                   # values = raw obs embeddings
        U = torch.linalg.solve_triangular(eye + A2, rhs, upper=False, unitriangular=True)

        # -- reads: r_i = sum_{writes p<i} beta^(m_i - c_p) (q_i.k_p) u_p
        e_r = (m.unsqueeze(2) - cnt.unsqueeze(1)).clamp_min(0.0)
        mask_r = lower.unsqueeze(0) * w.unsqueeze(1)              # p<i and p is a write
        pow_r = torch.pow(beta, e_r) * mask_r                     # [B,L,L]
        QK = torch.cat([torch.bmm(q_c, k_c.transpose(1, 2)),
                        torch.bmm(q_p, k_p.transpose(1, 2))], dim=0)
        R = torch.bmm(QK * pow_r.repeat(2, 1, 1), U)              # [2B,L,768]
        read_c, read_p = R[:B], R[B:]

        # -- output head (identical to champion, position-parallel)
        mix = torch.softmax(self.read_mix(core), dim=-1)
        target_read = mix[..., 0:1] * read_c + mix[..., 1:2] * read_p
        mem_h = self.read_to_h(target_read)
        fuse_in = torch.cat([core, mem_h], dim=-1)
        h = self.out_norm(core + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)
        pred = self.head(h) + torch.sigmoid(self.direct_gate(fuse_in)) * target_read

        pred = pred * vf
        h = h * vf
        return pred, h


def build(**params):
    return PScanMinGRUWYFastweights(**params)
