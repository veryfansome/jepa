"""
R13 arch: champion filebind trunk + a causal SYSTEM-IDENTITY summary channel
that FiLM-modulates the content prediction — "cat /etc/passwd" answered as
FEDORA's passwd, not the corpus-modal passwd.

THE TARGET (measured, v2 inner): cat is the weakest content margin (+.298;
wm .828 vs cross-image retrieve-by-cmd .530). The retrieval baseline is strong
precisely because common config files repeat across distros; the margin grows
only where the model predicts SYSTEM-SPECIFIC content. Every v2 sequence opens
with identity-revealing observations (uname / os-release / ls of telltale
paths), so the information is always in-context — but the champion gives it no
dedicated route into the content prediction:

  * The champion ARCH (filebind_verbquotient_viewmem) binds file identity
    WITHIN a trajectory: its file memory only returns content for files
    already observed in this sequence. A FIRST-TIME cat — the retrieval-
    baseline's home turf — reads nothing useful from memory; the prediction
    falls to the trunk + direct head, i.e. the corpus prior for that command.
  * The champion OBJECTIVE (antiretrieval ring) reweights in-batch negatives
    toward retrieval-confusable target pairs — it shapes WHERE loss pressure
    goes but supplies no mechanism for the net to ACT on system identity; and
    the sysblock batcher densifies SAME-system negatives, so cross-system
    same-command pairs are not even reliably present.

MECHANISM — two additions on the untouched champion machinery:

1. CAUSAL SYSTEM-SUMMARY STATE. A learned salience scorer reads each
   observation's trunk state and a gated running mean accumulates a bounded
   sys_d-dim summary: s_j = (sum_{i<j} g_i * v_i) / (sum_{i<j} g_i + eps) with
   g_i = sigmoid(salience(h_obs_i)) masked to valid observations, v_i =
   tanh(proj(h_obs_i)). Cumsum-based, strictly shifted so command j sees only
   observations of pairs < j (causal by construction). Training decides which
   observations are identity-bearing — the uname/os-release outputs that open
   every sequence are the obvious attractor, but nothing is hand-coded. This
   is a slow "context" state alongside the fast file memory — the neuroscience
   picture of a persistent contextual representation gating episodic recall
   (context + content two-stream models; cf. Ranganath 2010), and the same
   division as a speaker embedding conditioning an acoustic decoder.

2. ZERO-INITIALIZED SYSTEM FiLM ON THE PREDICTION (Perez et al.,
   arXiv:1709.07871). From [command state; system summary], a small net emits
   (gamma, shift) applied to the final command-position prediction in target
   space: pred <- pred * (1 + gamma) + shift. Zero-initialized, so at step 0
   the module IS the champion bit-for-bit and the channel fades in only where
   it reduces loss. Multiplicative + additive modulation in the frozen e5
   space is exactly the needed operation: take the corpus-modal embedding the
   trunk already produces for "cat /etc/passwd" and displace it toward the
   distro-specific variant — a per-system affine correction, the arch-side
   complement of the ring objective's demand to beat the cross-system foil.

WHY THE CAT MARGIN MOVES: the eval retrieves the true observation against
same-verb foils; for repeated config files the foils include the OTHER
systems' versions of the same file — the exact candidates retrieve-by-cmd
banks. A prediction displaced toward the current system's variant flips
precisely those decisions, and only those: wm top-1 rises on system-specific
content while the baseline (fixed) stays put, so the margin — not just the
absolute — moves. First-time cats are the dominant cat case (the file memory
covers re-reads), so the addressable mass is large.

Champion machinery retained verbatim: verb-quotient file memory, composite
[content; view-ctx] values, view FiLM, path memory, 3-way read mix, fuse /
direct gates, exact chunkwise lower-triangular delta solver. Strictly causal;
NaN-safe; ~240K added params at defaults.

Refs: FiLM (arXiv:1709.07871), global workspace / slow-context conditioning
(cf. speaker-conditioned decoders, e.g. arXiv:1806.04558), context-binding in
episodic memory (Ranganath 2010, "binding of item and context"), chunkwise
DeltaNet (arXiv:2406.06484).
"""

import math

import torch
import torch.nn as nn

D = 768

NAME = "r13_syscond_film_content"
DESCRIPTION = (
    "Champion filebind trunk + a causal gated-running-mean SYSTEM-IDENTITY summary "
    "(learned salience over past observations — uname/os-release open every sequence) "
    "feeding a zero-initialized FiLM that affinely displaces each command's content "
    "prediction toward the current system's variant: /etc/passwd on fedora predicted "
    "as fedora content, not the corpus-modal file. Exactly the champion at init."
)


class R13SyscondFilmContent(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        key_d=64,
        ctx_d=96,
        n_verb=8,
        film_hidden=128,
        sys_d=64,
        sysfilm_hidden=128,
        ffn_mult=2,
        dropout=0.1,
        chunk_size=16,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]
        if "chunk" in unused:
            chunk_size = unused["chunk"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.ctx_d = int(ctx_d)
        self.n_verb = max(1, int(n_verb))
        self.sys_d = max(8, int(sys_d))
        self.chunk_size = max(1, int(chunk_size))
        ffn_h = max(self.d, int(float(ffn_mult) * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(2, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        enc = nn.TransformerEncoderLayer(
            self.d,
            int(heads),
            ffn_h,
            float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, self.layers, enable_nested_tensor=False)

        # -- file memory: verb-quotient addressing over the raw command token.
        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        # -- path-state memory (champion channel, unchanged).
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        # -- FiLM view transform (champion channel, unchanged).
        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        # -- NEW: causal system-identity summary + zero-init system FiLM.
        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        self.read_mix = nn.Linear(self.d + 3, 3)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 3, self.d)
        self.direct_gate = nn.Linear(2 * self.d + 3, 1)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

        init_decay = (0.985 - 0.90) / 0.099
        self.logit_decay = nn.Parameter(torch.tensor(math.log(init_decay / (1.0 - init_decay))))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)

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
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _pad_steps(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        pad_shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def _verb_basis(self):
        """Differentiable Gram-Schmidt over the verb codebook -> orthonormal Q."""
        vs = []
        for i in range(self.n_verb):
            v = self.verb_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum() + 1e-8)
            vs.append(v)
        return torch.stack(vs, dim=0)  # [nv, key_d]

    def _quotient(self, k, Q):
        """Project k [B,N,K] onto the null space of span(Q) and unit-normalize."""
        coef = torch.matmul(k, Q.transpose(0, 1))          # [B,N,nv]
        return self._unit(k - torch.matmul(coef, Q))

    def _solve_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False)
        parts = []
        C = system.size(1)
        for i in range(C):
            yi = rhs[:, i, :]
            if parts:
                prev = torch.stack(parts, dim=1)
                corr = torch.bmm(system[:, i : i + 1, :i], prev).squeeze(1)
                yi = yi - corr
            yi = yi / system[:, i, i].unsqueeze(-1).clamp_min(1e-6)
            parts.append(yi)
        return torch.stack(parts, dim=1)

    def _chunked_delta_reads(self, q, k, value, beta, lam):
        """Exact chunkwise delta-rule associative memory (champion machinery).

        Strictly causal: within a chunk only strictly-lower pairs interact and
        the carried memory holds only writes from previous chunks, so the read
        at command index i sees writes from pairs < i only.
        """
        B, N, K = q.shape
        V = value.size(-1)
        if N == 0:
            return value.new_zeros(B, 0, V)

        dtype = value.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        beta = beta.to(dtype)
        lam = lam.to(dtype).clamp(0.90, 1.0)

        mem = value.new_zeros(B, K, V)
        outs = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            qc = q[:, start:end, :]
            kc = k[:, start:end, :]
            vc = value[:, start:end, :]
            bc = beta[:, start:end]
            lc = lam[:, start:end]
            C = end - start

            prefix = torch.cumprod(lc, dim=1)
            before = torch.cat(
                [torch.ones(B, 1, device=value.device, dtype=dtype), prefix[:, :-1]], dim=1
            )
            denom = prefix.clamp_min(1e-6)
            between = before.unsqueeze(2) / denom.unsqueeze(1)

            strict = torch.tril(torch.ones(C, C, device=value.device, dtype=torch.bool), diagonal=-1)
            strict = strict.unsqueeze(0).to(dtype)

            kk = torch.bmm(kc, kc.transpose(1, 2))
            lower = kk * between * bc.unsqueeze(1) * strict

            rhs = vc - before.unsqueeze(-1) * torch.bmm(kc, mem)
            eye = torch.eye(C, device=value.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            err = self._solve_lower(eye + lower, rhs)
            err = torch.nan_to_num(err, nan=0.0, posinf=1e4, neginf=-1e4)

            qk = torch.bmm(qc, kc.transpose(1, 2))
            weights = qk * between * bc.unsqueeze(1) * strict
            read = before.unsqueeze(-1) * torch.bmm(qc, mem) + torch.bmm(weights, err)
            outs.append(torch.nan_to_num(read, nan=0.0, posinf=1e4, neginf=-1e4))

            end_factor = prefix[:, -1]
            end_between = end_factor.unsqueeze(1) / denom
            contrib = torch.bmm(kc.transpose(1, 2), err * (bc * end_between).unsqueeze(-1))
            mem = end_factor.view(B, 1, 1) * mem + contrib
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.cat(outs, dim=1)

    # ---- forward -------------------------------------------------------------

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = torch.nan_to_num(h_base, nan=0.0, posinf=1e4, neginf=-1e4)
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        n_pair = L // 2

        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = h_base[:, 1::2, :]
        obs_tok = tok_emb[:, 1::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        active_pair = valid_cmd[:, :n_pair] & valid_obs

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        write_active = self._pad_steps(active_pair, n_cmd)
        beta = self._pad_steps(amount_pair, n_cmd) * write_active.to(x.dtype)

        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        lam = torch.where(write_active, decay.to(x.dtype).expand_as(beta), torch.ones_like(beta))

        # -- file memory: verb-quotient addresses, composite [content; view-ctx] values.
        Q = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), Q)
        k_file = self._quotient(self.file_write(x_cmd), Q)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        value_file = torch.cat([self._pad_steps(obs_tok, n_cmd), ctx], dim=-1)

        # -- path-state memory: champion channel over trunk states.
        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = self._pad_steps(obs_tok, n_cmd)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]

        # -- FiLM view transform: map stored content into the queried view.
        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        # -- NEW: causal system-identity summary (gated running mean over past obs).
        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1)      # [B, n_pair]
            sal = sal * valid_obs.to(sal.dtype)                       # padded obs contribute 0
            v_sys = torch.tanh(self.sys_val(h_obs))                   # bounded values
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)      # inclusive of pair j
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)                               # [B, n_pair, sys_d]
            # STRICT SHIFT: command j sees the summary of obs pairs < j only.
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        obs_for_prev = obs_tok * valid_obs.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_src = self._pad_steps(obs_for_prev, n_cmd - 1)
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), prev_src], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )

        mix = torch.softmax(self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1)), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        # -- NEW: zero-init system FiLM displaces command predictions toward the
        # current system's content variant. Identity at init (sysfilm_out = 0).
        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R13SyscondFilmContent(**params)
