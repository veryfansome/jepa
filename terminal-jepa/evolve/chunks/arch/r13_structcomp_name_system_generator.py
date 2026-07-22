"""
R13 arch: champion file-bound memory + a COMPOSITIONAL name-by-system content
generator — predict UNSEEN-file content from structural context instead of
looking it up.

THE GAP (measured, v2 inner): cat is the weakest content margin (+.298; wm .828
vs cross-image retrieve-by-cmd .530). The corpus baseline is strong because
common config files repeat across distros; the margin only grows from
SYSTEM-SPECIFIC content. The R12 champion already covers the LOOKUP side of
cat: its verb-quotient file memory returns content of files ALREADY OBSERVED in
this trajectory, and the anti-retrieval objective pushes predictions off the
corpus template. What NEITHER provides is a generative path for a file the
trajectory has NEVER opened: there is nothing in the file store under its
address, prev-obs is the wrong step, and the trunk must squeeze all evidence
through the d=176 state into one shared linear head. Yet the evidence is in
history — the ls of the parent directory that NAMED the file, sibling files
already read, and the distro identity accumulated across every observation. A
lookup cannot use that evidence by definition; composition can.

THREE ADDITIONS on the unchanged champion machinery (trunk, verb-quotient file
memory, path memory, FiLM view transform, 3 champion read channels, delta-rule
solver — all bit-retained):

1. SYSTEM FINGERPRINT s_i: a strictly-causal running mean of all PREVIOUS
   observation embeddings (pairs < i). Early steps (uname, /etc/os-release,
   ls /etc) make this a distro-identity vector; it is the "which system am I
   on" coordinate that turns a cross-distro template into a system-specific
   prediction.

2. NEIGHBORHOOD READ n_i: one cross-attention head in RAW embedding space —
   query = e5 embedding of the CURRENT command, keys = e5 embeddings of
   EARLIER commands (strictly j < i), values = their observations. Command
   embeddings of "ls /etc/nginx" and "cat /etc/nginx/nginx.conf" share the
   path prefix, so learned query/key projections can align a cat with the dir
   listing that contains its entry (name/size fields) or with sibling reads —
   evidence the champion's file memory cannot return for an address never
   written. Unlike retrieve-by-cmd, the fetched vector is never surfaced: it
   only CONDITIONS the generator, and the (unchanged) anti-retrieval objective
   keeps the final prediction off verbatim echoes.

3. NAME-x-SYSTEM MULTIPLICATIVE GENERATOR: f_name = tanh(W_n cmd_raw) binds
   file identity (the path is lexical content of the command embedding);
   f_ctx = tanh(W_c [proj(s_i); proj(n_i); h_cmd]) binds system + neighborhood
   context; the channel emits W_o(f_name * f_ctx) in 768-d target space — a
   Hadamard tensor-product binding (Smolensky 1990; holographic reduced
   representations, Plate 1995): content = template(name) modulated by system.
   W_o is ZERO-INITIALIZED, so at init the channel is exactly silent and the
   arch behaves as the champion (the 4-way read mix starts with ~0.04 weight
   on the empty channel; its zero-init bias keeps the other three at champion
   proportions).

Integration: the generator output joins the champion's 3-way softmax read mix
as a 4th channel (r_view / path / prev-obs / COMPOSE), with the rms read
features, fuse gate, and direct-output gate extended to 4 channels — same
gating discipline, same output path. Strictly causal: s_i and n_i use only
pairs < i (cumsum shifted by one; strict lower attention mask), so the
obs_t-perturbation guard holds for every cmd <= t.

WHY THE CAT MARGIN CAN MOVE: on an unseen file of a held-out image, retrieval
supplies the cross-distro template (.530 top-1) and the champion's memory
supplies nothing; the composed prediction name x (distro fingerprint +
parent-listing evidence) is precisely the residual the margin measures. On
already-observed files the champion channels win the mix and nothing is lost.

Refs: tensor-product representations (Smolensky 1990), holographic reduced
representations (Plate, IEEE TNN 1995), FiLM (arXiv:1709.07871), chunkwise
DeltaNet (arXiv:2406.06484), zero-init gated adapter channels (Flamingo,
arXiv:2204.14198), cortical "what x where" conjunctive coding (O'Reilly &
Rudy 2001).
"""

import math

import torch
import torch.nn as nn

D = 768

NAME = "r13_structcomp_name_system_generator"
DESCRIPTION = (
    "Champion filebind/verbquotient/viewmem arch + a zero-init COMPOSITIONAL channel "
    "that predicts UNSEEN-file content from structure: a strictly-causal system "
    "fingerprint (mean of prior observations) and a raw-embedding neighborhood "
    "attention read (the earlier ls/cat whose command names this file) multiplicatively "
    "bound with the command's lexical file identity (name x system Hadamard "
    "tensor-product), joining the read mix as a 4th channel."
)


class R13StructcompNameSystemGenerator(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        key_d=64,
        ctx_d=96,
        n_verb=8,
        film_hidden=128,
        ffn_mult=2,
        dropout=0.1,
        chunk_size=16,
        comp_rank=96,
        sysctx_d=64,
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
        self.chunk_size = max(1, int(chunk_size))
        self.comp_rank = max(8, int(comp_rank))
        self.sysctx_d = max(8, int(sysctx_d))
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

        # -- file memory: verb-quotient addressing over the raw command token (champion).
        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        # -- path-state memory (champion channel, unchanged).
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        # -- FiLM view transform (champion, unchanged).
        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        # -- NEW: compositional name-x-system generator ---------------------------
        # neighborhood attention in RAW embedding space (path-prefix association).
        self.nbr_q = nn.Linear(D, self.key_d, bias=False)
        self.nbr_k = nn.Linear(D, self.key_d, bias=False)
        # context projections (system fingerprint / neighborhood evidence).
        self.sys_proj = nn.Linear(D, self.sysctx_d)
        self.nbr_proj = nn.Linear(D, self.sysctx_d)
        # name / context factor maps + zero-init output binding.
        self.name_proj = nn.Linear(D, self.comp_rank)
        self.ctx_in = nn.Linear(2 * self.sysctx_d + self.d, self.comp_rank)
        self.comp_out = nn.Linear(self.comp_rank, D)
        nn.init.zeros_(self.comp_out.weight)
        nn.init.zeros_(self.comp_out.bias)
        # --------------------------------------------------------------------------

        # 4-way read mix (champion's 3 channels + compose), 4 rms features.
        self.read_mix = nn.Linear(self.d + 4, 4)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 4, self.d)
        self.direct_gate = nn.Linear(2 * self.d + 4, 1)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

        init_decay = (0.985 - 0.90) / 0.099
        self.logit_decay = nn.Parameter(torch.tensor(math.log(init_decay / (1.0 - init_decay))))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)
        # start the compose channel soft-off so the trained champion proportions
        # among its 3 channels are preserved at init (softmax bias -2 => ~0.04).
        with torch.no_grad():
            self.read_mix.bias.zero_()
            self.read_mix.bias[3] = -2.0

    # ---- helpers (champion, unchanged) --------------------------------------

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
        vs = []
        for i in range(self.n_verb):
            v = self.verb_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum() + 1e-8)
            vs.append(v)
        return torch.stack(vs, dim=0)

    def _quotient(self, k, Q):
        coef = torch.matmul(k, Q.transpose(0, 1))
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
        cmd_tok = tok_emb[:, 0::2, :]  # raw command embeddings (lexical name/path)

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

        # -- file memory: verb-quotient addresses, composite values (champion).
        Q = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), Q)
        k_file = self._quotient(self.file_write(x_cmd), Q)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        value_file = torch.cat([self._pad_steps(obs_tok, n_cmd), ctx], dim=-1)

        # -- path-state memory (champion).
        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = self._pad_steps(obs_tok, n_cmd)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]

        # -- FiLM view transform (champion).
        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        obs_for_prev = obs_tok * valid_obs.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_src = self._pad_steps(obs_for_prev, n_cmd - 1)
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), prev_src], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)

        # -- NEW: compositional name-x-system channel ------------------------------
        if n_pair:
            # (1) strictly-causal system fingerprint: mean of observations at pairs < i.
            sums = torch.cumsum(obs_for_prev, dim=1)
            cnts = torch.cumsum(valid_obs.to(dtype), dim=1)
            sums = torch.cat([tok_emb.new_zeros(B, 1, D), sums], dim=1)[:, :n_cmd]
            cnts = torch.cat([cnts.new_zeros(B, 1), cnts], dim=1)[:, :n_cmd]
            s_prev = sums / cnts.clamp_min(1.0).unsqueeze(-1)

            # (2) neighborhood read: raw-space attention, keys = earlier commands
            # (strictly j < i), values = their observations.
            qn = self.nbr_q(cmd_tok)                                   # [B, n_cmd, kd]
            kn = self.nbr_k(cmd_tok[:, :n_pair, :])                    # [B, n_pair, kd]
            scores = torch.bmm(qn, kn.transpose(1, 2)) / math.sqrt(self.key_d)
            strict_nbr = (
                torch.arange(n_pair, device=device).unsqueeze(0)
                < torch.arange(n_cmd, device=device).unsqueeze(1)
            )                                                          # [n_cmd, n_pair], j < i
            allowed = strict_nbr.unsqueeze(0) & valid_obs.unsqueeze(1)  # [B, n_cmd, n_pair]
            scores = scores.masked_fill(~allowed, -1e9)
            attn = torch.softmax(scores, dim=-1)
            attn = attn * allowed.any(dim=-1, keepdim=True).to(attn.dtype)
            n_read = torch.bmm(attn, obs_for_prev)                     # [B, n_cmd, D]
        else:
            s_prev = tok_emb.new_zeros(B, n_cmd, D)
            n_read = tok_emb.new_zeros(B, n_cmd, D)

        # (3) name-x-system Hadamard binding, zero-init output (silent at init).
        f_name = torch.tanh(self.name_proj(cmd_tok))
        ctx_cat = torch.cat(
            [self.sys_proj(s_prev), self.nbr_proj(n_read), h_cmd0.to(dtype)], dim=-1
        )
        f_ctx = torch.tanh(self.ctx_in(ctx_cat))
        r_comp = self.comp_out(f_name * f_ctx)
        r_comp = torch.nan_to_num(r_comp, nan=0.0, posinf=1e4, neginf=-1e4)
        # --------------------------------------------------------------------------

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        rc = r_comp.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rc.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )

        mix = torch.softmax(self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1)), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
            + mix[:, :, 3:4] * r_comp
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R13StructcompNameSystemGenerator(**params)
