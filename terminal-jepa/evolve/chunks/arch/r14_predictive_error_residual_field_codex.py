"""
R14 arch: champion filebind/syscond trunk plus a causal prediction-error
residual field for cat-content binding.

The stalled cat margin suggests that system identity alone is not enough: the
model needs to separate corpus-modal content from the file/body deviations that
distinguish one held-out Docker image from another. For each completed past
command-observation pair, this module computes an innovation residual

    e_j = z_obs_j - prior(cmd_j)

where prior(cmd_j) is the trunk head prediction before any memory read. A
precision-weighted shifted running summary and a strict lower-triangular
neighbor read expose only residuals from pairs j < i to command i. A
zero-initialized multiplicative generator binds the current command/name vector
with those residual-context codes and contributes a fourth target-space read
channel. At initialization the new channel is silent; causality follows from
the strict shift and j < i masks.
"""

import math

import torch
import torch.nn as nn

D = 768

NAME = "r14_predictive_error_residual_field_codex"
DESCRIPTION = (
    "R13 syscond/filebind trunk plus a causal prediction-error residual field: "
    "past obs minus the model's own command prior are accumulated with learned "
    "precision and strict neighbor reads, then a zero-init name x residual-context "
    "generator adds a fourth target-space read channel for system-variant cat bodies."
)


class R14PredictiveErrorResidualField(nn.Module):
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
        innov_d=96,
        innov_hidden=160,
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

        self.d = int(d)
        self.key_d = int(key_d)
        self.ctx_d = int(ctx_d)
        self.n_verb = max(1, int(n_verb))
        self.sys_d = max(8, int(sys_d))
        self.innov_d = max(8, int(innov_d))
        self.innov_hidden = max(16, int(innov_hidden))
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
        self.tf = nn.TransformerEncoder(enc, max(1, int(layers)), enable_nested_tensor=False)

        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)
        self.write_gate = nn.Linear(2 * self.d, 1)

        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        self.innov_sal = nn.Linear(2 * self.d + 1, 1)
        self.innov_val = nn.Linear(D, self.innov_d, bias=False)
        self.innov_q = nn.Linear(D, self.key_d, bias=False)
        self.innov_k = nn.Linear(D, self.key_d, bias=False)
        self.innov_near = nn.Linear(D, self.innov_d)
        self.innov_cmd = nn.Linear(D, self.innov_d)
        self.innov_name = nn.Linear(D, self.innov_hidden)
        self.innov_ctx = nn.Linear(self.d + self.sys_d + 3 * self.innov_d, self.innov_hidden)
        self.innov_out = nn.Linear(self.innov_hidden, D)
        nn.init.zeros_(self.innov_out.weight)
        nn.init.zeros_(self.innov_out.bias)

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
        with torch.no_grad():
            self.read_mix.bias.zero_()
            self.read_mix.bias[3] = -2.5

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

    def _quotient(self, k, qbasis):
        coef = torch.matmul(k, qbasis.transpose(0, 1))
        return self._unit(k - torch.matmul(coef, qbasis))

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
            strict = torch.tril(
                torch.ones(C, C, device=value.device, dtype=torch.bool), diagonal=-1
            ).unsqueeze(0).to(dtype)

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
        cmd_tok = tok_emb[:, 0::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        active_pair = valid_cmd[:, :n_pair] & valid_obs

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        write_active = self._pad_steps(active_pair, n_cmd)
        beta = self._pad_steps(amount_pair, n_cmd) * write_active.to(x.dtype)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        lam = torch.where(write_active, decay.to(x.dtype).expand_as(beta), torch.ones_like(beta))

        qbasis = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), qbasis)
        k_file = self._quotient(self.file_write(x_cmd), qbasis)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        value_file = torch.cat([self._pad_steps(obs_tok, n_cmd), ctx], dim=-1)

        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = self._pad_steps(obs_tok, n_cmd)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]
        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        r_view = torch.nan_to_num(
            (1.0 + film[..., :D]) * r_obs + film[..., D:],
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1) * valid_obs.to(x.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
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

        base_prior = self.head(self.out_norm(h_cmd0))
        if n_pair:
            resid = (obs_for_prev - base_prior[:, :n_pair, :].to(dtype))
            resid = resid * active_pair.unsqueeze(-1).to(dtype)
            resid = torch.nan_to_num(resid, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

            rnorm = (resid.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt().to(x.dtype)
            isal_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs, rnorm], dim=-1)
            isal = torch.sigmoid(self.innov_sal(isal_in)).squeeze(-1) * active_pair.to(x.dtype)
            ival = torch.tanh(self.innov_val(resid))

            num = torch.cumsum(isal.unsqueeze(-1) * ival, dim=1)
            sq = torch.cumsum(isal.unsqueeze(-1) * ival.pow(2), dim=1)
            den = torch.cumsum(isal, dim=1).unsqueeze(-1)
            mean = num / (den + 1e-6)
            var = (sq / (den + 1e-6) - mean.pow(2)).clamp_min(0.0)
            conf = den / (den + 4.0)
            incl = conf * mean * torch.rsqrt(var + 0.25)
            innov_sys = torch.cat([incl.new_zeros(B, 1, self.innov_d), incl], dim=1)
            innov_sys = self._pad_steps(innov_sys, n_cmd)

            qi = self._unit(self.innov_q(cmd_tok))
            ki = self._unit(self.innov_k(cmd_tok[:, :n_pair, :]))
            scores = torch.bmm(qi, ki.transpose(1, 2)) / math.sqrt(self.key_d)
            prior_idx = torch.arange(n_pair, device=device).unsqueeze(0)
            query_idx = torch.arange(n_cmd, device=device).unsqueeze(1)
            strict = prior_idx < query_idx
            allowed = strict.unsqueeze(0) & active_pair.unsqueeze(1)
            scores = scores.masked_fill(~allowed, -1e9)
            attn = torch.softmax(scores, dim=-1)
            attn = attn * allowed.any(dim=-1, keepdim=True).to(attn.dtype)
            near_resid = torch.bmm(attn, resid)
        else:
            innov_sys = x.new_zeros(B, n_cmd, self.innov_d)
            near_resid = tok_emb.new_zeros(B, n_cmd, D)

        near_small = torch.tanh(self.innov_near(near_resid))
        cmd_small = torch.tanh(self.innov_cmd(cmd_tok))
        name = torch.tanh(self.innov_name(cmd_tok))
        innov_context = torch.cat(
            [h_cmd0.to(dtype), s_cmd.to(dtype), innov_sys.to(dtype), near_small, cmd_small],
            dim=-1,
        )
        ictx = torch.tanh(self.innov_ctx(innov_context))
        r_innov = torch.nan_to_num(
            self.innov_out(name * ictx),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        ri = r_innov.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ri.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )

        mix = torch.softmax(self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1)), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
            + mix[:, :, 3:4] * r_innov
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd * (1.0 + sf[..., :D]) + sf[..., D:]
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(
            pred * valid.unsqueeze(-1).to(pred.dtype),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        h_out = torch.nan_to_num(
            h_out * valid.unsqueeze(-1).to(h_out.dtype),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        return pred, h_out


def build(**params):
    return R14PredictiveErrorResidualField(**params)
