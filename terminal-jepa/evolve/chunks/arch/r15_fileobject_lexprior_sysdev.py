"""
R15 hybrid arch: a single canonical FILE-OBJECT spine that unifies the three
R14 winners as disciplined readers/writers of ONE object state.

STRUCTURE-FIRST THESIS. The R14 archs each added a mechanism in a DIFFERENT
space — canon_fileobject stored a canonical code, pkm_content_lexicon read a
persistent 768-d correction, predictive_error_residual_field accumulated
768-d innovation residuals — and each entered the prediction as its own
independent additive channel. Independent channels in target space can
memorize inner-val idiosyncrasies separately (the R14 generalization gap:
inner gains, no final-test transfer). Here every mechanism is forced to be an
operation on ONE canonical file-object state c (canon_d-dim, bounded), and
only the view renderer maps object state to target space:

  1. EPISODIC OBJECT STATE (from r14_canon_fileobject_viewrender, verbatim
     machinery): each observation is encoded through its write-view chart
     (soft view coords from the command token; zero-init chart FiLM) into a
     bounded canonical code, stored under the champion's verb-quotient file
     address via the exact chunkwise delta solver. head/tail/cat/grep of a
     path REFINE one object instead of overwriting raw views.
  2. SEMANTIC OBJECT PRIOR (r14_pkm_content_lexicon, re-based): the
     product-key lexicon is addressed by (frozen command token x causal
     system summary) but its values now live IN THE CANONICAL CODE SPACE
     (dv = canon_d), so the persistent table is the corpus's prior over the
     object at this address — what /etc/passwd-like objects contain — not a
     free-floating output correction. A confidence gate fuses episodic code
     (when this trajectory has observed the object) with the prior (first
     visits — the dominant cat case): c = g*c_epi + (1-g)*c_prior.
     Complementary-learning-systems in one state variable: hippocampal
     episodic trace vs neocortical semantic prior, fused at recall
     (McClelland et al. 1995), and the lexicon read is regularized by
     construction — it can only help through the shared renderer.
  3. SYSTEM DEVIATION FIELD (r14_predictive_error residual field, re-based):
     for each completed pair, the innovation is computed in CODE space,
     e_j = code(obs_j) - prior(cmd_j) — how THIS system's object deviates
     from the corpus prior — salience-weighted, confidence-normalized,
     strictly shifted. Together with the champion's causal system summary it
     drives a zero-init FiLM ON THE OBJECT CODE: c_cond = c*(1+g)+b. This is
     the syscond idea moved from target space onto structure: "fedora-ness"
     transforms the object before rendering, so one learned deviation
     transfers across all of a system's files instead of being re-memorized
     per command.
  4. VIEW RENDERER (canon_fileobject, verbatim): the conditioned code is
     decoded and re-rendered into the queried view (query view coords +
     recovered write-view tag, zero-init FiLM) and enters the champion's
     read mix as the fourth channel, logit biased low at init.

The champion r13_syscond_film_content machinery (trunk, file/path delta
memories, view FiLM, system summary + target-space system FiLM, cued gates)
is retained verbatim; at init the model is near-champion (all new pathways
zero-init or low-gated). Strictly causal: writes readable only by later
commands (chunked-delta strict masks), system summary and deviation field
strictly shifted, view coords / lexicon query use only the command's own
token and past-shifted summaries. NaN-safe (tanh-bounded codes, clamps,
nan_to_num), deterministic, ~0.65M added params over the champion (~2.8M).

Refs: complementary learning systems (McClelland, McNaughton & O'Reilly
1995), product-key memories (Lample et al., arXiv:1907.05242), capsule /
view-conditioned rendering (arXiv:1710.09829; NeRF arXiv:2003.08934),
predictive-coding innovation (Rao & Ballard 1999), FiLM (arXiv:1709.07871),
chunkwise DeltaNet (arXiv:2406.06484).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "r15_fileobject_lexprior_sysdev"
DESCRIPTION = (
    "Canonical file-object spine unifying the R14 winners: view-chart episodic codes "
    "and a product-key lexicon PRIOR live in one canonical code space, fused by a "
    "confidence gate (episodic when observed, semantic prior on first visits); a "
    "code-space innovation field (obs code minus prior) plus the causal system summary "
    "FiLM-conditions the object by system identity before a view-conditioned renderer "
    "instantiates the queried view as the fourth read channel. Near-champion at init."
)


class R15FileobjectLexpriorSysdev(nn.Module):
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
        canon_d=128,
        n_view=6,
        render_hidden=96,
        devfilm_hidden=96,
        pk_nk=32,
        pk_d=32,
        pk_thalf=8,
        pk_topk=8,
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
        self.canon_d = max(16, int(canon_d))
        self.n_view = max(2, int(n_view))
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

        # -- champion file memory: verb-quotient addressing over the raw command token.
        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        # -- champion path-state memory.
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        # -- champion FiLM view transform on the raw file read.
        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        # -- champion causal system-identity summary + zero-init target-space system FiLM.
        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        # ---- THE FILE-OBJECT SPINE ------------------------------------------

        # (1) episodic object state: view-chart encoder writing canonical codes.
        self.view_cls = nn.Linear(self.d, self.n_view)
        self.canon_enc = nn.Linear(D, self.canon_d)
        self.enc_gain = nn.Linear(self.n_view, self.canon_d, bias=False)
        self.enc_bias = nn.Linear(self.n_view, self.canon_d, bias=False)
        nn.init.zeros_(self.enc_gain.weight)
        nn.init.zeros_(self.enc_bias.weight)

        # (2) semantic object prior: product-key lexicon with values IN code space.
        self.pk_nk = max(4, int(pk_nk))
        self.pk_d = max(8, int(pk_d))
        self.pk_thalf = max(1, min(int(pk_thalf), self.pk_nk))
        self.pk_topk = max(1, min(int(pk_topk), self.pk_thalf * self.pk_thalf))
        # sub-keys stored 3-d (shape-routed optimizer groups unchanged).
        self.pkm_keys = nn.Parameter(
            torch.randn(2, self.pk_nk, self.pk_d) / math.sqrt(self.pk_d)
        )
        self.pkm_values = nn.Parameter(
            torch.randn(self.pk_nk * self.pk_nk, self.canon_d) / math.sqrt(self.canon_d)
        )
        self.pkm_query = nn.Linear(D + self.sys_d, 2 * self.pk_d)
        self.pkm_qnorm = nn.LayerNorm(2 * self.pk_d)
        # episodic-vs-prior confidence fusion gate.
        self.conf_gate = nn.Linear(self.d + 2, 1)
        nn.init.zeros_(self.conf_gate.weight)
        nn.init.zeros_(self.conf_gate.bias)

        # (3) system deviation field in code space + zero-init object FiLM.
        self.dev_sal = nn.Linear(2 * self.d + 1, 1)
        self.dev_val = nn.Linear(self.canon_d, self.canon_d)
        dh = max(16, int(devfilm_hidden))
        self.devfilm_in = nn.Linear(self.d + self.sys_d + self.canon_d, dh)
        self.devfilm_out = nn.Linear(dh, 2 * self.canon_d)
        nn.init.zeros_(self.devfilm_out.weight)
        nn.init.zeros_(self.devfilm_out.bias)

        # (4) view-conditioned renderer: object code -> queried view embedding.
        self.canon_dec = nn.Linear(self.canon_d, D)
        rh = max(16, int(render_hidden))
        self.rfilm_in = nn.Linear(self.d + 2 * self.n_view, rh)
        self.rfilm_out = nn.Linear(rh, 2 * D)
        nn.init.zeros_(self.rfilm_out.weight)
        nn.init.zeros_(self.rfilm_out.bias)

        # 4-way read mix (r_view / path / prev_obs / object render).
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
        # the object-render channel fades in: its mix logit starts low.
        nn.init.zeros_(self.read_mix.bias)
        with torch.no_grad():
            self.read_mix.bias[3] = -2.0

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
        at command index i sees writes from pairs < i only. Generic in the
        value dimension.
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

    def _lexicon_prior(self, cmd_tok, s_cmd):
        """Product-key sparse read whose values live in canonical code space.
        cmd_tok [B,N,768] frozen command embeddings (lexical: the path is in
        them); s_cmd [B,N,sys_d] strictly-causal system summary. Returns a
        bounded [B,N,canon_d] prior over the object at this address."""
        q = self.pkm_qnorm(self.pkm_query(torch.cat([cmd_tok, s_cmd], dim=-1)))
        q1, q2 = q[..., : self.pk_d], q[..., self.pk_d :]
        scale = 1.0 / math.sqrt(self.pk_d)
        s1 = torch.matmul(q1, self.pkm_keys[0].transpose(0, 1)) * scale   # [B,N,nk]
        s2 = torch.matmul(q2, self.pkm_keys[1].transpose(0, 1)) * scale
        t = self.pk_thalf
        v1, i1 = s1.topk(t, dim=-1)                                       # [B,N,t]
        v2, i2 = s2.topk(t, dim=-1)
        cand = v1.unsqueeze(-1) + v2.unsqueeze(-2)                        # [B,N,t,t]
        slot = i1.unsqueeze(-1) * self.pk_nk + i2.unsqueeze(-2)           # [B,N,t,t]
        cand = cand.flatten(-2)
        slot = slot.flatten(-2)
        best, pos = cand.topk(self.pk_topk, dim=-1)                       # [B,N,k]
        idx = slot.gather(-1, pos)
        w = torch.softmax(best, dim=-1)
        vals = F.embedding(idx, self.pkm_values)                          # [B,N,k,canon_d]
        read = (w.unsqueeze(-1) * vals).sum(dim=-2)                       # [B,N,canon_d]
        return torch.tanh(torch.nan_to_num(read, nan=0.0, posinf=1e4, neginf=-1e4))

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

        # -- champion causal system-identity summary (computed early: the object
        # prior and the deviation field both consume it; strict shift keeps it causal).
        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1)      # [B, n_pair]
            sal = sal * valid_obs.to(sal.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
            # STRICT SHIFT: command j sees the summary of obs pairs < j only.
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        # -- champion file memory: verb-quotient addresses, [content; view-ctx] values.
        Q = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), Q)
        k_file = self._quotient(self.file_write(x_cmd), Q)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        obs_pad = self._pad_steps(obs_tok, n_cmd)
        value_file = torch.cat([obs_pad, ctx], dim=-1)

        # -- champion path-state memory over trunk states.
        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = obs_pad

        # -- SPINE (1): episodic object state. Observations are encoded through
        # their write-command's view chart into bounded canonical codes, stored
        # (with the write-view tag) under the SAME verb-quotient file address.
        vfeat = torch.softmax(self.view_cls(x_cmd), dim=-1)            # [B,n_cmd,n_view]
        code_w = torch.tanh(
            self.canon_enc(obs_pad) * (1.0 + self.enc_gain(vfeat)) + self.enc_bias(vfeat)
        )
        value_canon = torch.cat([code_w.to(dtype), vfeat.to(dtype)], dim=-1)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)
        read_canon = self._chunked_delta_reads(q_file, k_file, value_canon, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]

        # -- champion FiLM view transform on the raw file read.
        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        # -- SPINE (2): semantic object prior from the code-space lexicon,
        # addressed by (frozen command token x causal system summary).
        cmd_tok = tok_emb[:, 0::2, :].to(x.dtype)
        prior_code = self._lexicon_prior(cmd_tok, s_cmd.to(x.dtype))   # [B,n_cmd,canon_d]

        # -- SPINE (3): system deviation field in code space. Innovation of each
        # completed pair = observed code minus the lexicon prior at that address;
        # salience-weighted, confidence-normalized running mean, strictly shifted.
        if n_pair:
            e = code_w.to(x.dtype)[:, :n_pair, :] - prior_code[:, :n_pair, :]
            e = e * active_pair.unsqueeze(-1).to(e.dtype)              # bounded in [-2,2]
            ernorm = (e.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            dsal_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs, ernorm], dim=-1)
            dsal = torch.sigmoid(self.dev_sal(dsal_in)).squeeze(-1) * active_pair.to(x.dtype)
            dval = torch.tanh(self.dev_val(e))
            dnum = torch.cumsum(dsal.unsqueeze(-1) * dval, dim=1)
            dden = torch.cumsum(dsal, dim=1).unsqueeze(-1)
            dmean = dnum / (dden + 1e-6)
            dconf = dden / (dden + 4.0)
            d_incl = dconf * dmean
            # STRICT SHIFT: command j sees deviations of pairs < j only.
            dev_cmd = torch.cat([d_incl.new_zeros(B, 1, self.canon_d), d_incl], dim=1)
            dev_cmd = self._pad_steps(dev_cmd, n_cmd)
        else:
            dev_cmd = x.new_zeros(B, n_cmd, self.canon_d)

        # -- fuse episodic state with the semantic prior (complementary systems).
        epi_code = read_canon[..., : self.canon_d].clamp(-5.0, 5.0)
        r_vw = read_canon[..., self.canon_d :].clamp(-5.0, 5.0)
        epi_n = (epi_code.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt().to(x.dtype)
        pri_n = (prior_code.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        g_epi = torch.sigmoid(self.conf_gate(torch.cat([h_cmd0, epi_n, pri_n], dim=-1)))
        c_fused = g_epi.to(dtype) * epi_code + (1.0 - g_epi).to(dtype) * prior_code.to(dtype)

        # -- condition the OBJECT by system identity + deviation field (zero-init).
        df_h = torch.tanh(
            self.devfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype), dev_cmd], dim=-1))
        )
        df = self.devfilm_out(df_h).to(dtype)
        c_cond = c_fused * (1.0 + df[..., : self.canon_d]) + df[..., self.canon_d :]
        c_cond = torch.nan_to_num(c_cond, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        # -- SPINE (4): render the conditioned object into the queried view.
        base_dec = self.canon_dec(c_cond.to(x.dtype))
        rf_h = torch.tanh(
            self.rfilm_in(torch.cat([h_cmd0, vfeat.to(x.dtype), r_vw.to(x.dtype)], dim=-1))
        )
        rf = self.rfilm_out(rf_h)
        r_canon = torch.nan_to_num(
            base_dec * (1.0 + rf[..., :D]) + rf[..., D:], nan=0.0, posinf=1e4, neginf=-1e4
        ).to(dtype)

        obs_for_prev = obs_tok * valid_obs.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_src = self._pad_steps(obs_for_prev, n_cmd - 1)
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), prev_src], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        rc = r_canon.to(x.dtype)
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
            + mix[:, :, 3:4] * r_canon
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        # -- champion zero-init target-space system FiLM.
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
    return R15FileobjectLexpriorSysdev(**params)
