"""
R14 arch: a CANONICAL FILE-OBJECT memory with view-chart encoders and a
view-conditioned renderer — head/tail/cat/grep of one path become charts of
ONE stored object, not separate raw embeddings.

THE TARGET (measured, R14 inner diagnosis): tail has the lowest absolute wm
(.732) and cat's margin is STALLED (+.293 -> +.295 across two arch
generations). Both verbs are windows into one underlying file. The champion
(r13_syscond_film_content) binds views to one ADDRESS (verb-quotient keys) but
its file memory stores the raw 768-d embedding of the LAST-written view and a
zero-init FiLM affinely displaces it at read time. Two structural failures
follow:

  * DESTRUCTIVE OVERWRITE ACROSS VIEWS. The delta rule drives the slot toward
    the newest value. cat writes the whole file; a later `head` on the same
    path OVERWRITES that slot with the head-view embedding — the tail
    information the model had is erased. Views fight over one raw-value slot.
  * FiLM CANNOT CROSS VIEWS. An affine (gamma, shift) on the head-view
    embedding cannot produce the tail-view embedding: the frozen encoder
    pooled DISJOINT text for the two views. The champion's view transform can
    restyle a stored view, not re-render a different one.

MECHANISM — one new memory channel on the UNTOUCHED r13 champion (trunk, file
+ path delta memories, view FiLM, system FiLM, cued gates all verbatim):

1. VIEW-CHART ENCODER (write side). Each command gets soft view coordinates
   vfeat = softmax(view_cls(x_cmd)) (verbs + flags are in the raw command
   embedding; nothing hand-coded). Each observation is encoded to a bounded
   canonical code through its write-view's chart:
   code = tanh(enc(obs) * (1 + G vfeat) + B vfeat), with G,B zero-init so the
   chart starts as one shared encoder. Training pressure makes charts of the
   SAME file agree: a tail query reads a code written by cat, and the loss
   gradient flows through BOTH the renderer and cat's encoder, pulling
   code_cat(f) and code_tail(f) toward one canonical code_f — multi-view
   consistency, structure-from-motion in latent space. Once charts agree, the
   delta rule's overwrite-toward-latest becomes a harmless REFINEMENT (the new
   write says the same thing) instead of an erasure.

2. CANONICAL MEMORY. The code is stored — under the champion's existing
   verb-quotient file addresses (shared q_file/k_file, so the file binding
   already learned is reused for free) — via the same exact chunkwise delta
   solver, with the write-view coordinates appended as a value tag:
   value = [code (canon_d) ; vfeat (n_view)]. Cheap (V ~ 166 vs the raw
   channel's 864).

3. VIEW-CONDITIONED RENDERER (read side). pred_canon =
   dec(r_code) * (1 + gamma) + shift, where (gamma, shift) come zero-init from
   [query command state ; query view coords ; recovered write-view tag]: the
   canonical object is re-rendered INTO the queried view — the capsule /
   NeRF operation (pose-invariant instantiation code + view-conditioned
   rendering) rather than the champion's restyle-the-stored-view. Enters the
   prediction as a FOURTH read channel in the softmax mix, its logit biased
   low at init, so the module starts near-champion and the channel fades in
   only where it earns loss.

WHY THE MARGINS SHOULD MOVE: tail (+.364, lowest wm) — the dominant failure
is tail-after-cat/head on the same path; rendering from a canonical code that
cat populated is the first mechanism in this lineage that can emit content the
stored raw view embedding does not itself contain, and head-after-cat no
longer destroys it. cat (stalled +.295) — re-reads after partial views
(cat-after-head/tail/grep) currently return the partial view's raw embedding;
the canonical code accumulates all views, and the renderer emits the full-file
view. grep (within-traj baseline .513) — the renderer is conditioned on the
full query command state (pattern included), so it can render "matching lines
of THIS object" instead of echoing the stored output. ls/find keep their
champion pathways untouched.

Strictly causal: writes at pair i are readable only by commands > i (champion
chunked-delta guarantee); view coords use only each command's own token; the
system summary keeps its strict shift. NaN-safe (bounded codes, clamped reads,
nan_to_num), deterministic, ~0.5M added params at defaults.

Refs: capsule instantiation vs viewpoint (Sabour & Hinton, arXiv:1710.09829),
view-conditioned rendering from a canonical scene code (NeRF,
arXiv:2003.08934; GQN, Eslami et al. 2018), view-invariant object coding in
IT/perirhinal cortex (Booth & Rolls 1998), object files (Kahneman, Treisman &
Gibbs 1992), chunkwise DeltaNet (arXiv:2406.06484), FiLM (arXiv:1709.07871).
"""

import math

import torch
import torch.nn as nn

D = 768

NAME = "r14_canon_fileobject_viewrender"
DESCRIPTION = (
    "r13 champion + a canonical FILE-OBJECT memory: view-chart encoders map each "
    "observed head/tail/cat/grep of a path into ONE bounded canonical code stored "
    "under the existing verb-quotient file address, and a zero-init view-conditioned "
    "renderer re-renders that code into the queried view as a fourth read channel — "
    "so tail-after-cat renders the object instead of restyling the last raw view, "
    "and head-after-cat refines rather than erases. Near-champion at init."
)


class R14CanonFileobjectViewrender(nn.Module):
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
        canon_d=160,
        n_view=6,
        render_hidden=128,
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

        # -- champion system-identity summary + zero-init system FiLM.
        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        # -- NEW: canonical file-object channel.
        # soft view coordinates from the raw command projection (verb + flags live there).
        self.view_cls = nn.Linear(self.d, self.n_view)
        # shared canonical encoder + zero-init per-view chart (FiLM on the code).
        self.canon_enc = nn.Linear(D, self.canon_d)
        self.enc_gain = nn.Linear(self.n_view, self.canon_d, bias=False)
        self.enc_bias = nn.Linear(self.n_view, self.canon_d, bias=False)
        nn.init.zeros_(self.enc_gain.weight)
        nn.init.zeros_(self.enc_bias.weight)
        # view-conditioned renderer: canonical code -> queried view embedding.
        self.canon_dec = nn.Linear(self.canon_d, D)
        rh = max(16, int(render_hidden))
        self.rfilm_in = nn.Linear(self.d + 2 * self.n_view, rh)
        self.rfilm_out = nn.Linear(rh, 2 * D)
        nn.init.zeros_(self.rfilm_out.weight)
        nn.init.zeros_(self.rfilm_out.bias)

        # 4-way read mix (r_view / path / prev_obs / canonical render).
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
        # canonical channel fades in: its mix logit starts low.
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
        obs_pad = self._pad_steps(obs_tok, n_cmd)
        value_file = torch.cat([obs_pad, ctx], dim=-1)

        # -- path-state memory: champion channel over trunk states.
        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = obs_pad

        # -- NEW: canonical file-object memory. Each observation is encoded
        # through its write-command's view chart into a bounded canonical code;
        # the code (plus the write-view coordinates) is stored under the SAME
        # verb-quotient file address the champion already learns.
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

        # -- FiLM view transform (champion): map stored content into the queried view.
        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        # -- NEW: render the canonical object into the queried view.
        r_code = read_canon[..., : self.canon_d].clamp(-5.0, 5.0)
        r_vw = read_canon[..., self.canon_d :].clamp(-5.0, 5.0)
        base_dec = self.canon_dec(r_code.to(x.dtype))
        rf_h = torch.tanh(
            self.rfilm_in(torch.cat([h_cmd0, vfeat.to(x.dtype), r_vw.to(x.dtype)], dim=-1))
        )
        rf = self.rfilm_out(rf_h)
        r_canon = torch.nan_to_num(
            base_dec * (1.0 + rf[..., :D]) + rf[..., D:], nan=0.0, posinf=1e4, neginf=-1e4
        ).to(dtype)

        # -- causal system-identity summary (champion, gated running mean over past obs).
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

        # -- zero-init system FiLM (champion): displace toward the current system's variant.
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
    return R14CanonFileobjectViewrender(**params)
