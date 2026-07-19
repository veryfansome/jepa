"""Complementary-Learning-Systems dual-store memory on the delta-rule fastweights core.

The champion (r7_path_delta_fastweights) stores every completed (command, observation)
pair ONLY as a superposition in two delta-rule matrix memories — a "neocortical" store:
compact and smoothly interpolating, but interference-prone (a linear read mixes every
past association whose key overlaps the cue). CLS theory (McClelland, McNaughton &
O'Reilly 1995) and Neural Episodic Control (Pritzel et al. 2017, arXiv:1703.01988) say
the fix is a complementary VERBATIM store: keep the SAME (key -> observation)
associations twice — once superposed in the delta matrices, once verbatim in a
per-sequence episodic buffer — and arbitrate reads by cue-match confidence. An exact
revisit (same command at the same path state) is answered by a sharp softmax
pattern-completion over the verbatim store: a convex combination of real past
observation embeddings (on the retrieval manifold, zero interference). A
novel-but-related cue falls back to the matrices' smooth linear interpolation
(fastweights generalize; episodic recalls). Both stores share the champion's read/write
key spaces, so the two retrieval operators (softmax vs linear superposition; cf. Schlag
et al. 2021, arXiv:2102.11174) act on identical associations and a 4-way learned mix —
conditioned on the episodic match confidences — arbitrates between them. Closing the
loop, the episodic familiarity of the pending command also feeds the delta-rule WRITE
gate, so consolidation strength is informed by what the episodic store already knows.
Strictly causal: buffer slot t is appended only at observation position 2t+1, after
every read at or before command position 2t has already happened.
"""

import math

import torch
import torch.nn as nn


NAME = "r8_cls_dual_store_fastweights"
DESCRIPTION = (
    "CLS dual-store memory: the champion's delta-rule fastweight matrices (semantic, "
    "interpolating) plus a verbatim episodic buffer over the same key/value pairs "
    "(softmax pattern completion over real past observations), arbitrated by cue-match "
    "confidence, with episodic familiarity gating the delta-rule consolidation write."
)


class CLSDualStoreFastweights(nn.Module):
    def __init__(self, d=176, layers=2, key_d=64, ffn_mult=2, ep_beta=8.0, **unused):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = 768
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        ffn_h = max(self.d, int(ffn_mult * self.d))

        # ---- typed recurrent shell-state core (unchanged from the champion) ----
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

        # ---- shared key spaces (identical roles to the champion) ----
        self.content_read = nn.Linear(self.d, self.key_d, bias=False)
        self.content_write = nn.Linear(self.d, self.key_d, bias=False)
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        # ---- dual-store read arbitration ----
        # 4 pathways: content-matrix, path-matrix, content-episodic, path-episodic.
        # The mix is conditioned on the core state AND the two episodic match
        # confidences, so arbitration follows cue familiarity (CLS/NEC).
        self.read_mix = nn.Linear(self.d + 2, 4)
        self.read_to_h = nn.Linear(self.D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d, self.d)
        self.out_norm = nn.LayerNorm(self.d)

        # write gate now also sees the pending command's episodic familiarity
        self.write_gate = nn.Linear(2 * self.d + 2, 1)
        self.direct_gate = nn.Linear(2 * self.d, 1)
        self.head = nn.Linear(self.d, self.D)

        self.logit_decay = nn.Parameter(torch.tensor(math.log(0.985 / 0.015)))
        # episodic softmax sharpness: beta = softplus(p) + 1, init ~ep_beta
        init_p = math.log(max(math.exp(ep_beta - 1.0) - 1.0, 1e-4))
        self.ep_beta_c = nn.Parameter(torch.tensor(init_p))
        self.ep_beta_p = nn.Parameter(torch.tensor(init_p))

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

    @staticmethod
    def _read(mem, key):
        return torch.bmm(key.unsqueeze(1), mem).squeeze(1)

    def _delta_write(self, mem, key, value, amount, active, decay):
        old = self._read(mem, key)
        delta = value - old
        update = key.unsqueeze(2) * delta.unsqueeze(1) * amount.view(-1, 1, 1)
        active = active.float().view(-1, 1, 1)
        candidate = decay * mem + update
        return mem * (1.0 - active) + candidate * active

    def _episodic_read(self, q, keys, vals, ok, beta, val_dtype):
        """Softmax pattern completion over the verbatim store.
        q [B,kd]; keys [B,S,kd]; vals [B,S,D]; ok [B,S] bool.
        Returns (read [B,D] in val_dtype, maxsim [B,1] in q.dtype)."""
        sims = torch.einsum("bk,bsk->bs", q, keys)                    # [B,S] in [-1,1]
        neg = torch.finfo(sims.dtype).min
        logits = torch.where(ok, beta * sims, torch.full_like(sims, neg))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)                        # all-masked rows -> 0
        read = torch.einsum("bs,bsd->bd", attn.to(val_dtype), vals)
        has = ok.any(dim=-1, keepdim=True)
        maxsim = torch.where(ok, sims, torch.full_like(sims, -2.0)).max(dim=-1, keepdim=True).values
        maxsim = torch.where(has, maxsim, torch.zeros_like(maxsim))
        return read, maxsim

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

        # verbatim episodic store: one slot appended per completed pair (causal by
        # construction — slot t only exists after observation position 2t+1)
        ep_kc, ep_kp, ep_v, ep_ok = [], [], [], []

        pending_c = torch.zeros(B, self.key_d, device=device, dtype=x.dtype)
        pending_p = torch.zeros(B, self.key_d, device=device, dtype=x.dtype)
        pending_h = torch.zeros(B, self.d, device=device, dtype=x.dtype)
        pending_ms = torch.zeros(B, 2, device=device, dtype=x.dtype)
        pending_valid = torch.zeros(B, device=device, dtype=torch.bool)

        preds = []
        hs = []
        decay = torch.sigmoid(self.logit_decay)
        beta_c = nn.functional.softplus(self.ep_beta_c) + 1.0
        beta_p = nn.functional.softplus(self.ep_beta_p) + 1.0

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
            read_content = self._read(mem_content, q_content)
            read_path = self._read(mem_path, q_path)

            if ep_v:
                Kc = torch.stack(ep_kc, dim=1)                 # [B,S,kd]
                Kp = torch.stack(ep_kp, dim=1)
                V = torch.stack(ep_v, dim=1)                   # [B,S,D]
                OK = torch.stack(ep_ok, dim=1)                 # [B,S] bool
                ep_read_c, ms_c = self._episodic_read(q_content, Kc, V, OK, beta_c, dtype)
                ep_read_p, ms_p = self._episodic_read(q_path, Kp, V, OK, beta_p, dtype)
            else:
                ep_read_c = torch.zeros(B, self.D, device=device, dtype=dtype)
                ep_read_p = torch.zeros(B, self.D, device=device, dtype=dtype)
                ms_c = torch.zeros(B, 1, device=device, dtype=x.dtype)
                ms_p = torch.zeros(B, 1, device=device, dtype=x.dtype)
            ms = torch.cat([ms_c, ms_p], dim=-1)               # [B,2] episodic familiarity

            mix = torch.softmax(self.read_mix(torch.cat([core, ms], dim=-1)), dim=-1)
            target_read = (mix[:, 0:1] * read_content + mix[:, 1:2] * read_path
                           + mix[:, 2:3] * ep_read_c + mix[:, 3:4] * ep_read_p)
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
            pending_ms = torch.where(is_cmd.unsqueeze(-1), ms, pending_ms)
            pending_valid = torch.where(is_cmd, torch.ones_like(pending_valid), pending_valid)

            pair_active = is_obs & pending_valid
            if bool(pair_active.any()):
                # consolidation write, informed by episodic familiarity of the cue
                gate_in = torch.cat([pending_h, core, pending_ms], dim=-1)
                amount = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
                value = tok_emb[:, i, :]
                mem_content = self._delta_write(mem_content, pending_c.to(dtype), value,
                                                amount.to(dtype), pair_active, decay.to(dtype))
                mem_path = self._delta_write(mem_path, pending_p.to(dtype), value,
                                             amount.to(dtype), pair_active, decay.to(dtype))
                # verbatim episodic append of the SAME association (one-shot, no interference)
                ep_kc.append(pending_c)
                ep_kp.append(pending_p)
                ep_v.append(value)
                ep_ok.append(pair_active)

            pending_valid = torch.where(is_obs, torch.zeros_like(pending_valid), pending_valid)

        return torch.stack(preds, dim=1), torch.stack(hs, dim=1)


def build(**params):
    return CLSDualStoreFastweights(**params)
