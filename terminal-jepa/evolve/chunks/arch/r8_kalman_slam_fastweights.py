"""Kalman/RLS-SLAM fastweight predictor (wildcard cross-domain: robotics state estimation).

A shell exploration IS simultaneous localization and mapping: `cd` is odometry, `ls`/`cat` are
range measurements of a static map. The champion arch (r7_path_delta_fastweights_codex) writes
(command, observation) pairs into associative memories with the DELTA RULE — which is exactly the
LMS / Widrow-Hoff adaptive filter: a fixed-gain gradient step that ignores second-order structure.
EKF-SLAM's core insight is that a mapper must carry a COVARIANCE: repeated and correlated
measurements should be fused by their tracked uncertainty, and confidence in a retrieved map
element is a computed quantity (loop closure fires when revisiting a well-observed region drops
the predictive variance), not a guess.

This module upgrades both fastweight memories from LMS to exponentially-weighted RECURSIVE LEAST
SQUARES — the information-form Kalman filter for the static linear map obs ~= M^T key:

  write (key k unit-norm, value v, observation weight w, forgetting lam):
      g      = P k / (lam/w + k^T P k)          # Kalman gain from tracked covariance P
      M     <- M + g (v - M^T k)^T              # innovation update (least-squares, not LMS)
      P     <- (P - g (P k)^T) / lam            # covariance downdate + forgetting
  read (query q unit-norm):
      r = M^T q,   sigma2 = q^T P q             # prediction AND its predictive variance

Delta-rule is the degenerate case P = eta*I frozen. RLS instead solves the full least-squares
problem over all past pairs online: correlated keys (an `ls` of /etc and a `cat` of /etc/os-release
share key mass) are whitened rather than partially overwriting each other, and a novel query gets
the proper least-squares interpolation over everything seen. The variance sigma2 is fed back as
loop-closure confidence: it precision-weights the content-vs-path read mix and the direct
target-space output gate, so the memory speaks exactly when the current query direction has been
measured — a signal the champion's gates had to infer blindly from hidden features. Strictly
causal (same pending-pair write timing as the champion: a pair is written only at its observation
token), NaN-safe (all denominators >= lam/1 > 0, P symmetrized each write, variances clamped >= 0),
and bounded over the <=32-step sequences this model sees.
"""

import math

import torch
import torch.nn as nn

NAME = "r8_kalman_slam_fastweights"
DESCRIPTION = (
    "EKF-SLAM lens on the champion fastweights: both associative memories upgraded from "
    "delta-rule (LMS) to exponentially-weighted recursive least squares with a tracked "
    "per-memory covariance; the predictive variance q^T P q acts as loop-closure confidence "
    "that precision-weights the read mix and the direct target-space gate."
)


class KalmanSlamFastweights(nn.Module):
    def __init__(self, d=176, layers=2, key_d=64, ffn_mult=2, **unused):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = 768
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        ffn_h = max(self.d, int(ffn_mult * self.d))

        # ---- typed recurrent shell-state core (champion-matched) ----
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

        # ---- key/query maps for the two RLS map estimators ----
        self.content_read = nn.Linear(self.d, self.key_d, bias=False)
        self.content_write = nn.Linear(self.d, self.key_d, bias=False)
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        # ---- fusion (variance-aware: +2 loop-closure confidence features) ----
        self.read_mix = nn.Linear(self.d, 2)
        self.mix_prec = nn.Parameter(torch.tensor(0.5))    # softplus -> precision weight on -sigma2
        self.read_to_h = nn.Linear(self.D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 2, self.d)
        self.out_norm = nn.LayerNorm(self.d)

        self.write_gate = nn.Linear(2 * self.d, 1)
        self.direct_gate = nn.Linear(2 * self.d + 2, 1)
        self.head = nn.Linear(self.d, self.D)

        # RLS hyper-state: prior variance p0 and forgetting factor lam in (0.9, 1.0)
        self.p0_raw = nn.Parameter(torch.tensor(0.55))     # softplus(0.55) ~ 1.0
        self.lam_raw = nn.Parameter(torch.tensor(2.0))     # 0.9 + 0.1*sigmoid(2.0) ~ 0.988

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)

    # ---------------------------------------------------------------- helpers
    def _positional(self, L, device, dtype):
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(half, device=device, dtype=dtype)
                        * (-math.log(10000.0) / max(1, half - 1)))
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d // 2])
        return pe

    @staticmethod
    def _unit(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _read(mem, q):
        """mem [B,kd,D], q [B,kd] -> [B,D]  (M^T q)."""
        return torch.bmm(q.unsqueeze(1), mem).squeeze(1)

    @staticmethod
    def _variance(P, q):
        """P [B,kd,kd], q [B,kd] -> q^T P q  [B], clamped >= 0."""
        Pq = torch.bmm(P, q.unsqueeze(2)).squeeze(2)
        return (q * Pq).sum(-1).clamp_min(0.0)

    def _rls_write(self, mem, P, k, value, w, active, lam):
        """Weighted exponentially-forgetting RLS update, masked by `active` [B] bool.

        mem [B,kd,D], P [B,kd,kd], k [B,kd] unit key, value [B,D], w [B] obs weight in (0,1].
        """
        Pk = torch.bmm(P, k.unsqueeze(2)).squeeze(2)                    # [B,kd]
        kPk = (k * Pk).sum(-1).clamp_min(0.0)                           # [B]
        denom = lam / w.clamp(1e-3, 1.0) + kPk                          # [B] >= 0.9
        gain = Pk / denom.unsqueeze(1)                                  # [B,kd]
        err = value - self._read(mem, k)                                # [B,D] innovation
        mem_new = mem + gain.unsqueeze(2) * err.unsqueeze(1)            # [B,kd,D]
        P_new = (P - gain.unsqueeze(2) * Pk.unsqueeze(1)) / lam         # [B,kd,kd]
        P_new = 0.5 * (P_new + P_new.transpose(1, 2))                   # keep symmetric
        a = active.to(mem.dtype).view(-1, 1, 1)
        return mem * (1.0 - a) + mem_new * a, P * (1.0 - a) + P_new * a

    # ---------------------------------------------------------------- forward
    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device

        t = types.long().clamp(0, 1)
        valid = (~key_pad.bool()) if key_pad is not None else \
            torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype)[None]
        x = self.in_norm(x)
        dtype = x.dtype

        states = [torch.zeros(B, self.d, device=device, dtype=dtype)
                  for _ in range(self.layers)]

        p0 = nn.functional.softplus(self.p0_raw) + 0.05
        lam = 0.9 + 0.1 * torch.sigmoid(self.lam_raw)
        eye = torch.eye(self.key_d, device=device, dtype=dtype)
        mem_c = torch.zeros(B, self.key_d, self.D, device=device, dtype=dtype)
        mem_p = torch.zeros(B, self.key_d, self.D, device=device, dtype=dtype)
        P_c = (p0 * eye).expand(B, -1, -1).clone()
        P_p = (p0 * eye).expand(B, -1, -1).clone()

        pending_c = torch.zeros(B, self.key_d, device=device, dtype=dtype)
        pending_p = torch.zeros(B, self.key_d, device=device, dtype=dtype)
        pending_h = torch.zeros(B, self.d, device=device, dtype=dtype)
        pending_valid = torch.zeros(B, device=device, dtype=torch.bool)

        gamma = nn.functional.softplus(self.mix_prec)
        preds, hs = [], []

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

            # ---- reads with loop-closure confidence (predictive variance) ----
            q_c = self._unit(self.content_read(x[:, i, :]))
            q_p = self._unit(self.path_read(core))
            read_c = self._read(mem_c, q_c)
            read_p = self._read(mem_p, q_p)
            var_c = self._variance(P_c, q_c)                            # [B]
            var_p = self._variance(P_p, q_p)                            # [B]
            conf = 1.0 / (1.0 + torch.stack([var_c, var_p], dim=-1))    # [B,2] in (0,1]

            # precision-weighted content-vs-path mix (learned prior + -gamma*sigma2)
            mix_logits = self.read_mix(core) - gamma * torch.stack([var_c, var_p], dim=-1)
            mix = torch.softmax(mix_logits, dim=-1)
            target_read = mix[:, 0:1] * read_c + mix[:, 1:2] * read_p
            mem_h = self.read_to_h(target_read)

            fuse_in = torch.cat([core, mem_h, conf], dim=-1)
            h_i = self.out_norm(core + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)
            pred_i = self.head(h_i) + torch.sigmoid(self.direct_gate(fuse_in)) * target_read

            pred_i = torch.where(vi.unsqueeze(-1), pred_i, torch.zeros_like(pred_i))
            h_i = torch.where(vi.unsqueeze(-1), h_i, torch.zeros_like(h_i))
            preds.append(pred_i)
            hs.append(h_i)

            # ---- stage the pending pair at a command; write it at its observation ----
            new_c = self._unit(self.content_write(x[:, i, :]))
            new_p = self._unit(self.path_write(core))
            pending_c = torch.where(is_cmd.unsqueeze(-1), new_c, pending_c)
            pending_p = torch.where(is_cmd.unsqueeze(-1), new_p, pending_p)
            pending_h = torch.where(is_cmd.unsqueeze(-1), core, pending_h)
            pending_valid = torch.where(is_cmd, torch.ones_like(pending_valid), pending_valid)

            pair_active = is_obs & pending_valid
            if bool(pair_active.any()):
                w = torch.sigmoid(
                    self.write_gate(torch.cat([pending_h, core], dim=-1))).squeeze(-1)
                value = tok_emb[:, i, :].to(dtype)
                mem_c, P_c = self._rls_write(mem_c, P_c, pending_c, value, w, pair_active, lam)
                mem_p, P_p = self._rls_write(mem_p, P_p, pending_p, value, w, pair_active, lam)

            pending_valid = torch.where(is_obs, torch.zeros_like(pending_valid), pending_valid)

        return torch.stack(preds, dim=1), torch.stack(hs, dim=1)


def build(**params):
    return KalmanSlamFastweights(**params)
