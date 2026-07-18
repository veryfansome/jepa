"""Phase 1 networks (terminal-jepa.md §4): from-scratch token encoder (CLS latent),
compositional action encoder, AdaLN-conditioned predictor, reconstruction-twin decoder,
and the IDM ablation head. Total budget <=20M params."""

import torch
import torch.nn as nn

from . import data

D_Z = 256
D_ACT = 128


class TokenEncoder(nn.Module):
    def __init__(self, d_model=192, n_layers=4, n_heads=4, d_ff=512,
                 d_out=D_Z, max_len=640):
        super().__init__()
        self.d_out = d_out
        self.emb = nn.Embedding(len(data.TOKEN_VOCAB), d_model, padding_idx=data.PAD)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, ids):
        pad_mask = ids == data.PAD  # CLS at position 0 is never pad
        h = self.emb(ids) + self.pos[:, : ids.shape[1]]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.out(h[:, 0])


class SlotEncoder(nn.Module):
    """Token-latent variant (status doc queue item 4): K learned slot queries
    cross-attend to the encoded token sequence, giving a set-structured state code of
    n_slots*d_slot dims instead of the single CLS vector. Raw per-token latents don't
    fit this domain — the token sequence changes length/order as the tree changes, so
    there is no fixed grid to predict over; a fixed slot set is the analogue of
    Causal-JEPA's object slots. Returns the flattened code so every downstream consumer
    (losses, probes, audits) is unchanged."""

    def __init__(self, d_model=192, n_layers=4, n_heads=4, d_ff=512,
                 n_slots=16, d_slot=64, max_len=640):
        super().__init__()
        self.n_slots, self.d_slot = n_slots, d_slot
        self.d_out = n_slots * d_slot
        self.emb = nn.Embedding(len(data.TOKEN_VOCAB), d_model, padding_idx=data.PAD)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.queries = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.readout = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # No norm before the head: LayerNorm here destroys the per-sample scale that
        # SIGReg must set — the exact pattern LeWorldModel engineered out of their
        # encoder head (they cite it as preventing the anti-collapse objective from
        # optimizing). The CLS path has no pre-head norm either.
        self.out = nn.Linear(d_model, d_slot)

    def forward(self, ids):
        pad_mask = ids == data.PAD
        h = self.emb(ids) + self.pos[:, : ids.shape[1]]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        q = self.queries.expand(ids.shape[0], -1, -1)
        s, _ = self.readout(q, h, h, key_padding_mask=pad_mask)
        return self.out(s).reshape(ids.shape[0], self.d_out)


class AdaLNBlock(nn.Module):
    def __init__(self, d, d_cond, d_hidden):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, d_hidden), nn.GELU(), nn.Linear(d_hidden, d))
        self.cond = nn.Linear(d_cond, 3 * d)
        nn.init.zeros_(self.cond.weight)  # zero-init gate: identity at start (LeWorldModel)
        nn.init.zeros_(self.cond.bias)

    def forward(self, h, c):
        gamma, beta, gate = self.cond(c).chunk(3, dim=-1)
        return h + gate * self.mlp(self.norm(h) * (1 + gamma) + beta)


class Predictor(nn.Module):
    """(z_t, action) -> z_{t+1}. Context window W=1 per the plan; the AdaLN block stack
    generalizes to a causal transformer over longer W in Phase 3."""

    def __init__(self, d=D_Z, d_cond=D_ACT, n_blocks=4, d_hidden=1024):
        super().__init__()
        self.blocks = nn.ModuleList(
            AdaLNBlock(d, d_cond, d_hidden) for _ in range(n_blocks)
        )

    def forward(self, z, a):
        h = z
        for blk in self.blocks:
            h = blk(h, a)
        return h


class SlotPredictor(nn.Module):
    """Slot-set dynamics: self-attention across slots (interaction routing) + AdaLN
    action conditioning per block. Consumes and returns the flattened slot code."""

    def __init__(self, n_slots=16, d_slot=64, d_cond=D_ACT, n_blocks=4, n_heads=4):
        super().__init__()
        self.n_slots, self.d_slot = n_slots, d_slot
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(d_slot, n_heads, batch_first=True)
            for _ in range(n_blocks)
        )
        self.attn_norm = nn.ModuleList(nn.LayerNorm(d_slot) for _ in range(n_blocks))
        # Zero-init attention gates: the whole predictor must be the identity at init
        # (like the CLS Predictor's zero-init AdaLN), or prediction loss starts large
        # and encoder collapse is the fastest descent path — observed: z_std -> 7e-4.
        self.attn_gate = nn.ParameterList(
            nn.Parameter(torch.zeros(d_slot)) for _ in range(n_blocks)
        )
        self.mlps = nn.ModuleList(
            AdaLNBlock(d_slot, d_cond, 4 * d_slot) for _ in range(n_blocks)
        )

    def forward(self, z, a):
        b = z.shape[0]
        h = z.reshape(b, self.n_slots, self.d_slot)
        c = a.unsqueeze(1)  # broadcast action conditioning over slots
        for attn, norm, gate, mlp in zip(
            self.attn, self.attn_norm, self.attn_gate, self.mlps
        ):
            hn = norm(h)
            h = h + gate * attn(hn, hn, hn, need_weights=False)[0]
            h = mlp(h, c)
        return h.reshape(b, self.n_slots * self.d_slot)


class ReconDecoder(nn.Module):
    """Generative twin's head: causal LM over next-observation tokens, conditioned on
    (z_t, action) via a prepended embedding. Same encoder trunk as the JEPA arm."""

    def __init__(self, d_model=192, n_layers=4, n_heads=4, d_ff=512, max_len=640,
                 d_z=D_Z):
        super().__init__()
        self.cond = nn.Linear(d_z + D_ACT, d_model)
        self.emb = nn.Embedding(len(data.TOKEN_VOCAB), d_model, padding_idx=data.PAD)
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, len(data.TOKEN_VOCAB))

    def loss(self, z, a, target_ids):
        cond = self.cond(torch.cat([z, a], dim=-1)).unsqueeze(1)
        tok = self.emb(target_ids[:, :-1])
        h = torch.cat([cond, tok], dim=1) + self.pos[:, : target_ids.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(
            h.shape[1], device=h.device
        )
        h = self.decoder(h, mask=mask, is_causal=True)
        logits = self.head(h)
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=data.PAD,
        )


class IDMHead(nn.Module):
    """Inverse dynamics ablation arm: (z_t, z_{t+1}) -> action classification."""

    def __init__(self, d=D_Z, d_hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2 * d, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU()
        )
        self.verb = nn.Linear(d_hidden, len(data.VERB_INDEX))
        self.arg1 = nn.Linear(d_hidden, len(data.ARG_VOCAB))
        self.arg2 = nn.Linear(d_hidden, len(data.ARG_VOCAB))

    def loss(self, z_t, z_t1, labels):
        """Returns (total_loss, per-head metrics as detached tensors). arg1 — which
        path the action touched — is the load-bearing head for the layout-consistent
        path-code hypothesis; it must be observable separately, not averaged away."""
        h = self.trunk(torch.cat([z_t, z_t1], dim=-1))
        ce = nn.functional.cross_entropy
        logits = {"verb": self.verb(h), "arg1": self.arg1(h), "arg2": self.arg2(h)}
        head_losses = {k: ce(v, labels[:, i]) for i, (k, v) in enumerate(logits.items())}
        metrics = {}
        for i, (k, v) in enumerate(logits.items()):
            metrics[f"{k}_loss"] = head_losses[k].detach()
            metrics[f"{k}_acc"] = (v.argmax(-1) == labels[:, i]).float().mean().detach()
        return sum(head_losses.values()) / 3.0, metrics


ENCODER_TYPES = ("cls", "slot")


def build_models(encoder_type="cls"):
    action_encoder = TokenEncoder(
        d_model=128, n_layers=2, n_heads=4, d_ff=256, d_out=D_ACT, max_len=24
    )
    if encoder_type == "slot":
        return {
            "encoder": SlotEncoder(),
            "action_encoder": action_encoder,
            "predictor": SlotPredictor(),
        }
    return {
        "encoder": TokenEncoder(),
        "action_encoder": action_encoder,
        "predictor": Predictor(),
    }


def build_encoder(encoder_type="cls"):
    return SlotEncoder() if encoder_type == "slot" else TokenEncoder()


def load_encoder_state(enc, state):
    """Shared encoder-state loader for every tool that consumes checkpoints.
    Pre-audit slot checkpoints carry a since-removed pre-head LayerNorm; loading them
    without it changes the encoder's function, so warn loudly rather than crash (or
    silently strict-fail with an unexplained key error, as unshared loaders did)."""
    legacy = [k for k in state if k.startswith("norm.")]
    if legacy:
        print(f"WARNING: dropping legacy pre-head norm keys {legacy} — this "
              f"pre-audit checkpoint will NOT reproduce its original embeddings",
              flush=True)
        state = {k: v for k, v in state.items() if not k.startswith("norm.")}
    enc.load_state_dict(state)
    return legacy  # dropped-key list: callers must surface this in their artifacts


def param_count(modules):
    return sum(p.numel() for m in modules for p in m.parameters())
