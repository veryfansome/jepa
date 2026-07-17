"""Hippocampal episodic place-memory predictor: a causal transformer ("slow neocortex")
augmented with an explicit, non-parametric episodic memory ("fast hippocampus") that is
WRITTEN causally as the shell exploration unfolds and READ at every command position by
content-addressable pattern completion.

CROSS-DOMAIN MOTIVATION (neuroscience -> architecture)
------------------------------------------------------
A shell exploration is literal spatial navigation: `cd` moves the agent between places
(directories), and `ls`/`cat` are sensory observations AT a place. Two neuroscience results
translate directly:

  * Complementary Learning Systems / Neural Episodic Control (Pritzel et al. 2017,
    arXiv:1703.01988). The hippocampus is a fast, content-addressable episodic store that
    binds (context, observation) in ONE shot and later retrieves the observation from a
    partial cue by pattern completion (CA3 recurrence). It complements a slow neocortex
    that only gradually learns statistical regularities. Here the slow neocortex is the
    transformer (a learned cmd->obs map that must TRANSFER to unseen systems and is exactly
    what fails to transfer); the fast hippocampus is a per-sequence key-value store whose
    read needs NO transfer — within one exploration, a revisited/related place is completed
    one-shot from the actual observation that was seen.

  * Tolman-Eichenbaum Machine (Whittington et al. 2020, Cell). Generalization comes from
    FACTORIZING a structural place/grid code from sensory content: the place code indexes
    memory and transfers across environments even when the sensory content differs. Here the
    memory KEY is a factorized place code (the transformer's causal command-hidden context +
    a projection of the command embedding, which localizes the target path/file), kept
    SEPARATE from the stored VALUE (the raw observation embedding). So "which command, at
    which place, in which system" addresses the store, while the value it returns is a real
    observation embedding.

WHY THIS SHOULD RAISE THE HELD-OUT CONTENT-VERB MARGIN
------------------------------------------------------
The retrieval eval scores a prediction by squared-L2 distance to the true next observation
against same-verb foils. The transformer head emits an ARBITRARY 768-d vector; its learned
cmd->obs mapping may not transfer to an unseen image. The episodic read instead returns
    e_t = sum_{j<t} softmax(q_t . k_j) * obs_j
a CONVEX COMBINATION of ACTUAL standardized observation embeddings from strictly-past steps.
Two properties matter: (1) e_t is guaranteed to lie on the manifold of real observation
embeddings — precisely the space retrieval scores; (2) it requires no cross-system transfer,
because it copies observations the agent has already seen IN THIS exploration (repeated
commands, an `ls` of a directory whose file is later `cat`-ed, a revisited `cwd`). A
confidence gate (built from the read's max weight + entropy) fires the episodic pathway only
when memory is informative and otherwise relaxes to the parametric baseline, so the module is
a strict, learnable generalization of the baseline transformer.

DISTINCTION FROM EXISTING ARCH IMPLS AND FROM THE BASELINES
-----------------------------------------------------------
Unlike sysid (a broadcast global identity mean) and recency-ALiBi (a distance prior on the
transformer's own attention), this adds a SEPARATE read pathway whose VALUE is the raw
observation embedding and whose output directly forms the prediction (not a hidden feature a
head must decode). Full self-attention could in principle attend to past obs tokens, but its
output passes through value projections + layernorms and is NOT constrained to the obs
manifold; the explicit read is. It differs from the `retrieve_by_cmd` baseline (nearest
TRAIN command, cross-episode, no world model) because it retrieves over the CURRENT
sequence's strictly-past observations, keyed by a learned place code, fused with the
transformer; and from `copy_prev` (fixed t-1) because it is content-addressed over the whole
past, not the previous step.

CAUSALITY (leak-free by construction)
-------------------------------------
The transformer uses the standard upper-triangular causal mask, so the command hidden state
at position 2t depends only on tokens 0..2t (never its own observation 2t+1). The episodic
read at step t attends ONLY over entries j<t (strict lower triangular). Entry j's value is
obs_j at token position 2j+1 <= 2t-1 < 2t, so a command-position prediction at or before t
never depends on obs_t or any later observation. Disallowed entries get a true -inf logit
(exact zero weight; no future/pad leakage), and all-masked query rows are nan-to-num'd to a
zero read with the gate forced to zero via a has-memory mask. The per-genome no-leakage guard
(corrupt obs_t, check cmd_<=t predictions do not move) passes: verified that corrupting obs_3
leaves the step 0..3 command predictions bit-identical while legitimately changing step 4+.

NUMERICAL STABILITY
-------------------
No exponentials over sequence position and no division by an accumulated mass. The only
softmaxes are a standard causal transformer attention and the episodic read over a finite
[n,n] logit matrix with a bounded learned temperature (softplus + 0.5). Entropy uses a
clamped log. Padded positions are zeroed for the value mask; the harness reads only valid
command positions.

I/O CONTRACT
------------
forward(tok_emb [B,L,768], types [B,L] in {0,1}, key_pad [B,L] bool True=pad)
    -> (pred [B,L,768], h [B,L,d]); prediction at EVERY position (harness reads pred[:,0::2]).
~2.3M params at d=192, layers=4, heads=4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "hippo_episodic_place_read"
DESCRIPTION = (
    "Causal transformer (slow neocortex) + explicit non-parametric episodic memory (fast "
    "hippocampus): at each command position, a factorized place-code query pattern-completes "
    "a convex combination of strictly-past raw observation embeddings, confidence-gated into "
    "the parametric head. Strictly causal, on-manifold retrieval path, numerically bounded."
)


class HippoEpisodicWorldModel(nn.Module):
    """Baseline causal transformer front-end producing a per-position hidden state and a
    parametric prediction, plus a per-sequence episodic key-value read at command positions.

    Episodic read (per step t, keyed by a factorized place code):
        place_i = LayerNorm( h_cmd_i + cmd_proj(cmd_emb_i) )     # structural context + location
        q_t = Wq(place_t),  k_j = Wk(place_j)                    # query / keys
        logit_{t,j} = (q_t . k_j) / sqrt(kq) / temp,  temp = softplus(log_temp)+0.5
        e_t = sum_{j<t, j valid} softmax_j(logit_{t,j}) * obs_j  # value = raw std obs embedding
    Confidence-gated blend with the parametric prediction p_t at command positions:
        g_t = sigmoid(gate([h_cmd_t, max_j attn, entropy])) * has_memory
        pred_cmd_t = (1 - g_t) * p_t + g_t * e_t
    obs positions keep the parametric prediction (unused by the harness).
    """

    def __init__(self, d=192, layers=4, heads=4, dropout=0.1, max_len=64, kq=128,
                 no_history=False):
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.kq = kq
        # no_history=True is the matched-capacity ablation control: self-only attention AND the
        # episodic read disabled (the memory IS history by construction). Same modules/params.
        self.no_history = no_history

        # ---- slow neocortex: baseline-matched causal transformer front-end ----
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)          # 0=cmd, 1=obs
        self.pos_emb = nn.Embedding(max_len, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, D)

        # ---- fast hippocampus: factorized place-keyed episodic read ----
        self.cmd_proj = nn.Linear(D, d)              # sensory-command -> location component
        self.place_norm = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, kq)
        self.Wk = nn.Linear(d, kq)
        self.log_temp = nn.Parameter(torch.zeros(()))      # bounded read temperature
        self.gate = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        dev = tok_emb.device
        idx = torch.arange(L, device=dev).clamp(max=self.max_len - 1)
        x = self.proj(tok_emb) + self.type_emb(types.clamp(0, 1)) + self.pos_emb(idx)[None]

        if self.no_history:
            # self-only attention; ~eye alone forbids all cross-position flow, so drop the
            # key-padding mask — a padded query attending to itself is harmless (never read)
            # and avoids the all-masked-row NaN kernel divergence (CPU vs MPS).
            mask = ~torch.eye(L, device=dev, dtype=torch.bool)
            h = self.tf(x, mask=mask, src_key_padding_mask=None)
        else:
            mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1)
            h = self.tf(x, mask=mask, src_key_padding_mask=key_pad)  # [B,L,d]
        pred = self.head(h)                                          # [B,L,768] parametric
        if self.no_history:
            return pred, h                                          # episodic read disabled

        # ---- assemble per-step command/observation views (aligned by step) ----
        n = L // 2
        if n == 0:
            return pred, h
        hc = h[:, 0:2 * n:2]                     # [B,n,d]   command hidden (causal context)
        cmd_raw = tok_emb[:, 0:2 * n:2]          # [B,n,768] raw command embedding (location)
        obs_val = tok_emb[:, 1:2 * n:2]          # [B,n,768] raw standardized obs = episodic value
        valid_cmd = (~key_pad)[:, 0:2 * n:2]     # [B,n] bool valid command steps

        # ---- factorized place code (structural context + sensory-command location) ----
        place = self.place_norm(hc + self.cmd_proj(cmd_raw))        # [B,n,d]
        q = self.Wq(place)                                          # [B,n,kq]
        k = self.Wk(place)                                          # [B,n,kq]
        temp = F.softplus(self.log_temp) + 0.5
        logits = (q @ k.transpose(-2, -1)) / (self.kq ** 0.5) / temp   # [B,n,n]

        # strict causal read: query t may attend only to entries j < t, and only valid keys.
        strict = torch.tril(torch.ones(n, n, device=dev, dtype=torch.bool), -1)  # j<t
        allowed = strict[None] & valid_cmd[:, None, :]              # [B,n,n]
        logits = logits.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)                     # all-masked rows -> 0
        e = attn @ obs_val                                         # [B,n,768] convex combo (on-manifold)

        # ---- confidence-gated blend (neuromodulatory retrieval gate) ----
        maxw = attn.max(dim=-1, keepdim=True).values               # [B,n,1] sharpness of match
        ent = -(attn * torch.log(attn.clamp(min=1e-9))).sum(-1, keepdim=True)  # [B,n,1]
        has_mem = allowed.any(dim=-1, keepdim=True).to(x.dtype)    # 0 when no valid past entry
        g = torch.sigmoid(self.gate(torch.cat([hc, maxw, ent], dim=-1))) * has_mem  # [B,n,1]

        p_cmd = pred[:, 0:2 * n:2]                                  # parametric cmd predictions
        blended = (1.0 - g) * p_cmd + g * e                        # [B,n,768]
        pred = pred.clone()
        pred[:, 0:2 * n:2] = blended
        return pred, h


def build(d=192, layers=4, heads=4, dropout=0.1, max_len=64, kq=128, no_history=False):
    return HippoEpisodicWorldModel(d=d, layers=layers, heads=heads, dropout=dropout,
                                   max_len=max_len, kq=kq, no_history=no_history)

