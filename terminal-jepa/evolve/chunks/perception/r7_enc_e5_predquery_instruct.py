"""perception TIER-2 (encoder recipe): intfloat/e5-base-v2, the recorded champion encoder, used
ASYMMETRICALLY with an INSTRUCTION-CONDITIONED, PREDICTIVE command query.

Motivation / differential-help argument
---------------------------------------
The fitness is the MARGIN wm - max(baselines): it rewards an encoder that helps the WORLD MODEL
more than it helps the no-model baselines. The two objective-independent baselines
(retrieve_by_cmd, copy_prev) and the hard same-verb FOILS all live in the OBSERVATION (passage)
space. This recipe leaves that space BYTE-FOR-BYTE identical to the recorded e5 champion
(enc_e5_base): render_obs and pool are imported unchanged, so z_obs, the standardization stats
that touch obs, the foil geometry and every passage-space baseline are preserved. ONLY the command
render changes -> any margin delta is attributable to the command-side query.

e5-base-v2 is a shared-weight siamese retrieval encoder that marks a "question about a passage"
with the 'query: ' prefix and a passage with 'passage: '. The R6 asym recipe (bare 'query: '<cmd>)
was ~tied with symmetric 'passage: ' both sides -- the query subspace is usable but a bare shell
command is not a natural-language question, so it under-specifies the query. Here the command is
expanded into an explicit PREDICTIVE INSTRUCTION -- the exact task the world model performs at a
cmd position: given the command, anticipate the resulting output + working directory. This is the
efference-copy / forward-model idea (Pickering & Clark 2014): encode the action together with an
instruction to predict its sensory consequence, so the command embedding is pushed along the
contrastive direction e5 learned toward the PASSAGE manifold that answers a prediction, giving the
WM's cmd-position readout a starting geometry that already points at 'what comes next'. Because the
push is toward the cross-modal passage manifold (not command-command similarity), a fixed-cosine
lexical baseline (retrieve_by_cmd, which matches on command embeddings) can exploit it far less
than the learned cmd->obs WM head can -- the intended differential.

Contract / safety
-----------------
- MODEL 768-d, plain AutoModel.from_pretrained (standard BERT arch; NO trust_remote_code).
- e5-standard mean-pool (imported from enc_e5_base -> baseline.pool), NO L2-normalize (the harness
  z-scores z_obs and z_cmd separately, keeping the two roles scale-safe).
- Deterministic, data-independent render; no learned params, nothing fitted on our data.
- Causal/leak-free: render_cmd reads ONLY step['cmd'] (the action), never the output/cwd/exit that
  the command produces -- it embeds the QUESTION, not the answer. Anti-collapse is a property of the
  downstream objective, unchanged here.
"""

# Observation (passage) side + pooling are the recorded e5 champion's, UNCHANGED, so the target /
# eval / baseline / foil space is bit-identical to data/dockerfs-e5.
from evolve.chunks.perception.enc_e5_base import MODEL, render_obs, pool  # noqa: F401


def render_cmd(step):
    """The command as an instruction-conditioned QUERY that asks the encoder to represent the
    action as a *prediction* of its next observation. Reads only step['cmd'] (no leakage of the
    resulting output/cwd/exit). Kept short and in e5's 'query: ' subspace, matching how e5 was
    trained (query prefix), with the predictive task made explicit."""
    cmd = (step.get("cmd", "") or "").strip()
    return (
        "query: Given the shell command '"
        + cmd
        + "', predict the resulting terminal output and working directory."
    )

