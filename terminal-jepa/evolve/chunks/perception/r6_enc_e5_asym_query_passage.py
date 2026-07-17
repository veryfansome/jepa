"""perception TIER-2 (encoder recipe): intfloat/e5-base-v2 used ASYMMETRICALLY, the way e5 was
trained. e5 adds role prefixes 'query: ' and 'passage: ' and is designed for asymmetric retrieval
(a query about a passage). The archive recipe (enc_e5_base) used 'passage: ' for BOTH the command
and the observation, discarding that asymmetry. Here the shell COMMAND is rendered as a QUERY
(a question about what you are about to see) and the OBSERVATION stays a PASSAGE, identical to the
recorded e5 champion's obs render. This places the command token in e5's query subspace, which is
contrastively aligned to the passage subspace that is our retrieval target — so the world model's
cmd-position prediction starts from a geometry where the command already points at its observation.

Only the command side differs from enc_e5_base; the observation (passage) embeddings are byte-for-
byte the recorded champion's, so the same-verb foil geometry and retrieval space are preserved and
any margin change is attributable to the command-side asymmetry. 768-d, e5-standard mean-pool, no
L2-normalize (the harness z-scores z_obs and z_cmd separately, keeping the two roles scale-safe).
Loads via plain AutoModel.from_pretrained (standard BERT arch; no trust_remote_code)."""
from evolve.chunks.perception.baseline import pool, OBS_CAP

MODEL = "intfloat/e5-base-v2"


def render_obs(step):
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: cwd={step.get('cwd', '/')} exit={step.get('exit', 0)}\n{out}"


def render_cmd(step):
    # the command is a QUERY about the observation it will produce (e5 asymmetric role)
    return "query: " + step["cmd"]

