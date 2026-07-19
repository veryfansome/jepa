"""Build the inventor (mutation-operator) brief for a chunk, and extract code from a reply.
Both Claude subagents and Codex (`codex exec`) get the SAME brief, so mixing them is genuine
model/harness diversity (the ShinkaEvolve 'ensemble of frontier LLMs' idea) rather than an
apples-to-oranges comparison.

  python -m evolve.inventors objective          # print a chunk's brief to stdout

Bespoke briefs (objective / arch / target) are hand-written; the other chunks (optim / perception /
batcher / stream / head) use the generic brief driven by the baseline impl's docstring. Every brief
carries STANDING_RULES so both Claude and Codex inventors get the load-bearing rules (retry failed
traits recombined; look outside ML) in the one channel both reliably receive — the brief itself.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evolve import archive as A
from evolve import genome as G

ROOT = pathlib.Path(__file__).resolve().parent

# Rules every inventor must get regardless of dispatch channel. This is the ONLY context Codex
# inventors reliably receive (they don't inherit CLAUDE.md beyond the root AGENTS.md, and never
# auto-memory), so the load-bearing behaviors — retry-failed-recombined and cross-domain research,
# the two highest-leverage patterns from R6/R7 — live HERE in the brief, not only in the docs/memory.
STANDING_RULES = """
STANDING RULES (every inventor, every chunk):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot. Invent a genuinely different mechanism, or a
  novel RECOMBINATION of ideas already in the archive. Do not resubmit an impl listed above. Commit
  to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before is NOT off-limits. Evolution recombines: a
  trait that failed ALONE can win in a CHANGED context (stacked with a new winner, a different
  encoder/objective). You MAY retry a past idea in a new context; argue why the change could flip it.
- LOOK OUTSIDE ML — the biggest wins in this search came from CROSS-DOMAIN lenses. Search arxiv.org
  and read BEYOND machine learning (neuroscience: predictive coding, hippocampal/episodic memory,
  place & grid cells; biology; physics; information theory) and translate ONE concrete mechanism
  into code (equations, not metaphor). The hippocampal-memory and fast-weights architectures and the
  free-energy objective — the three largest gains — each came from this.
- NEVER touch the eval / retrieval metric / 3-way split / no-leakage guard; the harness re-checks and
  scores any violation −inf.
"""

# Chunks without a hand-written brief above use the generic brief, driven by the baseline impl's
# docstring (the authoritative contract): (what the chunk controls, its baseline filename).
GENERIC = {
    "optim": ("the OPTIMIZER + LR schedule for the fixed-step training run", "baseline_adamw.py"),
    "perception": ("PERCEPTION — the frozen encoder 'eyes' + the render/pooling recipe (evolved as a "
                   "data-side re-encode into a new data root via reencode.py / mv_encode.py)", "baseline.py"),
    "batcher": ("the BATCHER — how each training batch is composed (the in-batch negative pool the "
                "contrastive objectives rank against)", "baseline_uniform.py"),
    "stream": ("the STREAM — how the (command, observation) step sequence is laid out as tokens "
               "(single-vector interleave vs multi-vector)", "baseline_interleave.py"),
    "head": ("the readout HEAD — how the transformer's hidden state maps to the prediction, plus an "
             "optional train-only auxiliary task", "baseline_passthrough.py"),
}


def _impl_of(v):
    return v.get("impl") if isinstance(v, dict) else "(baseline numeric params)"


def _board(chunk):
    """A leaderboard string safe for any chunk (a genome may lack a newer chunk key)."""
    lb = A.leaderboard(10)
    return "\n".join(
        f'  {r["fitness"]:+.4f}  ({r.get("mode","?")})  '
        f'{r["chunks"].get(chunk, {}).get("impl", "·"):22s}  {(r.get("rationale") or "")[:58]}'
        for r in lb) or "  (no scored genomes yet)"


def _meta_digest():
    """Meta-scratchpad (ShinkaEvolve §3.3): the current best genome — the frontier your mutation
    stacks onto. Deterministic, from the archive; reaches every inventor via the brief."""
    champ = A.best()
    if not champ:
        return "CURRENT FRONTIER: (archive empty — you are mutating the R4 baseline)"
    recipe = ", ".join(f"{c}={_impl_of(v)}" for c, v in champ["chunks"].items())
    return ("CURRENT FRONTIER — the best genome so far; a one-chunk mutation replaces ONE piece and "
            "keeps the rest:\n"
            f"  {champ['id']} (held-out margin {champ['fitness']:+.4f}): {recipe}")


def _inspirations(chunk, k=2):
    """Inspiration exemplars (FunSearch/ShinkaEvolve): the CODE of the top-scoring impls for THIS
    chunk, so inventors build on / beat the current winners rather than only their names. Returns ''
    when the chunk isn't carried in genome dicts (e.g. perception, scored via a data root)."""
    seen, blocks = set(), []
    for r in A.leaderboard(20):
        v = r["chunks"].get(chunk)
        name = v.get("impl") if isinstance(v, dict) else None
        if not name or name in seen:
            continue
        f = ROOT / "chunks" / chunk / f"{name}.py"
        if not f.exists():
            continue
        seen.add(name)
        blocks.append(f"# ==== {name}  (held-out margin {r['fitness']:+.4f}) ====\n{f.read_text().rstrip()}")
        if len(blocks) >= k:
            break
    if not blocks:
        return ""
    sep = "-" * 80
    return (f"CURRENT TOP {chunk.upper()} IMPLS — study these; your job is to BEAT them (build on, "
            f"recombine, or invent something better). Do NOT just re-derive them:\n"
            f"{sep}\n" + "\n\n".join(blocks) + f"\n{sep}")


def _context(chunk):
    """The shared 'what's winning' block every brief injects: what's registered, the leaderboard,
    the frontier (meta-scratchpad), and the top impls' code (inspiration exemplars)."""
    return (f"ALREADY REGISTERED for this chunk (do NOT resubmit): {G.list_impls(chunk)}\n\n"
            f"LEADERBOARD (top genomes by held-out content-verb margin):\n{_board(chunk)}\n\n"
            f"{_meta_digest()}\n\n{_inspirations(chunk)}")


def objective_brief():
    contract = (ROOT / "chunks" / "objective" / "mse.py").read_text()
    ctx = _context("objective")
    return f"""You are an INVENTOR in an evolutionary search over the training OBJECTIVE (loss
function) of a JEPA-style shell world model. Your job: invent ONE genuinely NEW objective that
might beat what is already in the archive. Favor a real, different idea over a safe tweak — the
point of the search is DIVERSITY, so we can discover combinations a single designer would miss.

WHAT THE MODEL DOES: a causal transformer reads an interleaved (command, observation) history of
FROZEN ModernBERT embeddings and, at each command position, predicts the NEXT observation's
embedding. Targets are standardized (per-dim mean 0, std 1) 768-d vectors. Perception is frozen;
only this predictor learns. The downstream metric that fitness is computed from is RETRIEVAL:
given the prediction, rank the true next observation against same-verb foils by **squared
(L2) distance**; we report top-1. Fitness is the held-out **content-verb (ls+cat) margin** =
top1(world model) − max(top1 of lexical retrieve-by-command, a history-free MLP, copy-previous).

THE CONTRACT — your output is a complete Python module exposing `loss(pred, tgt) -> scalar`:
  pred: [n, 768] predicted next-obs embeddings at command positions (n = all cmd steps in the batch)
  tgt : [n, 768] the true standardized next-obs embeddings
  return: a scalar torch tensor with grad. `import torch` / `torch.nn.functional as F` as needed.
Because pred and tgt are the whole batch's cmd-position tensors, you MAY form batch-level
objectives (e.g. InfoNCE/contrastive over the n examples, variance/covariance regularizers).
HARD RULES: pure function of (pred, tgt) only; no file/network/global state; no in-place edits of
inputs; must be anti-collapse-safe (a constant prediction must NOT minimize it); keep it fast.
It will be hard-filtered: NaN/inf loss, or a model whose command-position prediction leaks the
answer, scores −inf.

THE BASELINE CONTRACT MODULE (mse.py), for the exact interface:
--------------------------------------------------------------------------------
{contract}--------------------------------------------------------------------------------

{ctx}

IDEA SPACE (non-exhaustive — invent beyond it): InfoNCE / contrastive with in-batch negatives
and a temperature (aligns the space to the retrieval task directly); a margin/triplet ranking
loss that mirrors the L2-distance retrieval metric; Huber/log-cosh for robustness to outlier
dims; a VICReg/SIGReg-style anti-collapse term added to MSE; normalizing predictions to the unit
sphere then MSE; hard-negative weighting; predicting a residual from the previous observation.
{STANDING_RULES}
OUTPUT FORMAT: output ONLY the complete Python module — inside a single ```python code fence,
nothing else. Start it with a short module docstring stating your idea and WHY it might raise the
retrieval margin, then `NAME = "<short_snake_case_unique_name>"`, `DESCRIPTION = "..."`, then the
`loss` function. No prose outside the code fence."""


def arch_brief():
    contract = (ROOT / "chunks" / "arch" / "baseline_transformer.py").read_text()
    baseline = (pathlib.Path(__file__).resolve().parent.parent / "realenv" / "seq_worldmodel.py").read_text()
    # pull just the SeqWorldModel class for reference
    start = baseline.find("class SeqWorldModel")
    end = baseline.find("\n\ndef collate", start)
    swm = baseline[start:end] if start >= 0 and end > start else "(see realenv/seq_worldmodel.py SeqWorldModel)"
    ctx = _context("arch")
    return f"""You are an INVENTOR in an evolutionary search over the ARCHITECTURE of a JEPA-style
shell world model. Invent ONE new predictor architecture that might beat the baseline causal
transformer. Favor a real structural idea over a hyperparameter tweak.

WHAT THE MODEL DOES: it reads an interleaved token stream cmd_0, obs_0, cmd_1, obs_1, … where each
token is a FROZEN 768-d ModernBERT embedding (commands at even positions, observations at odd),
tagged by type (0=cmd, 1=obs) and position, and at each COMMAND position predicts the next
observation's 768-d embedding. It MUST be CAUSAL: a command position's prediction may depend only
on the history up to and including that command — never its own observation (odd index just after
it) or anything later. Predictions are scored by retrieval (rank the true next-obs vs same-verb
foils by squared L2).

THE CONTRACT — your output is a complete Python module exposing `build(**params) -> nn.Module`:
  the module's forward(tok_emb [B,L,768], types [B,L] in {{0,1}}, key_pad [B,L] bool True=pad)
  returns (pred [B,L,768], h [B,L,dh]) — a prediction AND a hidden state at EVERY position; the
  harness reads command positions as pred[:, 0::2]. Map the frozen 768-d input in and a 768-d
  target out. `import torch` / `torch.nn as nn`. Self-contained; no file/network/global state.
HARD RULES: strictly CAUSAL (a per-genome no-leakage guard perturbs obs_t and REJECTS the genome
if any command-position prediction at or before t changes — score −inf); handle padding via
key_pad; keep params/compute modest (the baseline is d≈192, 4 layers). Broken/NaN/leaking → −inf.

THE BASELINE ARCH (baseline_transformer.py wraps this SeqWorldModel — study its interface):
--------------------------------------------------------------------------------
{contract}
--- SeqWorldModel (the reference implementation) ---
{swm}
--------------------------------------------------------------------------------

{ctx}

IDEA SPACE (invent beyond it): rotary/ALiBi positions instead of learned; a GLU/SwiGLU FFN; a
gated or highway residual; a two-stream design that routes cmd vs obs tokens differently; an
explicit learned "system-identity" summary token that aggregates the early uname/config tokens and
is broadcast to later positions; a causal SSM/Mamba-style state mixer instead of attention; deeper-
but-narrower; a retrieval/memory over the history; hierarchical (per-step then cross-step). Keep it
causal and the I/O contract exact.
{STANDING_RULES}
OUTPUT FORMAT: output ONLY the complete Python module inside a single ```python code fence — a
short docstring stating the architectural idea + why it may help, then NAME = "<snake_case_unique>",
DESCRIPTION, then the nn.Module class(es) and `build(**params)`. No prose outside the fence."""


def target_brief():
    contract = (ROOT / "chunks" / "target" / "identity.py").read_text()
    delta = (ROOT / "chunks" / "target" / "delta_prev.py").read_text()
    ctx = _context("target")
    return f"""You are an INVENTOR in an evolutionary search over the TARGET REPRESENTATION of a
JEPA-style shell world model — WHAT the model is trained to predict at each command position.
Invent ONE new target transform that might make the next-observation easier to predict and thus
raise held-out retrieval.

WHAT THE MODEL DOES: a causal transformer reads an interleaved (command, observation) history of
FROZEN 768-d ModernBERT embeddings and predicts the next observation's embedding. Currently it
predicts the absolute next-obs embedding z_obs (identity target) under an L2-InfoNCE objective (the
winning objective). The EVAL always ranks the true z_obs against same-verb foils by squared L2 on
the RECONSTRUCTED prediction, so your transform must be INVERTIBLE: a perfect prediction of your
target must reconstruct z_obs (near-)exactly.

THE CONTRACT — your output is a Python module exposing two PURE functions:
  make_target(z_obs, z_prev) -> [n,768]   # what the model trains to predict (objective compares
                                          #   the model's prediction to THIS)
  to_obs(pred, z_prev) -> [n,768]         # reconstruct the predicted next-obs for the retrieval eval
  z_obs = true next-obs embedding [n,768]; z_prev = the PREVIOUS observation embedding [n,768]
  (causally available; zeros at the first step). Require to_obs(make_target(z_obs,z_prev),z_prev) ≈
  z_obs. `import torch`. Pure functions of the args only — NO learned params, NO train stats, NO
  file/state (make_target sees only z_obs,z_prev; to_obs only pred,z_prev).

THE R4 DEFAULT (identity.py) and the delta baseline (delta_prev.py), for the exact interface:
--------------------------------------------------------------------------------
{contract}--- delta_prev.py ---
{delta}--------------------------------------------------------------------------------

{ctx}

The space is constrained to invertible functions of (z_obs, z_prev). IDEA SPACE: a PARTIAL residual
z_obs - alpha*z_prev (alpha in (0,1), reconstruct z_prev*alpha + pred) — interpolates identity and
delta and may hit a better bias/variance point; a per-dimension fixed reweighting that is inverted
in to_obs; predicting the delta scaled by a constant (z_obs - z_prev)/c to normalize its magnitude;
a "double residual" or momentum form; anything invertible that reduces the target's variance or
better matches the L2 retrieval geometry. Commit to the one you think most likely to help.
{STANDING_RULES}
OUTPUT FORMAT: output ONLY the complete Python module inside a single ```python code fence — a short
docstring (idea + why it may lower target variance / raise retrieval), then NAME = "<snake_case>",
DESCRIPTION, then make_target and to_obs. No prose outside the fence."""


def chunk_brief(chunk):
    """Generic brief for a chunk without a hand-written one above: the contract is the baseline
    impl's docstring (authoritative), plus the leaderboard, what's been tried, and STANDING_RULES.
    Reaches Codex and Claude inventors identically."""
    if chunk not in GENERIC:
        raise SystemExit(f"no brief for chunk {chunk!r}; bespoke: objective/arch/target; generic: {list(GENERIC)}")
    what, baseline_file = GENERIC[chunk]
    contract = (ROOT / "chunks" / chunk / baseline_file).read_text()
    ctx = _context(chunk)
    return f"""You are an INVENTOR in an evolutionary search over {what} of a JEPA-style shell world
model. Invent ONE genuinely NEW implementation of this chunk that might beat what is in the archive.

WHAT THE MODEL DOES: a causal transformer reads an interleaved (command, observation) history of
FROZEN text-encoder embeddings and, at each command position, predicts the next observation's
embedding. Fitness is the held-out **content-verb (ls+cat) margin** = top1(world model) − max(top1
of lexical retrieve-by-command, a history-free MLP, copy-previous), scored by next-obs RETRIEVAL
(rank the true next-obs vs same-verb foils by squared L2).

THE CONTRACT is defined by the reference baseline below — its **docstring documents the exact
interface** your module must expose (the functions + signatures). Read it, match it exactly, and
keep your module self-contained (no file/network/global state unless the baseline itself uses it):
--------------------------------------------------------------------------------
{contract}--------------------------------------------------------------------------------

{ctx}
{STANDING_RULES}
OUTPUT FORMAT: output ONLY the complete Python module inside a single ```python code fence — a short
docstring (idea + why it may raise the margin), then NAME = "<snake_case_unique>", DESCRIPTION, then
the implementation matching the baseline's contract exactly. No prose outside the fence."""


def extract_code(text):
    """Pull the python module out of an inventor reply: the last ```python fence, else the last
    generic fence, else the raw text. Returns the code string."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (fences[-1] if fences else text).strip()


if __name__ == "__main__":
    chunk = sys.argv[1] if len(sys.argv) > 1 else "objective"
    bespoke = {"objective": objective_brief, "arch": arch_brief, "target": target_brief}
    print(bespoke[chunk]() if chunk in bespoke else chunk_brief(chunk))
