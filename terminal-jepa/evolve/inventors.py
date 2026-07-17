"""Build the inventor (mutation-operator) brief for a chunk, and extract code from a reply.
Both Claude subagents and Codex (`codex exec`) get the SAME brief, so mixing them is genuine
model/harness diversity (the ShinkaEvolve 'ensemble of frontier LLMs' idea) rather than an
apples-to-oranges comparison.

  python -m evolve.inventors objective          # print the objective-chunk brief to stdout
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evolve import archive as A
from evolve import genome as G

ROOT = pathlib.Path(__file__).resolve().parent


def objective_brief():
    contract = (ROOT / "chunks" / "objective" / "mse.py").read_text()
    tried = G.list_impls("objective")
    lb = A.leaderboard(10)
    board = "\n".join(
        f'  {r["fitness"]:+.4f}  ({r.get("mode","?")})  {r["chunks"]["objective"]["impl"]:16s}  '
        f'{(r.get("rationale") or "")[:70]}' for r in lb) or "  (no scored genomes yet)"
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

ALREADY IN THE ARCHIVE (do NOT just resubmit these): {tried}
LEADERBOARD (fitness = held-out content-verb margin; higher is better):
{board}

IDEA SPACE (non-exhaustive — invent beyond it): InfoNCE / contrastive with in-batch negatives
and a temperature (aligns the space to the retrieval task directly); a margin/triplet ranking
loss that mirrors the L2-distance retrieval metric; Huber/log-cosh for robustness to outlier
dims; a VICReg/SIGReg-style anti-collapse term added to MSE; normalizing predictions to the unit
sphere then MSE; hard-negative weighting; predicting a residual from the previous observation.

OUTPUT FORMAT: output ONLY the complete Python module — inside a single ```python code fence,
nothing else. Start it with a short module docstring stating your idea and WHY it might raise the
retrieval margin, then `NAME = "<short_snake_case_unique_name>"`, `DESCRIPTION = "..."`, then the
`loss` function. No prose outside the code fence."""


def extract_code(text):
    """Pull the python module out of an inventor reply: the last ```python fence, else the last
    generic fence, else the raw text. Returns the code string."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (fences[-1] if fences else text).strip()


if __name__ == "__main__":
    chunk = sys.argv[1] if len(sys.argv) > 1 else "objective"
    print({"objective": objective_brief}[chunk]())
