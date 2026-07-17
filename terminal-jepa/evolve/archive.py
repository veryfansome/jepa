"""The program archive: append-only archive/genomes.jsonl of every scored genome, plus a
leaderboard and ShinkaEvolve-style fitness+novelty weighted parent sampling. Durable *insights*
(what works / what collapses) live in auto-memory, not here — this is the raw record."""

import json
import math
import pathlib
import random

ARCHIVE = pathlib.Path(__file__).resolve().parent / "archive" / "genomes.jsonl"
NEG = float("-inf")


def append(record):
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load():
    if not ARCHIVE.exists():
        return []
    return [json.loads(l) for l in open(ARCHIVE) if l.strip()]


def _valid(recs):
    return [r for r in recs if isinstance(r.get("fitness"), (int, float)) and r["fitness"] > NEG]


def _best_per_id(recs):
    """Collapse to one record per genome id, preferring full-mode then higher fitness."""
    by_id = {}
    for r in recs:
        key = (r.get("mode") == "full", r["fitness"])
        if r["id"] not in by_id or key > by_id[r["id"]][0]:
            by_id[r["id"]] = (key, r)
    return [v[1] for v in by_id.values()]


def leaderboard(top=10):
    items = _best_per_id(_valid(load()))
    items.sort(key=lambda r: (r.get("mode") == "full", r["fitness"]), reverse=True)
    return items[:top]


def best():
    lb = leaderboard(1)
    return lb[0] if lb else None


def sample_parent(seed=0, lam=10.0):
    """p_i ∝ σ(λ·(F_i − median F)) · 1/(1 + offspring_i) — performance × novelty (ShinkaEvolve
    §4.1). Returns a genome-shaped dict (id/parent/generation/chunks/...) or None."""
    all_recs = load()
    items = _best_per_id(_valid(all_recs))
    if not items:
        return None
    fits = sorted(r["fitness"] for r in items)
    med = fits[len(fits) // 2]
    offspring = {}
    for r in all_recs:
        p = r.get("parent")
        if p:
            offspring[p] = offspring.get(p, 0) + 1
    weights = [1.0 / (1.0 + math.exp(-lam * (r["fitness"] - med))) / (1.0 + offspring.get(r["id"], 0))
               for r in items]
    return random.Random(f"parent:{seed}").choices(items, weights=weights, k=1)[0]
