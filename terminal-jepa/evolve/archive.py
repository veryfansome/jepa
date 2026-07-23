"""The program archive: append-only archive/genomes.jsonl of every scored genome, plus a
leaderboard and ShinkaEvolve-style fitness+novelty weighted parent sampling. Durable *insights*
(what works / what collapses) live in auto-memory, not here — this is the raw record."""

import json
import math
import pathlib
import random

ARCHIVE = pathlib.Path(__file__).resolve().parent / "archive" / "genomes.jsonl"
NEG = float("-inf")

# Margins are only comparable within one bench version (the fitness classes and baseline max
# changed at the dockerfs2-v2.0 mint), so selection surfaces default to the active bench.
# Records are assigned by their data root; pre-field records are all v1-era.
# ACTIVE_BENCH flips to "v3" only at the dockerfs3 re-baseline commit (prereg §6 step 14).
ACTIVE_BENCH = "v2"


def _bench_of(rec):
    root = str(rec.get("data", ""))
    if "dockerfs3" in root:
        return "v3"
    return "v2" if "dockerfs2" in root else "v1"


def append(record):
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load():
    if not ARCHIVE.exists():
        return []
    return [json.loads(l) for l in open(ARCHIVE) if l.strip()]


def _valid(recs, bench=None):
    """Scored genomes that count as FITNESS: exclude final-test records — the final-test split is
    validation-only and must never drive parent selection / the leaderboard (else the optimization
    leaks into the held-out-of-held-out set). bench='v1'/'v2' keeps one bench version's records
    (margins are not comparable across versions); None keeps all."""
    return [r for r in recs if isinstance(r.get("fitness"), (int, float)) and r["fitness"] > NEG
            and r.get("split", "inner") != "final"
            and (bench is None or _bench_of(r) == bench)]


def _best_per_id(recs):
    """Collapse to one record per genome id, preferring full-mode then higher fitness."""
    by_id = {}
    for r in recs:
        key = (r.get("mode") == "full", r["fitness"])
        if r["id"] not in by_id or key > by_id[r["id"]][0]:
            by_id[r["id"]] = (key, r)
    return [v[1] for v in by_id.values()]


def leaderboard(top=10, bench=ACTIVE_BENCH):
    items = _best_per_id(_valid(load(), bench=bench))
    items.sort(key=lambda r: (r.get("mode") == "full", r["fitness"]), reverse=True)
    return items[:top]


def best(bench=ACTIVE_BENCH):
    lb = leaderboard(1, bench=bench)
    return lb[0] if lb else None


def sample_parent(seed=0, lam=10.0, bench=ACTIVE_BENCH):
    """p_i ∝ σ(λ·(F_i − median F)) · 1/(1 + offspring_i) — performance × novelty (ShinkaEvolve
    §4.1). Returns a genome-shaped dict (id/parent/generation/chunks/...) or None."""
    all_recs = load()
    items = _best_per_id(_valid(all_recs, bench=bench))
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
