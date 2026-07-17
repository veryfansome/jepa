"""Genome = a JSON-able dict selecting one implementation per chunk (+ params). The registry
resolves a chunk-impl name to code. v0 evolves the `objective` chunk (+ arch/optim numeric
params); other chunks default to the R4 baseline."""

import importlib
import pathlib

CHUNKS_DIR = pathlib.Path(__file__).resolve().parent / "chunks"


# ---- baseline genome (the R4 world model) ----------------------------------------------

def baseline_genome():
    return {
        "id": "gen0-baseline",
        "parent": None,
        "generation": 0,
        "inventor": "seed",
        "chunk_changed": None,
        "rationale": "R4 baseline: causal transformer over cmd/obs frozen embeddings, MSE loss.",
        "chunks": {
            "objective": {"impl": "mse", "params": {}},
            "arch": {"d": 192, "layers": 4, "heads": 4, "dropout": 0.1},
            "optim": {"lr": 3e-4, "wd": 1e-4, "steps": 4000, "bs": 64},
        },
    }


# ---- registry --------------------------------------------------------------------------

def load_objective(genome):
    """Import the objective impl module named by the genome and return its `loss` callable.
    Raises a clear error if the impl is missing or malformed."""
    name = genome["chunks"]["objective"]["impl"]
    mod = importlib.import_module(f"evolve.chunks.objective.{name}")
    if not hasattr(mod, "loss"):
        raise AttributeError(f"objective impl '{name}' has no loss(pred, tgt) function")
    return mod.loss


def load_optim(genome):
    """Return (make_fn, bs). optim = {"impl": name, "bs": B} uses the optim registry (optimizer +
    LR schedule); legacy {"lr","wd","steps","bs"} maps to constant AdamW with those values. make_fn
    (params, steps) -> (optimizer, scheduler_or_None)."""
    o = genome["chunks"]["optim"]
    bs = o.get("bs", 64)
    if "impl" in o:
        mod = importlib.import_module(f"evolve.chunks.optim.{o['impl']}")
        if not hasattr(mod, "make"):
            raise AttributeError(f"optim impl '{o['impl']}' has no make(params, steps)")
        p = dict(o.get("params", {}))
        return (lambda params, steps: mod.make(params, steps, **p)), bs
    lr, wd = o.get("lr", 3e-4), o.get("wd", 1e-4)
    import torch
    return (lambda params, steps: (torch.optim.AdamW(params, lr=lr, weight_decay=wd), None)), bs


def load_target(genome):
    """Return the target-chunk module (make_target/to_obs). Defaults to identity (the R4 target)
    when a genome has no target chunk, so existing genomes are unchanged."""
    t = genome["chunks"].get("target", {"impl": "identity"})
    mod = importlib.import_module(f"evolve.chunks.target.{t['impl']}")
    for fn in ("make_target", "to_obs"):
        if not hasattr(mod, fn):
            raise AttributeError(f"target impl '{t['impl']}' has no {fn}")
    return mod


def load_arch(genome):
    """Return (build_fn, params) for the arch chunk. arch = {"impl": name, "params": {...}} uses
    the arch registry (a swappable model module); legacy {"d","layers","heads","dropout"} maps to
    the baseline transformer. build_fn(**params) -> nn.Module with SeqWorldModel's I/O contract."""
    a = genome["chunks"]["arch"]
    if "impl" in a:
        mod = importlib.import_module(f"evolve.chunks.arch.{a['impl']}")
        if not hasattr(mod, "build"):
            raise AttributeError(f"arch impl '{a['impl']}' has no build(**params) function")
        return mod.build, dict(a.get("params", {}))
    mod = importlib.import_module("evolve.chunks.arch.baseline_transformer")
    return mod.build, {k: a[k] for k in ("d", "layers", "heads", "dropout") if k in a}


def list_impls(chunk="objective"):
    d = CHUNKS_DIR / chunk
    return sorted(p.stem for p in d.glob("*.py") if p.stem != "__init__")


def validate(genome):
    """Cheap structural check before spending a training run on a genome."""
    c = genome.get("chunks", {})
    for k in ("objective", "arch", "optim"):
        if k not in c:
            raise ValueError(f"genome missing chunk '{k}'")
    if c["objective"]["impl"] not in list_impls("objective"):
        raise ValueError(f"unknown objective impl '{c['objective']['impl']}' "
                         f"(have {list_impls('objective')})")
    a = c["arch"]
    if "impl" in a:
        if a["impl"] not in list_impls("arch"):
            raise ValueError(f"unknown arch impl '{a['impl']}' (have {list_impls('arch')})")
    elif a["d"] % a["heads"] != 0:  # legacy numeric baseline transformer
        raise ValueError(f"arch d={a['d']} not divisible by heads={a['heads']}")
    return True
