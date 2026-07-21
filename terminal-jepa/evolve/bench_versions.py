"""Benchmark-version registry (bench-constitution §4/§5): the FROZEN per-version verb-class
tables and baseline sets. The prereg docs are authoritative; this module is their executable
mirror, and the harness asserts a v2 root's recorded classes match before scoring.

v1 semantics are bit-identical to the historical harness (content ls+cat, no ok-masking,
no within-traj baseline) — every archived margin must reproduce exactly.
"""

import json
import pathlib

VERSIONS = {
    "v1": {
        "content": ("ls", "cat"),
        "ok_masked_verbs": (),          # no step-level exclusions
        "within_traj_in_max": False,
    },
    # dockerfs2-prereg.md Amendment 2 (frozen 2026-07-21)
    "dockerfs2-v2.0": {
        "content": ("ls", "cat", "head", "tail", "stat", "find", "grep"),
        "ok_masked_verbs": ("grep",),   # grep-MISS (exit!=0 or empty) excluded from fitness
        "within_traj_in_max": True,
    },
}


def resolve(data_root):
    """Version spec for a data root: reads bench_version from summary.json (absent → v1),
    and for v2 roots asserts the recorded verb_classes match the frozen table."""
    s = pathlib.Path(data_root) / "summary.json"
    ver = "v1"
    recorded = None
    if s.exists():
        try:
            js = json.loads(s.read_text())
            ver = js.get("bench_version", "v1")
            recorded = js.get("verb_classes")
        except Exception:
            ver = "v1"
    if ver not in VERSIONS:
        raise ValueError(f"unknown bench_version '{ver}' in {s} — register it in bench_versions.py "
                         f"(a new version requires its own ratified prereg)")
    spec = VERSIONS[ver]
    if recorded is not None and ver != "v1":
        want = sorted(spec["content"])
        got = sorted(recorded.get("content", []))
        if want != got:
            raise ValueError(f"class-table mismatch for {data_root}: prereg {want} vs recorded {got} "
                             f"— the prereg is authoritative (constitution §4)")
    return dict(spec, version=ver)
