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
    # dockerfs2-prereg.md Amendment 2 as amended by Amendment 3 (stat → semi-echo:
    # the %n path-echo channel scores 0.81-0.85 for a zero-parameter echo predictor)
    "dockerfs2-v2.0": {
        "content": ("ls", "cat", "head", "tail", "find", "grep"),
        "ok_masked_verbs": ("grep",),   # grep-MISS (exit!=0 or empty) excluded from fitness
        "within_traj_in_max": True,
    },
}


def resolve(data_root):
    """Version spec for a data root: reads bench_version from summary.json (absent → v1),
    and for v2 roots asserts the recorded verb_classes match the frozen table."""
    root = pathlib.Path(data_root)
    s = root / "summary.json"
    ver = "v1"
    recorded = None
    if s.exists():
        js = json.loads(s.read_text())   # unparseable summary must FAIL, never fall back to v1
        ver = js.get("bench_version", "v1")
        recorded = js.get("verb_classes")
    else:
        # fail-closed sniff (review-B2 blocker): v2 jsonl steps always carry `meta`; a root
        # with meta-bearing data but no summary must never silently score under v1 classes
        for split in ("train", "val"):
            tj = root / f"{split}.jsonl"
            if not tj.exists():
                continue
            with open(tj) as f:
                first = f.readline()
            if '"meta"' in first:
                raise ValueError(f"{data_root}: train.jsonl carries v2 step meta but summary.json "
                                 f"is missing — refusing the silent v1 fallback (constitution §4). "
                                 f"Re-encode with the summary-copying reencode/mv_encode.")
    if ver not in VERSIONS:
        raise ValueError(f"unknown bench_version '{ver}' in {s} — register it in bench_versions.py "
                         f"(a new version requires its own ratified prereg)")
    spec = VERSIONS[ver]
    if recorded is not None and ver != "v1":
        # full-table mirror check (round-3 fix): content AND semi_echo/excluded/mode rule
        ref = {"content": sorted(spec["content"]), "semi_echo": ["stat"],
               "excluded": ["cd", "uname"],
               "grep_mode_rule": "exit!=0 or empty output => miss (excluded)"}
        got_full = {"content": sorted(recorded.get("content", [])),
                    "semi_echo": sorted(recorded.get("semi_echo", [])),
                    "excluded": sorted(recorded.get("excluded", [])),
                    "grep_mode_rule": recorded.get("grep_mode_rule", "")}
        if got_full != ref:
            raise ValueError(f"class-table mismatch for {data_root}: prereg {ref} vs recorded "
                             f"{got_full} — the prereg is authoritative (constitution §4)")
        want = sorted(spec["content"])
        got = sorted(recorded.get("content", []))
        if want != got:
            raise ValueError(f"class-table mismatch for {data_root}: prereg {want} vs recorded {got} "
                             f"— the prereg is authoritative (constitution §4)")
    return dict(spec, version=ver)
