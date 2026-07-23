"""Benchmark-version registry (bench-constitution §4/§5): the FROZEN per-version verb-class
tables and baseline sets. The prereg docs are authoritative; this module is their executable
mirror, and the harness asserts a v2 root's recorded classes match before scoring.

v1 semantics are bit-identical to the historical harness (content ls+cat, no ok-masking,
no within-traj baseline) — every archived margin must reproduce exactly.
"""

import hashlib
import json
import pathlib

# The frozen v3 class table (prereg §1.1 "Classes-file authority"): sha-pinned; resolve()
# loads it and asserts a v3 root's recorded classes_sha matches verbatim (fail-closed). ONE
# global file; the harness consumes the set of content CELLS, not verbs.
CLASSES_JSON = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "dockerfs3-classes.json"

# v3 fitness-role map (prereg §7 / design §8.5, §9.5): the six measured classes + under-floor.
# ONLY "content" enters fitness; everything else is excluded from the content pool.
V3_CONTENT_CLASSES = ("content",)
V3_EXCLUDED_CLASSES = ("semi-echo", "ack", "echo/const", "sim", "noisy-excluded", "under-floor")

# The seven baseline arms (prereg §4.4 / design §10.3), frozen. The first four are the ratified
# v2-continuity arms; the last three are the v3 SST/wtm additions (precompute_baselines.py).
V3_ARMS = ("retrieve_by_cmd", "no_history", "copy_prev", "within_traj",
           "within_traj_mut", "sst", "sst_composite")

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
    # dockerfs3-prereg.md §1.1 / design §9.5, §8.1, §4.4. STATIC parts only — the `content`
    # SET (content CELLS, not verbs) is loaded per-root from CLASSES_JSON by resolve() and
    # asserted against the root's classes_sha. `ok_masked_verbs` is EMPTY: v3 does per-STEP
    # cell rewrite + the Finding-1 axis-2' echo purge (§4.6) in _data_tensors, not verb-mode
    # masking. Cell pseudo-verb strings ("sig|mode|scope[|created-obs]") are ATOMIC keys.
    "dockerfs3-v3.0": {
        "content": None,                       # DYNAMIC — resolve() fills the content-cell set
        "ok_masked_verbs": (),                 # cell rewrite + echo purge instead (§4.6/§9.5)
        "within_traj_in_max": True,
        "cell_based": True,                    # _data_tensors branches on this
        "arms": V3_ARMS,                       # the seven-arm baseline max (§4.4)
        "axis2p_purge_thresh": 0.96,           # Finding-1 per-step echo purge (§4.6, pinned)
        "content_classes": V3_CONTENT_CLASSES,
        "excluded_classes": V3_EXCLUDED_CLASSES,
        "classes_file": str(CLASSES_JSON),
    },
}


def _classes_bytes_and_sha():
    """(raw bytes, sha256 hex) of the frozen class table file. The sha is over the file bytes
    verbatim (the prereg pins THIS file); the mint stamps the same sha into summary.json."""
    b = CLASSES_JSON.read_bytes()
    return b, hashlib.sha256(b).hexdigest()


def _content_cells(classes_js, content_classes):
    """The set of content CELLS the harness pools/scores (design §9.5): every classes.json row
    whose measured `class` is in content_classes. Cell keys are the ATOMIC 'cell' strings."""
    cells = set()
    for row in classes_js.get("rows", []):
        if row.get("class") in content_classes:
            cells.add(row["cell"])
    return frozenset(cells)


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
                raise ValueError(f"{data_root}: {split}.jsonl carries v2 step meta but summary.json "
                                 f"is missing — refusing the silent v1 fallback (constitution §4). "
                                 f"Re-encode with the summary-copying reencode/mv_encode.")
    if ver not in VERSIONS:
        raise ValueError(f"unknown bench_version '{ver}' in {s} — register it in bench_versions.py "
                         f"(a new version requires its own ratified prereg)")
    spec = VERSIONS[ver]
    if str(ver).startswith("dockerfs3"):
        return _resolve_v3(data_root, spec)
    if recorded is None and ver != "v1":
        raise ValueError(f"{data_root}: bench_version {ver} but summary.json records no "
                         f"verb_classes — malformed v2 root (constitution §4)")
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


def _resolve_v3(data_root, spec):
    """v3 spec for a dockerfs3 root (prereg §1.1 / design §9.5). Loads the FROZEN class table
    (CLASSES_JSON), builds the content-CELL set (fitness-role map: only `content` cells enter
    fitness), and returns the resolved spec {content: <content-cell frozenset>, cell_based: True,
    within_traj_in_max: True, arms: <7-arm list>, axis2p_purge_thresh, classes_sha, ...}.

    FAIL-CLOSED (the mint-vs-scoring class-table match, §1.1): a v3 root whose summary.json
    records no classes_sha, or whose classes_sha != the sha256 of CLASSES_JSON, RAISES — a v3
    root scored against a class table different from the one it was minted under is impossible."""
    root = pathlib.Path(data_root)
    summ = root / "summary.json"
    if not summ.exists():
        raise ValueError(f"{data_root}: v3-policy root with no summary.json (fail-closed, §1.1)")
    js = json.loads(summ.read_text())
    recorded_sha = js.get("classes_sha")
    if not recorded_sha:
        raise ValueError(f"{data_root}: bench_version {js.get('bench_version')} but summary.json "
                         f"records no classes_sha — a v3 root MUST pin the class table it was "
                         f"minted under (fail-closed, §1.1)")
    _, cur_sha = _classes_bytes_and_sha()
    if recorded_sha != cur_sha:
        raise ValueError(f"class-table sha mismatch for {data_root}: recorded classes_sha "
                         f"{recorded_sha} != {CLASSES_JSON.name} sha {cur_sha} — the mint's class "
                         f"table and the scoring class table must match verbatim (fail-closed, §1.1)")
    classes_js = json.loads(CLASSES_JSON.read_text())
    content = _content_cells(classes_js, spec["content_classes"])
    if not content:
        raise ValueError(f"{CLASSES_JSON.name}: no rows classified as content "
                         f"{spec['content_classes']} — refusing an empty content pool (§9.5)")
    return dict(spec, version=js.get("bench_version"), content=content,
                classes_sha=cur_sha)


# ---------------------------------------------------------------- v3-policy scoring-side infra
# The full dockerfs3-v3.0 VERSIONS entry (the (sig, mode, state_scope) cell table + fitness-role
# map) is gated on `dockerfs3-classes.json` and lands with it. The helpers below are the
# classes-file-INDEPENDENT scoring-side infra (dockerfs3-prereg §7): they let the reencode/
# mv_encode stampers, the harness cached-encode gate, and load_perception_for_root fail-closed on
# a v3 root NOW, keyed only on the root's own declaration. They never touch v1/v2 roots.

def is_v3_policy(data_root):
    """True iff the root's summary.json declares a dockerfs3 (v3) bench policy. Lightweight
    detection (no classes.json load); False for a missing/unparseable summary, for v1 (no
    summary), and for dockerfs2-v2.0. Once the full v3 VERSIONS entry lands, `resolve()` is the
    authority; this stays the cheap predicate the fail-closed gates branch on."""
    s = pathlib.Path(data_root) / "summary.json"
    if not s.exists():
        return False
    try:
        js = json.loads(s.read_text())
    except Exception:
        return False
    return str(js.get("bench_version", "")).startswith("dockerfs3")


def classes_sha(data_root):
    """The v3 classes-file sha recorded on a root (cache_meta.json, then summary.json); None for
    non-v3/unstamped roots. Consumed only by the v3 base_cache key (§13.2)."""
    root = pathlib.Path(data_root)
    for name in ("cache_meta.json", "summary.json"):
        p = root / name
        if p.exists():
            try:
                v = json.loads(p.read_text()).get("classes_sha")
            except Exception:
                v = None
            if v:
                return v
    return None


def require_v3_cache(data_root):
    """Fail-closed staleness gate for a v3-policy root (§13.2): the root-level cache_meta.json must
    exist and carry {cache_format: 3, bench_version, policy_sha, classes_sha} consistent with the
    root's summary.json, AND summary.json must carry the perception stamp {perception:{impl,model,
    content_sha}}. Any absence/mismatch RAISES — a v3 root scored against a v2-era or partial cache
    is impossible by construction. NEVER call this on a v1/v2 root (they pass straight through)."""
    root = pathlib.Path(data_root)
    summ = root / "summary.json"
    if not summ.exists():
        raise ValueError(f"{data_root}: v3-policy root with no summary.json (fail-closed, §13.2)")
    js = json.loads(summ.read_text())
    cm_path = root / "cache_meta.json"
    if not cm_path.exists():
        raise ValueError(f"{data_root}: v3-policy root missing cache_meta.json — refusing to load a "
                         f"stamp-less v3 cache (fail-closed, §13.2)")
    cm = json.loads(cm_path.read_text())
    if cm.get("cache_format") != 3:
        raise ValueError(f"{data_root}: cache_meta.json cache_format={cm.get('cache_format')!r} != 3 "
                         f"(fail-closed, §13.1)")
    for fld in ("bench_version", "policy_sha", "classes_sha"):
        cv, jv = cm.get(fld), js.get(fld)
        # B1: reject FALSY stamps, not only mismatched — a pre-B1 root (or a hand-edited cache)
        # carrying no policy_sha/classes_sha would otherwise pass this guard vacuously (None==None),
        # exactly the gap that let an unstamped v3 root reach scoring. A v3 root MUST pin all three.
        if not cv or not jv:
            raise ValueError(f"{data_root}: v3 {fld} is empty (cache_meta={cv!r}, summary={jv!r}) "
                             f"— a v3 root MUST carry a non-empty {fld} (fail-closed, §13.2)")
        if cv != jv:
            raise ValueError(f"{data_root}: cache_meta.json {fld}={cv!r} != summary.json "
                             f"{jv!r} — stale/mismatched v3 cache (fail-closed, §13.2)")
    if not ((js.get("perception") or {}).get("content_sha")):
        raise ValueError(f"{data_root}: v3-policy root lacking the perception stamp "
                         f"{{perception:{{impl,model,content_sha}}}} (fail-closed, §10.3/§13.1)")
    return cm
