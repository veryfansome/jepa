"""Scoring-side guards for the dockerfs3 (v3.0) pre-mint fixes B1 + S2 (GO-gate 2026-07-23).

B1: evolve.bench_versions.require_v3_cache rejects FALSY (not only mismatched) classes_sha/
policy_sha/bench_version stamps — the guard fires per its advertised contract instead of passing
vacuously (None == None) on a pre-B1 unstamped root.

S2: evolve.harness._demote_lowcov + its wiring in _v3_cell_verbs implement the per-split G-COV
coverage-demotion — a content cell with < _V3_GCOV_FLOOR surviving content steps in the CURRENT
eval split is rewritten to the excluded "<cell>-lowcov" pseudo-verb (out of the pooled fitness
margin, into the report battery). The frozen classes.json is never touched.

Docker-free and encoder-free: fabricated cell strings for the helper; FakeWorld-minted real
recorded steps + zero embeddings for the wired _data_tensors path.
"""

import json
import pathlib
import random
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import realenv.collect_docker as C
from realenv import seq_worldmodel as M
from evolve import bench_versions as BV
from evolve import harness as H
from tests.fakeworld_v3 import FakeWorld

IMAGE = "alpine:latest"
TMPL = C.sst_error_templates(IMAGE)


def _fake_collect_image_v3(image, n_seqs, seq_len, seed, ref=None, arm="full"):
    steps = [{"cmd": "pwd", "output": "/", "exit": 0, "cwd": "/",
              "meta": {"verb": "pwd", "sig": "pwd", "mode": "hit", "arm": "pwd",
                       "state_scope": "native"}}]
    seqs = [{"image": image, "system_id": "fake", "arm": arm, "seq_idx": 0,
             "ws_manifest_sha256": "0" * 64, "steps": steps}]
    return image, seqs, "fake", {"steps": 1}, [{"seq_idx": 0, "step": 0, "dur_ms": 5}]


def _mint_v3_root():
    """A tiny B1-stamped v3 root (fake collect_image, no docker)."""
    out = pathlib.Path(tempfile.mkdtemp())
    orig = C.collect_image_v3
    C.collect_image_v3 = _fake_collect_image_v3
    try:
        summary = C.collect(str(out), ["img-a", "img-b"], [], 4, 28, 0, 2, policy="v3", arm="full")
    finally:
        C.collect_image_v3 = orig
    return out, summary


# --------------------------------------------------------------- B1: require_v3_cache falsy reject

def test_require_v3_cache_rejects_falsy_stamps():
    out, summ = _mint_v3_root()
    js = json.loads((out / "summary.json").read_text())
    js["perception"] = {"impl": "enc_e5_base", "model": "m", "content_sha": "deadbeef"}
    (out / "summary.json").write_text(json.dumps(js))
    cm = {"cache_format": 3, "bench_version": js["bench_version"],
          "policy_sha": js["policy_sha"], "classes_sha": js["classes_sha"],
          "built_summary_sha": "x"}
    (out / "cache_meta.json").write_text(json.dumps(cm))
    BV.require_v3_cache(str(out))            # consistent, non-empty stamps -> passes

    for fld in ("policy_sha", "classes_sha", "bench_version"):
        jbad = dict(js); jbad[fld] = ""
        cbad = dict(cm); cbad[fld] = ""      # both falsy: pre-B1 this passed (== None/"")
        (out / "summary.json").write_text(json.dumps(jbad))
        (out / "cache_meta.json").write_text(json.dumps(cbad))
        try:
            BV.require_v3_cache(str(out))
        except ValueError as e:
            assert "empty" in str(e) and fld in str(e), (fld, str(e))
        else:
            raise AssertionError(f"require_v3_cache accepted a FALSY {fld}")
    # restore consistent stamps
    (out / "summary.json").write_text(json.dumps(js))
    (out / "cache_meta.json").write_text(json.dumps(cm))
    BV.require_v3_cache(str(out))


# --------------------------------------------------------------- S2: _demote_lowcov helper

def test_demote_lowcov_helper_fires_on_low_cov_cell():
    content = frozenset({"cat|hit|native", "ls|hit|native"})
    # 600 of cat|hit|native (>= floor, retained) + 100 of ls|hit|native (< floor, demoted)
    verbs = ["cat|hit|native"] * 600 + ["ls|hit|native"] * 100
    new_verbs, lowcov = H._demote_lowcov(verbs, content)
    assert lowcov == {"ls|hit|native": 100}, lowcov
    assert new_verbs.count("cat|hit|native") == 600         # above-floor untouched
    assert new_verbs.count("ls|hit|native") == 0
    assert new_verbs.count("ls|hit|native-lowcov") == 100   # demoted to excluded pseudo-verb
    assert "ls|hit|native-lowcov" not in content            # -> out of the fitness pool


def test_demote_lowcov_boundary_is_exactly_the_floor():
    floor = H._V3_GCOV_FLOOR
    assert floor == 500
    cell = frozenset({"x|hit|native"})
    _, at = H._demote_lowcov(["x|hit|native"] * floor, cell)
    _, below = H._demote_lowcov(["x|hit|native"] * (floor - 1), cell)
    assert at == {} and below == {"x|hit|native": floor - 1}


def test_demote_lowcov_noop_when_all_cells_covered():
    content = frozenset({"a|hit|native"})
    verbs = ["a|hit|native"] * 800 + ["a|hit|native-echo"] * 3  # echo already out of `content`
    new_verbs, lowcov = H._demote_lowcov(verbs, content)
    assert lowcov == {} and new_verbs == verbs               # identity when covered


# --------------------------------------------------------------- S2: wired through _v3_cell_verbs

def _mint_steps(n_sessions=4, length=32):
    """Real recorded v3 steps from FakeWorld sessions (docker-free)."""
    out = []
    for seed in range(n_sessions):
        box = FakeWorld(IMAGE, TMPL)
        dirs, files = box.image_dirs_files()
        rng = random.Random(f"dockerfs:{seed}:{IMAGE}:full:0")
        boot = C.v3_bootstrap(box, seed, files, C._v2_probe(box, dirs, files), rng)
        sess = C.gen_sequence_v3(box, dirs, files, rng, length, IMAGE, TMPL, boot, arm="full")
        out.append([C._step_record(s) for s in sess.steps])
    return out


def test_v3_cell_verbs_demotes_all_content_at_tiny_scale():
    """Wired path: _data_tensors -> _v3_cell_verbs applies the coverage-demotion. On a handful of
    real minted sequences EVERY content cell is far below the 500-step floor, so all content steps
    demote to '<cell>-lowcov', diag['lowcov'] is populated, and NO surviving verb is in the frozen
    content-cell set (the pooled fitness margin is empty at this scale — exactly the report-battery
    routing S2 specifies)."""
    root, _ = _mint_v3_root()
    spec = BV.resolve(str(root))
    assert spec.get("cell_based") and spec["content"]

    sessions = _mint_steps()
    evalset, raw_seqs = [], []
    for steps in sessions:
        n = len(steps)
        evalset.append({"z_obs": torch.zeros(n, M.D), "z_cmd": torch.zeros(n, M.D),
                        "cmds": [s["cmd"] for s in steps], "image": IMAGE,
                        "ok": [s["exit"] == 0 and bool(s["output"]) for s in steps]})
        raw_seqs.append({"image": IMAGE, "steps": steps})

    out = H._data_tensors(evalset, spec, raw_seqs=raw_seqs)
    verbs = out["verbs"]
    diag = out["_v3diag"]

    # at least one content cell was present pre-demotion (otherwise the test is vacuous)
    n_lowcov_steps = sum(1 for v in verbs if v.endswith("-lowcov"))
    assert diag["lowcov"], "no content cells found to demote — FakeWorld emitted no content hits"
    assert n_lowcov_steps == sum(diag["lowcov"].values())
    # every content cell was below the floor -> none survives in the fitness pool
    assert not any(v in spec["content"] for v in verbs), \
        "a content cell survived despite being below the coverage floor"
    # demoted cells are the ex-content cells; the -lowcov key is not in the frozen content set
    for cell, cnt in diag["lowcov"].items():
        assert cell in spec["content"] and cnt < H._V3_GCOV_FLOOR
        assert (cell + "-lowcov") not in spec["content"]


if __name__ == "__main__":
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)}/{len(fns)} v3 scoring-gate tests passed")
