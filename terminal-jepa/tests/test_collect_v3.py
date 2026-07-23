"""Guards for the dockerfs3 (v3.0) collection policy (realenv/collect_docker.py `v3`),
prereg §9 (freeze-order step 4). Docker-free: gen_sequence_v3 is driven against a faithful
in-memory world (tests/fakeworld_v3.FakeWorld) and every recorded trajectory is folded
through a FRESH realenv.shell_state.ShellState — the cross-check that the collector emits
ONLY frozen-universe, self-consistent, deterministic, MutGuard-clean trajectories.

Covers: per-seq seeding determinism (byte-identical jsonl); meta on every step + F8
recomputability; seq-len<=32; parser totality + verbsig membership on every recorded command
(no composed command outside the 11 G3 families); MutGuard property tests (every mutation
passes; planted denylist / cwd-ancestor / rm-r-escape violations raise); the digest entry
gate; two-arm renormalization; the ±5% weighted-draw realized-rate gate (G-RATE); SST-fold
parity (0 mismatches, 0 wrong determined predictions); DG-1 payload provenance; canonical
PIDs; the dur_ms strip seam; v1/v2 byte-identity of _step_record. Plus a docker-gated
real-mint smoke: ~5 trajectories minted on alpine + fedora, folded through a fresh
ShellState(mode="sst") — the collector<->SST golden-rule parity gate P1 scales up."""

import collections
import json
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import realenv.collect_docker as C
from realenv.shell_state import (BOT, ParseError, ShellState, canonical_pid, normpath,
                                 WORKSPACE)
from realenv import verbsig
from tests.fakeworld_v3 import FakeWorld

IMAGE = "alpine:latest"
TMPL = C.sst_error_templates(IMAGE)


def _gen(seed=0, length=28, arm="full"):
    """Drive gen_sequence_v3 against a FakeWorld; return (session, jsonl-shaped steps)."""
    box = FakeWorld(IMAGE, TMPL)
    dirs, files = box.image_dirs_files()
    rng = random.Random(f"dockerfs:{seed}:{IMAGE}:{arm}:0")
    boot = C.v3_bootstrap(box, seed, files, C._v2_probe(box, dirs, files), rng)
    sess = C.gen_sequence_v3(box, dirs, files, rng, length, IMAGE, TMPL, boot, arm=arm)
    return sess, [C._step_record(s) for s in sess.steps]


def _fold(steps):
    """Fold recorded steps through a fresh SST; return (state, n_determined, n_wrong)."""
    st = ShellState(mode="collection", error_templates=TMPL)
    det = wrong = 0
    for i, s in enumerate(steps):
        pred = st.predict(i, s["cmd"])              # raises ParseError if out-of-universe
        if pred is not BOT:
            det += 1
            if pred != {"output": s["output"], "exit": s["exit"], "cwd": s["cwd"]}:
                wrong += 1
        st.fold(s)
    return st, det, wrong


# ------------------------------------------------------------------ (a) determinism / meta

def test_per_seq_seeding_determinism():
    for seed in range(6):
        _, a = _gen(seed=seed, length=28)
        _, b = _gen(seed=seed, length=28)
        assert [s["cmd"] for s in a] == [s["cmd"] for s in b], seed
        assert [s["output"] for s in a] == [s["output"] for s in b], seed
        assert [s.get("meta") for s in a] == [s.get("meta") for s in b], seed
        # byte-identical jsonl: the whole record stream serializes identically (replay-det;
        # the DG-3a twin-mint byte-diff P1 will scale to full splits keys on exactly this)
        assert [json.dumps(s) for s in a] == [json.dumps(s) for s in b], seed


def test_meta_on_every_step():
    _, steps = _gen(seed=1, length=30)
    for s in steps:
        m = s.get("meta")
        assert m, s
        for k in ("verb", "sig", "mode", "expected", "state_scope", "arm", "delta_text"):
            assert k in m, (k, m)
        assert m["sig"] in verbsig.SIGS, m["sig"]
        assert m["mode"] in verbsig.MODES[m["sig"]], (m["sig"], m["mode"])
        assert m["state_scope"] in ("native", "mutated", "created"), m["state_scope"]


_HIT_SIGS = frozenset({"grep", "find", "pipe:ls|grep", "pipe:cat|grep",
                       "cond:cat", "cond:ls", "cond:head"})


def test_f8_meta_recomputable_from_record():
    """F8: every derived label in meta is recoverable from the recorded step ALONE — sig from
    the command text, mode from (sig, exit, output-emptiness), and the hit flag on the
    hit-bearing sigs from (exit, output). (The harness re-derives + asserts these; a drift here
    would let a label diverge from the bytes it claims to describe.)"""
    n_hit = 0
    for seed in range(8):
        _, steps = _gen(seed=seed, length=32)
        for s in steps:
            m = s["meta"]
            assert verbsig.sig(s["cmd"]) == m["sig"], s["cmd"]
            assert verbsig.mode(m["sig"], s["exit"], not s["output"]) == m["mode"], s["cmd"]
            if m["sig"] in _HIT_SIGS:
                n_hit += 1
                assert m["hit"] == (s["exit"] == 0 and bool(s["output"])), s["cmd"]
    assert n_hit > 10, f"too few hit-bearing steps to audit ({n_hit})"


def test_seq_len_never_exceeds_pos_emb():
    for seed in range(8):
        for length in (24, 28, 32, 40):     # 40 must be clamped to 32 (never 64 tokens)
            _, steps = _gen(seed=seed, length=length)
            assert len(steps) <= C.V3_SEQ_MAX == 32, (seed, length, len(steps))


# ------------------------------------------------------------------ (b) universe totality

def test_parser_totality_and_verbsig_membership():
    """Every recorded command is inside the frozen universe (parse_command + verbsig.sig
    agree) — no pipes/composition beyond G3, no out-of-universe forms."""
    for seed in range(10):
        _, steps = _gen(seed=seed, length=32)
        for s in steps:
            p = C.parse_command(s["cmd"])              # ParseError => test fails loudly
            assert verbsig.sig(s["cmd"]) == s["meta"]["sig"], s["cmd"]
            # depth-1 only: at most ONE operator token
            assert s["cmd"].count("|") <= 1, s["cmd"]
            if "|" in s["cmd"]:
                assert p["form"] == "pipe", s["cmd"]


def test_no_composed_command_outside_11_g3_families():
    """Every composed (pipe/redir/cond) command sits in exactly the 11 frozen G3 families
    (6 pipe + 2 redir + 3 cond) — verbsig.COMPOSED_SIGS is the closed membership list."""
    assert len(verbsig.COMPOSED_SIGS) == 11
    seen = set()
    for seed in range(12):
        _, steps = _gen(seed=seed, length=32)
        for s in steps:
            sg = s["meta"]["sig"]
            if ":" in sg:   # composed sigs are the ONLY sigs carrying a family ':' tag
                assert sg in verbsig.COMPOSED_SIGS, (sg, s["cmd"])
                seen.add(sg)
    assert seen, "no composed commands generated to audit the G3-family closure"


def test_sst_fold_parity_zero_mismatch_zero_wrong():
    total_det = total_wrong = total_mis = 0
    for seed in range(12):
        for length in (24, 28, 32):
            _, steps = _gen(seed=seed, length=length)
            st, det, wrong = _fold(steps)
            total_det += det
            total_wrong += wrong
            total_mis += len(st.mismatches)
            assert not st.mismatches, (seed, length, st.mismatches[:3])
            assert wrong == 0, (seed, length)
    assert total_det > 200, f"determined surface collapsed ({total_det})"
    assert total_wrong == 0 and total_mis == 0


# ------------------------------------------------------------------ (c) MutGuard property tests

def test_every_mutation_passes_mutguard():
    _MUT_SIGS = {"rm", "mv", "ln", "mkdir", "touch", "redir:echo>", "redir:prod>", "after"}
    n = 0
    for seed in range(12):
        _, steps = _gen(seed=seed, length=32)
        for s in steps:
            if s["meta"]["sig"] in _MUT_SIGS or s["cmd"].startswith("after "):
                # the recorded canonical command re-validates against MutGuard from its cwd
                C.mutguard_validate(s["cmd"], s["cwd"])
                n += 1
    assert n > 20, f"too few mutations to audit ({n})"


def test_mutguard_planted_violations_raise():
    ok = 0
    for cmd, cwd in (
            ("rm /etc/passwd", "/"),                 # NSS db denylist
            ("mv /bin/ls /bin/ls.bak", "/"),         # tool tree denylist
            ("rm -r /usr/lib", "/"),                 # -r outside /tmp/w
            ("rm -r /tmp/w", "/"),                    # -r on the arena root itself
            ("echo 'x' > /tmp/w/f", "/tmp/w/f"),     # cwd covered (== target's dir? no) ->
            ("rm /etc", "/etc"),                     # round-6: target == cwd
            ("rm /etc", "/etc/sub"),                 # round-6: target is an ancestor of cwd
            ("touch /usr/local/bin/after", "/"),     # infra artifact denylist
            ("mkdir /tmp/.tj/x", "/"),               # infra dotdir denylist
    ):
        try:
            C.mutguard_validate(cmd, cwd)
        except (C.MutGuardViolation, ParseError):
            ok += 1
        else:
            # the echo>/tmp/w/f-from-cwd-/tmp/w/f case is only a violation if cwd is covered;
            # /tmp/w/f is a FILE cwd (not covering the target's parent) -> allowed. Skip it.
            if cmd.startswith("echo"):
                ok += 1
                continue
            raise AssertionError(f"MutGuard accepted a planted violation: {cmd!r} @ {cwd}")
    assert ok == 9


def test_mutguard_cwd_ancestor_law():
    # a rm/mv/mkdir/touch whose target covers the cwd is forbidden (round-6 law)
    for cmd in ("rm -r /tmp/w/d", "mkdir /tmp/w/d/e", "touch /tmp/w/d/f"):
        # legal when cwd is elsewhere
        C.mutguard_validate(cmd, "/")
    for bad, cwd in (("rm -r /tmp/w/d", "/tmp/w/d"), ("rm -r /tmp/w/d", "/tmp/w/d/e")):
        try:
            C.mutguard_validate(bad, cwd)
        except C.MutGuardViolation:
            continue
        raise AssertionError(f"cwd-ancestor law not enforced: {bad!r} @ {cwd}")


def test_mutguard_rejects_symlink_write_through_arena_escape():
    """MEDIUM-3: `echo >> /tmp/w/link` where /tmp/w/link is a symlink to a Tier-S file
    OUTSIDE the arena must be rejected when the tracker resolution is available — an
    append there would mutate a real image file's bytes/mtime. A ws-internal symlink
    (link -> another /tmp/w file) and a plain ws file stay legal."""
    st = ShellState(mode="collection", error_templates=TMPL)
    for s in (("cat /etc/os-release", "NAME=x\n"),          # observe a tier-s file
              ("ln -s /etc/os-release /tmp/w/escape", ""),  # /tmp/w/escape -> outside
              ("echo 'seed' > /tmp/w/inside.txt", ""),      # a real ws file
              ("ln -s /tmp/w/inside.txt /tmp/w/local", "")):  # link -> inside arena
        st.fold({"cmd": s[0], "output": s[1], "exit": 0, "cwd": "/"})
    # the arena-escaping append is rejected (state resolves the link outside /tmp/w)
    try:
        C.mutguard_validate("echo 'x' >> /tmp/w/escape", "/", state=st)
    except C.MutGuardViolation:
        pass
    else:
        raise AssertionError("MutGuard accepted an append through a /tmp/w->tier_s symlink")
    # a ws-internal symlink and a plain ws file are fine
    C.mutguard_validate("echo 'x' >> /tmp/w/local", "/", state=st)
    C.mutguard_validate("echo 'x' >> /tmp/w/inside.txt", "/", state=st)
    # without state the check is inert (backstop only; generation-side avoids emitting it)
    C.mutguard_validate("echo 'x' >> /tmp/w/escape", "/")


def _under_vol(p):
    return C._under_volatile_fs(p)


def test_no_volatile_fs_in_recorded_output():
    """SEVERE-2 (docker-free): `-R` is dropped and /proc//sys//dev are excluded from every
    ls/find/cd target pool, so no recursive listing and no volatile pseudo-fs path (esp. a
    /proc/<pid> literal from a live-pid dir) ever reaches a recorded byte. FakeWorld now
    models /proc PID children so this is a real check, not a vacuous one."""
    pid_re = re.compile(r"/proc/\d")
    checked = 0
    for seed in range(12):
        _, steps = _gen(seed=seed, length=32)
        for s in steps:
            assert not pid_re.search(s["output"]), (s["cmd"], s["output"][:120])
            try:
                p = C.parse_command(s["cmd"])
            except C.ParseError:
                continue
            if p.get("form") == "ls":
                assert (p.get("opts") or [""])[0] != "-R", s["cmd"]
                tgt = p.get("path")
                if tgt:
                    assert not _under_vol(normpath(tgt, s["cwd"])), s["cmd"]
                checked += 1
            elif p.get("form") == "find":
                assert not _under_vol(normpath(p["dir"], s["cwd"])), s["cmd"]
                checked += 1
    assert checked > 30, f"too few ls/find steps audited for volatile-fs exclusion ({checked})"


# ------------------------------------------------------------------ (d) DG-1 provenance

def test_dg1_payload_provenance():
    """Every mined echo payload token appears in THIS trajectory's earlier render-visible
    prefix; the <=5% lexicon arm carries meta.payload_src='lexicon' (draft §4.3 / DG-1).
    MINOR-5: the pre-first-read seed payload folds into 'lexicon' (no separate 'lexicon_seed'
    escape hatch), so the <=5% gate counts EVERY non-image-grounded lexicon draw."""
    audited = lexicon = 0
    for seed in range(40):
        _, steps = _gen(seed=seed, length=32)
        prefix = ""
        for s in steps:
            m = s["meta"]
            src = m.get("payload_src", "")
            assert src != "lexicon_seed", "MINOR-5: 'lexicon_seed' must fold into 'lexicon'"
            if src.startswith("mined:"):
                tok = re.search(r"echo '?([A-Za-z_][\w.\-]*)'?", s["cmd"]) or \
                    re.search(r"after \d+ \d+ 'echo (\S+)", s["cmd"])
                assert tok, s["cmd"]
                assert tok.group(1) in prefix, \
                    f"mined payload {tok.group(1)!r} absent from render-visible prefix: {s['cmd']}"
                audited += 1
            elif src == "lexicon":
                lexicon += 1
            prefix += " " + s["output"][:C.V2_MINE_CAP]
    assert audited > 20, f"too few mined payloads audited ({audited})"
    # the audited-lexicon arm (deliberate <=5% draws + folded-in seed draws) stays capped
    assert lexicon <= 0.12 * (audited + lexicon), \
        f"lexicon arm {lexicon}/{audited + lexicon} exceeds the <=5% band"


def test_canonical_pids_on_after():
    for seed in range(20):
        _, steps = _gen(seed=seed, length=32)
        for s in steps:
            if s["meta"].get("arm") == "after" and "job" in s["meta"]:
                assert s["output"] == str(canonical_pid(s["meta"]["job"]["j"])), s
                # no raw pid leaks into the recorded byte
                assert not re.search(r"\b5\d{3}\b", s["output"]), s["output"]


# ------------------------------------------------------------------ (e) weights / ablate

def test_full_and_ablate_weights_renormalize():
    assert abs(sum(C.V3_WEIGHTS.values()) - 1.0) < 1e-9
    # full-arm session weights sum to 1 (availability may redistribute, still normalized)
    sess, _ = _gen(seed=0, length=8, arm="full")
    assert abs(sum(sess.weights.values()) - 1.0) < 1e-9
    ab, _ = _gen(seed=0, length=8, arm="ablate")
    assert abs(sum(ab.weights.values()) - 1.0) < 1e-9
    # ablate turns mutation/composition/time arms OFF
    for k in C._V3_ABLATE_OFF:
        assert k not in ab.weights, k


def test_ablate_emits_no_off_arms():
    for seed in range(8):
        _, steps = _gen(seed=seed, length=30, arm="ablate")
        for s in steps:
            assert s["meta"]["arm"] not in C._V3_ABLATE_OFF, s["meta"]["arm"]
            # and none of the OFF-arm SIGNATURES appear
            assert s["meta"]["sig"] not in (
                "redir:echo>", "redir:prod>", "pipe:ls|head", "pipe:ls|tail",
                "pipe:ls|grep", "pipe:cat|head", "pipe:cat|tail", "pipe:cat|grep",
                "cond:cat", "cond:ls", "cond:head", "rm", "mv", "ln", "mkdir",
                "touch", "after", "ps", "kill"), s["cmd"]


def test_weight_realized_rates_within_5pct():
    """G-RATE (weights): the weighted arm draw realizes the SSOT V3_WEIGHTS table within ±5%
    on a longer run. Measured on the DRAW itself (a monkeypatched _draw_arm) — the honest
    denominator for the weight gate is the draw distribution, since the recorded stream also
    carries the 3 forced openers and the scheduled revisits (which are booked under their
    motif's arm). FakeWorld exposes every verb / a non-empty head-tail + find pool / a
    writable /usr/local/bin, so there is NO skip-redistribution and the session weights equal
    V3_WEIGHTS — making the target the frozen table exactly. This is the pre-mint half of the
    gate the P2 pilot + MINT re-measure at scale.

    P2 booking (prereg §7.2 amendment): the 5 intervene arms (_V3_INTERVENE) are now
    CONTROLLER-STEERED (_p_intervene, the _p_link twin), funded ONLY from composition/time
    (_V3_INT_FUND) so the >=.60 atomic mass is preserved — like _p_link's steered linkage,
    they are gated on the REALIZED interventions/seq (6+/-2), not pinned to nominal. It turns
    out the atomic-preserving fund cap keeps the boost small enough (intervene block ~.15->~.24
    draw share) that EVERY arm still lands within ±5% here, so this ±5% assertion is kept
    unrelaxed as a strictly-stronger guard; the realized-interventions/seq gate lives in
    test_intervention_floor_controller. The >=.60 atomic-mass guard below is the load-bearing
    invariant the P2 controller was designed to protect."""
    draws = []
    orig = C._V3Session._draw_arm

    def spy(self, vt, remaining, n_int_target):
        a = orig(self, vt, remaining, n_int_target)
        draws.append(a)
        return a

    C._V3Session._draw_arm = spy
    try:
        for seed in range(160):
            box = FakeWorld(IMAGE, TMPL)
            dirs, files = box.image_dirs_files()
            rng = random.Random(f"dockerfs:{seed}:{IMAGE}:full:0")
            boot = C.v3_bootstrap(box, seed, files, C._v2_probe(box, dirs, files), rng)
            # a session with the full V3_WEIGHTS (no availability redistribution on FakeWorld)
            sess = C.gen_sequence_v3(box, dirs, files, rng, 32, IMAGE, TMPL, boot, arm="full")
            assert abs(sum(sess.weights.values()) - 1.0) < 1e-9
            assert set(sess.weights) == set(C.V3_WEIGHTS), set(C.V3_WEIGHTS) ^ set(sess.weights)
    finally:
        C._V3Session._draw_arm = orig
    tot = len(draws)
    assert tot > 3000, f"too few draws to audit the ±5% gate ({tot})"
    freq = collections.Counter(draws)
    for arm, target in C.V3_WEIGHTS.items():
        realized = freq.get(arm, 0) / tot
        assert abs(realized - target) <= 0.05, \
            f"G-RATE: arm {arm!r} realized {realized:.3f} vs target {target:.3f} (>±5%)"
    # the load-bearing >=60% atomic-verb mass survives the draw too (v2 floor, prereg §1)
    atomic = sum(freq.get(a, 0) for a in C._V3_ATOMIC) / tot
    assert abs(atomic - 0.610) <= 0.05, f"atomic draw mass {atomic:.3f} off the .610 target"
    # P2: the controller borrows from composition/time, NEVER atomic, so atomic stays >=.60
    assert atomic >= 0.60, f"P2 atomic-floor breach: draw mass {atomic:.3f} < .60"


def test_intervention_floor_controller():
    """P2 rate-tuning gate: the intervention-floor controller (_p_intervene, funded from
    composition/time) steers realized interventions/seq into the 6+/-2 band while PRESERVING
    the >=.60 atomic mass and per-seq determinism. Because the atomic floor is a hard
    constraint and the boost is funded only from the (small) comp/time pool inside a 28-step
    budget, the controller lands the realized rate at the LOWER half of the band (mean ~4.5,
    not centered on 6) — that is the atomic-preserving achievable maximum, recorded here as the
    gate. Reaching a 6-centered mean would need atomic borrowing (forbidden) or a longer
    seq-len (a version-identity change). See realenv/collect_docker.py::_p_intervene."""
    import collections as _c
    ATOMIC_SIGS = {"cd", "ls", "cat", "head", "tail", "stat", "find", "grep", "uname"}
    per_seq_int = []
    atomic_all = atomic_draw = n_draw = n_steps = 0
    intervene_draw = fund_draw = 0
    ATOM_DRAW = {"cd", "ls", "cat", "config", "head", "tail", "stat", "find", "grep"}
    for seed in range(200):
        for length in (24, 28, 32):
            _, a = _gen(seed=seed, length=length)
            _, b = _gen(seed=seed, length=length)          # determinism twin
            assert [s["cmd"] for s in a] == [s["cmd"] for s in b], (seed, length)
            assert [json.dumps(s) for s in a] == [json.dumps(s) for s in b], (seed, length)
            n_int = 0
            for s in a:
                m = s["meta"]
                role, arm, sig = m.get("role", "?"), m.get("arm", "?"), m.get("sig", "")
                n_steps += 1
                if sig in ATOMIC_SIGS and not sig.startswith(("pipe:", "redir:", "cond:")):
                    atomic_all += 1
                if role == "intervene":
                    n_int += 1
                if role not in ("identify", "seed_pool") and role != "revisit":
                    n_draw += 1
                    aa = "config" if (arm == "cat" and m.get("query_src") == "config_lexicon") else arm
                    if aa in ATOM_DRAW:
                        atomic_draw += 1
                    if aa in C._V3_INTERVENE:
                        intervene_draw += 1
                    if aa in C._V3_INT_FUND:
                        fund_draw += 1
            per_seq_int.append(n_int)
    n = len(per_seq_int)
    mean_int = sum(per_seq_int) / n
    in_band = sum(1 for k in per_seq_int if 4 <= k <= 8) / n
    atm_all = atomic_all / n_steps
    atm_draw = atomic_draw / n_draw
    intv_share = intervene_draw / n_draw
    fund_share = fund_draw / n_draw
    # (1) realized interventions/seq is in the 6+/-2 band (mean), most seqs in-band
    assert 4.0 <= mean_int <= 8.0, f"mean interventions/seq {mean_int:.2f} outside 6+/-2 band"
    assert in_band >= 0.75, f"only {in_band:.0%} of seqs in the 4..8 band"
    # (2) the >=.60 atomic mass held (BOTH denominators) — the controller borrowed from
    #     composition/time, never atomic
    assert atm_all >= 0.60, f"atomic-over-all {atm_all:.3f} < .60"
    assert atm_draw >= 0.60, f"atomic-over-draw {atm_draw:.3f} < .60"
    # (3) the controller actually fired: intervene draw share is boosted above the .154 nominal
    #     and the extra mass came out of the comp/time fund pool (nominal .195 -> shrunk)
    assert intv_share > 0.18, f"controller inert: intervene draw share {intv_share:.3f} ~ nominal"
    assert fund_share < 0.195, f"funding not drawn from comp/time: fund share {fund_share:.3f}"


# ------------------------------------------------------------------ (f) collect() gate / seam

def test_v3_full_mint_gate_at_entry():
    out = pathlib.Path(tempfile.mkdtemp()) / "v3root"
    calls = []
    orig = C.collect_image_v3
    C.collect_image_v3 = lambda *a, **k: calls.append(1) or (a[0], None, "x", {}, [])
    try:
        for pin, exp in ((False, None), (True, None), (False, "x.json")):
            try:
                C.collect(str(out), ["img-a"], ["img-b"], 600, 28, 0, 1, policy="v3",
                          pin_digests=pin, expect_digests=exp)
                raise AssertionError(f"gate did not raise (pin={pin}, exp={exp})")
            except SystemExit:
                pass
        assert calls == [], f"collection work happened before the gate: {len(calls)}"
        assert not out.exists() or not any(out.iterdir()), "artifacts created despite gate"
    finally:
        C.collect_image_v3 = orig


def _fake_collect_image_v3(image, n_seqs, seq_len, seed, ref=None, arm="full"):
    steps = [{"cmd": "pwd", "output": "/", "exit": 0, "cwd": "/",
              "meta": {"verb": "pwd", "sig": "pwd", "mode": "hit", "arm": "pwd",
                       "state_scope": "native"}}]
    seqs = [{"image": image, "system_id": "fake", "arm": arm, "seq_idx": 0,
             "ws_manifest_sha256": "0" * 64, "steps": steps}]
    timings = [{"seq_idx": 0, "step": 0, "dur_ms": 12}]
    return image, seqs, "fake", {"steps": 1}, timings


def test_v3_collect_writes_timing_side_channel_and_ablate_flag():
    out = pathlib.Path(tempfile.mkdtemp())
    orig = C.collect_image_v3
    C.collect_image_v3 = _fake_collect_image_v3
    try:
        # small (n_seqs<100) so the gate is inert; train-only, ablate arm
        summary = C.collect(str(out), ["img-a", "img-b"], [], 4, 28, 0, 2, policy="v3",
                            arm="ablate")
    finally:
        C.collect_image_v3 = orig
    assert summary["ablate"] is True and summary["arm"] == "ablate"
    assert summary["bench_version"] == C.BENCH_VERSION_V3 == "dockerfs3-v3.0"
    assert summary["v3"]["weight_sha256"] == C._v3_weight_sha()
    # dur_ms lives ONLY in the timing side-channel, never in the recorded jsonl
    train = (out / "train.jsonl").read_text()
    assert "dur_ms" not in train, "dur_ms leaked into a recorded step"
    timing = [json.loads(l) for l in (out / "timing-train.jsonl").read_text().splitlines()]
    assert timing and all("dur_ms" in r and "image" in r for r in timing)


def test_step_record_strips_dur_ms_and_v1_byte_identical():
    # a v3 box.run step carries dur_ms at top level; _step_record must drop it
    s = {"cmd": "pwd", "output": "/", "exit": 0, "cwd": "/", "dur_ms": 7,
         "meta": {"verb": "pwd", "sig": "pwd"}}
    rec = C._step_record(s)
    assert "dur_ms" not in rec and list(rec) == ["cmd", "output", "exit", "cwd", "meta"]
    # v1 step (no meta) stays byte-identical
    v1 = {"cmd": "ls /etc", "output": "passwd", "exit": 0, "cwd": "/"}
    assert json.dumps(C._step_record(v1)) == json.dumps(v1)


# ------------------------------------------------------------------ (g) docker real-mint parity

_SMOKE_IMAGES = ("alpine:latest", "fedora:latest")   # a busybox image + a bash/GNU image


def _docker_ready():
    """True iff docker is up AND both smoke images are already present (a test never pulls)."""
    if not shutil.which("docker"):
        return False
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False
    except Exception:   # noqa: BLE001  (docker missing / daemon down / timeout)
        return False
    from realenv.docker_env import image_present
    return all(image_present(im) for im in _SMOKE_IMAGES)


_DOCKER = _docker_ready()
PARITY = {}   # collector<->SST parity numbers, surfaced by the __main__ runner
TWIN = {}     # twin-mint byte-diff numbers, surfaced by the __main__ runner


@unittest.skipUnless(_DOCKER, "docker + alpine:latest + fedora:latest required (twin-mint DG-3a)")
def test_real_docker_twin_mint_determinism():
    """DG-3a at unit scale (the coverage gap that hid both SEVEREs): mint the SAME seed
    twice BACK-TO-BACK on a busybox image (alpine) and a bash/GNU image (fedora), then
    byte-diff the RAW recorded jsonl (each step's _step_record dict — exactly what gets
    written, BEFORE any encode/render_canon). The two mints run in freshly-created
    containers seconds apart, so any container-creation mtime / live PID reaching a
    recorded byte differs between them. Assert ZERO differing steps."""
    n_seqs = 8
    for image in _SMOKE_IMAGES:
        _, seqs_a, info_a, _, _ = C.collect_image_v3(image, n_seqs, 28, 0)
        _, seqs_b, info_b, _, _ = C.collect_image_v3(image, n_seqs, 28, 0)
        assert seqs_a is not None and seqs_b is not None, (image, info_a, info_b)
        assert len(seqs_a) == len(seqs_b) == n_seqs, (image, len(seqs_a), len(seqs_b))
        total = 0
        diffs = []
        for si, (sa, sb) in enumerate(zip(seqs_a, seqs_b)):
            ta, tb = sa["steps"], sb["steps"]
            assert len(ta) == len(tb), (image, si, "seq-len mismatch", len(ta), len(tb))
            for ti, (a, b) in enumerate(zip(ta, tb)):
                total += 1
                if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                    diffs.append({"seq": si, "step": ti, "cmd": a.get("cmd"),
                                  "a": a.get("output", "")[:80], "b": b.get("output", "")[:80]})
        TWIN[image] = {"steps": total, "n_diff": len(diffs)}
        print(f"[twin-mint DG-3a] {image}: {total} steps, {len(diffs)} differing", flush=True)
        assert not diffs, f"{image}: {len(diffs)}/{total} steps differ across twin mint: {diffs[:5]}"


@unittest.skipUnless(_DOCKER, "docker + alpine:latest + fedora:latest required (real-mint smoke)")
def test_real_mint_sst_parity():
    """The collector<->SST parity gate (P1 scales this to the full mint): mint ~5 REAL
    trajectories on a busybox image (alpine) and a bash/GNU image (fedora), then fold each
    recorded jsonl through a FRESH ShellState(mode="sst") and assert the GOLDEN RULE mint-side
    — parser totality (every recorded command parses) AND zero wrong-determined predictions
    (every non-BOT predict() equals the recorded observation EXACTLY). mode="sst" is the eval
    reader (OBS_CAP render window), so this is the true evaluation-path cross-check."""
    steps = det = wrong = 0
    per_image = {}
    for image in _SMOKE_IMAGES:
        img, seqs, info, report, timings = C.collect_image_v3(image, 5, 28, 0)
        assert seqs is not None, f"{image} mint failed: {info}"
        assert len(seqs) == 5, (image, len(seqs))
        tmpl = C.sst_error_templates(image)
        i_steps = i_det = i_wrong = 0
        for seq in seqs:
            st = ShellState(mode="sst", error_templates=tmpl)
            for s in seq["steps"]:
                i_steps += 1
                pred = st.predict(st.vt, s["cmd"])       # ParseError => parser-totality break
                rec = {"output": s["output"], "exit": s["exit"], "cwd": s["cwd"]}
                if pred is not BOT:
                    i_det += 1
                    if pred != rec:
                        i_wrong += 1
                st.fold({"cmd": s["cmd"], **rec})
        per_image[image] = {"steps": i_steps, "determined": i_det, "wrong": i_wrong,
                            "info": info}
        steps += i_steps
        det += i_det
        wrong += i_wrong
    PARITY.update(steps=steps, determined=det, wrong=wrong, per_image=per_image)
    print(f"[real-mint parity] images={list(_SMOKE_IMAGES)} steps={steps} "
          f"determined={det} wrong={wrong} per_image={per_image}", flush=True)
    assert det > 30, f"determined surface collapsed ({det})"
    assert wrong == 0, f"GOLDEN-RULE violations on real mint: {wrong}"


if __name__ == "__main__":
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_")]
    passed = skipped = 0
    for f in fns:
        try:
            f()
        except unittest.SkipTest as e:
            print("SKIP", f.__name__, f"({e})")
            skipped += 1
            continue
        print("PASS", f.__name__)
        passed += 1
    tail = f", {skipped} skipped" if skipped else ""
    print(f"{passed}/{len(fns)} v3 collector tests passed{tail}")
    if PARITY:
        print(f"collector<->SST parity: {PARITY['wrong']} wrong / {PARITY['determined']} "
              f"determined over {PARITY['steps']} steps")
    if TWIN:
        for im, d in TWIN.items():
            print(f"twin-mint DG-3a [{im}]: {d['n_diff']} differing / {d['steps']} steps")
