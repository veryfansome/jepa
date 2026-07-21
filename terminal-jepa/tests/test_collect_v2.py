"""Guards for the dockerfs2 (v2.0) collection policy (realenv/collect_docker.py `v2`):
the DockerBox._exec contract (review-B item 6), determinism of the policy's pure helpers
(global crc32 K assignment, lexicon content-hashes), meta threading through _step_record,
and the policy's pre-registered invariants on a stubbed box (no Docker needed): meta on
every step, no pipes, no grep -c, no -i on intended misses, global-K head/tail with the
2K+1 floor by rejection, render-visible-prefix mining, byte-identical determinism."""

import fnmatch
import hashlib
import json
import pathlib
import random
import re
import shlex
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import realenv.collect_docker as C
import realenv.docker_env as DE


# ------------------------------------------------------------------ fake box (no Docker)

def _mk_content(name, n):
    words = ["root", "daemon", "server", "listen", "kernel", "module", "cache", "socket"]
    return "".join(f"{words[i % len(words)]} {name} line{i} value{i * 7}\n" for i in range(n))


FAKE_DIRS = ["/etc", "/etc/conf.d", "/etc/ssl", "/usr", "/usr/lib", "/usr/bin",
             "/usr/share", "/var", "/var/log"]

FAKE_FS = {
    "/etc/os-release": _mk_content("osrel", 45),
    "/etc/passwd": _mk_content("passwd", 45),
    "/etc/hosts": _mk_content("hosts", 45),
    "/etc/conf.d/net.conf": _mk_content("netconf", 45),
    "/etc/ssl/openssl.cnf": _mk_content("openssl", 45),
    "/usr/lib/libc.so": _mk_content("libc", 45),
    "/usr/lib/os-release": _mk_content("usrosrel", 45),
    "/usr/bin/tool.sh": _mk_content("tool", 45),
    "/usr/share/readme.txt": _mk_content("readme", 45),
    "/var/log/boot.log": _mk_content("bootlog", 45),
    "/etc/hostname": "fakebox\n",              # 1 line: must be REJECTED from head/tail pool
    "/etc/shells": _mk_content("shells", 5),   # 5 lines < 2K+1 for every K: rejected too
    # marker token past the render-visible mining prefix (V2_MINE_CAP=500 output chars, F7):
    # must NEVER be mined into a query pool. Two lines (one long first line) so the 2K+1
    # floor rejects the file from the head/tail pool — a tail would legitimately SHOW the
    # marker inside its own visible prefix, which is not what this guard tests.
    "/etc/late.conf": ("aa " * 200) + "\n" + "ZZLATEMARKER hidden\n",
}


def _parent(p):
    p = p.rstrip("/") or "/"
    return "/" if p == "/" else (p.rsplit("/", 1)[0] or "/")


class FakeBox:
    """Offline DockerBox stand-in: same run()/_exec() contracts over a tiny fake fs."""

    def __init__(self):
        self.fs = dict(FAKE_FS)
        self.dirs = list(FAKE_DIRS)
        self.files = sorted(self.fs)
        self.dirset, self.fileset = set(self.dirs) | {"/"}, set(self.files)
        self.cwd = "/"

    def _exec(self, script):
        if script.startswith("for t in head tail stat find grep"):
            return "head\ntail\nstat\nfind\ngrep\n", "", 0
        if script.startswith("for g in "):   # F2 find-pair probe: `glob|first-hit` per hit
            d = re.search(r"find '([^']*)' -maxdepth 3", script).group(1)
            lines = []
            for g in C.GLOB_LEXICON:
                r = self.run(f"find {d} -maxdepth 3 -name '{g}'")
                hit = r["output"].split("\n")[0] if r["output"] else ""
                if hit:
                    lines.append(f"{g}|{hit}")
            return ("\n".join(lines) + "\n") if lines else "", "", 0
        paths = [p for p in re.findall(r"'([^']*)'", script) if p.startswith("/")]
        if script.startswith(("stat -c '%s %n' ", "wc -c ")):
            out = "".join(f"{len(self.fs[p])} {p}\n" for p in paths if p in self.fs)
            return out, "", 0
        if script.startswith("wc -l "):
            out = "".join(f"{self.fs[p].count(chr(10))} {p}\n" for p in paths if p in self.fs)
            return out, "", 0
        raise AssertionError(f"unexpected probe script: {script[:60]}")

    def _ls(self, target):
        t = target.rstrip("/") or "/"
        if t in self.fileset:
            return t.rsplit("/", 1)[1], 0
        if t not in self.dirset:
            return f"ls: {t}: No such file or directory", 1
        pref = "" if t == "/" else t
        names = sorted({p[len(pref) + 1:].split("/")[0]
                        for p in self.files + self.dirs if p.startswith(pref + "/")})
        return "\n".join(names), 0

    def run(self, cmd):
        args = shlex.split(cmd)
        verb = args[0]
        out, code = "", 0
        if verb == "uname":
            out = "Linux fakebox 6.1.0"
        elif verb == "cd":
            tgt = args[1] if len(args) > 1 else "/"
            new = (_parent(self.cwd) if tgt == ".." else self.cwd if tgt == "."
                   else (tgt.rstrip("/") or "/") if tgt.startswith("/")
                   else self.cwd.rstrip("/") + "/" + tgt)
            if new in self.dirset:
                self.cwd = new
            else:
                out, code = f"sh: cd: {tgt}: No such file or directory", 1
        elif verb == "cat":
            out, code = (self.fs[args[-1]], 0) if args[-1] in self.fs \
                else (f"cat: {args[-1]}: No such file or directory", 1)
        elif verb == "ls":
            tgt = next((a for a in args[1:] if not a.startswith("-")), self.cwd)
            out, code = self._ls(tgt)
        elif verb in ("head", "tail"):
            k, path = int(args[2]), args[3]
            if path in self.fs:
                lines = self.fs[path].splitlines()
                keep = lines[:k] if verb == "head" else lines[-k:]
                out = "\n".join(keep)
            else:
                out, code = f"{verb}: {path}: No such file or directory", 1
        elif verb == "stat":
            path = args[-1]
            if path in self.fs:
                out = f"{path} {len(self.fs[path])} regular file 644"
            elif path in self.dirset:
                out = f"{path} 4096 directory 755"
            else:
                out, code = f"stat: {path}: No such file or directory", 1
        elif verb == "find":
            base = args[1].rstrip("/") or "/"
            md = int(args[args.index("-maxdepth") + 1])
            ty = args[args.index("-type") + 1] if "-type" in args else None
            glob = args[args.index("-name") + 1]
            pool = (self.dirs if ty != "f" else []) + (self.files if ty != "d" else [])
            hits = []
            for p in sorted(pool):
                pref = "" if base == "/" else base
                if not p.startswith(pref + "/"):
                    continue
                rel = p[len(pref) + 1:]
                if rel.count("/") + 1 <= md and fnmatch.fnmatch(p.rsplit("/", 1)[1], glob):
                    hits.append(p)
            out = "\n".join(hits)
        elif verb == "grep":
            tok, path = args[-2], args[-1]
            ci = "-i" in args
            if path not in self.fs:
                out, code = f"grep: {path}: No such file or directory", 2
            else:
                needle = tok.lower() if ci else tok
                hits = [ln for ln in self.fs[path].splitlines()
                        if needle in (ln.lower() if ci else ln)][:8]
                out, code = ("\n".join(hits), 0) if hits else ("", 1)
        else:
            out, code = f"sh: {verb}: not found", 127
        return {"cmd": cmd, "output": out, "exit": code, "cwd": self.cwd}


def _gen(seed=0, length=28):
    box = FakeBox()
    steps = C.gen_sequence_v2(box, box.dirs, box.files, random.Random(seed), length)
    return box, steps


# ------------------------------------------------------------------ (a) _exec contract

def test_exec_contract():
    """Review-B item 6: DockerBox._exec returns (stdout, stderr, returncode) as STRINGS
    (decoded from captured bytes), and the timeout path returns ("", "command timed out", 124)."""
    box = DE.DockerBox.__new__(DE.DockerBox)
    box.cid, box.cmd_timeout = "fakecid", 8

    class R:
        stdout, stderr, returncode = b"out", b"err", 3

    calls = {}

    def fake_run(argv, **kw):
        calls["argv"] = argv
        assert "text" not in kw and "encoding" not in kw, "F1: must capture bytes"
        return R

    orig = DE.subprocess.run
    DE.subprocess.run = fake_run
    try:
        res = box._exec("echo hi")
        assert res == ("out", "err", 3)
        assert calls["argv"][:5] == ["docker", "exec", "fakecid", "/bin/sh", "-c"]
        assert calls["argv"][5] == "echo hi"

        def raising(argv, **kw):
            raise DE.subprocess.TimeoutExpired(argv, kw.get("timeout"))
        DE.subprocess.run = raising
        assert box._exec("sleep 99") == ("", "command timed out", 124)
    finally:
        DE.subprocess.run = orig


def test_exec_binary_decodes_with_replacement():
    """Review-B F1 (BLOCKER): binary exec output becomes replacement-char mojibake (as a
    real terminal shows), never the host-side "executor error: 'utf-8' codec ..." artifact."""
    box = DE.DockerBox.__new__(DE.DockerBox)
    box.cid, box.cmd_timeout = "fakecid", 8

    class R:
        stdout = b"\x7fELF\xff\xfe\x00binary\x89tail"
        stderr = b"warn\xc3("
        returncode = 0

    orig = DE.subprocess.run
    DE.subprocess.run = lambda argv, **kw: R
    try:
        out, err, code = box._exec("cat /bin/busybox")
    finally:
        DE.subprocess.run = orig
    assert code == 0
    assert "executor error" not in out and "executor error" not in err
    assert "�" in out and out.startswith("\x7fELF") and out.endswith("tail")
    assert err.startswith("warn") and "�" in err


# ------------------------------------------------------------------ (b) pure-helper determinism

def test_k_assignment_global_deterministic():
    """K = crc32(path) % KSET, image-independent (function of the path alone), stable."""
    for p in ("/etc/passwd", "/etc/os-release", "/usr/lib/libc.so", "/etc/hosts"):
        k = C.v2_k_for(p)
        assert k == C.HEADTAIL_KS[zlib.crc32(p.encode()) % len(C.HEADTAIL_KS)]
        assert k == C.v2_k_for(p) and k in C.HEADTAIL_KS


def test_lexicon_hashes_deterministic():
    h1, h2 = C.lexicon_hashes(), C.lexicon_hashes()
    assert h1 == h2
    for lex, key in ((C.QUERY_LEXICON, "query_lexicon_sha256"),
                     (C.MISS_LEXICON, "miss_lexicon_sha256"),
                     (C.GLOB_LEXICON, "glob_lexicon_sha256")):
        assert h1[key] == hashlib.sha256(json.dumps(lex, sort_keys=True).encode()).hexdigest()
    assert h1["headtail_ks"] == [3, 5, 10, 20]


def test_p_link_deterministic_and_bounded():
    assert C._p_link([0, 0], 0.6) == 0.6
    for lk, tot in ((0, 10), (5, 10), (10, 10), (3, 4)):
        p = C._p_link([lk, tot], 0.6)
        assert 0.20 <= p <= 0.97 and p == C._p_link([lk, tot], 0.6)
    assert C._p_link([0, 10], 0.6) > 0.6 > C._p_link([10, 10], 0.6)  # steers toward target


def test_headtail_pool_floor_by_rejection():
    """The 2K+1 line floor drops short files from the pool; K itself is never reassigned."""
    box = FakeBox()
    st = C._v2_probe(box, box.dirs, box.files)
    assert "/etc/hostname" not in st["kmap"] and "/etc/shells" not in st["kmap"]
    assert st["kmap"], "no head/tail-eligible files in the fake fs"
    for p, k in st["kmap"].items():
        assert k == C.v2_k_for(p)
        assert st["lines"][p] >= 2 * k + 1


# ------------------------------------------------------------------ (c) policy + meta threading

def test_meta_on_every_step_and_invariants():
    box, steps = _gen(seed=0, length=28)
    assert len(steps) == 28
    for s in steps:
        m = s.get("meta")
        assert m and "verb" in m and "linked" in m and "query_src" in m
        assert "|" not in s["cmd"], f"pipe in recorded cmd: {s['cmd']}"
        cverb = s["cmd"].split()[0]
        if cverb == "grep":
            assert " -c" not in s["cmd"], "grep -c is dropped from the v2 toolset"
            if m["intended_miss"]:
                assert " -i" not in s["cmd"], "-i on an intended miss"
        if cverb in ("head", "tail"):
            path = s["cmd"].split()[-1]
            assert int(s["cmd"].split()[2]) == C.v2_k_for(path) == m["k"]
    # step 0 records the availability probe + skips + lexicon version
    m0 = steps[0]["meta"]
    assert m0["availability"] == {v: True for v in C.V2_NEW_VERBS}
    assert m0["skipped_verbs"] == [] and m0["lexicon_version"] == C.V2_LEXICON_VERSION
    # linked steps occur and record a provenance pointer into the same sequence
    n_linked = 0
    for i, s in enumerate(steps):
        if not s["meta"]["linked"]:
            continue
        n_linked += 1
        src = s["meta"]["query_src"]
        assert "@" in src, f"linked step without obs provenance: {src}"
        assert int(src.split("@")[1]) < i, "provenance must be strictly earlier"
    assert n_linked, "no linked step in 28 steps"


def test_mining_only_from_visible_prefix():
    """Review-A fix: a token that only appears past output[:1000] must never be mined into
    the grep query pools (linked or transplant)."""
    for seed in range(6):
        _, steps = _gen(seed=seed, length=32)
        for s in steps:
            assert "ZZLATEMARKER" not in s["cmd"], f"mined past the visible prefix: {s['cmd']}"


def test_policy_deterministic():
    _, a = _gen(seed=7, length=26)
    _, b = _gen(seed=7, length=26)
    assert [s["cmd"] for s in a] == [s["cmd"] for s in b]
    assert [s["meta"] for s in a] == [s["meta"] for s in b]


def test_step_record_threads_meta():
    """collect_image writes meta through _step_record; v1 steps (no meta) stay byte-identical."""
    s = {"cmd": "grep -F -m 8 'root' /etc/passwd", "output": "root:x:0", "exit": 0,
         "cwd": "/", "meta": {"verb": "grep", "linked": True, "query_src": "cat@2",
                              "intended_miss": False, "flags": "-F"}}
    rec = C._step_record(s)
    assert rec["meta"] == s["meta"]
    assert list(rec) == ["cmd", "output", "exit", "cwd", "meta"]
    v1 = {"cmd": "ls /etc", "output": "passwd", "exit": 0, "cwd": "/"}
    rec1 = C._step_record(v1)
    assert "meta" not in rec1 and json.dumps(rec1) == json.dumps(v1)


def test_image_report_counters():
    box, steps = _gen(seed=1, length=28)
    rep = C.v2_image_report(box)
    assert rep["avail"] == {v: True for v in C.V2_NEW_VERBS} and rep["skipped_verbs"] == []
    assert rep["steps"] == len(steps) == sum(rep["verb_counts"].values())
    from collections import Counter
    assert rep["verb_counts"] == dict(Counter(s["meta"]["verb"] for s in steps))
    assert rep["linked_counts"] == dict(Counter(s["meta"]["verb"] for s in steps
                                                if s["meta"]["linked"]))
    assert rep["intended_miss"] == sum(1 for s in steps if s["meta"].get("intended_miss"))
    assert rep["intended_empty"] == sum(1 for s in steps if s["meta"].get("intended_empty"))
    assert rep["find_pool"]["hits"] >= 1 and rep["find_pool"]["empties"] >= 1


# ------------------------------------------------------------------ (d) review-B fixes

def test_find_pool_verified_pairs_and_empty_rate():
    """Review-B F2 (BLOCKER): every find command draws a probe-verified (dir, glob) pair —
    hit-pool draws realize hits, the deliberate-empty arm realizes empties — with an
    intended-empty rate near the controlled 0.2."""
    box = FakeBox()
    st = C._v2_probe(box, box.dirs, box.files)
    hit_pairs = {(d, g) for d, g, _mds, _t in st["find_hits"]}
    empty_pairs = set(st["find_empties"])
    assert hit_pairs and empty_pairs
    for d, g, mds, _t in st["find_hits"]:   # probe ground truth at every verified depth
        for md in mds:
            assert box.run(f"find {d} -maxdepth {md} -name '{g}'")["output"], (d, g, md)
    n_find = n_empty = 0
    for seed in range(30):
        b = FakeBox()
        steps = C.gen_sequence_v2(b, b.dirs, b.files, random.Random(seed), 30)
        for s in steps:
            m = s["meta"]
            if m["verb"] != "find":
                continue
            n_find += 1
            args = shlex.split(s["cmd"])
            d, g = args[1], args[args.index("-name") + 1]
            if m["intended_empty"]:
                n_empty += 1
                assert (d, g) in empty_pairs, f"empty draw outside verified pool: {s['cmd']}"
                assert s["output"] == "" and m["hit"] is False
            else:
                assert (d, g) in hit_pairs, f"hit draw outside verified pool: {s['cmd']}"
                assert s["output"] and m["hit"] is True
    assert n_find >= 40, f"too few find steps to audit ({n_find})"
    rate = n_empty / n_find
    assert 0.08 <= rate <= 0.35, f"intended-empty rate {rate:.3f} not ~0.2"


def test_meta_hit_recorded_for_grep_and_find():
    """Review-B F8: meta.hit == (exit == 0 and non-empty output) on every grep and find
    step (labels recoverable from the jsonl without re-execution); other verbs carry none."""
    n = 0
    for seed in range(8):
        _, steps = _gen(seed=seed, length=30)
        for s in steps:
            m = s["meta"]
            if m["verb"] in ("grep", "find"):
                n += 1
                assert m["hit"] == (s["exit"] == 0 and bool(s["output"])), s["cmd"]
            else:
                assert "hit" not in m
    assert n, "no grep/find steps generated"


def _fake_collect_image(image, n_seqs, seq_len, seed, policy="baseline", ref=None):
    steps = [{"cmd": "ls", "output": "etc", "exit": 0, "cwd": "/",
              "meta": {"verb": "ls", "linked": False, "query_src": "cwd"}}]
    return image, [{"image": image, "system_id": "fake", "steps": steps}], "fake", {}


def test_train_only_never_truncates_val():
    """Review-B F6: an empty val list must not open (truncate) val.jsonl. Also checks the
    F9 summary block (frozen class table + bench_version) and deterministic submitted-image
    order in the jsonl."""
    import tempfile
    out = pathlib.Path(tempfile.mkdtemp())
    (out / "val.jsonl").write_text('{"keep": 1}\n')
    orig = C.collect_image
    C.collect_image = _fake_collect_image
    try:
        summary = C.collect(out, ["img-a", "img-b"], [], 1, 4, 0, 2, policy="v2")
    finally:
        C.collect_image = orig
    assert (out / "val.jsonl").read_text() == '{"keep": 1}\n', "val.jsonl was truncated"
    assert summary["val_steps"] == 0
    imgs = [json.loads(ln)["image"] for ln in (out / "train.jsonl").read_text().splitlines()]
    assert imgs == ["img-a", "img-b"], f"jsonl not in submitted-image order: {imgs}"
    assert summary["bench_version"] == C.BENCH_VERSION == "dockerfs2-v2.0"
    assert summary["verb_classes"] == C.V2_VERB_CLASSES == summary["v2"]["verb_classes"]
    assert summary["verb_classes"]["content"] == ["ls", "cat", "head", "tail", "find", "grep"]
    assert summary["verb_classes"]["semi_echo"] == ["stat"]
    assert summary["verb_classes"]["grep_mode_rule"] == "exit!=0 or empty output => miss (excluded)"
    written = json.loads((out / "summary.json").read_text())
    assert written["verb_classes"] == C.V2_VERB_CLASSES


def test_v2_mint_aborts_on_failed_image():
    """Review-B mint integrity: a v2-policy run must RAISE (not skip) when an image fails —
    a preregistered one-run mint must never silently produce a partial dataset."""
    import tempfile
    out = pathlib.Path(tempfile.mkdtemp())

    def failing(image, *a, **kw):
        if image == "img-bad":
            return image, None, "could not pull img-bad", {}
        return _fake_collect_image(image, *a, **kw)

    orig = C.collect_image
    C.collect_image = failing
    try:
        try:
            C.collect(out, ["img-a", "img-bad"], [], 1, 4, 0, 1, policy="v2")
            raised = False
        except RuntimeError as e:
            raised = True
            assert "img-bad" in str(e)
        assert raised, "v2 collect() did not raise on a skipped image"
        # v1 policies keep the old skip behavior
        C.collect(out, ["img-a", "img-bad"], [], 1, 4, 0, 1, policy="baseline")
    finally:
        C.collect_image = orig
