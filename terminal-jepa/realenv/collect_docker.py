"""Collect real shell trajectories from Docker images (Phase R2, 2026-07-16 redesign).

Each trajectory is a SEQUENCE that explores a real Linux filesystem: identify the system
(uname + cat a config file), then navigate and inspect (cd / ls / cat) with option and
target variety over the image's real paths. The world model must predict later
observations from the accumulated history. Split by held-out IMAGE (unseen system types)
— the fair "does it generalize to new places/systems" test, NOT the unreasonable
"infer an unseen tool" test.

Tools (initial set, per user): uname, cat (system config files), ls, cd.
The v2 mint policy (`--policy v2`, dockerfs2) expands to 9 verbs — see gen_sequence_v2 and
benchmarks/dockerfs2-prereg.md (Amendment 1 is the adopted-policy spec).

Usage:
  python3 -m realenv.collect_docker --out data/dockerfs --seqs-per-image 300 --seq-len 16
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import pathlib
import random
import re
import subprocess
import zlib

from realenv.docker_env import DockerBox, hostname_for, image_present, pull
from realenv import shell_state as SS
from realenv import verbsig
from realenv.shell_state import (ShellState, ParseError, canonical_pid, normpath,
                                 parent_of, basename_of, WORKSPACE, PS_PATH,
                                 TIME_FREE_LS, parse_command)
from realenv.render_canon import canon_ls_l_text

TRAIN_IMAGES = ["alpine:latest", "ubuntu:latest", "debian:stable-slim", "python:3.12-slim"]
VAL_IMAGES = ["fedora:latest", "redis:alpine", "nginx:alpine"]  # held-out system types

CONFIG_FILES = ["/etc/os-release", "/etc/hostname", "/etc/issue", "/proc/version",
                "/etc/passwd", "/etc/group", "/etc/hosts", "/etc/resolv.conf",
                "/etc/shells", "/etc/profile"]
UNAME_OPTS = ["-a", "-s", "-m", "-r", "-n", "-o", "-v", "-sm", "-sr", ""]
# options common to busybox and GNU ls (kept safe across images)
LS_OPTS = ["", "-l", "-a", "-la", "-R", "-1", "-lh", "-lt", "-lS", "-ld", "-i", "-lr", "-ln"]


def gen_sequence(box, dirs, files, rng, length):
    steps = []

    def do(cmd):
        steps.append(box.run(cmd))

    do("uname " + rng.choice(UNAME_OPTS))
    do("cat " + rng.choice(CONFIG_FILES))
    for _ in range(max(0, length - 2)):
        act = rng.choices(["cd", "ls", "cat", "config"], weights=[0.32, 0.4, 0.2, 0.08])[0]
        if act == "cd":
            tgt = rng.choice(dirs + ["..", ".", "..", "/", box.cwd] if dirs else ["..", "/", "."])
            do(f"cd {tgt}")
        elif act == "ls":
            opt = rng.choice(LS_OPTS)
            tgt = (" " + rng.choice(dirs)) if (dirs and rng.random() < 0.45) else ""
            do(f"ls {opt}{tgt}".strip())
        elif act == "cat" and files:
            do("cat " + rng.choice(files))
        else:
            do("cat " + rng.choice(CONFIG_FILES))
    return steps


def gen_sequence_diverse(box, dirs, files, rng, length):
    """Exploration-policy variant (the `exploration` evolve chunk): higher training-data
    diversity than the baseline policy — (1) richer system identity: read TWO distinct config
    files at the open; (2) higher distinct-target COVERAGE: cycle through per-sequence SHUFFLED
    dir/file lists instead of uniform-random sampling (so a sequence visits many distinct paths
    rather than repeating a few); (3) more file-content: higher `cat`-of-file weight, always with
    a target. Hypothesis: more diverse (command, observation) pairs on the train systems → better
    generalization to the unseen held-out systems (same held-out val as baseline)."""
    steps = []

    def do(cmd):
        steps.append(box.run(cmd))

    do("uname " + rng.choice(UNAME_OPTS))
    cfgs = rng.sample(CONFIG_FILES, min(2, len(CONFIG_FILES)))
    for c in cfgs:
        do("cat " + c)
    dcycle = dirs[:]; rng.shuffle(dcycle); di = 0
    fcycle = files[:]; rng.shuffle(fcycle); fi = 0
    for _ in range(max(0, length - 1 - len(cfgs))):
        act = rng.choices(["cd", "ls", "cat", "config"], weights=[0.28, 0.34, 0.30, 0.08])[0]
        if act == "cd" and dcycle:
            do(f"cd {dcycle[di % len(dcycle)]}"); di += 1
        elif act == "ls":
            opt = rng.choice(LS_OPTS)
            tgt = (" " + dcycle[di % len(dcycle)]) if dcycle else ""
            di += 1
            do(f"ls {opt}{tgt}".strip())
        elif act == "cat" and fcycle:
            do("cat " + fcycle[fi % len(fcycle)]); fi += 1
        else:
            do("cat " + rng.choice(CONFIG_FILES))
    return steps


def gen_sequence_levy_novelty(box, dirs, files, rng, length):
    """Exploration-policy variant (the `exploration` evolve chunk): forage the REAL directory
    TREE as an intermittent inverse-square Levy walk with count-based novelty, instead of the
    baseline's uniformly-random global teleports.

    Confound control: the action/verb selection weights and fallbacks are IDENTICAL to the
    baseline policy and every action emits exactly ONE command (a multi-level descent is a single
    `cd /a/b/c`), so the TRAIN verb mix and observation-type marginals match baseline within noise
    (unlike `diverse`, which shifted cat 15676->22371). Only WHICH concrete paths appear changes:
      - cd: sample a Levy step-length L (power law, alpha=2 -> P(L>=k)~k^-1, capped) and descend a
            path of L novelty-weighted children in ONE command (intensive phase); with small prob
            return to the parent, or make a rare global novelty jump (extensive phase / heavy tail);
      - ls: option identical to baseline; target (same ~0.45 arg rate) is a novel child of cwd;
      - cat: prefer a novelty-weighted file inside the current subtree (coherent with history).
    Visit counts over directories persist across this image's sequences via the reused `box`.
    Hypothesis: spatially coherent local descent makes the accumulated history genuinely predictive
    of the next observation (transition structure), while the Levy heavy tail + count-based novelty
    cover the whole tree -> better generalization to the unseen held-out systems."""
    from collections import defaultdict

    st = getattr(box, "_levy_state", None)
    if st is None:
        def parent_of(p):
            p = p.rstrip("/") or "/"
            if p == "/":
                return "/"
            par = p.rsplit("/", 1)[0]
            return par or "/"
        children = defaultdict(list)
        for d in dirs:
            children[parent_of(d)].append(d)
        st = {"children": children, "all_dirs": list(dirs), "vc": defaultdict(int)}
        box._levy_state = st
    children, all_dirs, vc = st["children"], st["all_dirs"], st["vc"]

    ALPHA = 2.0     # inverse-square Levy: intensive-phase step length P(L>=k) ~ k^-(ALPHA-1)
    MAXJUMP = 8

    steps = []

    def do(cmd):
        steps.append(box.run(cmd))
        vc[box.cwd] += 1  # count-based novelty over visited directories

    def novel_choice(pool):
        if not pool:
            return None
        w = [1.0 / (1.0 + vc[p]) for p in pool]
        return rng.choices(pool, weights=w)[0]

    def global_jump():
        sample = rng.sample(all_dirs, min(64, len(all_dirs))) if all_dirs else []
        tgt = novel_choice(sample)
        do("cd " + tgt) if tgt else do("cd /")

    # system identity: identical to baseline (preserve verb mix / observation marginals)
    do("uname " + rng.choice(UNAME_OPTS))
    do("cat " + rng.choice(CONFIG_FILES))

    for _ in range(max(0, length - 2)):
        act = rng.choices(["cd", "ls", "cat", "config"], weights=[0.32, 0.4, 0.2, 0.08])[0]
        cwd = box.cwd
        if act == "cd":
            r = rng.random()
            if r < 0.12:
                global_jump()                              # extensive phase / heavy Levy tail
            elif r < 0.30 and cwd != "/":
                do("cd ..")                                 # intermittent return (ascend)
            else:                                           # intensive phase: Levy-length descent
                u = rng.random()
                L = min(MAXJUMP, int((1.0 - u) ** (-1.0 / (ALPHA - 1.0))))
                c = cwd
                for _ in range(L):
                    kids = children.get(c, [])
                    if not kids:
                        break
                    c = novel_choice(kids)
                if c != cwd:
                    do("cd " + c)
                elif cwd != "/":
                    do("cd ..")
                else:
                    global_jump()
        elif act == "ls":
            opt = rng.choice(LS_OPTS)
            kids = children.get(cwd, [])
            if kids and rng.random() < 0.45:
                do(f"ls {opt} {novel_choice(kids)}".strip())
            else:
                do(f"ls {opt}".strip())
        elif act == "cat":
            if cwd != "/":
                pref = cwd if cwd.endswith("/") else cwd + "/"
                pool = [f for f in files if f.startswith(pref)]
            else:
                pool = []
            if not pool:
                pool = files
            if pool:
                if len(pool) > 128:
                    pool = rng.sample(pool, 128)
                do("cat " + novel_choice(pool))
            else:
                do("cat " + rng.choice(CONFIG_FILES))
        else:
            do("cat " + rng.choice(CONFIG_FILES))
    return steps


# ============================== dockerfs2 (v2.0) mint policy ==============================
# The adopted policy of benchmarks/dockerfs2-prereg.md Amendment 1: SYNTHESIS on the
# adversarial-coverage base with every review-A fix, plus the navreal grafts named in the
# review verdict (cd->ls orient motif, observed-child descent, richer query_src provenance).
# Review-A fixes applied here: global crc32 K over KSET (image-independent; 2K+1 line floor
# by REJECTION from the head/tail pool, never K reassignment); linkage pools mined ONLY from
# the render-visible prefix (output[:1000]); transplant-miss primary for grep + small audited
# lexicon-miss arm, no -i on intended misses; per-verb within-sequence used-sets; explicit
# imports; per-image `command -v` availability probe with recorded skips; grep -c DROPPED;
# meta={verb, linked, query_src, ...} on every step.

V2_LEXICON_VERSION = "v2.0"

# ~40 config-domain tokens for grep probes (natural hits AND natural near-misses: every
# token is plausible in *some* /etc file, so a miss is a semantically hard negative).
QUERY_LEXICON = [
    "root", "daemon", "nologin", "localhost", "hostname", "nameserver", "domain",
    "search", "PATH", "HOME", "SHELL", "export", "umask", "VERSION_ID",
    "PRETTY_NAME", "ID_LIKE", "NAME", "listen", "server", "worker", "include",
    "user", "group", "port", "socket", "timeout", "buffer", "cache", "error",
    "log", "ssl", "charset", "alias", "module", "kernel", "mount", "systemd",
    "getty", "tcp", "bind",
]

# Small audited lexicon-miss arm: config-domain tokens common in SOME config dialects but
# rare in the files these images ship (near-miss by design; realized rate audited at review B).
MISS_LEXICON = [
    "LoadModule", "ProxyPass", "ServerAdmin", "innodb_buffer", "max_connections",
    "keepalive_requests", "fastcgi_param", "LogFormat", "TimeoutSec",
    "OOMScoreAdjust", "vm.swappiness", "net.ipv4", "MACs", "KexAlgorithms",
    "PermitRootLogin", "session_required",
]

# find -name globs (busybox & GNU safe; no shell metachars beyond *).
GLOB_LEXICON = [
    "*.conf", "*.d", "*.so*", "*.sh", "*.cfg", "*.txt", "*.py", "*.list",
    "*.service", "lib*", "*ssl*", "*.pem", "*.crt", "*rc", "*.gz", "*.h",
    "*.json", "*.ini", "os-release", "*.cnf",
]

HEADTAIL_KS = (3, 5, 10, 20)
V2_NEW_VERBS = ("head", "tail", "stat", "find", "grep")
# Review-B F7: linkage evidence must survive the e5 256-token encoder window — roughly the
# first ~1000 chars of the RENDER, which prefixes cwd/exit before the output. Mining only
# output[:500] keeps every mined fact safely inside what the encoder actually sees.
V2_MINE_CAP = 500

BENCH_VERSION = "dockerfs2-v2.0"
# Frozen verb-class table (constitution §4; dockerfs2-prereg Amendment 2 AS AMENDED BY
# Amendment 3 — stat reclassified semi-echo). Written verbatim into summary.json's v2 block
# by collect(); evolve/bench_versions.py asserts a root's recorded classes match (F9).
V2_VERB_CLASSES = {
    "content": ["ls", "cat", "head", "tail", "find", "grep"],
    "grep_mode_rule": "exit!=0 or empty output => miss (excluded)",
    "semi_echo": ["stat"],
    "excluded": ["uname", "cd"],
}

# 9-verb loop mix. Old loop mass = .17+.23+.16+.04 = .60; plus 3 forced old opener steps,
# realized old-verb mass ~ (3 + (L-3)*.60)/L = 65% at L=24 (>= 60%). Unavailable new verbs
# redistribute to cat, and orient flips add ls -> old mass only grows.
V2_WEIGHTS = {"cd": 0.17, "ls": 0.23, "cat": 0.16, "config": 0.04,
              "head": 0.08, "tail": 0.07, "stat": 0.07, "find": 0.08, "grep": 0.10}

_STRATUM_W = {"etc": 0.40, "usr": 0.40, "other": 0.20}  # config-heavy vs structure-heavy coverage
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]{3,23}")
_PROBE_SIZE_CAP = 65536   # line-count only files <= 64 KB (8s timeout headroom)
_PROBE_MAX_STAT = 3000
_PROBE_MAX_WC = 1200
_PROBE_CHUNK = 200
_PROBE_MAX_FIND_DIRS = 64   # review-B2 hard budget (was 16: collapsed glob-sparse images)
_PROBE_MIN_HIT_PAIRS = 10   # iterate probe dirs until this floor of distinct (dir,glob) hits
_PROBE_ANCHOR_DIRS = ("/etc", "/usr/share", "/usr/lib")  # always probed first (glob-rich)


def _stable(s):
    """Deterministic string hash (zlib.crc32; builtin hash() is salted per-process)."""
    return zlib.crc32(s.encode("utf-8", "ignore"))


def _sq(s):
    """POSIX single-quote (grep tokens / globs embedded in recorded commands)."""
    return "'" + str(s).replace("'", "'\\''") + "'"


def _parent(p):
    p = p.rstrip("/") or "/"
    return "/" if p == "/" else (p.rsplit("/", 1)[0] or "/")


def v2_k_for(path):
    """Review-A fix: ONE K per file, crc32-assigned over the full KSET GLOBALLY
    (image-independent — the same shared file gets the same K on every image, preserving
    cross-image command identity for the class protocol). The 2K+1 line floor is enforced
    by REJECTION from the head/tail pool (_v2_probe), never by K reassignment."""
    return HEADTAIL_KS[_stable(path) % len(HEADTAIL_KS)]


def lexicon_hashes():
    """Content hashes for summary.json (constitution S1: lexicons are version identity)."""
    def h(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
    return {"lexicon_version": V2_LEXICON_VERSION,
            "query_lexicon_sha256": h(QUERY_LEXICON),
            "miss_lexicon_sha256": h(MISS_LEXICON),
            "glob_lexicon_sha256": h(GLOB_LEXICON),
            "headtail_ks": list(HEADTAIL_KS),
            "verb_weights": V2_WEIGHTS}


def _find_probe_script(d):
    """F2: one in-band probe script per candidate dir — for every glob in GLOB_LEXICON, run
    the loosest mint-form find (-maxdepth 3, no -type) and echo `glob|first-hit` when it
    hits. `head -1` short-circuits; never a recorded step."""
    globs = " ".join(_sq(g) for g in GLOB_LEXICON)
    return ("for g in " + globs + "; do h=$(find " + _sq(d)
            + " -maxdepth 3 -name \"$g\" 2>/dev/null | head -1); "
            + "[ -n \"$h\" ] && echo \"$g|$h\"; done")


def _stratum(p):
    if p.startswith("/etc"):
        return "etc"
    if p.startswith(("/usr", "/lib", "/bin", "/sbin")):
        return "usr"
    return "other"


def _make_cycler(pool, rng):
    """Stratified (etc/usr/other) shuffled cycler: dense coverage of both the config-heavy
    /etc and structure-heavy /usr subtrees, no uniform-teleport bias."""
    strata = {}
    for p in pool:
        strata.setdefault(_stratum(p), []).append(p)
    for s in strata.values():
        rng.shuffle(s)
    order = [s for s in ("etc", "usr", "other") if strata.get(s)]
    idx = {s: 0 for s in order}
    weights = [_STRATUM_W[s] for s in order]

    def nxt():
        if not order:
            return None
        s = rng.choices(order, weights=weights)[0]
        p = strata[s][idx[s] % len(strata[s])]
        idx[s] += 1
        return p
    return nxt


def _p_link(counter, target):
    """Small proportional controller steering the REALIZED linkage rate to the prereg
    target (early-sequence linked pools are thin; a constant draw prob would undershoot).
    Deterministic; meta.linked always records ground truth."""
    lk, tot = counter
    if tot == 0:
        return target
    return min(0.97, max(0.20, target + (target - lk / tot) * 2.5))


def _v2_probe(box, dirs, files):
    """Once per image (cached on box, like _levy_state): `command -v` availability probe
    (in-band, never a recorded step; skips recorded), file sizes, line counts, and the
    head/tail pool (global-K files passing the 2K+1 floor by rejection). Uses NO rng
    (candidate selection is crc32-sorted) -> image-deterministic. Also owns the per-image
    verb/linkage counters merged into summary.json via v2_image_report()."""
    st = getattr(box, "_v2_state", None)
    if st is not None:
        return st

    # -- per-image verb availability (command -v), skips recorded --
    out, _, _ = box._exec(
        "for t in head tail stat find grep; do command -v $t >/dev/null 2>&1 && echo $t; done")
    present = set(out.split())
    avail = {v: (v in present) for v in V2_NEW_VERBS}
    skipped = [v for v in V2_NEW_VERBS if not avail[v]]

    # -- file sizes (metadata only; fast) --
    sizes = {}
    cand = sorted(files, key=_stable)[:_PROBE_MAX_STAT]
    for i in range(0, len(cand), _PROBE_CHUNK):
        chunk = cand[i:i + _PROBE_CHUNK]
        if avail["stat"]:
            out, _, _ = box._exec("stat -c '%s %n' " + " ".join(_sq(p) for p in chunk)
                                  + " 2>/dev/null")
        else:
            out, _, _ = box._exec("wc -c " + " ".join(_sq(p) for p in chunk) + " 2>/dev/null")
        for ln in out.split("\n"):
            parts = ln.strip().split(None, 1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].startswith("/"):
                sizes[parts[1]] = int(parts[0])

    # -- line counts on small files -> head/tail eligibility --
    lines = {}
    wc_cand = [p for p in cand if 0 < sizes.get(p, -1) <= _PROBE_SIZE_CAP][:_PROBE_MAX_WC]
    for i in range(0, len(wc_cand), _PROBE_CHUNK):
        chunk = wc_cand[i:i + _PROBE_CHUNK]
        out, _, _ = box._exec("wc -l " + " ".join(_sq(p) for p in chunk) + " 2>/dev/null")
        for ln in out.split("\n"):
            parts = ln.strip().split(None, 1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].startswith("/"):
                lines[parts[1]] = int(parts[0])

    # -- head/tail pool: 2K+1 line floor by REJECTION against the file's GLOBAL K --
    # 2K+1 guarantees: head -n K f != cat f (no head==cat dup in the foil pool), and
    # head -n K f is line-disjoint from tail -n K f (no head==tail dup).
    kmap, k_buckets = {}, {k: [] for k in HEADTAIL_KS}
    for p in wc_cand:
        k = v2_k_for(p)
        if lines.get(p, 0) >= 2 * k + 1:
            kmap[p] = k
            k_buckets[k].append(p)

    # -- size bands for cat target balance (the v1 truncation lesson) --
    bands = {"s": [], "m": [], "l": [], "u": []}
    for p in files:
        sz = sizes.get(p)
        if sz is None:
            bands["u"].append(p)
        elif sz <= 1024:
            bands["s"].append(p)
        elif sz <= 8192:
            bands["m"].append(p)
        else:
            bands["l"].append(p)

    small_files = [p for p in cand if 0 < sizes.get(p, -1) <= _PROBE_SIZE_CAP]

    # -- find-pair probe (review-B F2): verify (dir, glob) hit/empty pairs per image so the
    # find arm draws from ground truth instead of an independent dir x glob product
    # (365/374 pilot finds were empty). The first hit's depth bounds which -maxdepth values
    # are hit-verified (a hit at depth <= 2 hits at both 2 and 3); no output at -maxdepth 3
    # verifies empty at both. Deterministic: no rng, crc32-sorted dirs, lexicon-order globs.
    find_hits, find_empties = [], []   # (dir, glob, verified_mds, hit_type) / (dir, glob)
    files_set, dirs_set = set(files), set(dirs)
    probe_order = ([d for d in _PROBE_ANCHOR_DIRS if d in dirs_set]
                   + [d for d in sorted(dirs, key=_stable) if d not in _PROBE_ANCHOR_DIRS])
    probe_dirs = probe_order[:_PROBE_MAX_FIND_DIRS] if avail["find"] else []
    for d in probe_dirs:
        if len({(hd, hg) for hd, hg, _m, _t in find_hits}) >= _PROBE_MIN_HIT_PAIRS \
                and probe_dirs.index(d) >= len(_PROBE_ANCHOR_DIRS):
            break  # floor met after anchors — stop spending execs
        out, _, _ = box._exec(_find_probe_script(d))
        hit_by_glob = {}
        for ln in out.split("\n"):
            if "|" not in ln:
                continue
            g, hit = ln.split("|", 1)
            if g in GLOB_LEXICON and g not in hit_by_glob and hit.startswith("/"):
                hit_by_glob[g] = hit
        base = d.rstrip("/")
        for g in GLOB_LEXICON:
            hit = hit_by_glob.get(g)
            if hit is None:
                find_empties.append((d, g))
                continue
            rel = hit[len(base):].strip("/") if hit.startswith(base) else hit.strip("/")
            depth = rel.count("/") + 1 if rel else 0
            htype = "f" if hit in files_set else ("d" if hit in dirs_set else None)
            find_hits.append((d, g, (2, 3) if depth <= 2 else (3,), htype))

    st = {"avail": avail, "skipped": skipped, "sizes": sizes, "lines": lines,
          "kmap": kmap, "k_buckets": {k: v for k, v in k_buckets.items() if v},
          "cat_bands": {b: v for b, v in bands.items() if v},
          "small_files": small_files,
          "find_hits": find_hits, "find_empties": find_empties,
          "counters": {"verbs": {}, "linked": {}, "intended_miss": 0, "intended_empty": 0,
                       "steps": 0, "daemon_errs": 0}}
    box._v2_state = st
    return st


def v2_image_report(box):
    """Per-image availability/skip/verb/linkage counters for summary.json ({} for v1 policies)."""
    st = getattr(box, "_v2_state", None)
    if st is None:
        return {}
    c = st["counters"]
    return {"avail": dict(st["avail"]), "skipped_verbs": list(st["skipped"]),
            "verb_counts": dict(c["verbs"]), "linked_counts": dict(c["linked"]),
            "intended_miss": c["intended_miss"], "intended_empty": c["intended_empty"],
            "steps": c["steps"],
            "find_pool": {"hits": len(st["find_hits"]), "empties": len(st["find_empties"])}}


V2_STORE_CAP = 65536  # stored-output cap (review-B2: binary cats were ~60% of bytes at
                      # mint scale; render sees ~1KB — 64KB keeps fidelity far above the window)


def gen_sequence_v2(box, dirs, files, rng, length):
    """dockerfs2 mint policy (prereg Amendment 1 synthesis).

    9 verbs, old-verb mass >= 60%; history-linked head/tail/stat targets (~60%) and grep
    queries (~50%) with per-step meta {verb, linked, query_src, ...}; ~20% deliberate grep
    misses (transplant primary, small lexicon arm, never -i); global-K + 2K+1 rejection floor
    for head/tail; per-verb within-sequence used-sets; navreal grafts: descend into
    OBSERVED children + orient (ls) after arriving somewhere new; /etc-vs-/usr stratified
    cyclers; cat size-band balancing; busybox-and-GNU-safe flags; single commands (no pipes);
    linkage mined ONLY from the render-visible output prefix."""
    st = _v2_probe(box, dirs, files)
    steps = []
    files_set, dirs_set = set(files), set(dirs)

    # per-sequence history state (dicts as ordered sets: set iteration order is hash-salted
    # across processes -> never rng.choice over a set)
    seen_paths = {}     # path -> "ls_obs@i" | "find_obs@i"   (paths ENTAILED by earlier obs)
    file_tokens = {}    # file -> {token: "cat@i" | ...}      (tokens seen IN that file's obs)
    tokens_global = {}  # token -> (src_file, "cat@i")
    used = {"ht": set(), "stat": set(), "find": set(), "grep": set()}
    # [linked, total]; "hts" is SHARED across head/tail/stat (prereg: aggregate ~60% target)
    ctr = {"hts": [0, 0], "grep": [0, 0]}
    need_orient = [False]                  # navreal graft: just arrived somewhere un-listed

    dir_cyc = _make_cycler(dirs, rng)
    stat_cyc = _make_cycler(files + dirs, rng)
    cat_cycs = {b: _make_cycler(pool, rng) for b, pool in st["cat_bands"].items()}
    cat_band_order = [b for b in ("s", "m", "l", "u") if b in cat_cycs]
    cat_band_w = {"s": 0.35, "m": 0.35, "l": 0.10, "u": 0.20}

    def do(cmd, meta):
        s = box.run(cmd)
        if meta["verb"] in ("grep", "find"):
            # F8: record the REALIZED outcome at step-record time (authoritative rule:
            # exit != 0 => miss) so labels are recoverable from the jsonl alone.
            meta["hit"] = s["exit"] == 0 and bool(s["output"])
        if len(s["output"]) > V2_STORE_CAP:
            # Amendment 4: stored-output cap (binary payloads); render window unaffected
            s["output"] = s["output"][:V2_STORE_CAP]
            meta["trunc_stored"] = True
        s["meta"] = meta
        steps.append(s)
        c = st["counters"]
        c["verbs"][meta["verb"]] = c["verbs"].get(meta["verb"], 0) + 1
        if meta.get("linked"):
            c["linked"][meta["verb"]] = c["linked"].get(meta["verb"], 0) + 1
        if meta.get("intended_miss"):
            c["intended_miss"] += 1
        if meta.get("intended_empty"):
            c["intended_empty"] += 1
        c["steps"] += 1
        # F5 fail-fast: a dead container answers every exec with a daemon error — abort the
        # image rather than silently minting host-artifact observations.
        if s["output"].startswith("Error response from daemon"):
            c["daemon_errs"] += 1
            if c["daemon_errs"] >= 5:
                raise RuntimeError(f"dead container ({getattr(box, 'image', '?')}): "
                                   f"5 consecutive daemon errors, last cmd {cmd!r}")
        else:
            c["daemon_errs"] = 0
        return s

    def mine_ls(base, text, idx):
        for ln in text[:V2_MINE_CAP].split("\n"):
            toks = ln.split()
            if not toks:
                continue
            name = toks[-1]
            path = (base.rstrip("/") + "/" + name) if base != "/" else "/" + name
            if (path in files_set or path in dirs_set) and path not in seen_paths:
                seen_paths[path] = f"ls_obs@{idx}"

    def mine_find(text, idx):
        for ln in text[:V2_MINE_CAP].split("\n"):
            p = ln.strip()
            if (p in files_set or p in dirs_set) and p not in seen_paths:
                seen_paths[p] = f"find_obs@{idx}"

    def mine_tokens(fpath, text, idx, verb):
        ft = file_tokens.setdefault(fpath, {})
        for t in _TOKEN_RE.findall(text[:V2_MINE_CAP])[:60]:
            if t not in ft:
                ft[t] = f"{verb}@{idx}"
            if t not in tokens_global:
                tokens_global[t] = (fpath, f"{verb}@{idx}")

    def do_cat(src="size_band_cycler"):
        band = rng.choices(cat_band_order,
                           weights=[cat_band_w[b] for b in cat_band_order])[0] \
            if cat_band_order else None
        path = cat_cycs[band]() if band else (rng.choice(files) if files else None)
        if path is None:
            path = rng.choice(CONFIG_FILES)
        s = do("cat " + path, {"verb": "cat", "linked": False, "query_src": src, "band": band})
        if s["exit"] == 0 and s["output"]:
            mine_tokens(path, s["output"], len(steps) - 1, "cat")

    # ---- opening: system identity (old verbs, v1-continuous) ----
    m0 = {"verb": "uname", "linked": False, "query_src": "identify",
          "availability": dict(st["avail"]), "skipped_verbs": list(st["skipped"]),
          "lexicon_version": V2_LEXICON_VERSION}
    do("uname " + rng.choice(UNAME_OPTS), m0)
    cfg = rng.choice(CONFIG_FILES)
    s0 = do("cat " + cfg, {"verb": "cat", "linked": False, "query_src": "identify"})
    if s0["exit"] == 0 and s0["output"]:
        mine_tokens(cfg, s0["output"], len(steps) - 1, "cat")
    # third opener: seed the history-linkage pool BEFORE any new verb can fire
    # (old verb -> old mass rises; alternates config-heavy vs structure-heavy roots)
    root = rng.choice(["/etc", "/etc", "/usr/lib", "/usr/bin", "/usr/share"])
    s1 = do(f"ls -1 {root}", {"verb": "ls", "linked": False, "query_src": "seed_pool"})
    if s1["exit"] == 0 and s1["output"]:
        mine_ls(root, s1["output"], len(steps) - 1)

    # availability-adjusted verb weights (skips -> cat; old mass only grows)
    w = dict(V2_WEIGHTS)
    for v in st["skipped"]:
        w["cat"] += w.pop(v)
    if not st["kmap"]:                      # tiny image: no head/tail-eligible files
        for v in ("head", "tail"):
            if v in w:
                w["cat"] += w.pop(v)
    if not st["find_hits"] and not st["find_empties"] and "find" in w:
        w["cat"] += w.pop("find")           # F2: no probe-verified pairs -> find untargetable
    verbs, weights = list(w), [w[v] for v in w]

    for _ in range(max(0, length - 3)):
        act = rng.choices(verbs, weights=weights)[0]
        if need_orient[0] and act != "ls" and rng.random() < 0.35:
            act = "ls"                      # navreal graft: operator orients after arriving
        idx = len(steps)

        if act == "cd":
            cwd = box.cwd
            obs_kids = [d for d in seen_paths if d in dirs_set and _parent(d) == cwd]
            r = rng.random()
            if obs_kids and r < 0.50:       # navreal graft: descend into an OBSERVED child
                tgt = rng.choice(obs_kids)
                m = {"verb": "cd", "linked": True, "query_src": seen_paths[tgt],
                     "nav": "observed_child"}
            elif r < 0.65 and cwd != "/":
                tgt = ".."
                m = {"verb": "cd", "linked": False, "query_src": "nav_ascend", "nav": "ascend"}
            elif dirs and r < 0.90:
                tgt = dir_cyc() or "/"
                m = {"verb": "cd", "linked": False, "query_src": "nav_cycler", "nav": "cycler"}
            else:
                tgt = rng.choice(["..", ".", "/", cwd])
                m = {"verb": "cd", "linked": False, "query_src": "nav_reset", "nav": "reset"}
            do(f"cd {tgt}", m)
            if box.cwd != cwd:
                need_orient[0] = True

        elif act == "ls":
            opt = rng.choice(LS_OPTS)
            base = box.cwd
            tgt = ""
            if need_orient[0]:              # orient: list where we just arrived
                src = "orient_cwd"
                need_orient[0] = False
            elif dirs and rng.random() < 0.45:
                base = dir_cyc()
                tgt = " " + base
                src = "target_cycler"
            else:
                src = "cwd"
            s = do(f"ls {opt}{tgt}".strip(), {"verb": "ls", "linked": False, "query_src": src})
            if s["exit"] == 0 and s["output"] and "R" not in opt:  # -R: header noise, skip mining
                mine_ls(base, s["output"], idx)

        elif act == "cat":
            do_cat()

        elif act == "config":
            do("cat " + rng.choice(CONFIG_FILES),
               {"verb": "cat", "linked": False, "query_src": "config_lexicon"})

        elif act in ("head", "tail"):
            kmap = st["kmap"]
            linked_cands = [p for p in seen_paths if p in kmap and (act, p) not in used["ht"]]
            if linked_cands and rng.random() < _p_link(ctr["hts"], 0.60):
                by_k = {}
                for p in linked_cands:
                    by_k.setdefault(kmap[p], []).append(p)
                k = rng.choice(sorted(by_k))       # balance K bands (target-length balance)
                path = rng.choice(by_k[k])
                linked, src = True, seen_paths[path]
            else:
                bks = sorted(st["k_buckets"])
                path = None
                for _try in range(6):
                    k = rng.choice(bks)
                    p = rng.choice(st["k_buckets"][k])
                    if (act, p) not in used["ht"]:
                        path = p
                        break
                if path is None:
                    do_cat("headtail_fallback")
                    continue
                linked, src = False, "file_pool"
            used["ht"].add((act, path))
            ctr["hts"][0] += linked
            ctr["hts"][1] += 1
            s = do(f"{act} -n {kmap[path]} {path}",
                   {"verb": act, "linked": linked, "query_src": src, "k": kmap[path]})
            if s["exit"] == 0 and s["output"]:
                mine_tokens(path, s["output"], idx, act)

        elif act == "stat":
            cands = [p for p in seen_paths if p not in used["stat"]]
            if cands and rng.random() < _p_link(ctr["hts"], 0.60):
                path = rng.choice(cands)
                linked, src = True, seen_paths[path]
            else:
                path = None
                for _try in range(6):
                    p = stat_cyc()
                    if p and p not in used["stat"]:
                        path = p
                        break
                if path is None:
                    do_cat("stat_fallback")
                    continue
                linked, src = False, "path_pool"
            used["stat"].add(path)
            ctr["hts"][0] += linked
            ctr["hts"][1] += 1
            do(f"stat -c '%n %s %F %a' {path}",
               {"verb": "stat", "linked": linked, "query_src": src})

        elif act == "find":
            # F2: draw from the probe-verified pools — hit pairs by default, a deliberate-
            # empty arm at a controlled ~20% (mirrors the grep-miss design: absence is an
            # outcome; meta records intended_empty). -maxdepth comes only from the pair's
            # hit-verified depths; -type only matches the verified hit's type (empty pairs
            # take any modifier: subsets of empty stay empty).
            hits, empties = st["find_hits"], st["find_empties"]
            intended_empty = (rng.random() < 0.20 and bool(empties)) or not hits
            pool = empties if intended_empty else hits
            key = None
            for _try in range(6):
                if intended_empty:
                    d, g = rng.choice(pool)
                    md = rng.choice((2, 3))
                    ty = rng.choice(("", "-type f", "-type d"))
                else:
                    d, g, mds, htype = rng.choice(pool)
                    md = rng.choice(mds)
                    ty = rng.choice(("", f"-type {htype}")) if htype else ""
                key = (d, md, ty, g)
                if key not in used["find"]:
                    break
            used["find"].add(key)
            d, md, ty, g = key
            cmd = f"find {d} -maxdepth {md}" + (f" {ty}" if ty else "") + f" -name {_sq(g)}"
            s = do(cmd, {"verb": "find", "linked": False, "query_src": f"glob:{g}",
                         "pool": "empty" if intended_empty else "hit",
                         "intended_empty": bool(intended_empty)})
            if s["exit"] == 0 and s["output"]:
                mine_find(s["output"], idx)

        else:  # grep (Amendment 1: -c dropped; transplant-miss primary; no -i on misses)
            r = rng.random()
            tok = path = src = None
            linked = miss = False
            p_hit = _p_link(ctr["grep"], 0.50)
            if r < p_hit and file_tokens:
                # linked guaranteed-hit: token mined earlier from THIS file's obs
                fpool = [f for f in file_tokens if file_tokens[f]]
                if fpool:
                    fp = rng.choice(fpool)
                    tpool = [t for t in file_tokens[fp] if (t, fp) not in used["grep"]]
                    if tpool:
                        tok = rng.choice(tpool)
                        path = fp
                        linked, src = True, file_tokens[fp][tok]
            if tok is None and r < p_hit + 0.12 and tokens_global and st["small_files"]:
                # deliberate near-miss (PRIMARY miss arm): transplant a token OBSERVED in
                # file A into file B (token->file binding attack: plausible, file-absent)
                for _try in range(8):
                    t = rng.choice(list(tokens_global))
                    b = rng.choice(st["small_files"])
                    if tokens_global[t][0] != b and t not in file_tokens.get(b, {}) \
                            and (t, b) not in used["grep"]:
                        tok, path, miss = t, b, True
                        src = f"miss_transplant:{tokens_global[t][1]}"
                        break
            if tok is None and tokens_global and rng.random() < 0.50:
                # self-binding hit: grep a mined token in its OWN source file (entailed by
                # the earlier observation that surfaced it -> linked, guaranteed-hit-biased)
                for _try in range(8):
                    t = rng.choice(list(tokens_global))
                    b = tokens_global[t][0]
                    if (t, b) not in used["grep"]:
                        tok, path, linked = t, b, True
                        src = f"token_selfbind:{tokens_global[t][1]}"
                        break
            if tok is None and st["small_files"]:
                # lexicon probe (natural hit-or-miss) or the small lexicon-miss arm
                lex, tag = (MISS_LEXICON, "miss_lexicon") if rng.random() < 0.10 \
                    else (QUERY_LEXICON, "lexicon")
                for _try in range(8):
                    t = rng.choice(lex)
                    b = rng.choice(st["small_files"])
                    if (t, b) not in used["grep"]:
                        tok, path, src = t, b, tag
                        miss = tag == "miss_lexicon"
                        break
            if tok is None:
                do_cat("grep_fallback")
                continue
            used["grep"].add((tok, path))
            ctr["grep"][0] += linked
            ctr["grep"][1] += 1
            fl = "-F" if miss else ("-F -i" if rng.random() < 0.20 else "-F")
            s = do(f"grep {fl} -m 8 {_sq(tok)} {path}",
                   {"verb": "grep", "linked": linked, "query_src": src,
                    "intended_miss": miss, "flags": fl})
            if s["exit"] == 0 and s["output"]:
                mine_tokens(path, s["output"], idx, "grep")

    return steps


# ============================== dockerfs3 (v3.0) mint policy ==============================
# gen_sequence_v3 (benchmarks/dockerfs3-prereg.md §1 / draft §4-§7): the v2 flat
# verb-mixture wrapped in an event scheduler + a collection-mode ShellState (the ONE
# tracker, realenv.shell_state) driven alongside. The command UNIVERSE is exactly the
# frozen shell_state.parse_command grammar (v2 atomics + G3 depth-1 composition + the
# audited process forms) — EVERY emitted command parses or the fold raises (parser
# totality is a mint gate). MutGuard (homed here) re-validates every mutation string.
# Determinism: per-seq rng, canonical PIDs, store-time virtualization of every recorded
# byte, no wall-clock in the jsonl (dur_ms -> timing side-channel). v1/v2 policies are
# byte-identical: v3 is a NEW POLICIES entry and _step_record's meta passthrough is
# additive (a v1/v2 step carries no v3 meta keys, so its record is unchanged).

BENCH_VERSION_V3 = "dockerfs3-v3.0"

# --- the prereg §1 SSOT weight table (copied VERBATIM; total = 1.000 exactly) ---
V3_WEIGHTS = {
    "cd": 0.105, "ls": 0.140, "cat": 0.100, "config": 0.025,
    "head": 0.048, "tail": 0.042, "stat": 0.042, "find": 0.048, "grep": 0.060,
    "pwd": 0.031, "uptime": 0.010, "sleep": 0.005,
    "m_rm": 0.035, "m_mv": 0.035, "m_ln": 0.022, "m_redirect": 0.050,
    "mkdir_touch": 0.012, "pipe": 0.075, "cond": 0.025,
    "after": 0.030, "ps": 0.030, "kill": 0.020, "echo_bare": 0.010,
}
_V3_ATOMIC = {"cd", "ls", "cat", "config", "head", "tail", "stat", "find", "grep"}
_V3_MUTATION = {"m_rm", "m_mv", "m_ln", "mkdir_touch"}
_V3_COMPOSITION = {"pipe", "cond", "m_redirect"}
_V3_TIME = {"after", "ps", "kill", "uptime", "sleep"}
# the ablate arm turns mutation / composition / time OFF and renormalizes the rest
# (atomic + pwd + echo-bare); TRAIN-ONLY (prereg §1 / §6, --train-only enforced).
_V3_ABLATE_OFF = _V3_MUTATION | _V3_COMPOSITION | _V3_TIME
assert abs(sum(V3_WEIGHTS.values()) - 1.0) < 1e-9, sum(V3_WEIGHTS.values())

V3_SEQ_LEN = 28          # 28 +/- 4, max 32 = 64 tokens (pos_emb(64) — never exceed)
V3_SEQ_MAX = 32
V3_MAX_MUTATIONS = 8     # draft §4.2 caps
V3_MAX_TIER_S = 4
V3_MAX_JOBS = 3
V3_INTERVENTIONS = (4, 8)   # 6 +/- 2 interventions/seq (controller target, per-seq randint)
V3_LEX_ARM = 0.05        # <=5% audited-lexicon payload arm (meta.payload_src="lexicon")

# --- intervention-floor controller (P2 rate-tuning; the v2 `_p_link` twin) ---
# The 5 role=="intervene" arms (mutation block + the REDIR_W redirect). The nominal .154
# draw block realizes only ~2.4 interventions/seq (P2 measure) — a pure cap never reaches
# the 6+/-2 floor. `_p_intervene` steers the REALIZED interventions/seq up to the per-seq
# target, exactly mirroring `_p_link`'s proportional linkage steering.
_V3_INTERVENE = _V3_MUTATION | {"m_redirect"}
# The boost is funded ONLY from composition (pipe/cond) + time (after/ps/kill/uptime/sleep)
# — NEVER from atomic — so the load-bearing >=.60 atomic mass is preserved by construction
# (moving weight between the block and this pool leaves each atomic arm's weight untouched,
# and the total is conserved, so the atomic draw FRACTION is invariant to the boost).
_V3_INT_FUND = {"pipe", "cond"} | _V3_TIME
_V3_INT_KP = 3.5         # proportional gain (mirrors _p_link's 2.5 family); pilot 1.5..3.5.
#                          Controller is fund-limited (see below), so kp/draws_est barely move
#                          the realized rate (sweep: 4.46..4.56 across kp 2.5..3.5, est 10..14).
_V3_INT_DRAWS_EST = 11   # est. weight-draws/seq at target load (28 - 3 openers - ~14 revisit/
#                          fire steps once interventions are boosted). tgt_frac = target/est.
_V3_INT_PMAX = 0.60      # controller per-draw block-prob ceiling (like _p_link's 0.97 bound)
_V3_INT_PMIN = 0.05
_V3_INT_FUND_MAX = 1.0   # max fraction of the comp/time pool a single draw may borrow (the
#                          fund>=0 clamp; the achievable-max bound when atomic is preserved)


def _p_intervene(n_mut, draws, target, draws_est=_V3_INT_DRAWS_EST, kp=_V3_INT_KP):
    """Proportional controller (the v2 `_p_link` twin) steering the REALIZED interventions/seq
    to `target` (6+/-2). The nominal .154 intervene block realizes only ~2.4/seq; a pure cap
    never reaches the floor. Returns the per-DRAW probability the intervene block should carry
    this step; the caller funds any boost from composition/time and clamps the fund pool >= 0,
    so the effective probability is min(this, the fund-availability cap). Deterministic
    (per-seq state only); meta.mut_affected / n_mut always record ground truth."""
    tgt_frac = target / max(1, draws_est)
    if draws <= 0:
        return min(_V3_INT_PMAX, max(_V3_INT_PMIN, tgt_frac))
    realized = n_mut / draws
    return min(_V3_INT_PMAX, max(_V3_INT_PMIN, tgt_frac + (tgt_frac - realized) * kp))

_ARTIFACT_BUSYBOX = str(pathlib.Path(__file__).resolve().parent.parent
                        / "benchmarks" / "artifacts" / "busybox-arm64")
_P0_TEMPLATES_PATH = (pathlib.Path(__file__).resolve().parent.parent
                      / "benchmarks" / "p0" / "error-templates.json")

# normalized workspace-filename lexicon (sha-hashed selection -> cross-image command
# identity; content is image-specific, so the held-out-image split keeps biting).
WS_NAME_LEXICON = ("notes", "data", "conf", "readme", "list", "cache", "log",
                   "info", "meta", "record", "scratch", "entry", "value", "index")
# small audited payload lexicon (the <=5% arm, meta.payload_src="lexicon"). Every token
# passes shell_state._check_payload (leading letter, no backslash/$/backtick/dash).
PAYLOAD_LEXICON = ("alpha_tok", "beta_tok", "gamma_tok", "delta_tok", "epsilon_tok",
                   "marker_one", "marker_two", "seed_value", "audit_token", "probe_ref")

# --- the after helper + tj3-ps wrapper (draft §3.3 / Annex P0 UD-9 Route B) ---
AFTER_HELPER = (
    "#!/bin/sh\n"
    "exec </dev/null >/dev/null 2>&1\n"
    'j="$1"; shift 2\n'
    'mkfifo "/tmp/.tj/g$j" 2>/dev/null\n'
    'read _ < "/tmp/.tj/g$j"\n'
    "( sleep 5; kill -9 $$ ) &\n"
    "w=$!\n"
    'sh -c "$*"\n'
    ': > "/tmp/.tj/d$j"\n'
    'kill "$w" 2>/dev/null\n')
TJ3PS_WRAPPER = '#!/bin/sh\nexec /usr/local/bin/busybox-tj3 ps "$@"\n'


def _load_p0_templates():
    try:
        return json.loads(_P0_TEMPLATES_PATH.read_text())
    except (OSError, ValueError):
        return {}


_P0_TEMPLATES = _load_p0_templates()

# shell_state predict/mining consumes these keys with {path}/{pid} placeholders; the P0
# harvest keys the same facts as "<verb>_missing"/"rm_isdir" with a {p} placeholder and a
# literal probe pid (99999). Translate here so the SST reads its native shape (F3: it
# fails closed on a malformed entry, so a missing key is BOT, never a guess).
_TMPL_KEYMAP = {"cat": "cat_missing", "ls": "ls_missing", "head": "head_missing",
                "tail": "tail_missing", "stat": "stat_missing", "grep": "grep_missing",
                "cd": "cd_missing", "rm": "rm_missing", "mv": "mv_missing",
                "rm_isdir": "rm_isdir", "kill": "kill_missing"}


def sst_error_templates(image):
    """Per-image {sst_key: {template, exit}} for ShellState, translated from the P0
    harvest (benchmarks/p0/error-templates.json). {p}->{path}; kill's probe pid 99999
    ->{pid}; empty/malformed entries dropped (the SST then BOTs that surface)."""
    tbl = _P0_TEMPLATES.get(image, {})
    out = {}
    for sk, hk in _TMPL_KEYMAP.items():
        e = tbl.get(hk)
        if not isinstance(e, dict):
            continue
        t = e.get("template", e.get("text"))
        if not t or "exit" not in e:
            continue
        t = t.replace("99999", "{pid}") if sk == "kill" else t.replace("{p}", "{path}")
        out[sk] = {"template": t, "exit": e["exit"]}
    return out


# --- MutGuard (draft §4.4; homed here, collection-side only, never eval-path) ---

class MutGuardViolation(RuntimeError):
    """A mutation string escaped the templates/denylist/laws — a collector bug; the
    mint fails fast (the F5/daemon_errs precedent: a preregistered one-run mint must
    never silently emit an out-of-policy world-mutation)."""


# LOAD_BEARING denylist (draft §4.4 union). Workspace paths (/tmp/w/...) are exempt.
_LB_DIR_PREFIXES = ("/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/lib",
                    "/usr/libexec", "/usr/local/bin")
_LB_EXACT = frozenset({"/etc/passwd", "/etc/group", "/etc/shadow", "/etc/nsswitch.conf",
                       "/usr/local/bin/after", "/usr/local/bin/tj3-ps",
                       "/usr/local/bin/busybox-tj3"})
_LB_NAME_GLOBS = ("ld-", "ld.so", "libc", "libm", "libpthread", "musl", "busybox")


def _load_bearing(p, tool_paths=()):
    """Draft §4.4: infra artifacts, NSS/identity dbs, the tool tree and loader/musl/
    busybox libraries are untouchable (a passwd edit flips every later `ls -l` uid->name
    resolution image-wide — a dependency no path-keyed ledger can represent)."""
    if p in _LB_EXACT or p in tool_paths:
        return True
    if p == "/tmp/.tj" or p.startswith("/tmp/.tj/"):
        return True
    if p.startswith("/lib"):                 # /lib, /lib64, /libexec globs
        return True
    for pre in _LB_DIR_PREFIXES:
        if p == pre or p.startswith(pre + "/"):
            return True
    base = basename_of(p)
    return any(base.startswith(g) for g in _LB_NAME_GLOBS)


def mutguard_validate(cmd, cwd, tool_paths=(), state=None):
    """Re-validate a TEMPLATE-constructed mutation string (draft §4.4 + the round-6 law).
    op in the mutation whitelist; workspace or denylist-clean Tier-S targets only; `-r`
    only strictly under /tmp/w (depth>=2); no `-f`/globs/multi-redirect (the parser
    already bans these — checked here in depth); NEVER the cwd or an ancestor of it (the
    SST stale-cwd class, defended collection-side). MEDIUM-3: when `state` (the collection
    ShellState) is supplied, a workspace WRITE-THROUGH form (redir / after-effect / touch)
    whose target symlink-RESOLVES outside /tmp/w is rejected — an `ln -s <tier_s> /tmp/w/link`
    followed by `echo >> /tmp/w/link` would otherwise mutate a real image file. Raises
    MutGuardViolation on any breach. Returns the parse structure."""
    p = parse_command(cmd)               # ParseError => not the frozen universe
    form = p["form"]
    cwdn = normpath(cwd)
    if form == "redir":
        targets = [normpath(p["dst"], cwd)]
    elif form == "rm":
        tgt = normpath(p["path"], cwd)
        targets = [tgt]
        if p["recursive"] and not (tgt.startswith(WORKSPACE + "/") and tgt.count("/") >= 2):
            raise MutGuardViolation(f"rm -r only strictly under {WORKSPACE}/ (depth>=2): {cmd!r}")
    elif form == "mv":
        targets = [normpath(p["src"], cwd), normpath(p["dst"], cwd)]
    elif form == "ln":
        targets = [normpath(p["link"], cwd)]    # the LINK is the write; target is aliased
    elif form in ("mkdir", "touch"):
        targets = [normpath(p["path"], cwd)]
    elif form == "after":
        targets = [normpath(p["effect_parsed"]["dst"], cwd)]   # the /tmp/w task-log append
    else:
        raise MutGuardViolation(f"MutGuard on non-mutation form {form!r}: {cmd!r}")
    for tgt in targets:
        if cwdn == tgt or cwdn.startswith(tgt.rstrip("/") + "/"):
            raise MutGuardViolation(
                f"round-6 law: mutation of {tgt!r} covers cwd {cwdn!r}: {cmd!r}")
        if not (tgt == WORKSPACE or tgt.startswith(WORKSPACE + "/")):
            if _load_bearing(tgt, tool_paths):
                raise MutGuardViolation(f"LOAD_BEARING denylist hit: {tgt!r}: {cmd!r}")
        elif state is not None and form in ("redir", "after", "touch"):
            # MEDIUM-3: a workspace WRITE-THROUGH must land inside the arena. Resolve the
            # target's final-component symlink chain through the tracker; reject an escape
            # (a /tmp/w link pointing at a Tier-S file outside /tmp/w).
            r = state._resolve(tgt)
            if not (r == WORKSPACE or r.startswith(WORKSPACE + "/")):
                raise MutGuardViolation(
                    f"workspace write resolves outside {WORKSPACE}: {tgt!r}->{r!r}: {cmd!r}")
    return p


# --- container bootstrap (unrecorded; draft §3.3 / §4.2 / Annex P0) ---

def _seed_workspace(box, files, probe, rng):
    """Tier-W seeding (draft §4.2): copy a crc32-selected image-specific sample of small
    real files into /tmp/w under normalized names from WS_NAME_LEXICON. Cross-image
    command identity (same names), image-specific content (held-out split bites).
    Returns the manifest [(wsname, src, size)]; the collector hashes it into step-0 meta."""
    sizes = probe.get("sizes", {})
    cand = sorted((f for f in files
                   if 0 < sizes.get(f, 65537) <= 65536 and not _load_bearing(f)),
                  key=_stable)
    manifest = []
    used = set()
    for i, src in enumerate(cand[:12]):
        name = WS_NAME_LEXICON[_stable(src) % len(WS_NAME_LEXICON)]
        while name in used:                     # de-dup by appending an index
            name = WS_NAME_LEXICON[(_stable(src) + len(used)) % len(WS_NAME_LEXICON)] \
                + str(len(used))
        used.add(name)
        wsname = f"{WORKSPACE}/{name}.seed"
        if box.cp_in(src, wsname):
            manifest.append((wsname, src, sizes.get(src, 0)))
    return manifest


def v3_bootstrap(box, seed, files, probe, rng):
    """Unrecorded, once per container (draft §3.3): install the vendored busybox tj3-ps
    (UD-9 Route B), the tj3-ps wrapper, the `after` helper + /tmp/.tj dotdir, create the
    /tmp/w arena and seed it. Returns the bootstrap dict (availability, ws manifest hash,
    /bin/sh identity)."""
    box.cp_in(_ARTIFACT_BUSYBOX, "/usr/local/bin/busybox-tj3")
    box._exec("chmod +x /usr/local/bin/busybox-tj3 2>/dev/null")
    box._exec("printf %s " + _sq(TJ3PS_WRAPPER)
              + " > /usr/local/bin/tj3-ps && chmod +x /usr/local/bin/tj3-ps")
    box._exec("mkdir -p /tmp/.tj")
    wr, _, wcode = box._exec(
        "printf %s " + _sq(AFTER_HELPER)
        + " > /usr/local/bin/after && chmod +x /usr/local/bin/after && echo ok")
    after_ok = wr.strip().endswith("ok") and wcode == 0
    box._exec("mkdir -p " + WORKSPACE)
    manifest = _seed_workspace(box, files, probe, rng)
    man_hash = hashlib.sha256(
        json.dumps([[n, s, z] for n, s, z in manifest], sort_keys=True).encode()).hexdigest()
    shid, _, _ = box._exec("readlink -f /bin/sh 2>/dev/null || echo /bin/sh")
    return {"after_ok": after_ok, "ws_manifest": manifest, "ws_manifest_sha256": man_hash,
            "sh": shid.strip()}


# --- the v3 session (the event scheduler + tracker-driven arms) ---

_V3_TIME_FREE_LS = tuple(sorted(TIME_FREE_LS - {""}))    # ("-1","-1a","-a","-a1")
# mutation-adjacent / revisit ls forms drop -i/-lt/-lS (nondet template avoidance, §5.5).
# SEVERE-2 fix: `-R` DROPPED — a recursive listing at cwd=/ recurses /proc//sys//dev and
# renders live PID dirs (replay-varying) + risks the 8s timeout on fat images (DG-10b).
_V3_SAFE_LS = ("", "-l", "-a", "-la", "-1", "-lh", "-ld", "-lr", "-ln")

# SEVERE-2 fix: /proc, /sys, /dev are volatile pseudo-filesystems (PID churn, self-links,
# container-start mtimes) — the world model's world is the real IMAGE filesystem, so these
# are excluded from every flat-ls / find / cd target pool (below) and never listed directly.
_V3_VOLATILE_FS = ("/proc", "/sys", "/dev")


def _under_volatile_fs(p):
    """True iff p is one of /proc//sys//dev or lives under it (SEVERE-2 exclusion)."""
    return any(p == pre or p.startswith(pre + "/") for pre in _V3_VOLATILE_FS)


# v3 host-fingerprint exclusions (determinism re-review, 2026-07-23): the raw jsonl must be
# cross-host replayable and DG-3c date-scanner clean. /proc/version leaks the kernel build-date
# (a literal date -> DG-3c false positive) and lives under a volatile fs (SEVERE-2 consistency);
# /etc/resolv.conf carries the HOST DNS nameserver (flips on a different host). Both are dropped
# from the v3 cat pool. os-release/hostname/hosts/issue/passwd/group stay (image-constant,
# --hostname-pinned). These are v3-ONLY constants; the shared CONFIG_FILES/UNAME_OPTS are
# untouched so v1/v2 records stay byte-identical.
_V3_CONFIG_FILES = [c for c in CONFIG_FILES
                    if c not in ("/proc/version", "/etc/resolv.conf") and not _under_volatile_fs(c)]
# uname -a and -v embed the kernel build-date; every other form (-s/-m/-r/-n/-o/-sm/-sr/"") does not.
_V3_UNAME_OPTS = [o for o in UNAME_OPTS if o not in ("-a", "-v")]
# store-time mask for the kernel build-date in the top-level system_id (uname -a) — the one
# remaining date carrier once -a/-v are out of the verb pool. Preserves distro/arch/hostname
# identity (the syscond context) while removing the DG-3c-tripping literal date.
_KERNEL_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?\b")


def _canon_system_id(sysid):
    """Mask the kernel build-date token in a system_id string (store-time, v3 only)."""
    return _KERNEL_DATE_RE.sub("<date>", sysid)


def _is_ls_l_render(cmd):
    """True iff cmd is an atomic `-l`-family ls (a long listing whose rows carry a
    date/time triplet). Pipes / cond reads only use names-only `ls -1`, so a `-l`
    render is atomic-only — this is the store-time canon_ls_l_text gate in do()."""
    try:
        p = parse_command(cmd)
    except ParseError:
        return False
    if p.get("form") != "ls":
        return False
    opts = p.get("opts") or []
    return bool(opts) and "l" in opts[0]


class _V3Session:
    """One trajectory. Drives a collection-mode ShellState alongside the box: every
    recorded step is folded (predict BEFORE fold, meta.expected cached), pools route
    through the tracker's live view, and canonical PIDs / store-time virtualization make
    every recorded byte replay-deterministic."""

    def __init__(self, box, dirs, files, rng, length, image, error_templates,
                 boot, arm="full"):
        # SEVERE-2: exclude /proc//sys//dev from the navigable dir pool (cd targets,
        # flat-ls targets, observed-child mining) — the world is the real image fs.
        dirs = [d for d in dirs if not _under_volatile_fs(d)]
        self.box, self.dirs, self.files, self.rng = box, dirs, files, rng
        self.length = min(V3_SEQ_MAX, max(4, length))
        self.image, self.arm, self.boot = image, arm, boot
        self.st = ShellState(mode="collection", error_templates=error_templates)
        self.probe = _v2_probe(box, dirs, files)
        # SEVERE-2: the find pool is built once by _v2_probe (shared with the v2 policy,
        # so it is NOT filtered there — v2 byte-identity must hold). Drop volatile-fs base
        # dirs here so the v3 find arm never targets /proc//sys//dev.
        self.find_hits = [h for h in self.probe["find_hits"]
                          if not _under_volatile_fs(h[0])]
        self.find_empties = [e for e in self.probe["find_empties"]
                             if not _under_volatile_fs(e[0])]
        self.steps, self.timings = [], []
        self.daemon_errs = 0
        self.dirs_set, self.files_set = set(dirs), set(files)
        # tier-S expendable real files (denylist-clean, <=64KB), tracker-routed at draw
        sizes = self.probe.get("sizes", {})
        self.tier_s_pool = sorted(
            (f for f in files if 0 < sizes.get(f, 65537) <= 65536 and not _load_bearing(f)),
            key=_stable)
        # history-linkage pools (v2 mining discipline, render-visible prefix only)
        self.seen_paths, self.file_tokens, self.tokens_global = {}, {}, {}
        self.last_obs_of = {}                # path -> step idx of its last pre-mutation read
        self.dir_cyc = _make_cycler(dirs, rng)
        self.cat_cycs = {b: _make_cycler(pool, rng)
                         for b, pool in self.probe["cat_bands"].items()}
        self.cat_band_order = [b for b in ("s", "m", "l", "u") if b in self.cat_cycs]
        # world-mutation bookkeeping
        self.n_mut = self.n_tier_s = 0
        self._n_draws = 0                    # per-seq weight-draw counter (P2 controller)
        self.next_j = 1
        self.real_of = {}                    # canonical cpid -> real pid (kill/ps xlate)
        self.ws_files = []                   # workspace files this trajectory created
        self.ws_dirs = []                    # workspace dirs (mkdir; rm -r targets)
        self.pending = []                    # scheduled probes [(due_vt, fn)]
        self.pending_kill_barrier = None     # (realpid, kind) for the NEXT step's prologue
        self.counters = {"sigs": {}, "arms": {}, "linked": 0, "revisits": 0,
                         "intended_miss": 0, "mutations": 0, "jobs": 0, "skips": []}
        # availability-adjusted weights
        self.weights = self._arm_weights()

    # ---- weights / availability ----

    def _arm_weights(self):
        w = dict(V3_WEIGHTS)
        avail = self.probe["avail"]
        for v in ("head", "tail", "stat", "find", "grep"):     # v2 skip-redistribute
            if not avail.get(v, True):
                w["cat"] += w.pop(v, 0.0)
                self.counters["skips"].append(v)
        if not self.probe["kmap"]:
            for v in ("head", "tail"):
                w["cat"] += w.pop(v, 0.0)
        if not (self.find_hits or self.find_empties):    # SEVERE-2: post-exclusion pools
            w["cat"] += w.pop("find", 0.0)
        if not self.boot.get("after_ok"):                      # /usr/local/bin unwritable
            for v in ("after", "ps", "kill"):
                w["cat"] += w.pop(v, 0.0)
                self.counters["skips"].append(v)
        if self.arm == "ablate":                               # mutation/time/composition OFF
            for v in list(w):
                if v in _V3_ABLATE_OFF:
                    w.pop(v, None)
        tot = sum(w.values())
        return {k: v / tot for k, v in w.items()} if tot else w

    # ---- the recorder ----

    def vt(self):
        return len(self.steps)

    def do(self, cmd, meta, prologue="", extra_timeout=0, exec_cmd=None,
           store_out=None, store_exit=None, target=None, after_j=None):
        """Record one step: the CANONICAL command string `cmd` (folded through the SST),
        an optional distinct `exec_cmd` executed in-container (PID de-virtualization),
        optional store-time virtualized (store_out/store_exit) bytes, and (after_j) the
        `echo $!` -> canonical-PID virtualization for a bgjob launch. Prepends the UNRECORDED
        fire/barrier prologue for jobs due at this step (draft §3.2). Fills the F8 meta
        columns (sig/mode/expected/state_scope/hit/delta_text), folds, fail-fasts on
        daemon/timeout/125 errors."""
        vt = self.vt()
        assert self.st.vt == vt, (self.st.vt, vt)
        try:
            meta["sig"] = verbsig.sig(cmd)
        except ValueError:
            raise MutGuardViolation(f"emitted out-of-universe command: {cmd!r}")
        meta.setdefault("verb", "ps" if cmd.startswith(PS_PATH) else cmd.split()[0])
        pred = self.st.predict(vt, cmd)          # pure; BEFORE fold (fold-order discipline)
        meta["expected"] = "bot" if pred is None else \
            ("ok" if pred["exit"] == 0 else "err%d" % pred["exit"])
        if target is not None:
            meta["state_scope"] = self.st.state_scope_of(target)   # pre-fold scope
            meta.setdefault("pre_obs_step", self.last_obs_of.get(
                normpath(target, self.st.cwd), -1))
        else:
            meta.setdefault("state_scope", "native")
        # UNRECORDED prologue: fire due jobs + a pending post-signal barrier (draft §3.2/§3.3)
        fpro, fextra = self._fire_prologue(vt)
        merged = "; ".join(x for x in (fpro, prologue) if x)
        s = self.box.run(exec_cmd if exec_cmd is not None else cmd,
                         prologue=merged, extra_timeout=extra_timeout + fextra)
        s["cmd"] = cmd                            # recorded == canonical, not exec_cmd
        if after_j is not None:
            cpid = canonical_pid(after_j)
            raw = s["output"].strip().split("\n")[-1] if s["output"].strip() else ""
            if s["exit"] == 0 and raw.isdigit():
                self.real_of[cpid] = raw
                s["output"], s["exit"] = str(cpid), 0    # store-time PID virtualization
            # else: preserve the real failure — SST._fold_after won't register the job, so no
            # fire is scheduled and belief stays consistent (mint-integrity)
        elif store_out is not None:
            s["output"] = store_out
        if store_exit is not None:
            s["exit"] = store_exit
        dur = s.pop("dur_ms", 0)
        out = s["output"]
        if out.startswith("Error response from daemon"):
            self.daemon_errs += 1
            if self.daemon_errs >= 5:
                raise RuntimeError(f"dead container ({self.image}): 5 daemon errors")
        else:
            self.daemon_errs = 0
        if s["exit"] == 124:
            raise RuntimeError(f"timeout at step {vt} ({self.image}): {cmd!r} (DG-10b)")
        if s["exit"] == 125:
            raise RuntimeError(f"host executor error at step {vt} ({self.image}): {cmd!r}")
        if len(out) > V2_STORE_CAP:
            s["output"] = out[:V2_STORE_CAP]
            meta["trunc_stored"] = True
        # SEVERE-1 STORE-TIME canon: an `-l`-family ls render carries dir/file mtimes, and
        # dirs like /, /etc, /tmp are docker-bind-mounted / bootstrap-touched at CONTAINER-
        # CREATION time (not image-constant), so their mtime differs across a twin mint. Mask
        # the date/time triplet on EVERY long-listing row BEFORE the step is stored, so the RAW
        # jsonl is replay-deterministic (DG-3a diffs raw bytes, before render_canon can rescue
        # it). predict() is BOT for -l (R6 _ls_predict), so this never touches collector<->SST
        # golden-rule parity; the 3-token mask preserves the 9-field row the -l splice reads.
        # render_canon re-applies canon_ls_l_text at encode time (defense in depth). uptime/ps
        # renders are already store-virtualized to canonical bytes above.
        if _is_ls_l_render(cmd):
            s["output"] = canon_ls_l_text(s["output"])
        meta["mode"] = verbsig.mode(meta["sig"], s["exit"], not s["output"])
        if meta["sig"] in ("grep", "find", "pipe:ls|grep", "pipe:cat|grep",
                           "cond:cat", "cond:ls", "cond:head"):
            meta["hit"] = s["exit"] == 0 and bool(s["output"])
        s["meta"] = meta
        self.steps.append(s)
        self.timings.append(dur)
        self.st.fold(s)                          # advance the ONE tracker
        meta["delta_text"] = self.st.delta_text()
        self.counters["sigs"][meta["sig"]] = self.counters["sigs"].get(meta["sig"], 0) + 1
        self.counters["arms"][meta.get("arm", "?")] = \
            self.counters["arms"].get(meta.get("arm", "?"), 0) + 1
        if meta.get("linked"):
            self.counters["linked"] += 1
        if meta.get("role") == "revisit":
            self.counters["revisits"] += 1
        if meta.get("intended_miss") or meta.get("intended_outcome", "").startswith("miss"):
            self.counters["intended_miss"] += 1
        return s

    # ---- mining (v2 discipline: render-visible prefix only) ----

    def _mine_ls(self, base, text, idx):
        for ln in text[:V2_MINE_CAP].split("\n"):
            toks = ln.split()
            if not toks:
                continue
            name = toks[-1]
            path = (base.rstrip("/") + "/" + name) if base != "/" else "/" + name
            if (path in self.files_set or path in self.dirs_set) \
                    and path not in self.seen_paths:
                self.seen_paths[path] = f"ls_obs@{idx}"

    def _mine_find(self, text, idx):
        for ln in text[:V2_MINE_CAP].split("\n"):
            p = ln.strip()
            if (p in self.files_set or p in self.dirs_set) and p not in self.seen_paths:
                self.seen_paths[p] = f"find_obs@{idx}"

    def _mine_tokens(self, fpath, text, idx, verb):
        ft = self.file_tokens.setdefault(fpath, {})
        for t in _TOKEN_RE.findall(text[:V2_MINE_CAP])[:60]:
            if t not in ft:
                ft[t] = f"{verb}@{idx}"
            if t not in self.tokens_global:
                self.tokens_global[t] = (fpath, f"{verb}@{idx}")

    # ---- tracker-routed pool helpers (§4.5) ----

    def _dead(self, path):
        r = self.st._resolve(normpath(path, self.st.cwd))
        return self.st._stat(r)[0] == "dead"

    def _live_pool(self, paths):
        return [p for p in paths if not self._dead(p)]

    def _resolves_in_ws(self, path):
        """MEDIUM-3: a /tmp/w path whose symlink-resolved node stays inside the arena —
        safe to WRITE THROUGH. An append (`echo >> /tmp/w/link`) onto a symlink that
        resolves to a Tier-S file OUTSIDE /tmp/w would mutate a real image file's
        bytes/mtime (feeding the SEVERE-1 ls -l nondeterminism on a real path)."""
        r = self.st._resolve(normpath(path, self.st.cwd))
        return r == WORKSPACE or r.startswith(WORKSPACE + "/")

    def _mined_payload(self):
        """DG-1 provenance law (draft §4.3): an echo payload is a token mined from THIS
        trajectory's render-visible prefix; the deliberate <=5% audited-lexicon arm carries
        payload_src='lexicon'. MINOR-5 fix: the unavoidable seed payload (a payload write in
        the first slots, before any read has mined a token) is a PAYLOAD_LEXICON draw exactly
        like the audited arm and is indistinguishable in the record, so it is labelled
        'lexicon' too — folded into the <=5% DG-1 lexicon gate (the gate is what hard-caps the
        realized lexicon fraction) rather than escaping it under a separate 'lexicon_seed'."""
        if self.tokens_global:
            if self.rng.random() < V3_LEX_ARM:
                return self.rng.choice(PAYLOAD_LEXICON), "lexicon"
            tok, (src, prov) = self.rng.choice(list(self.tokens_global.items()))
            return tok, f"mined:{prov}"
        return self.rng.choice(PAYLOAD_LEXICON), "lexicon"

    # ---- openers (3 v2-style, old-verb mass) ----

    def _open(self):
        self.do("uname " + self.rng.choice(_V3_UNAME_OPTS),
                {"arm": "open", "role": "identify"})
        cfg = self.rng.choice([c for c in _V3_CONFIG_FILES if not self._dead(c)] or _V3_CONFIG_FILES)
        s = self.do("cat " + cfg, {"arm": "open", "role": "identify"}, target=cfg)
        if s["exit"] == 0 and s["output"]:
            self._mine_tokens(cfg, s["output"], len(self.steps) - 1, "cat")
            self.last_obs_of[normpath(cfg)] = len(self.steps) - 1
        root = self.rng.choice(["/etc", "/etc", "/usr/lib", "/usr/bin", "/usr/share"])
        s = self.do(f"ls -1 {root}", {"arm": "open", "role": "seed_pool"}, target=root)
        if s["exit"] == 0 and s["output"]:
            self._mine_ls(root, s["output"], len(self.steps) - 1)

    # ---- flat read arms ----

    def _cat(self, arm="cat", config=False):
        if config:
            pool = self._live_pool(_V3_CONFIG_FILES) or _V3_CONFIG_FILES
            path = self.rng.choice(pool)
            src = "config_lexicon"
        else:
            band = self.rng.choices(
                self.cat_band_order,
                weights=[{"s": .35, "m": .35, "l": .10, "u": .20}[b]
                         for b in self.cat_band_order])[0] if self.cat_band_order else None
            path = self.cat_cycs[band]() if band else None
            if path is None or self._dead(path):
                pool = self._live_pool(self.files) or _V3_CONFIG_FILES
                path = self.rng.choice(pool)
            src = "size_band_cycler"
        s = self.do("cat " + path, {"arm": arm, "role": "read", "query_src": src},
                    target=path)
        if s["exit"] == 0 and s["output"]:
            self._mine_tokens(path, s["output"], len(self.steps) - 1, "cat")
            self.last_obs_of[normpath(path, self.st.cwd)] = len(self.steps) - 1

    def _ls(self, opt=None, target=None, arm="ls", role="read"):
        opt = self.rng.choice(_V3_SAFE_LS) if opt is None else opt
        base = self.st.cwd
        tgt = ""
        if target is not None:
            base, tgt = target, " " + target
        elif self.dirs and self.rng.random() < 0.45:
            base = self.dir_cyc() or self.st.cwd
            if base and not self._dead(base):
                tgt = " " + base
            else:
                base = self.st.cwd
        # SEVERE-2: an `-l`-family listing of the CONTENTS of "/" renders the /proc//sys//dev
        # mount rows, whose link-count (= live PID/subdir count for /proc) is replay-VOLATILE
        # even after the store-time time mask (and can shift column widths). "/" is the only
        # dir holding these pseudo-fs mounts — downgrade a root content-listing -l to a
        # names-only form (the pseudo-fs NAMES are constant; only the -l metadata churns). `-ld`
        # lists "/" ITSELF (link-count = the fixed subdir count of /, time already masked) so it
        # is left intact. The names-only opt stays inside the frozen universe (`` / `-a`).
        if "l" in opt and "d" not in opt and normpath(base, self.st.cwd) == "/":
            opt = "-a" if "a" in opt else ""
        s = self.do(f"ls {opt}{tgt}".strip(), {"arm": arm, "role": role,
                                               "query_src": "cwd" if not tgt else "target"},
                    target=base if tgt else None)
        if s["exit"] == 0 and s["output"] and "R" not in opt:
            self._mine_ls(base, s["output"], len(self.steps) - 1)
        return s

    def _cd(self):
        cwd = self.st.cwd
        obs_kids = [d for d in self._live_pool(self.seen_paths)
                    if d in self.dirs_set and parent_of(d) == cwd]
        r = self.rng.random()
        if obs_kids and r < 0.50:
            tgt, nav = self.rng.choice(obs_kids), "observed_child"
        elif r < 0.62 and cwd != "/":
            tgt, nav = "..", "ascend"
        elif self.dirs and r < 0.90:
            tgt = self.dir_cyc() or "/"
            if self._dead(tgt):
                tgt = "/"
            nav = "cycler"
        else:
            tgt, nav = "/", "reset"
        # round-6 defense: never let cd's own move be into a path we might later mutate —
        # cd is not a mutation, but the tracker keeps cwd honest either way.
        self.do(f"cd {tgt}", {"arm": "cd", "role": "nav", "nav": nav,
                              "linked": nav == "observed_child",
                              "query_src": self.seen_paths.get(tgt, nav)})

    def _pwd(self):
        self.do("pwd", {"arm": "pwd", "role": "nav"})

    def _headtail(self, verb):
        kmap = self.probe["kmap"]
        linked = [p for p in self._live_pool(self.seen_paths) if p in kmap]
        p_link = _p_link([self.counters["linked"], max(1, len(self.steps))], 0.60)
        if linked and self.rng.random() < p_link:
            path = self.rng.choice(linked)
            is_linked, src = True, self.seen_paths[path]
        else:
            bucket = self._live_pool([p for k in self.probe["k_buckets"]
                                      for p in self.probe["k_buckets"][k]])
            if not bucket:
                return self._cat("headtail_fallback")
            path, is_linked, src = self.rng.choice(bucket), False, "file_pool"
        k = kmap[path]
        s = self.do(f"{verb} -n {k} {path}",
                    {"arm": verb, "role": "read", "linked": is_linked,
                     "query_src": src, "k": k}, target=path)
        if s["exit"] == 0 and s["output"]:
            self._mine_tokens(path, s["output"], len(self.steps) - 1, verb)

    def _stat(self):
        cands = self._live_pool([p for p in self.seen_paths]) or \
            self._live_pool(self.files + self.dirs)
        if not cands:
            return self._cat("stat_fallback")
        path = self.rng.choice(cands)
        self.do(f"stat -c '%n %s %F %a' {path}",
                {"arm": "stat", "role": "read",
                 "linked": path in self.seen_paths,
                 "query_src": self.seen_paths.get(path, "path_pool")}, target=path)

    def _find(self):
        hits, empties = self.find_hits, self.find_empties   # SEVERE-2: /proc//sys//dev excluded
        intended_empty = (self.rng.random() < 0.20 and bool(empties)) or not hits
        if intended_empty and not empties:
            return self._cat("find_fallback")
        if intended_empty:
            d, g = self.rng.choice(empties)
            md, ty = self.rng.choice((2, 3)), self.rng.choice(("", "-type f", "-type d"))
        else:
            d, g, mds, htype = self.rng.choice(hits)
            md = self.rng.choice(mds)
            ty = self.rng.choice(("", f"-type {htype}")) if htype else ""
        cmd = f"find {d} -maxdepth {md}" + (f" {ty}" if ty else "") + f" -name {_sq(g)}"
        s = self.do(cmd, {"arm": "find", "role": "read", "query_src": f"glob:{g}",
                          "intended_outcome": "miss-never-mutated" if intended_empty else "hit"},
                    target=d)
        if s["exit"] == 0 and s["output"]:
            self._mine_find(s["output"], len(self.steps) - 1)

    def _grep(self):
        small = self._live_pool(self.probe["small_files"])
        r = self.rng.random()
        tok = path = src = None
        linked = miss = False
        fpool = [f for f in self.file_tokens if self.file_tokens[f] and not self._dead(f)]
        if r < 0.55 and fpool:
            fp = self.rng.choice(fpool)
            tpool = list(self.file_tokens[fp])
            if tpool:
                tok = self.rng.choice(tpool)
                path, linked, src = fp, True, self.file_tokens[fp][tok]
        if tok is None and r < 0.75 and self.tokens_global and small:
            for _ in range(8):                   # transplant miss (token from A into B)
                t = self.rng.choice(list(self.tokens_global))
                b = self.rng.choice(small)
                if self.tokens_global[t][0] != b and t not in self.file_tokens.get(b, {}):
                    tok, path, miss, src = t, b, True, f"miss_transplant:{self.tokens_global[t][1]}"
                    break
        if tok is None and small:
            tok, path, src = self.rng.choice(QUERY_LEXICON), self.rng.choice(small), "lexicon"
        if tok is None:
            return self._cat("grep_fallback")
        s = self.do(f"grep -F -m 8 {_sq(tok)} {path}",
                    {"arm": "grep", "role": "read", "linked": linked, "query_src": src,
                     "intended_miss": miss}, target=path)
        if s["exit"] == 0 and s["output"]:
            self._mine_tokens(path, s["output"], len(self.steps) - 1, "grep")

    def _uptime(self):
        vt = self.vt()
        # store-time virtualization of BOTH bytes: output -> the vt-clock render AND exit -> 0
        # (the SST predicts uptime as an always-ok synthetic surface, _ok(render_uptime), like
        # ps). Without store_exit, an image lacking uptime (fedora: real exit 127) records the
        # synthetic render under a 127 exit — a golden-rule violation the SST-parity gate flags.
        self.do("uptime", {"arm": "uptime", "role": "read"},
                store_out=SS.render_uptime(vt), store_exit=0)

    def _sleep(self):
        self.do("sleep " + self.rng.choice(("0", "1")),
                {"arm": "sleep", "role": "decor"}, store_out="")

    def _echo_bare(self):
        tok, prov = self._mined_payload()
        self.do("echo " + tok, {"arm": "echo_bare", "role": "echo", "payload_src": prov})

    # ---- mutation / motif arms (all MutGuard-validated) ----

    def _emit_mut(self, cmd, meta, target=None, exec_cmd=None):
        mutguard_validate(cmd, self.st.cwd, self.probe.get("tool_paths", ()), state=self.st)
        self.n_mut += 1
        self.counters["mutations"] += 1
        meta.setdefault("mut_id", f"m{self.n_mut}")
        meta["mut_affected"] = True
        return self.do(cmd, meta, target=target, exec_cmd=exec_cmd)

    def _new_ws_name(self):
        base = self.rng.choice(WS_NAME_LEXICON)
        return f"{WORKSPACE}/{base}{self.n_mut}.txt"

    def _mkdir_touch(self):
        if self.rng.random() < 0.5:
            d = f"{WORKSPACE}/d{self.n_mut}"
            self._emit_mut(f"mkdir {d}", {"arm": "mkdir_touch", "role": "intervene",
                                          "state_scope": "created"}, target=d)
            self.ws_dirs.append(d)
            self._schedule(self.vt() + self.rng.randint(1, 3),
                           lambda dd=d: self._ls(opt=self.rng.choice(_V3_TIME_FREE_LS),
                                                  target=dd, arm="mkdir_touch", role="revisit"))
        else:
            f = self._new_ws_name()
            self._emit_mut(f"touch {f}", {"arm": "mkdir_touch", "role": "intervene",
                                          "state_scope": "created"}, target=f)
            self.ws_files.append(f)

    def _m_redirect(self):
        """G3 REDIR_W (booked under composition). echo-payload create/append (observed
        content) OR a blind PROD > WSF (tracker-blind honest-margin surface)."""
        blind = self.rng.random() < 0.45 and self.dirs
        name = self._new_ws_name()
        if blind:
            src = self.dir_cyc()
            if not src or self._dead(src):
                src = "/etc"
            cmd = f"ls -1 {src} > {name}"
            meta = {"arm": "m_redirect", "role": "intervene", "state_scope": "created",
                    "ws_target": True, "ws_observed": False, "sig_family": "redir:prod>"}
        else:
            tok, prov = self._mined_payload()
            op = ">>" if (self.ws_files and self.rng.random() < 0.35) else ">"
            if op == ">>":
                # MEDIUM-3: never append THROUGH a /tmp/w symlink that resolves to a real
                # image file outside the arena (mutguard_validate rejects it as a backstop).
                app_pool = [f for f in self.ws_files if self._resolves_in_ws(f)]
                if app_pool:
                    name = self.rng.choice(app_pool)
                else:
                    op = ">"          # no in-arena append target: overwrite a fresh ws file
            cmd = f"echo {_sq(tok)} {op} {name}"
            meta = {"arm": "m_redirect", "role": "intervene", "state_scope": "created",
                    "ws_target": True, "ws_observed": True, "payload_src": prov,
                    "chain_depth": 2 if op == ">>" else 1}
        self._emit_mut(cmd, meta, target=name)
        if name not in self.ws_files:
            self.ws_files.append(name)
        # readback (>=70% of writes): names-only ls or a content cat
        if self.rng.random() < 0.80:
            def _rb(n=name, observed=not blind):
                if observed:
                    r = self.do("cat " + n, {"arm": "m_redirect", "role": "revisit",
                                             "ws_target": True}, target=n)
                    if r["exit"] == 0 and r["output"]:
                        self._mine_tokens(n, r["output"], len(self.steps) - 1, "cat")
                else:
                    self._ls(opt=self.rng.choice(_V3_TIME_FREE_LS), target=WORKSPACE,
                             arm="m_redirect", role="revisit")
            self._schedule(self.vt() + self.rng.randint(1, 3), _rb)

    def _m_rm(self):
        """CUD/RM-listing motif: rm a workspace file (or a Tier-S file), then a listing
        revisit (>=1 survivor) and an error revisit (report-only)."""
        target, recursive = None, False
        live_ws_dirs = self._live_pool(self.ws_dirs)
        live_ws_files = self._live_pool(self.ws_files)
        live_tier_s = self._live_pool(self.tier_s_pool)
        if live_ws_dirs and self.rng.random() < 0.30:
            target, recursive = self.rng.choice(live_ws_dirs), True   # rm -r under /tmp/w
        elif live_ws_files and self.rng.random() < 0.65:
            target = self.rng.choice(live_ws_files)
        elif live_tier_s and self.n_tier_s < V3_MAX_TIER_S:
            target = self.rng.choice(live_tier_s)
            self.n_tier_s += 1
        elif live_ws_files:
            target = self.rng.choice(live_ws_files)
        else:
            return self._mkdir_touch()
        cmd = f"rm -r {target}" if recursive else f"rm {target}"
        parent = parent_of(normpath(target, self.st.cwd))
        self._emit_mut(cmd, {"arm": "m_rm", "role": "intervene", "state_scope": "mutated",
                             "victim_observed": normpath(target) in self.last_obs_of},
                       target=target)
        for lst in (self.ws_files, self.ws_dirs):
            if target in lst:
                lst.remove(target)
        # listing-delta revisit + an error revisit (report-only, D3-excluded)
        self._schedule(self.vt() + self.rng.randint(1, 3),
                       lambda p=parent: self._ls(opt=self.rng.choice(_V3_TIME_FREE_LS),
                                                 target=p, arm="m_rm", role="revisit"))
        self._schedule(self.vt() + self.rng.randint(2, 4),
                       lambda t=target: self.do("cat " + t,
                                                {"arm": "m_rm", "role": "revisit",
                                                 "intended_outcome": "miss-reverted"},
                                                target=t))

    def _m_mv(self):
        """MV-displacement: mv a Tier-S file to <F>.bak (content transported), then read
        the .bak (content), the vacated F (error, report), and a listing delta."""
        live_s = self._live_pool(self.tier_s_pool)
        if live_s and self.n_tier_s < V3_MAX_TIER_S and self.rng.random() < 0.6:
            src = self.rng.choice(live_s)
            # unique suffix guarantees destination absence (§4.2 dst-absence rule)
            dst = src + self.rng.choice((".bak", ".old", ".orig")) + str(self.n_mut)
            self.n_tier_s += 1
            self._emit_mut(f"mv {src} {dst}",
                           {"arm": "m_mv", "role": "intervene", "state_scope": "mutated",
                            "victim_observed": normpath(src) in self.last_obs_of},
                           target=src)
            self._schedule(self.vt() + self.rng.randint(1, 2),
                           lambda d=dst: self._cat_path(d, "m_mv", "revisit"))
            self._schedule(self.vt() + self.rng.randint(2, 4),
                           lambda p=parent_of(normpath(src)):
                           self._ls(opt=self.rng.choice(_V3_TIME_FREE_LS), target=p,
                                    arm="m_mv", role="revisit"))
        elif self.ws_files:
            src = self.rng.choice(self.ws_files)
            dst = f"{WORKSPACE}/moved{self.n_mut}.txt"
            self._emit_mut(f"mv {src} {dst}",
                           {"arm": "m_mv", "role": "intervene", "state_scope": "mutated"},
                           target=src)
            self.ws_files.remove(src)
            self.ws_files.append(dst)
            self._schedule(self.vt() + self.rng.randint(1, 2),
                           lambda d=dst: self._cat_path(d, "m_mv", "revisit"))
        else:
            self._mkdir_touch()

    def _cat_path(self, path, arm, role):
        # a dead target renders the real error (report-only, D3-excluded) — folded fine
        s = self.do("cat " + path, {"arm": arm, "role": role, "ws_target":
                                    path.startswith(WORKSPACE + "/")}, target=path)
        if s["exit"] == 0 and s["output"]:
            self._mine_tokens(path, s["output"], len(self.steps) - 1, "cat")
            self.last_obs_of[normpath(path, self.st.cwd)] = len(self.steps) - 1
        return s

    def _m_ln(self):
        """LN chain: ln -s <F> /tmp/w/<link>, then readlink + cat-through (= F's content);
        p=.35 chain extension rm <F> -> cat <link> (dangling, report). LN-CONTRAST low
        rate: a hard ln twin whose content survives rm."""
        srcs = self._live_pool(self.ws_files) or self._live_pool(self.tier_s_pool)
        if not srcs:
            return self._mkdir_touch()
        src = self.rng.choice(srcs)
        link = f"{WORKSPACE}/link{self.n_mut}"
        srcv = self.st.fs.get(normpath(src, self.st.cwd))
        hard = (self.rng.random() < 0.25 and srcv is not None
                and srcv["kind"] == "file" and srcv.get("linkness_known"))
        if hard:
            self._emit_mut(f"ln {src} {link}",
                           {"arm": "m_ln", "role": "intervene", "state_scope": "created",
                            "motif": "ln_contrast"}, target=link)
        else:
            self._emit_mut(f"ln -s {src} {link}",
                           {"arm": "m_ln", "role": "intervene", "state_scope": "created",
                            "motif": "ln_chain"}, target=link)
            self._schedule(self.vt() + self.rng.randint(1, 2),
                           lambda l=link: self.do("readlink " + l,
                                                  {"arm": "m_ln", "role": "revisit"}, target=l))
        self.ws_files.append(link)
        self._schedule(self.vt() + self.rng.randint(1, 3),
                       lambda l=link: self._cat_path(l, "m_ln", "revisit"))

    # ---- composition arms (G3) ----

    def _prod_path(self):
        """A tracker-live producer path (§6.4: producers only from existence-verified
        paths so the pipe/redir mode rule stays sound)."""
        pool = self._live_pool(self.seen_paths) or self._live_pool(self.files)
        return self.rng.choice(pool) if pool else None

    def _pipe(self):
        prod_kind = self.rng.choice(("ls", "cat"))
        if prod_kind == "ls":
            d = self.dir_cyc()
            if not d or self._dead(d):
                d = "/etc"
            prod = f"ls -1 {d}"
        else:
            f = self._prod_path()
            if f is None:
                return self._cat("pipe_fallback")
            prod = f"cat {f}"
        filt_kind = self.rng.choice(("head", "tail", "grep"))
        if filt_kind in ("head", "tail"):
            filt = f"{filt_kind} -n {self.rng.choice((3, 5, 10))}"
        else:
            tok = self.rng.choice(list(self.tokens_global) or QUERY_LEXICON)
            filt = f"grep -F -m 8 {_sq(tok)}"
        cmd = f"{prod} | {filt}"
        self.do(cmd, {"arm": "pipe", "role": "read"})

    def _cond(self):
        pool = self._live_pool(self.seen_paths) or self._live_pool(self.files)
        if not pool:
            return self._cat("cond_fallback")
        p = self.rng.choice(pool)
        testop = self.rng.choice(("-e", "-f", "-s"))
        read_kind = self.rng.choice(("cat", "ls", "head"))
        if read_kind == "cat":
            read = f"cat {p}"
        elif read_kind == "ls":
            read = f"ls -1 {p}"
        else:
            k = self.probe["kmap"].get(p, self.rng.choice((3, 5)))
            read = f"head -n {k} {p}"
        cmd = f"[ {testop} {p} ] && {read}"
        self.do(cmd, {"arm": "cond", "role": "read"}, target=p)

    # ---- time / process arms ----

    def _after(self):
        if self.counters["jobs"] >= V3_MAX_JOBS:
            return self._ps()
        j = self.next_j
        # schedule-aware launch guard (§3.3 S3): launch_vt + K + slack <= L-1
        room = self.length - 1 - self.vt()
        ks = [k for k in (2, 3, 5, 8) if k + 2 <= room]
        if not ks:
            return self._ps()
        k = self.rng.choice(ks)
        tok, prov = self._mined_payload()
        effect = f"echo {tok} >> {WORKSPACE}/task{j}.log"
        cmd = f"after {j} {k} '{effect}' & echo $!"
        mutguard_validate(cmd, self.st.cwd, self.probe.get("tool_paths", ()), state=self.st)
        launch_vt = self.vt()
        # exec the REAL launch; do() virtualizes `echo $!` -> the canonical cpid (after_j)
        self.do(cmd, {"arm": "after", "role": "launch", "state_scope": "created",
                      "payload_src": prov,
                      "job": {"j": j, "K": k, "launch_vt": launch_vt,
                              "fire_vt_planned": launch_vt + k, "phase": "launch"}},
                exec_cmd=cmd, after_j=j, extra_timeout=1)
        if j in self.st.jobs:                    # the SST registered the launch
            self.next_j += 1
            self.counters["jobs"] += 1
            # firing is SST-decided (_fires_due); the fire-script prologue is injected by
            # do() at the DUE step. Schedule only the fire+delta job-log readback.
            self._schedule(launch_vt + k + self.rng.randint(0, 1),
                           lambda jj=j: self._job_readback(jj))

    def _job_readback(self, j):
        self._cat_path(f"{WORKSPACE}/task{j}.log", "after", "revisit")

    def _ps(self):
        cmd = f"{PS_PATH} -o pid,stat,args"
        # the SST job table is the canonical ps authority (draft §5.4); the real tj3-ps
        # output is canonicalized to exactly this render (store-time virtualization)
        pred = self.st.predict(self.vt(), cmd)
        self.do(cmd, {"arm": "ps", "role": "read", "ps_tier": "T2"},
                store_out=pred["output"] if pred is not None else "", store_exit=0,
                exec_cmd=cmd)

    def _kill(self):
        alive = [(j, job) for j, job in self.st.jobs.items()
                 if job["state"] in ("waiting", "stopped", "stopped_pending_term")]
        dead = [(j, job) for j, job in self.st.jobs.items()
                if job["state"] in ("fired", "killed")]
        miss = (self.rng.random() < 0.20 and bool(dead)) or not alive
        if miss and not dead:
            return self._ps()
        j, job = self.rng.choice(dead if miss else alive)
        cpid = job["cpid"]
        # -CONT is meaningful only on a stopped job; STOP on waiting, TERM/-9 general
        st = job["state"]
        if st == "stopped":
            sig = self.rng.choice(("-CONT", "", "-9"))
        elif st == "stopped_pending_term":
            sig = self.rng.choice(("-CONT", "-9"))
        elif miss:
            sig = self.rng.choice(("", "-9", "-0"))
        else:
            sig = self.rng.choice(("", "-STOP", "-9", "-0"))
        cmd = ("kill " + (sig + " " if sig else "") + str(cpid)).strip()
        pred = self.st.predict(self.vt(), cmd)
        realpid = self.real_of.get(cpid, str(cpid))
        exec_cmd = ("kill " + (sig + " " if sig else "") + realpid).strip()
        self.do(cmd, {"arm": "kill", "role": "signal", "signal": sig or "TERM",
                      "intended_miss": bool(miss)},
                store_out=pred["output"] if pred is not None else "",
                store_exit=pred["exit"] if pred is not None else 0, exec_cmd=exec_cmd)
        # a post-signal barrier belongs to the NEXT step's prologue (draft §3.3 ruling)
        if not miss and sig in ("", "-STOP", "-9"):
            self.pending_kill_barrier = (realpid,
                                         {"": "TERM", "-STOP": "-STOP", "-9": "-9"}[sig])

    # ---- scheduler ----

    def _schedule(self, due, fn):
        self.pending.append({"due": min(due, self.length - 1), "fn": fn,
                             "issued": self.vt()})

    def _fire_prologue(self, vt):
        """Assemble the UNRECORDED prologue for step vt: fire-scripts for the jobs the SST
        will fire this step (st._fires_due — the ONE authority, so the real container and
        the tracker fire the SAME set) + a pending post-signal barrier (draft §3.2/§3.3).
        Returns (prologue, extra_timeout)."""
        frags, extra = [], 0
        for j in self.st._fires_due(vt):
            rp = self.real_of.get(canonical_pid(j), str(canonical_pid(j)))
            frags.append(
                f"echo go > /tmp/.tj/g{j}; t=0; "
                f"while [ ! -e /tmp/.tj/d{j} ] && [ $t -lt 50 ]; "
                f"do sleep 0.1; t=$((t+1)); done; "
                f"while [ -d /proc/{rp} ] && [ $t -lt 50 ]; do sleep 0.1; t=$((t+1)); done")
            extra += 5
        if self.pending_kill_barrier is not None:
            rp, kind = self.pending_kill_barrier
            if kind in ("TERM", "-9"):
                frags.append(f"tk=0; while [ -d /proc/{rp} ] && [ $tk -lt 50 ]; "
                             f"do sleep 0.1; tk=$((tk+1)); done")
            elif kind == "-STOP":
                frags.append(f"tk=0; while ! grep -q '^State:.T' /proc/{rp}/status 2>/dev/null "
                             f"&& [ $tk -lt 50 ]; do sleep 0.1; tk=$((tk+1)); done")
            self.pending_kill_barrier = None
            extra += 5
        return ("; ".join(frags), extra)

    def _due_probe(self, vt):
        """A scheduled probe/fire-readback that is due (<= vt) pre-empts the weight draw
        (deterministic given rng); the earliest one runs this step, one per step."""
        due = sorted((p for p in self.pending if p["due"] <= vt),
                     key=lambda p: (p["due"], p["issued"]))
        if not due:
            return False
        p = due[0]
        self.pending.remove(p)
        p["fn"]()
        return True

    # ---- the loop ----

    def generate(self):
        self._open()
        n_interventions_target = self.rng.randint(*V3_INTERVENTIONS)
        while self.vt() < self.length:
            vt = self.vt()
            remaining = self.length - vt
            # a due scheduled probe/fire-readback pre-empts the draw
            if self._due_probe(vt):
                continue
            arm = self._draw_arm(vt, remaining, n_interventions_target)
            self._dispatch(arm)
        # flush: run any still-pending readbacks that fit (best effort, in-bounds only)
        return self.steps

    def _draw_arm(self, vt, remaining, n_int_target):
        self._n_draws += 1
        w = dict(self.weights)
        # pacing (draft §4.2): no motif starts in the last 5 slots; cap interventions at the
        # per-seq target (6+/-2) and the hard <=8 ceiling.
        tail = remaining <= 5
        capped = self.n_mut >= min(n_int_target, V3_MAX_MUTATIONS)
        if tail or capped:                       # drop the intervene block late / once capped
            for k in list(w):
                if k in _V3_INTERVENE or k == "after":
                    w.pop(k, None)
        else:
            # P2 intervention-floor controller: steer REALIZED interventions/seq up to the
            # target, borrowing the extra draw mass ONLY from composition/time (never atomic),
            # so the >=.60 atomic mass is preserved by construction. The rescale sets the
            # intervene block's per-draw probability to `p` while keeping intra-block and
            # intra-fund SSOT ratios (a scalar multiply per group), and conserves the total —
            # so atomic/pwd/echo draw shares are invariant to the boost.
            block_arms = [k for k in w if k in _V3_INTERVENE]     # w-order => deterministic
            fund_arms = [k for k in w if k in _V3_INT_FUND]
            block = sum(w[k] for k in block_arms)
            fund = sum(w[k] for k in fund_arms)
            if block_arms and fund_arms and block > 0.0 and fund > 0.0:
                tot0 = sum(w[k] for k in w)                       # conserved (~1.0)
                p = _p_intervene(self.n_mut, self._n_draws - 1, n_int_target)
                # never overdraw the fund pool (keep >= 1-_V3_INT_FUND_MAX of it): this bounds
                # p to the fund-availability cap — the achievable max when atomic is preserved.
                p = min(p, (block + fund * _V3_INT_FUND_MAX) / tot0)
                want = p * tot0                                   # desired absolute block weight
                if want > block:
                    bscale = want / block                        # >1, preserves block ratios
                    fscale = (fund - (want - block)) / fund       # <1, preserves fund ratios
                    for k in block_arms:
                        w[k] *= bscale
                    for k in fund_arms:
                        w[k] *= fscale
        if self.counters["jobs"] >= V3_MAX_JOBS:
            w.pop("after", None)
        if not w:
            return "cat"
        tot = sum(w.values())
        arms = list(w)
        return self.rng.choices(arms, weights=[w[a] / tot for a in arms])[0]

    def _dispatch(self, arm):
        if arm == "cd":
            self._cd()
        elif arm == "ls":
            self._ls()
        elif arm == "cat":
            self._cat()
        elif arm == "config":
            self._cat(config=True)
        elif arm in ("head", "tail"):
            self._headtail(arm)
        elif arm == "stat":
            self._stat()
        elif arm == "find":
            self._find()
        elif arm == "grep":
            self._grep()
        elif arm == "pwd":
            self._pwd()
        elif arm == "uptime":
            self._uptime()
        elif arm == "sleep":
            self._sleep()
        elif arm == "echo_bare":
            self._echo_bare()
        elif arm == "mkdir_touch":
            self._mkdir_touch()
        elif arm == "m_redirect":
            self._m_redirect()
        elif arm == "m_rm":
            self._m_rm()
        elif arm == "m_mv":
            self._m_mv()
        elif arm == "m_ln":
            self._m_ln()
        elif arm == "pipe":
            self._pipe()
        elif arm == "cond":
            self._cond()
        elif arm == "after":
            self._after()
        elif arm == "ps":
            self._ps()
        elif arm == "kill":
            self._kill()
        else:
            self._cat()


def gen_sequence_v3(box, dirs, files, rng, length, image, error_templates, boot, arm="full"):
    """dockerfs3 mint policy (prereg §1 / draft §4-§7): the v2 flat verb-mixture wrapped in
    an event scheduler + a collection-mode ShellState (the ONE tracker). Emits ONLY frozen-
    universe commands (parser totality is a mint gate), routes every pool draw through the
    tracker's live view, MutGuard-validates every mutation, virtualizes every recorded byte
    (canonical PIDs, uptime->vt, ps->render_ps), and covers the atomic / mutation /
    composition (G3) / time-process arms per the SSOT weight table. Returns the step list
    (each step's dur_ms lives in `.timings`, routed to the timing side-channel by the
    collector, never in the jsonl)."""
    sess = _V3Session(box, dirs, files, rng, length, image, error_templates, boot, arm=arm)
    sess.generate()
    return sess


POLICIES = {"baseline": gen_sequence, "diverse": gen_sequence_diverse,
            "levy_novelty": gen_sequence_levy_novelty, "v2": gen_sequence_v2,
            "v3": gen_sequence_v3}


def _step_record(s):
    """Step dict for the jsonl: the v1 fields, plus meta when the policy attached it (v2
    linkage/provenance). v1 policies set no meta -> their records stay byte-identical."""
    rec = {"cmd": s["cmd"], "output": s["output"], "exit": s["exit"], "cwd": s["cwd"]}
    if "meta" in s:
        rec["meta"] = s["meta"]
    return rec


def collect_image(image, n_seqs, seq_len, seed, policy="baseline", ref=None):
    ref = ref or image  # --pin-digests: run by digest reference; record/seed by tag
    if not image_present(ref) and not pull(ref):
        return image, None, f"could not pull {ref}", {}
    try:
        # F4: fixed hostname from the image TAG (not the digest ref) — stable across pinned
        # and unpinned runs; kills the per-run 12-hex container-ID nonce in observations.
        box = DockerBox(ref, hostname=hostname_for(image))
        sysid = box.system_id()
        dirs, files = box.enumerate()
        if not dirs:
            dirs = ["/etc", "/var", "/usr", "/"]
        rng = random.Random(f"dockerfs:{seed}:{image}")
        seqs = []
        for i in range(n_seqs):
            box.cwd = "/"  # each sequence starts fresh at root
            ln = rng.randint(max(4, seq_len - 4), seq_len + 4)
            seqs.append({"image": image, "system_id": sysid,
                         "steps": [_step_record(s)
                                   for s in POLICIES[policy](box, dirs, files, rng, ln)]})
        report = v2_image_report(box)
        box.close()
        return image, seqs, f"{len(dirs)} dirs / {len(files)} files", report
    except Exception as e:  # noqa: BLE001
        return image, None, f"error: {e}", {}


def _v3_image_report(sessions):
    """Aggregate per-image v3 counters (arm/sig realized mix, mutation/job/linkage counts)
    across a container's sessions for summary.json (report-only; F8-recoverable from meta)."""
    agg = {"sigs": {}, "arms": {}, "linked": 0, "revisits": 0, "intended_miss": 0,
           "mutations": 0, "jobs": 0, "steps": 0, "skips": []}
    for sess in sessions:
        c = sess.counters
        for k in ("sigs", "arms"):
            for kk, vv in c[k].items():
                agg[k][kk] = agg[k].get(kk, 0) + vv
        for k in ("linked", "revisits", "intended_miss", "mutations", "jobs"):
            agg[k] += c[k]
        agg["steps"] += len(sess.steps)
        agg["skips"] = sorted(set(agg["skips"]) | set(c["skips"]))
    return agg


def collect_image_v3(image, n_seqs, seq_len, seed, ref=None, arm="full"):
    """dockerfs3 collect_image (prereg §1/§7): a pristine PROBE container per image (§3.5),
    then a FRESH --init container per trajectory (draft §3.1) with the unrecorded bootstrap
    (vendored busybox tj3-ps + after helper + /tmp/.tj + /tmp/w seed), per-seq RNG stream
    `dockerfs:{seed}:{image}:{arm}:{seq_idx}`, seq-len 28±4 (<=32; pos_emb(64)). Returns
    (image, seqs, info, report, timings) — timings feed the timing-<split> side-channel
    (dur_ms is stripped from every recorded step by _step_record)."""
    ref = ref or image
    if not image_present(ref) and not pull(ref):
        return image, None, f"could not pull {ref}", {}, []
    try:
        probe_box = DockerBox(ref, hostname=hostname_for(image))
        sysid = _canon_system_id(probe_box.system_id())  # v3: mask kernel build-date (DG-3c)
        dirs, files = probe_box.enumerate()
        if not dirs:
            dirs = ["/etc", "/var", "/usr", "/"]
        probe = _v2_probe(probe_box, dirs, files)      # warm probe products (pristine box)
        probe_box.close()
        error_templates = sst_error_templates(image)
        seqs, timings, sessions = [], [], []
        for i in range(n_seqs):
            rng = random.Random(f"dockerfs:{seed}:{image}:{arm}:{i}")
            ln = min(V3_SEQ_MAX, rng.randint(max(4, seq_len - 4), seq_len + 4))
            box = DockerBox(ref, hostname=hostname_for(image), init=True,
                            label=f"tj3-mint={seed}")
            box._v2_state = probe                       # reuse probe products on the mint box
            try:
                boot = v3_bootstrap(box, seed, files, probe, rng)
                sess = gen_sequence_v3(box, dirs, files, rng, ln, image,
                                       error_templates, boot, arm=arm)
            finally:
                box.close()                             # fresh-container-per-trajectory teardown
            sessions.append(sess)
            seqs.append({"image": image, "system_id": sysid, "arm": arm, "seq_idx": i,
                         "ws_manifest_sha256": boot["ws_manifest_sha256"],
                         "steps": [_step_record(s) for s in sess.steps]})
            for step_idx, dur in enumerate(sess.timings):
                timings.append({"seq_idx": i, "step": step_idx, "dur_ms": dur})
        return (image, seqs, f"{len(dirs)} dirs / {len(files)} files",
                _v3_image_report(sessions), timings)
    except Exception as e:  # noqa: BLE001
        return image, None, f"error: {e}", {}, []


def resolve_digests(images):
    """--pin-digests: resolve each tag to its sha256 RepoDigest via docker inspect (pulling
    if absent). The table is recorded in summary.json and collection proceeds by digest."""
    table = {}
    for im in dict.fromkeys(images):
        if not image_present(im) and not pull(im):
            raise RuntimeError(f"pin-digests: could not pull {im}")
        r = subprocess.run(["docker", "image", "inspect", "--format",
                            "{{index .RepoDigests 0}}", im], capture_output=True, text=True)
        ref = r.stdout.strip()
        if r.returncode != 0 or "@sha256:" not in ref:
            raise RuntimeError(f"pin-digests: no RepoDigest for {im}: {r.stderr.strip() or ref}")
        table[im] = ref
    return table


def _v3_weight_sha():
    return hashlib.sha256(
        json.dumps(V3_WEIGHTS, sort_keys=True).encode()).hexdigest()


def collect(out_dir, train_imgs, val_imgs, n_seqs, seq_len, seed, workers, policy="baseline",
            pin_digests=False, expect_digests=None, arm="full"):
    # Amendment 5/7 gate, at FUNCTION ENTRY (round-5 fix: the nested/late form was
    # bypassable without --pin-digests and burned the collection hour before aborting).
    # v3 mirrors it (prereg §1 one-mint rule; §6 digest entry gate).
    if policy in ("v2", "v3") and n_seqs >= 100 and not (pin_digests and expect_digests):
        raise SystemExit(f"{policy} full mint requires BOTH --pin-digests AND --expect-digests "
                         f"(prereg digest entry gate)")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in ("summary.json", "emb-seq-train.pt", "emb-seq-val.pt",
                       "timing-train.jsonl", "timing-val.jsonl"):
        stale = out / stale_name
        if stale.exists():
            stale.unlink()   # aborted/reused dirs must never leave resolvable stale artifacts
    digests = resolve_digests(train_imgs + val_imgs) if pin_digests else {}
    if expect_digests:
        want = json.loads(pathlib.Path(expect_digests).read_text())
        drift = {i: (want.get(i), digests.get(i)) for i in set(want) | set(digests)
                 if want.get(i) != digests.get(i)}
        if drift:
            raise SystemExit(f"DIGEST DRIFT vs {expect_digests} — the mint must run on the "
                             f"audited bytes (prereg Amendment 5): {drift}")
    reports = {}  # split -> image -> availability/skip/verb/linkage counters

    def run_split(images, path, split):
        if not images:
            return 0  # F6: --train-only must never open (truncate) the other split's file
        n_steps = 0
        timing_rows = []                     # (image, seq_idx, step, dur_ms) side-channel
        with open(path, "w") as fh, cf.ThreadPoolExecutor(max_workers=workers) as ex:
            # futures kept in submitted-image order -> deterministic jsonl layout
            if policy == "v3":
                futs = [(im, ex.submit(collect_image_v3, im, n_seqs, seq_len, seed,
                                       digests.get(im), arm)) for im in images]
            else:
                futs = [(im, ex.submit(collect_image, im, n_seqs, seq_len, seed, policy,
                                       digests.get(im))) for im in images]
            for im, fut in futs:
                res = fut.result()
                if policy == "v3":
                    image, seqs, info, report, timings = res
                else:
                    image, seqs, info, report = res
                    timings = []
                if seqs is None:
                    if policy in ("v2", "v3"):
                        # mint integrity: a preregistered one-run mint must never silently
                        # produce a partial dataset (DG-10a fail-fast)
                        raise RuntimeError(f"[{split}] {image} failed ({info}) — aborting mint")
                    print(f"  [{split}] SKIP {image}: {info}", flush=True)
                    continue
                if report:
                    reports.setdefault(split, {})[image] = report
                for s in seqs:
                    fh.write(json.dumps(s) + "\n")
                    n_steps += len(s["steps"])
                for t in timings:
                    timing_rows.append(dict(t, image=image))
                print(f"  [{split}] {image}: {len(seqs)} seqs ({info})", flush=True)
        if policy == "v3":
            # the timing side-channel: dur_ms lives ONLY here (gitignored, excluded from
            # replay byte-diffs); wall-clock never enters a recorded/diffed byte (draft §3.4)
            with open(out / f"timing-{split}.jsonl", "w") as tf:
                for row in timing_rows:
                    tf.write(json.dumps(row) + "\n")
        return n_steps

    tr_steps = run_split(train_imgs, out / "train.jsonl", "train")
    va_steps = run_split(val_imgs, out / "val.jsonl", "val")
    summary = {"seed": seed, "seqs_per_image": n_seqs, "seq_len": seq_len,
               "train_images": train_imgs, "val_images_heldout": val_imgs,
               "train_steps": tr_steps, "val_steps": va_steps, "policy": policy}
    if policy.startswith("v2"):
        # F9 (constitution §4): the frozen class table + version identity, verbatim from
        # prereg Amendment 2 as amended by Amendment 3. Top-level copies are what
        # evolve/bench_versions.resolve() reads and asserts against.
        summary["bench_version"] = BENCH_VERSION
        summary["verb_classes"] = V2_VERB_CLASSES
        summary["v2"] = dict(lexicon_hashes(), bench_version=BENCH_VERSION,
                             verb_classes=V2_VERB_CLASSES, images=reports)
    if policy == "v3":
        # version identity + the ablate flag (prereg §7: bench_versions recognizes the ablate
        # arm via this summary flag, NEVER by sniffing for an absent val.jsonl).
        summary["bench_version"] = BENCH_VERSION_V3
        summary["arm"] = arm
        summary["ablate"] = (arm == "ablate")
        summary["v3"] = {"bench_version": BENCH_VERSION_V3, "arm": arm,
                         "weight_sha256": _v3_weight_sha(), "weights": V3_WEIGHTS,
                         "seq_len_band": [V3_SEQ_LEN - 4, V3_SEQ_MAX], "images": reports,
                         # MINOR-4: the G-RATE (prereg §3) denominator is the arm-DRAW
                         # distribution, NOT recorded steps — scheduled revisits + motif
                         # children book under their spawning arm by design (a rm draw emits
                         # 2 revisits under arm=m_rm), and scheduled probes pre-empt the draw.
                         "g_rate_denominator": "arm_draw_distribution"}
    if digests:
        summary["image_digests"] = digests
        # constitution §9: bind the published bytes to the audited mint bytes
        summary["artifact_sha256"] = {
            f"{s}.jsonl": hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest()
            for s in ("train", "val") if (out / f"{s}.jsonl").exists()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dockerfs")
    ap.add_argument("--seqs-per-image", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-images", default=",".join(TRAIN_IMAGES))
    ap.add_argument("--val-images", default=",".join(VAL_IMAGES))
    ap.add_argument("--policy", default="baseline", choices=list(POLICIES))
    ap.add_argument("--train-only", action="store_true", help="collect train split only (reuse existing val)")
    ap.add_argument("--pin-digests", action="store_true",
                    help="resolve image tags to sha256 digests (recorded in summary.json; collection runs by digest)")
    ap.add_argument("--expect-digests", default=None,
                    help="JSON {image: digest} table; abort on drift (mint gate, prereg Amendment 5)")
    ap.add_argument("--arm", default="both", choices=("full", "ablate", "both"),
                    help="v3 mint arm: 'both' (default) runs the full arm then the paired "
                         "train-only ablate arm into <out>-ablate (prereg §1 two-arm mint)")
    ap.add_argument("--ablate-seqs", type=int, default=None,
                    help="v3 ablate-arm seqs/image (default: half the full arm, per prereg §1)")
    args = ap.parse_args(argv)
    val_imgs = [] if args.train_only else args.val_images.split(",")
    train_imgs = args.train_images.split(",")

    def _one(out, seqs, arm, vals):
        s = collect(out, train_imgs, vals, seqs, args.seq_len, args.seed, args.workers,
                    args.policy, pin_digests=args.pin_digests,
                    expect_digests=args.expect_digests, arm=arm)
        print(json.dumps(s, indent=1))
        return s

    if args.policy == "v3" and args.arm == "both":
        # prereg §1 two-arm mint (one run, same seeds/digests): full arm both splits, then the
        # paired ablate arm (mutation/time/composition OFF, renormalized) TRAIN-ONLY (F6).
        _one(args.out, args.seqs_per_image, "full", val_imgs)
        _one(args.out.rstrip("/") + "-ablate",
             args.ablate_seqs or max(1, args.seqs_per_image // 2), "ablate", [])
    else:
        _one(args.out, args.seqs_per_image,
             args.arm if args.policy == "v3" and args.arm != "both" else "full", val_imgs)


if __name__ == "__main__":
    main()
