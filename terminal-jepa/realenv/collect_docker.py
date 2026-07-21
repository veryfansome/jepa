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


POLICIES = {"baseline": gen_sequence, "diverse": gen_sequence_diverse,
            "levy_novelty": gen_sequence_levy_novelty, "v2": gen_sequence_v2}


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


def collect(out_dir, train_imgs, val_imgs, n_seqs, seq_len, seed, workers, policy="baseline",
            pin_digests=False, expect_digests=None):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in ("summary.json", "emb-seq-train.pt", "emb-seq-val.pt"):
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
    reports = {}  # split -> image -> v2 availability/skip/verb/linkage counters

    def run_split(images, path, split):
        if not images:
            return 0  # F6: --train-only must never open (truncate) the other split's file
        n_steps = 0
        with open(path, "w") as fh, cf.ThreadPoolExecutor(max_workers=workers) as ex:
            # futures kept in submitted-image order -> deterministic jsonl layout
            futs = [ex.submit(collect_image, im, n_seqs, seq_len, seed, policy,
                              digests.get(im)) for im in images]
            for fut in futs:
                image, seqs, info, report = fut.result()
                if seqs is None:
                    if policy.startswith("v2"):
                        # mint integrity: a preregistered one-run mint must never silently
                        # produce a partial dataset
                        raise RuntimeError(f"[{split}] {image} failed ({info}) — aborting v2 mint")
                    print(f"  [{split}] SKIP {image}: {info}", flush=True)
                    continue
                if report:
                    reports.setdefault(split, {})[image] = report
                for s in seqs:
                    fh.write(json.dumps(s) + "\n")
                    n_steps += len(s["steps"])
                print(f"  [{split}] {image}: {len(seqs)} seqs ({info})", flush=True)
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
    if digests:
        summary["image_digests"] = digests
        # constitution §9: bind the published bytes to the audited mint bytes
        summary["artifact_sha256"] = {
            f"{s}.jsonl": hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest()
            for s in ("train", "val") if (out / f"{s}.jsonl").exists()}
        # a full-scale v2 mint must run digest-gated (Amendment 5; prose rule made code)
        if policy == "v2" and n_seqs >= 100 and not expect_digests:
            raise SystemExit("v2 full mint requires --expect-digests "
                             "benchmarks/dockerfs2-digests.json (prereg Amendment 5/6)")
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
    args = ap.parse_args(argv)
    summary = collect(args.out, args.train_images.split(","),
                      [] if args.train_only else args.val_images.split(","),
                      args.seqs_per_image, args.seq_len, args.seed, args.workers, args.policy,
                      pin_digests=args.pin_digests, expect_digests=args.expect_digests)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
