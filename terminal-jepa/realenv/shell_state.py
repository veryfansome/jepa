"""Symbolic shell-state tracker (SST) — the ONE shell-state authority of dockerfs3.

Ratified by benchmarks/dockerfs3-design-draft.md §10.1–§10.2 (verdict:cross-lens D-B5:
one tracker module is simultaneously the collection-time state authority, the meta
labeler, the axis-3 detector, and the strongest baseline arm) and frozen against
benchmarks/dockerfs3-prereg.md §4.1 (command universe / BNF) + §4.2 (exit vocabulary,
canonical delta format). Sha-pinned eval-path code once frozen: post-freeze edits
require a dated prereg amendment + re-baseline. Never a genome chunk.

CONTRACT
  ShellState folds a sequence of RENDERED step records {cmd, output, exit, cwd} —
  exactly the fields the encoder sees (render parity; meta is NEVER read) — into:
    - a belief-overlay filesystem (Overlay["fs"]: created / appended / deleted /
      moved / symlink / hardlink states, payload text where observed or derivable,
      tombstones, provenance, fs_clock — the FsLedger view);
    - the tracked cwd ("/" at t=0 — protocol constant, draft §10.1);
    - the composition ws sub-structure (workspace files: producer, observed-capture
      vs blind-capture, append count, known line count — draft §6.3);
    - the 5-state job table {waiting, stopped, stopped_pending_term, fired, killed}
      driven by the after-launch and kill-family forms with canonical PIDs (§5.3);
      a cpid absent from the table is never-launched (kill ⇒ "No such process").
  predict(step_index, cmd) -> {"output","exit","cwd"} where DETERMINED, else BOT
  (None). The determined/BOT split is load-bearing: axis-3 `sim` classification and
  the sst baseline arm both key on it. Determined surfaces (draft §10.2 R1–R9):
    R1 pwd; R2 cd (entailed targets only); R3 mutation own-step acks (entailed
    existence; absence errors via the probe-harvested per-image error-template dict,
    §3.5 — no hand-authored dialect enum); R4 echo (literal join; BNF bans
    backslashes); R5 reads of KNOWN content — appends are determined only for
    trajectory-created files whose full byte content incl. trailing newline is known
    from the writing commands (docker_env rstrips trailing newlines, so
    merely-observed content has trailing_nl_known=False and appends onto it => BOT);
    R6 ls edit-replay — requires an UNCAPPED prior names-only render (or a dir
    created this trajectory); one-per-line forms only; -l splices are ledger-only;
    R7 find replay — identical command, uncapped prior render, subtree untouched
    since; R8 errors on known-dead paths via the template table; R9 process — the
    shared 5-state job simulator; canonical ps renders (render_ps below is THE
    canonical format authority for the vendored /usr/local/bin/tj3-ps, UD-9 Route B);
    kill acks; job-log readbacks with known payloads. SST-G3: the composed grammar
    (pipe / redirect / cond) is evaluated over belief state (draft §6.5, DG-6).
  Everything else — unobserved image content, blind-capture reads, partial dirs,
  capped renders — is BOT.

ROUND-4 HARDENING (adversarial differential review, 2026-07-23; all conservative):
  grep never replays NUL-bearing content (dialect-divergent binary heuristic);
  tail needs trailing_nl_known (a rstripped observation under-counts lines);
  head/tail/grep windows mirror the record channel's rstrip; textual '..'
  collapse requires every erased intermediate to be a known-live linkness-known
  dir (else predict is BOT and folds mine/claim nothing — cd stays logical);
  content mined at linkness-unknown nodes is invalidated by any rm/mv/truncate/
  append destroying equal (or uncertain) bytes — the alias-staleness law;
  rm/ln/touch acks need the relevant kinds known; ls -l splices only certain
  9-field rows (no phantom children); mutations on RUNTIME_MOUNT_PATHS or under
  /proc//sys//dev are BOT; _resolve is hop-capped (symlink cycles -> BOT);
  the parser additionally bans unquoted double quotes, glob characters (an
  unterminated '[' stays literal — /usr/bin/[ is in-universe) and tilde;
  cat-success never entails regular-file for cond -f; observed content,
  listings and link targets under /proc//sys are never re-served.

ROUND-5 HARDENING (adversarial differential review, 2026-07-23; all conservative):
  F1 severed hardgroups — a conservative fold that severs a hardgroup (mv to an
  uncertain landing, forgotten '..' candidates) degrades every remaining peer to
  unknown content at severing time and flags it `severed`: the linkness_known
  exemption in the alias-staleness law applies only to groups that remain fully
  tracked. F2 fire failure channel — _fire_due routes its landing through the
  SAME soundness predicate as _pred_redir (writability + not-a-dir + fully-known
  chain + live-dir parent); an unsound landing degrades (claims nothing, mints
  nothing) and predict-side speculative fires mark the chain 'degraded' (reads/
  listings/conds/mutations there go BOT); _stat's reserved task-log entailment
  requires /tmp/w itself known-live; _wchain's never-minted-workspace rule
  requires every ancestor to be an SST-known live dir; a failed redirect's
  created-empty claim needs a live-dir parent. F3 absence-revival law (the dual
  of the alias law) — tombstones mined through uncertain resolution (template/
  cond-miss; linkness unknown) are DROPPED on any successful creation event
  (echo>/mkdir/touch/ln/mv-in/fire) anywhere: uncertain absence is only valid
  while the world has not created new paths; lstat-certain deadness is
  unaffected. F4 dir-mine staleness — entries/entries_complete/kind mined at
  linkness-unknown dir nodes are invalidated when a real dir with a consistent
  listing is destroyed (rm -r/mv), and on uncertain destructions (mirrors the
  content law). F5 the parser bans leading-dash tokens in every RELATIVE path
  argument position and the grep TOK position (real tools parse them as
  options; absolute '/...' paths stay in; frozen template option positions are
  exempt by position). F6 the R7 find-replay cache is keyed on the RAW command
  string (quoted-form collisions are impossible).

ROUND-6 HARDENING (adversarial live-battery review, 2026-07-23; all conservative):
  F1 stale-cwd — docker_env prologues every step with `cd <tracked-cwd> 2>/dev/null`
  and never re-anchors on failure, so once a tombstone/forget/conservative-move
  covers the cwd (or an ancestor) non-cd commands silently run from the exec start
  dir while the recorded cwd stays stale and every later cd (absolute included)
  fails. A latch (_cwd_stale), set by any such coverage, forces ALL cwd-dependent
  surfaces (pwd, cd, bare/relative ls, every relative-arg resolution) to BOT and
  suppresses cwd-dependent FOLDS (their record does not describe self.cwd) until a
  recorded cwd re-anchors on a live dir (the path is re-created, so the prologue's
  cd succeeds again). Absolute-arg commands are unaffected. §16 POLICY LAW (defense
  in depth, for the collector's MutGuard — NOT implemented here): MutGuard should
  forbid mutating the current cwd or any of its ancestors, so the collector never
  emits this shape; the SST holds on the universe regardless. F2 grep -i folds
  bytes/ASCII (C locale) — an icase match is determined only when the token AND the
  compared text are ASCII (an ASCII fold is byte-identical), else BOT (Python
  str.lower over-folds Unicode: İ->i̇, K->k, ſ->s, GRÜN). F3 error templates use the
  {template|text, exit} shape of benchmarks/p0/error-templates.json — predict/mining
  FAIL CLOSED on a string-only entry or a missing per-dialect exit (grep ENOENT=2
  both dialects, ls ENOENT=1 busybox/2 GNU — never a guessed code). F4 a failed
  recursive rm is NOT a no-op — it destroys depth-first before hitting EBUSY (bind
  mounts) / ENOTDIR ('link/' through a symlink), so a code!=0 recursive rm that is
  not a template-matched ENOENT folds conservatively (forget the subtree — the link
  AND, for a slashed arg, the resolved dir — + alias/dir-mine invalidation +
  revival). F5 the relative find-replay cache is cwd-blind (keyed on the raw string)
  — the replay is guarded on normpath(dir, self.cwd) == the cached resolved dir.

TWO VISIBILITY MODES (one implementation, a mode flag; §10.1):
  mode="collection"  the collector's live state authority (pool filtering,
                     meta.expected, state_scope, fs_clock, delta_text audit copy).
                     Evidence = the stored output (capped only at STORE_CAP).
  mode="sst"         render-parity folding for encode/eval: evidence is truncated
                     to the OBS_CAP=1600 render window before any mining, so nothing
                     past the render cap can enter belief (DG-4c visibility parity).
  Transition code is shared; the modes differ ONLY in the evidence extractor.

TOUCHED-SET PROPAGATION LAW (draft §10.2, verdict:eval-split M3 — written law):
  a mutation touches its target path, the target's parent (listing changes), all
  descendant paths for dir moves/removals, and the link node for ln; symlink-mediated
  reads touch both the link and its target. Overlay["touched"] maps path -> fs_clock
  of the last touch (set semantics + recency, consumed by render_canon's -l masks
  and the R7 untouched-since precondition).

PARSER TOTALITY (mint gate): parse_command() is TOTAL over the frozen universe —
  the v2 atomic templates (option lists pinned to the actual v2 mint: UNAME_OPTS /
  LS_OPTS below) + the G3 depth-1 grammar (prereg §4.1 BNF: exactly one operator,
  echo-redirect = ONE quoted 'PAYLOAD', destination a file strictly under /tmp/w/)
  + the audited process forms (`after j K 'echo TOK >> /tmp/w/task<j>.log' & echo
  $!` — the §3.3 audited effect only —, kill family, the vendored tj3-ps template).
  Any command outside it raises ParseError (an AssertionError): fail loudly, never
  guess. parse_command is the SINGLE universe authority — verbsig.sig() delegates
  membership to it (one totality, prereg §4.1); shell_state never imports verbsig.
  Deterministic, no wall-clock, no randomness, stdlib only.

Static knowledge (declared, §10.1 "frozen static knowledge"): "/", "/tmp" and the
Tier-W arena "/tmp/w" exist at t=0 (the workspace is protocol-seeded in-band; its
CONTENTS are unknown until observed). cwd starts at "/".
"""

import hashlib

# ---------------------------------------------------------------- constants

OBS_CAP = 1600     # render window (seq_worldmodel.OBS_CAP; duplicated as a frozen constant)
STORE_CAP = 65536  # collection-mode stored-output cap (collect_docker.V2_STORE_CAP)
WORKSPACE = "/tmp/w"
PS_PATH = "/usr/local/bin/tj3-ps"   # UD-9 Route B: the vendored busybox ps template
TIME_FREE_LS = {"", "-1", "-a", "-1a", "-a1"}   # names-only forms (draft §4.6)
# v2 mint option lists — pinned copies of collect_docker.py's UNAME_OPTS / LS_OPTS
# (minus the bare "" entry): the universe accepts EXACTLY the v2 atomic templates.
# The prereg §3.2 template-avoidance row (-i/-lt/-lS dropped for mutation-adjacent
# draws) is a POLICY rule, not a universe restriction — those opts ARE v2 commands.
UNAME_OPTS = ("-a", "-s", "-m", "-r", "-n", "-o", "-v", "-sm", "-sr")
LS_OPTS = ("-l", "-a", "-la", "-R", "-1", "-lh", "-lt", "-lS", "-ld", "-i", "-lr", "-ln")
_LS_OPT_SET = frozenset(LS_OPTS) | (TIME_FREE_LS - {""})   # + v3 TIME_FREE revisit forms
JOB_STATES = ("waiting", "stopped", "stopped_pending_term", "fired", "killed")
TESTOPS = ("-e", "-f", "-d", "-s")
BOT = None   # predict() bottom: not determined
# Chars that make GNU quotearg shell-escape (single-quote) an error-message filename while
# busybox always-quotes — a determined error-template prediction for such a path is
# dialect-divergent, so the SST BOTs it (P1-review DG-4b).
_QUOTEARG_SPECIAL = frozenset(" \t\n:'\"()[]{}|&;<>*?$`\\!#~")
# Round-4 class 5 (writability): docker bind-mounts (render_canon.RUNTIME_MOUNT_PATHS
# is the canonical copy; duplicated here like OBS_CAP — shell_state can't import
# render_canon back) and the read-only pseudo-filesystems. Mutations predicted on
# these are BOT, never determined-ok; reads are unaffected by THIS list.
RUNTIME_MOUNT_PATHS = frozenset({"/etc/resolv.conf", "/etc/hostname", "/etc/hosts"})
UNWRITABLE_PREFIXES = ("/proc", "/sys", "/dev")
# Round-4 class 8 (volatility): observed content/listings/link targets under these
# prefixes are never re-served (PIDs churn, uptime advances) — reads there are BOT.
VOLATILE_PREFIXES = ("/proc", "/sys")
# Round-4 class 6: _resolve's hop cap exceeded (symlink cycle) — a sentinel path
# that can never be a real fs key; _stat reports it 'unknown', so consumers go BOT.
ELOOP = "\x00:ELOOP"


class _Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


PARTIAL = _Sentinel("PARTIAL")   # content seen but capped/incomplete
UNKNOWN = _Sentinel("UNKNOWN")   # content never observed (blind capture / image file)


class ParseError(AssertionError):
    """Command outside the frozen universe (prereg §4.1) — parser totality is a mint gate."""


def canonical_pid(j):
    """Canonical PID of job j (draft §5.2). Scheme 100+10j is the working default;
    it is prereg-OPEN until the DG-8b watermark pricing freezes it by amendment."""
    return 100 + 10 * int(j)


def exit_cls(code):
    """The frozen TOTAL exit_cls mapping, specifics-first (prereg §4.2):
    {0 ok, 1 err, 2 usage, 126 not-executable, 127 not-found}; 124 = pilot-only class
    (zero at mint, DG-10b); 125 / >=128 / negative => ABORT family — raise, never
    classify (the no-producer rule would be falsified)."""
    code = int(code)
    if code == 124:
        return 124
    if code == 125 or code >= 128 or code < 0:
        raise ValueError(f"exit {code} is abort-family (prereg §4.2), never a class")
    if code in (0, 1, 2, 126, 127):
        return code
    return 1   # remaining nonzero -> err catch-all


# ---------------------------------------------------------------- canonical renders

def render_uptime(vt):
    """Canonical store-time uptime render (draft §5.1): clock/users/load masked to
    fixed tokens, elapsed virtualized to the virtual step-clock vt."""
    return f"00:00:00 up {vt}, 0 users, load average: 0.00, 0.00, 0.00"


def render_ps(rows):
    """THE canonical ps format (UD-9 Route B: ONE format on all images; the collector
    canonicalizes real tj3-ps output to this, the SST emits it directly). rows =
    [(pid, stat, args)] — already virtualized, self/transient rows already dropped."""
    lines = [f"{'PID':<5} {'STAT':<4} ARGS"]
    for pid, stat, args in sorted(rows):
        lines.append(f"{pid:<5} {stat:<4} {args}".rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------- path utilities

def normpath(path, cwd="/"):
    """Normalize a path (absolute or cwd-relative): resolves '.', '..', '//'."""
    if not path.startswith("/"):
        path = cwd.rstrip("/") + "/" + path
    parts = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
        else:
            parts.append(seg)
    return "/" + "/".join(parts)


def slashed(patharg):
    """F3 (round-2 review): a trailing-slash arg '<p>/' only resolves when p is a
    directory — normpath must NOT silently equate it with p for prediction, and
    fold-side evidence keyed on the exact path form stays conservative."""
    return isinstance(patharg, str) and patharg != "/" and patharg.endswith("/")


def logical_cwd(target, cwd):
    """F9 (round-3 review): the LOGICAL cwd `cd -L` realizes (what `pwd` prints,
    docker_env.py's recorded cwd). Never resolves symlinks; '.'/'..' processed
    textually; internal duplicate slashes collapse; a leading '//' (exactly two)
    is preserved (POSIX; verified on busybox ash + dash)."""
    base = target if target.startswith("/") else \
        (cwd if cwd.endswith("/") else cwd + "/") + target
    root = "//" if base.startswith("//") and not base.startswith("///") else "/"
    parts = []
    for seg in base.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
        else:
            parts.append(seg)
    return root + "/".join(parts)


def parent_of(p):
    p = p.rstrip("/") or "/"
    return "/" if p == "/" else (p.rsplit("/", 1)[0] or "/")


def basename_of(p):
    return p.rstrip("/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------- lexer

def _lex(cmd):
    """Tokenize a recorded command: POSIX single-quote aware (incl. the '\\'' escape
    that _sq emits), with | > >> & && as standalone operator tokens.
    Returns [(token, was_quoted)]."""
    toks, i, n = [], 0, len(cmd)
    while i < n:
        c = cmd[i]
        if c in " \t":
            i += 1
            continue
        if c == "'":
            j, buf = i + 1, []
            while True:
                if j >= n:
                    raise ParseError(f"unterminated quote in {cmd!r}")
                if cmd[j] == "'":
                    if cmd[j:j + 4] == "'\\''":   # the _sq escape for an embedded '
                        buf.append("'")
                        j += 4
                        continue
                    j += 1
                    break
                buf.append(cmd[j])
                j += 1
            toks.append(("".join(buf), True))
            i = j
            continue
        if cmd[i:i + 2] in (">>", "&&"):
            toks.append((cmd[i:i + 2], False))
            i += 2
            continue
        if c in ">|&":
            toks.append((c, False))
            i += 1
            continue
        j = i
        while j < n and cmd[j] not in " \t'>|&":
            j += 1
        toks.append((cmd[i:j], False))
        i = j
    return toks


def _unquoted_view(cmd):
    """cmd with the CONTENT of single-quoted spans (incl. the '\\'' escape) blanked
    to \\x00, same length/positions — the documented-exclusion scans (backslash,
    backtick, ';', '<', '$') see only unquoted characters, mirroring _lex's quoting.
    Raises ParseError on an unterminated quote (same as _lex)."""
    out, i, n = [], 0, len(cmd)
    while i < n:
        if cmd[i] != "'":
            out.append(cmd[i])
            i += 1
            continue
        out.append("'")
        j = i + 1
        while True:
            if j >= n:
                raise ParseError(f"unterminated quote in {cmd!r}")
            if cmd[j] == "'":
                if cmd[j:j + 4] == "'\\''":
                    out.append("\x00" * 4)
                    j += 4
                    continue
                out.append("'")
                j += 1
                break
            out.append("\x00")
            j += 1
        i = j
    return "".join(out)


# ---------------------------------------------------------------- parser (TOTAL)

_SIMPLE_VERBS = ("uname", "cd", "pwd", "ls", "cat", "head", "tail", "stat", "find",
                 "grep", "echo", "rm", "mv", "ln", "readlink", "mkdir", "touch",
                 "kill", "uptime", "sleep")
_STAT_FMT = "%n %s %F %a"   # the ONE pinned time-free stat format (draft §4.6)


def _err(cmd, why):
    raise ParseError(f"outside the frozen command universe (prereg §4.1): {cmd!r} — {why}")


def _parse_int(cmd, s, what):
    if not s.isdigit():
        _err(cmd, f"non-integer {what}: {s!r}")
    return int(s)


def _parse_prod(cmd, toks):
    """PROD ::= 'ls -1' D | 'cat' F  (G3; find-producers pruned, ls -l banned)."""
    words = [t for t, _ in toks]
    if len(words) == 3 and words[0] == "ls" and words[1] == "-1":
        return {"kind": "ls", "dir": _dashless(cmd, words[2])}
    if len(words) == 2 and words[0] == "cat":
        return {"kind": "cat", "file": _dashless(cmd, words[1])}
    _err(cmd, f"not a G3 producer: {' '.join(words)!r}")


def _parse_filt(cmd, toks):
    """FILT ::= 'head -n' K | 'tail -n' K | 'grep -F -m 8' TOK."""
    words = [t for t, _ in toks]
    if len(words) == 3 and words[0] in ("head", "tail") and words[1] == "-n":
        return {"kind": words[0], "k": _parse_int(cmd, words[2], "K")}
    if len(words) == 5 and words[:4] == ["grep", "-F", "-m", "8"]:
        return {"kind": "grep", "tok": _dashless(cmd, words[4], "grep TOK")}
    _err(cmd, f"not a G3 filter: {' '.join(words)!r}")


def _parse_read(cmd, toks):
    """READ P ::= cat P | ls -1 P | head -n K P  (COND arm)."""
    words = [t for t, _ in toks]
    if len(words) == 2 and words[0] == "cat":
        return {"form": "cat", "path": _dashless(cmd, words[1])}
    if len(words) == 3 and words[0] == "ls" and words[1] == "-1":
        return {"form": "ls", "opts": ["-1"], "path": _dashless(cmd, words[2])}
    if len(words) == 4 and words[0] == "head" and words[1] == "-n":
        return {"form": "head", "k": _parse_int(cmd, words[2], "K"),
                "path": _dashless(cmd, words[3])}
    _err(cmd, f"not a COND READ: {' '.join(words)!r}")


def _parse_simple(cmd, toks):
    words = [t for t, _ in toks]
    v = words[0]
    args = words[1:]
    if v == PS_PATH:
        if args not in ([], ["-o", "pid,stat,args"]):
            _err(cmd, "tj3-ps takes only the frozen '-o pid,stat,args' template")
        return {"form": "ps"}
    if v not in _SIMPLE_VERBS:
        _err(cmd, f"unknown verb {v!r}")
    if v == "uname":
        # pinned to the v2 mint templates: bare uname or ONE option from UNAME_OPTS
        if len(args) > 1 or (args and args[0] not in UNAME_OPTS):
            _err(cmd, f"uname template is 'uname [OPT]' with OPT in {UNAME_OPTS}")
        return {"form": "uname", "opts": args}
    if v == "cd":
        if len(args) > 1:
            _err(cmd, "cd takes at most one target")
        if args:
            _dashless(cmd, args[0], "cd target")   # round-5 F5 ('cd -' is OLDPWD)
        return {"form": "cd", "target": args[0] if args else ""}
    if v == "pwd":
        if args:
            _err(cmd, "pwd takes no arguments")
        return {"form": "pwd"}
    if v == "ls":
        opts = [a for a in args if a.startswith("-")]
        paths = [a for a in args if not a.startswith("-")]
        if len(paths) > 1 or len(opts) > 1:
            _err(cmd, "ls takes at most one option token and one path")
        # pinned to the v2 mint LS_OPTS plus the v3 TIME_FREE revisit forms
        if opts and opts[0] not in _LS_OPT_SET:
            _err(cmd, f"ls option {opts[0]!r} outside the v2 mint template list")
        return {"form": "ls", "opts": opts, "path": paths[0] if paths else None}
    if v == "cat":
        if len(args) != 1:
            _err(cmd, "cat takes exactly one path")
        return {"form": "cat", "path": _dashless(cmd, args[0])}
    if v in ("head", "tail"):
        if len(args) != 3 or args[0] != "-n":
            _err(cmd, f"{v} template is '{v} -n K PATH'")
        return {"form": v, "k": _parse_int(cmd, args[1], "K"),
                "path": _dashless(cmd, args[2])}
    if v == "stat":
        if len(args) != 3 or args[0] != "-c" or args[1] != _STAT_FMT:
            _err(cmd, f"stat template is stat -c '{_STAT_FMT}' PATH")
        return {"form": "stat", "path": _dashless(cmd, args[2])}
    if v == "find":
        # find D -maxdepth K [-type f|d] -name GLOB   (v2 mint template)
        if len(args) not in (5, 7) or args[1] != "-maxdepth" or args[-2] != "-name":
            _err(cmd, "find template is 'find D -maxdepth K [-type f|d] -name GLOB'")
        ftype = None
        if len(args) == 7:
            if args[3] != "-type" or args[4] not in ("f", "d"):
                _err(cmd, "find -type must be f or d")
            ftype = args[4]
        # round-5 F6: "raw" (the exact command string) keys the R7 replay cache —
        # the reconstructed unquoted flattening of two DIFFERENT quoted commands
        # can collide (cache-key injection); the raw string never can.
        return {"form": "find", "dir": _dashless(cmd, args[0], "find dir"),
                "maxdepth": _parse_int(cmd, args[2], "maxdepth"),
                "type": ftype, "glob": args[-1], "raw": cmd}
    if v == "grep":
        # grep -F [-i] -m 8 TOK PATH   (v2 mint template; -c dropped)
        rest = list(args)
        if not rest or rest[0] != "-F":
            _err(cmd, "grep template starts 'grep -F'")
        rest.pop(0)
        icase = bool(rest) and rest[0] == "-i"
        if icase:
            rest.pop(0)
        if len(rest) != 4 or rest[0] != "-m" or rest[1] != "8":
            _err(cmd, "grep template is 'grep -F [-i] -m 8 TOK PATH'")
        return {"form": "grep", "icase": icase,
                "tok": _dashless(cmd, rest[2], "grep TOK"),
                "path": _dashless(cmd, rest[3])}
    if v == "echo":
        if not args:
            _err(cmd, "bare echo needs a payload")
        _check_payload(cmd, args[0], joined=" ".join(args))
        return {"form": "echo", "text": " ".join(args)}
    if v == "rm":
        rec = bool(args) and args[0] == "-r"
        rest = args[1:] if rec else args
        if len(rest) != 1:
            _err(cmd, "rm template is 'rm [-r] PATH'")
        return {"form": "rm", "recursive": rec, "path": _dashless(cmd, rest[0])}
    if v == "mv":
        if len(args) != 2:
            _err(cmd, "mv template is 'mv SRC DST'")
        return {"form": "mv", "src": _dashless(cmd, args[0]),
                "dst": _dashless(cmd, args[1])}
    if v == "ln":
        sym = bool(args) and args[0] == "-s"
        rest = args[1:] if sym else args
        if len(rest) != 2:
            _err(cmd, "ln template is 'ln [-s] TARGET LINK'")
        return {"form": "ln", "symbolic": sym, "target": _dashless(cmd, rest[0]),
                "link": _dashless(cmd, rest[1])}
    if v == "readlink":
        if len(args) != 1:
            _err(cmd, "readlink template is 'readlink PATH'")
        return {"form": "readlink", "path": _dashless(cmd, args[0])}
    if v in ("mkdir", "touch"):
        if len(args) != 1:
            _err(cmd, f"{v} template is '{v} PATH'")
        return {"form": v, "path": _dashless(cmd, args[0])}
    if v == "kill":
        # kill [-STOP|-CONT|-9|-0] CPID  (bare kill = TERM; -INT excluded, UD-1)
        sig = "TERM"
        rest = list(args)
        if rest and rest[0].startswith("-"):
            flag = rest.pop(0)
            sigs = {"-STOP": "STOP", "-CONT": "CONT", "-9": "KILL", "-0": "0"}
            if flag not in sigs:
                _err(cmd, f"kill signal {flag!r} outside the frozen family")
            sig = sigs[flag]
        if len(rest) != 1:
            _err(cmd, "kill template is 'kill [-SIG] CPID'")
        return {"form": "kill", "sig": sig, "cpid": _parse_int(cmd, rest[0], "cpid")}
    if v == "uptime":
        if args:
            _err(cmd, "uptime takes no arguments")
        return {"form": "uptime"}
    if v == "sleep":
        if len(args) != 1 or args[0] not in ("0", "1"):
            _err(cmd, "foreground sleep is restricted to {0,1} (draft §5.1)")
        return {"form": "sleep", "k": int(args[0])}
    _err(cmd, f"unhandled verb {v!r}")


def _glob_bearing(tok):
    """Round-4 class 7 (universe seam): an unquoted token the shell would
    glob-expand — contains '*' or '?' or a POTENTIAL bracket expression ('['
    with a later ']' in the same word). An unterminated '[' is literal per
    POSIX, which keeps the real /usr/bin/[ mint path in-universe and lets the
    cond form's standalone '[' / ']' tokens pass untouched. Quoted tokens
    (e.g. the v2 templates' -name '*.conf') are never scanned."""
    if "*" in tok or "?" in tok:
        return True
    i = tok.find("[")
    return i >= 0 and "]" in tok[i + 1:]


def _dashless(cmd, tok, what="path"):
    """Round-5 F5 (leading-dash seam): a leading-dash token in a PATH argument
    position (necessarily relative — absolute paths start '/') or the grep TOK
    position is parsed as an OPTION by real tools (busybox `touch '-x'` errors
    'unrecognized option'; grep eats '-v' as a flag and re-reads its argv) —
    outside the universe. The FROZEN template option positions ('-n K', '-m 8',
    '-c FMT', '-maxdepth K', the -name GLOB operand, kill signals) are exempt
    by position: they are pinned template tokens, never free arguments."""
    if tok.startswith("-"):
        _err(cmd, f"leading-dash {what} {tok!r} (round-5 seam: real tools parse"
                  " it as an option)")
    return tok


def _check_payload(cmd, first, joined=None):
    """F13 (round-3 review) — the §4.3 payload charset LAW, enforced on the payload
    CONTENT (quoted or not): backslash (dash's XSI echo interprets '\\n' etc.),
    '$' and backtick (the after helper shell would expand them) are banned, and a
    leading-dash payload is banned (busybox echo eats '-n'/'-e' as flags). Applies
    to bare echo, the echo-redirect payload, and the after-effect TOK."""
    text = joined if joined is not None else first
    for ch in "\\$`":
        if ch in text:
            _err(cmd, f"payload charset law (§4.3): {ch!r} banned in echo payloads")
    if first.startswith("-"):
        _err(cmd, "payload charset law (§4.3): leading-dash echo payload (flag-ambiguous)")


def parse_command(cmd):
    """TOTAL parser over the frozen command universe (prereg §4.1): v2 atomic
    templates (option lists pinned to the actual v2 mint) + G3 depth-1 composed
    forms + the audited process forms. Raises ParseError on anything else —
    parser totality is a mint gate, and this parser is the SINGLE universe
    authority (verbsig.sig delegates membership here)."""
    if "\n" in cmd or "\r" in cmd:
        _err(cmd, "embedded newline")
    view = _unquoted_view(cmd)        # documented exclusions scan unquoted chars only
    for ch, why in (("\\", "backslash excluded (payload charset, draft §4.3)"),
                    ("`", "backtick excluded"),
                    (";", "';' excluded"),
                    ("<", "'<'/'<<<'/REDIR_IN excluded"),
                    ('"', "double quote excluded (round-4 seam: the SST would alias"
                          " the quoted and bare spellings of one path)"),
                    ("~", "unquoted tilde excluded (round-4 seam: tilde expansion)")):
        if ch in view:
            _err(cmd, why)
    toks = _lex(cmd)
    if not toks:
        _err(cmd, "empty")
    for t, q in toks:
        # round-4 class 7: unquoted glob-bearing words leave the universe — the
        # shell would expand them, so the literal text is not the executed path
        if not q and _glob_bearing(t):
            _err(cmd, f"unquoted shell-expansion characters in {t!r} (round-4 seam)")
    words = [t for t, _ in toks]
    unq = [t for t, q in toks if not q]
    if words[0] != "after" and "$" in view:
        _err(cmd, "'$' outside the after form")

    if words[0] == "after":
        # after j K 'effect' & echo $!   (the ONE canonical launch shape, §3.3);
        # the effect is ONLY the audited echo-append form onto the job's own log:
        #   echo TOK >> /tmp/w/task<j>.log
        if len(toks) != 7 or words[4] != "&" or words[5] != "echo" or words[6] != "$!":
            _err(cmd, "after template is \"after j K 'effect' & echo $!\"")
        j = _parse_int(cmd, words[1], "job index")
        k = _parse_int(cmd, words[2], "K")
        if cmd.strip() != f"after {words[1]} {words[2]} '{words[3]}' & echo $!":
            _err(cmd, "after launch must be the ONE canonical single-spaced shape")
        eview = _unquoted_view(words[3])
        for ch in ("\\", "`", ";", "<", '"', "~", "$"):
            # round-4 seam: the effect string re-enters a helper shell UNQUOTED —
            # the outer scan blanked it (quoted span), so scan it separately
            if ch in eview:
                _err(cmd, f"{ch!r} excluded inside the after effect (round-4 seam)")
        etoks = _lex(words[3])
        for t, q in etoks:
            if not q and _glob_bearing(t):
                _err(cmd, f"unquoted shell-expansion characters in after effect {t!r}")
        ewords = [t for t, _ in etoks]
        if len(etoks) != 4 or ewords[0] != "echo" or ewords[2] != ">>":
            _err(cmd, "after effect must be the audited echo-append form (draft §3.3)")
        if ewords[3] != f"{WORKSPACE}/task{j}.log":
            _err(cmd, f"after effect must append to {WORKSPACE}/task{j}.log (draft §4.6)")
        _check_payload(cmd, ewords[1])   # F13: the charset law covers after TOKs too
        eff = {"form": "redir", "op": ">>",
               "prod": {"kind": "echo", "payload": ewords[1]}, "dst": ewords[3]}
        return {"form": "after", "j": j, "K": k, "effect": words[3], "effect_parsed": eff}

    if "|" in unq:
        i = next(i for i, (t, q) in enumerate(toks) if t == "|" and not q)
        return {"form": "pipe", "prod": _parse_prod(cmd, toks[:i]),
                "filt": _parse_filt(cmd, toks[i + 1:])}

    redir = [i for i, (t, q) in enumerate(toks) if t in (">", ">>") and not q]
    if redir:
        if len(redir) != 1:
            _err(cmd, "exactly one operator (depth-1 grammar)")
        i = redir[0]
        if len(toks) != i + 2:
            _err(cmd, "redirect needs exactly one destination")
        dst, dst_quoted = toks[i + 1]
        if dst_quoted:
            _err(cmd, "redirect destination must be an unquoted workspace path")
        # WSF: a FILE destination strictly UNDER the workspace (dst == /tmp/w
        # rejected) — F12 (round-3 review): checked on the NORMALIZED path too,
        # so '..'-escapes of the Tier-W arena are out of the universe.
        if not dst.startswith(WORKSPACE + "/") or dst == WORKSPACE + "/" \
                or not normpath(dst).startswith(WORKSPACE + "/"):
            _err(cmd, f"redirection target must be a file under {WORKSPACE}/")
        left = toks[:i]
        if left and left[0][0] == "echo" and not left[0][1]:
            # BNF §4.1: "echo" 'PAYLOAD' — exactly ONE single-quoted payload token
            if len(left) != 2 or not left[1][1] or left[1][0] == "":
                _err(cmd, "echo redirect payload must be ONE single-quoted 'PAYLOAD'")
            _check_payload(cmd, left[1][0])      # F13: the §4.3 charset law
            prod = {"kind": "echo", "payload": left[1][0]}
        else:
            prod = _parse_prod(cmd, left)
        return {"form": "redir", "op": toks[i][0], "prod": prod, "dst": dst}

    if words[0] == "[":
        # [ TESTOP P ] && READ P   (ONE conditional form; same P both sides)
        if len(words) < 6 or words[3] != "]" or words[4] != "&&":
            _err(cmd, "cond template is '[ TESTOP P ] && READ P'")
        if words[1] not in TESTOPS:
            _err(cmd, f"TESTOP must be one of {TESTOPS}")
        _dashless(cmd, words[2], "tested path")   # round-5 F5
        read = _parse_read(cmd, toks[5:])
        if read["path"] != words[2]:
            _err(cmd, "COND READ path must equal the tested path")
        return {"form": "cond", "testop": words[1], "path": words[2], "read": read}

    if "&" in unq or "&&" in unq:
        _err(cmd, "'&'/'&&' outside the after/cond templates")
    return _parse_simple(cmd, toks)


# =================================================================== ShellState

def _sha(text):
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


class ShellState:
    """The one tracker (draft §10.1 Overlay schema, normative). Fold order discipline:
    predict(t, cmd) BEFORE fold(step_t); fold steps strictly in trajectory order."""

    def __init__(self, mode="collection", error_templates=None):
        assert mode in ("collection", "sst"), mode
        self.mode = mode
        # error templates: probe-harvested per-image table (§3.5); key -> str or
        # {"text": fmt, "exit": n}; fmt placeholders {path} / {pid}. predict() answers
        # error steps ONLY through this table (no hand-authored dialect enum) — a
        # missing key means BOT, never a guess.
        self.error_templates = dict(error_templates or {})
        self.cwd = "/"                       # protocol constant at t=0
        self._cwd_stale = False              # F1 (round-6): the stale-cwd latch — set
                                             # when a tombstone/forget/move covers the
                                             # tracked cwd (or an ancestor); lifted when
                                             # a recorded cwd re-anchors on a live dir
        self.fs = {}                         # path -> VNode dict
        self.ws = {}                         # wsf -> {producer, observed, appends, content_lines}
        self.jobs = {}                       # j -> {cpid, K, effect, state, launch_vt, deferrals}
        self.fs_clock = 0
        self.touched = {}                    # path -> fs_clock of last touch (set + recency)
        self.vt = 0                          # virtual step-clock = index of next step to fold
        self.mismatches = []                 # (vt, cmd, note) — collection-mode audit trail
        self._find_cache = {}                # cmd -> (output, fs_clock_at_obs, dir)
        self._delta = []                     # canonical fs-delta ops of the LAST folded step
        for p, kind in (("/", "dir"), ("/tmp", "dir"), (WORKSPACE, "dir")):
            # declared static knowledge; link-ness protocol-certain (the workspace
            # is protocol-seeded as a real dir; / and /tmp are real dirs on all mints)
            self.fs[p] = self._vnode(kind, UNKNOWN, "image", linkness_known=True)

    # ------------------------------------------------------------- vnode plumbing

    def _vnode(self, kind, content, provenance, **kw):
        # linkness_known — the LINK-CONSERVATISM LAW (round-3 review): link-ness is
        # definitively known ONLY for nodes the SST itself created (mkdir/touch/ln/
        # redirect-create) or learned via a successful readlink. Image-inherited
        # nodes (cat/ls/find/cond evidence) keep False: a cat success proves
        # readable content, NOT non-link-ness (kind "file" means file-or-symlink);
        # a template-matched absence error proves only "does not resolve".
        # severed (round-5 F1): True once the node's inode became reachable via a
        # path the SST no longer tracks (a conservative fold severed its hardgroup)
        # — such a node loses the linkness_known exemption in
        # _mined_alias_invalidate forever (the exemption is only sound for groups
        # that remain FULLY tracked).
        v = {"kind": kind, "content": content, "deleted": False, "link_target": None,
             "fs_clock": self.fs_clock, "provenance": provenance, "payload_sha256": None,
             "trailing_nl_known": False, "entries": None, "entries_complete": None,
             "hardgroup": None, "linkness_known": False, "severed": False}
        v.update(kw)
        if isinstance(v["content"], str):
            v["payload_sha256"] = _sha(v["content"])
        return v

    def _stat(self, path):
        """Belief existence: ('alive', vnode) | ('dead', vnode) | ('unknown', None)."""
        v = self.fs.get(path)
        if v is None:
            par = self.fs.get(parent_of(path))
            # F2 (round-3 review): a VISIBLE (non -a) listing never renders
            # dotfiles, so it entails absence only for non-dot basenames; dotfile
            # absence needs entries_complete == "all".
            if par is not None and not par["deleted"] and par.get("entries_complete") \
                    and basename_of(path) not in par["entries"] \
                    and (par["entries_complete"] == "all"
                         or not basename_of(path).startswith(".")):
                return "dead", None          # complete parent listing entails absence
            if parent_of(path) == WORKSPACE and basename_of(path).startswith("task") \
                    and basename_of(path).endswith(".log"):
                # reserved job-effect namespace: absent at t=0 (Tier-W lexicon
                # cannot mint it; the JOB-chain pre-fire absence check keys here).
                # Round-5 F2: the entailment needs the workspace itself known-live
                # — a tombstoned /tmp/w kills the reservation (the certain-absent
                # verdict would otherwise feed _wchain a phantom creation slot).
                wv = self.fs.get(WORKSPACE)
                if wv is not None and not wv["deleted"] and wv["kind"] == "dir":
                    return "dead", None
            return "unknown", None
        return ("dead", v) if v["deleted"] else ("alive", v)

    def _resolve(self, path):
        """Follow FINAL-component symlink chains through KNOWN links (bounded); the
        policy creates only flat links, so intermediate-component links don't occur.
        Round-4 class 6: hop-capped at 40 (kernels ELOOP around that order) — a
        chain that exceeds the cap (symlink cycle) returns the ELOOP sentinel,
        which _stat reports as 'unknown', so every consumer goes BOT."""
        for _ in range(40):
            v = self.fs.get(path)
            if v is None or v["deleted"] or v["kind"] != "symlink" or not v["link_target"]:
                return path
            path = normpath(v["link_target"], parent_of(path))
        return ELOOP

    def _touch(self, *paths):
        for p in paths:
            self.touched[p] = self.fs_clock

    @staticmethod
    def _dead_certain(v):
        """A 'dead' _stat verdict is lstat-certain (the PATH itself is absent, not
        merely non-resolving) when it came from entailment (v is None: complete
        parent listing / reserved job-log namespace) or from an SST-performed
        deletion (linkness_known tombstone). Template-mined absences through
        resolving reads (cat/cd/head/...) keep linkness_known False: the path may
        be a dangling symlink (LINK-CONSERVATISM LAW, round-3)."""
        return v is None or bool(v.get("linkness_known"))

    @staticmethod
    def _volatile(path):
        """Round-4 class 8: /proc and /sys churn between reads (PIDs, uptime,
        self-links) — observed content/listings/link targets there are never
        re-served as determined."""
        return any(path == pre or path.startswith(pre + "/")
                   for pre in VOLATILE_PREFIXES)

    @staticmethod
    def _unwritable(*paths):
        """Round-4 class 5: mutations on docker bind-mounts (EBUSY) or under the
        read-only pseudo-filesystems fail or diverge by dialect — predicted
        mutations there are BOT, never determined-ok. Reads are unaffected."""
        for p in paths:
            if p in RUNTIME_MOUNT_PATHS:
                return True
            if any(p == pre or p.startswith(pre + "/")
                   for pre in UNWRITABLE_PREFIXES):
                return True
        return False

    @staticmethod
    def _mount_ancestor(path):
        """A recursive/rename mutation of an ANCESTOR of a runtime bind-mount hits
        EBUSY on the mount row — never determined-ok (round-4 class 5)."""
        pre = path.rstrip("/") + "/"
        return any(m.startswith(pre) for m in RUNTIME_MOUNT_PATHS)

    @staticmethod
    def _relarg(a):
        """A cwd-relative path argument: not absolute, not empty/None. Bare/relative
        path resolution keys off self.cwd, so such an argument is cwd-dependent."""
        return isinstance(a, str) and a != "" and not a.startswith("/")

    def _cwd_dependent(self, p):
        """F1 (round-6): whether predicting/folding this command's outcome depends on
        the tracked cwd — pwd, ANY cd (docker_env's `cd <cwd> && cd <t>` prologue
        short-circuits on a dead cwd, so absolute cd fails too), bare ls, or a
        RELATIVE path argument (resolved via normpath(arg, self.cwd)). Absolute-only
        commands are cwd-independent."""
        form = p["form"]
        if form in ("pwd", "cd"):
            return True
        if form == "ls":
            return p.get("path") is None or self._relarg(p.get("path"))
        rel = self._relarg
        if form in ("cat", "head", "tail", "stat", "grep", "rm", "readlink",
                    "mkdir", "touch"):
            return rel(p.get("path"))
        if form == "mv":
            return rel(p.get("src")) or rel(p.get("dst"))
        if form == "ln":
            return rel(p.get("target")) or rel(p.get("link"))
        if form == "find":
            return rel(p.get("dir"))
        if form == "cond":
            return rel(p.get("path"))
        if form in ("redir", "pipe"):
            prod = p["prod"]
            pp = prod.get("file") if prod["kind"] == "cat" else \
                (prod.get("dir") if prod["kind"] == "ls" else None)
            return rel(pp)          # the redirect dst is always an absolute /tmp/w/ path
        return False                # echo, uname, uptime, sleep, kill, ps, after

    def _cwd_anchored(self):
        """F1 (round-6): the tracked cwd resolves to a known-live directory, i.e.
        docker_env's `cd <cwd> 2>/dev/null` prologue succeeds. When it does not (a
        tombstone/forget/move covered the cwd or an ancestor), non-cd commands run
        from the exec start dir while the recorded cwd stays stale, so cwd-dependent
        surfaces desync. Used to lift the stale latch once a recorded cwd re-anchors
        on a live dir (e.g. the path is re-created)."""
        r = self._resolve(normpath(self.cwd))
        if r == ELOOP:
            return False
        st, v = self._stat(r)
        return st == "alive" and v is not None and v["kind"] == "dir"

    def _mark_cwd_stale_if_covered(self, paths):
        """F1 (round-6): a tombstone / forget / conservative-move of any of `paths`
        that covers the tracked cwd (the cwd itself or an ancestor) desyncs the cd
        prologue from belief — latch cwd-dependent predictions/folds until a recorded
        cwd re-anchors on a live dir (fold() lifts the latch via _cwd_anchored)."""
        cwd = normpath(self.cwd)
        for pth in paths:
            base = normpath(pth)
            if cwd == base or cwd.startswith(base.rstrip("/") + "/"):
                self._cwd_stale = True
                return

    def _dotdot_sound(self, patharg, cwd=None):
        """Round-4 class 2: normpath's textual '..' collapse is belief-sound only
        when every erased intermediate is a known-live, LINKNESS-KNOWN directory —
        POSIX resolution requires the intermediate to exist (ENOENT otherwise),
        and a symlink intermediate re-roots '..' at its TARGET's parent, so a
        dir-from-ls-evidence (possible symlink-to-dir) is not enough. Paths
        without '..' are trivially sound; '/..' at the root is '/' (POSIX)."""
        path = patharg if patharg.startswith("/") else \
            (cwd or self.cwd).rstrip("/") + "/" + patharg
        segs = path.split("/")
        if ".." not in segs:
            return True
        parts = []
        for seg in segs:
            if seg in ("", "."):
                continue
            if seg == "..":
                if parts:
                    inter = "/" + "/".join(parts)
                    v = self.fs.get(inter)
                    if v is None or v["deleted"] or v["kind"] != "dir" \
                            or not v.get("linkness_known"):
                        return False
                    parts.pop()
            else:
                parts.append(seg)
        return True

    def _workspace_chain_live(self, path):
        """Round-5 F2: every ancestor of `path` strictly inside the workspace
        (up to and including /tmp/w itself) is a known-live, LINKNESS-KNOWN dir
        (mkdir-created or the protocol-seeded arena root). Only then is the
        never-minted certain-absent argument sound: the arena never lost
        tracking on the way down."""
        anc = parent_of(path)
        while True:
            v = self.fs.get(anc)
            if v is None or v["deleted"] or v["kind"] != "dir" \
                    or not v.get("linkness_known"):
                return False
            if anc == WORKSPACE:
                return True
            anc = parent_of(anc)

    def _sever_group(self, member):
        """Round-5 F1: a conservative fold is about to lose track of `member`'s
        path while its inode stays live (mv to an uncertain landing site, a
        forgotten '..' candidate). Severing loses inode tracking: every REMAINING
        hardgroup peer degrades to unknown content NOW and is flagged `severed`
        (it permanently loses the linkness_known exemption in
        _mined_alias_invalidate — future untracked writes through the escaped
        side can change the shared bytes invisibly). Caller bumps fs_clock."""
        v = self.fs.get(member)
        if v is None or not v.get("hardgroup"):
            return
        group = v["hardgroup"]
        group.discard(member)
        for r in group:
            w = self.fs.get(r)
            if w is not None and not w["deleted"]:
                w["content"] = UNKNOWN
                w["trailing_nl_known"] = False
                w["payload_sha256"] = None
                w["severed"] = True
                w["fs_clock"] = self.fs_clock
                self._touch(r)

    def _revive_uncertain_absences(self):
        """Round-5 F3 (the ABSENCE-REVIVAL LAW — the dual of the round-4 alias-
        staleness law): a tombstone mined through UNCERTAIN resolution (an error
        template match / a cond -e miss; linkness_known False — the path may be
        a dangling symlink) proves only 'did not resolve THEN'. Any subsequent
        successful creation event anywhere in the world (echo>/mkdir/touch/ln/
        mv-in/fire) can make such a path resolve again, so every uncertain
        tombstone is DROPPED (reads go BOT, never a stale determined error).
        lstat-certain deadness (SST-performed rm/mv, entailed absence) and
        complete-listing entailments are maintained by the folds themselves and
        are unaffected."""
        for q in [k for k, v in self.fs.items()
                  if v["deleted"] and not v.get("linkness_known")]:
            del self.fs[q]

    def _forget_path(self, path):
        """Round-4 class 2 conservative fold: a SUCCESSFUL mutation whose textual
        '..' collapse is not belief-sound landed at an uncertain real path. Drop
        all belief at the textual candidate (and its subtree), break the textual
        parent's listing completeness, and touch — never claim, never tombstone.
        (A symlink intermediate could stale a THIRD dir's complete listing; such
        '..'-mutations are outside every mint policy, accepted residual.)"""
        self.fs_clock += 1
        pre = path.rstrip("/") + "/"
        for q in [k for k in self.fs if k == path or k.startswith(pre)]:
            self._sever_group(q)                 # round-5 F1: peers degrade NOW
            self.fs.pop(q)
            self.ws.pop(q, None)
            self._touch(q)
        par = self.fs.get(parent_of(path))
        if par is not None and par.get("entries_complete"):
            par["entries_complete"] = None       # completeness no longer trusted
        self._touch(path, parent_of(path))
        self._mark_cwd_stale_if_covered([path])  # F1: a forget covering the cwd desyncs
        # the uncertain mutation may have CREATED something somewhere (round-5 F3)
        self._revive_uncertain_absences()

    def _alias_basis(self, path, v):
        """Round-4 class 3 helper: (applies, basis) — the byte content an
        UNKNOWN-alias reader of `path` would have served before a mutation.
        basis is that content when exactly comparable (str), None when uncertain.
        Dirs never serve byte content; a KNOWN symlink resolving to a dead/absent
        target was already unreadable, so destroying it strands no mined bytes."""
        if v is not None and v["kind"] == "dir":
            return False, None
        if v is not None and v["kind"] == "symlink" and v.get("linkness_known"):
            final = self._resolve(path)
            fv = self.fs.get(final)
            if fv is None:
                return True, None                # target unknown: uncertain
            if fv["deleted"] or fv["kind"] == "dir":
                return False, None               # nothing byte-readable was lost
            c = fv["content"]
            return True, (c if isinstance(c, str) else None)
        c = v["content"] if v is not None else None
        return True, (c if isinstance(c, str) else None)

    def _mined_alias_invalidate(self, basis, exclude=frozenset()):
        """Round-4 class 3 (alias staleness): a mutation destroyed or changed the
        bytes reachable through some path; content MINED at a linkness-unknown
        node may alias that inode (image symlinks are invisible to the SST), so
        its replay would serve stale bytes. A true alias must share bytes —
        content equality (modulo the record channel's trailing-newline rstrip)
        is the NECESSARY condition: invalidate exactly the equal-content mined
        nodes; when the mutated bytes are not exactly known (basis None — capped/
        partial/unobserved), invalidate every mined-content node. SST-created
        nodes (linkness_known) are never mined evidence and keep their bytes;
        `exclude` carries the mutated node's own hardgroup peers (they share the
        inode legitimately and keep the content, per POSIX link semantics).
        Round-5 F1: the linkness_known exemption applies only to nodes whose
        hardgroup remained fully tracked — a `severed` node's inode has an
        untracked alias, so its bytes are as invalidatable as mined evidence."""
        cmp = basis.rstrip("\n") if isinstance(basis, str) else None
        for q, w in self.fs.items():
            if q in exclude or w["deleted"] \
                    or (w.get("linkness_known") and not w.get("severed")) \
                    or not isinstance(w["content"], str):
                continue
            if cmp is None or w["content"].rstrip("\n") == cmp:
                w["content"] = UNKNOWN
                w["trailing_nl_known"] = False
                w["payload_sha256"] = None
                w["fs_clock"] = self.fs_clock
                self._touch(q)

    @staticmethod
    def _dir_alias_basis(v):
        """Round-5 F4 helper: (applies, basis) — whether destroying this node can
        strand LISTING/KIND facts mined at an aliasing path, and the exact
        all-entries listing it held when known. Destroying a known non-dir
        (file / symlink — rm/mv operate on the LINK, never its target dir) can
        never stale a dir mine; a kind-unknown node MAY be a real dir (basis
        unknowable); a dir node's listing is the basis only when 'all'-complete."""
        if v is None or v["kind"] is None:
            return True, None
        if v["kind"] != "dir":
            return False, None
        if v.get("entries_complete") == "all" and v["entries"] is not None:
            return True, frozenset(v["entries"])
        return True, None

    def _mined_dir_invalidate(self, basis, exclude=frozenset()):
        """Round-5 F4 (dir-mine staleness — the listing/kind dual of the round-4
        content law): a mutation destroyed (or may have destroyed) a REAL
        directory; entries/entries_complete/kind MINED at a linkness-unknown dir
        node may alias that dir (an image symlink-to-dir is invisible to the
        SST), so their replay would serve stale determined listings and -d
        truths. A true alias must share the listing — set CONSISTENCY with the
        destroyed listing is the necessary condition: keep only nodes whose
        known listing facts contradict the basis; with basis None (destroyed
        listing not exactly known) invalidate every candidate. SST-created dirs
        (linkness_known) are never mined evidence and keep their facts."""
        vis = None if basis is None else \
            frozenset(n for n in basis if not n.startswith("."))
        for q, w in self.fs.items():
            if q in exclude or w["deleted"] or w.get("linkness_known") \
                    or w["kind"] != "dir":
                continue
            if basis is not None:
                ents = w["entries"] or set()
                comp = w.get("entries_complete")
                if not ents <= basis:
                    continue                     # holds a name the basis lacks
                if comp == "all" and frozenset(ents) != basis:
                    continue                     # complete but misses basis names
                if comp == "visible" and not vis <= ents:
                    continue                     # misses a visible basis name
            w["kind"] = None
            w["entries"] = None
            w["entries_complete"] = None
            w["fs_clock"] = self.fs_clock
            self._touch(q)

    def _wchain(self, dst):
        """LINK-CONSERVATISM LAW (round-3 review): follow the WRITE chain from dst
        through SST-known symlinks. Returns (final, chain, fully_known) where
        fully_known means the landing site of a redirect/append through dst is
        certain: every hop's link-ness is SST-known and the final node is a
        linkness-known file, a linkness-certain absent slot, or a never-minted
        path strictly inside the protocol-seeded workspace (Tier-W arena: absent
        at t=0 unless the SST minted it). Anything else — image-inherited nodes,
        kind-unknown nodes, chains leaving the workspace into unknown space —
        is NOT fully known and the write must degrade, never claim."""
        chain, cur = [], dst
        for _ in range(8):
            chain.append(cur)
            v = self.fs.get(cur)
            if v is None:
                st, _ = self._stat(cur)
                if st == "dead":
                    return cur, chain, True          # entailed-absent creation slot
                # round-5 F2: the never-minted-workspace certain-absent rule needs
                # every ancestor between the path and the arena root to be an
                # SST-known live dir — a conservative-fold shadow (kind None), an
                # ls-mined dir (possible symlink), a missing or tombstoned
                # ancestor all mean the arena lost tracking there, so the real
                # subtree may contain untracked nodes (the path itself included).
                return cur, chain, (cur.startswith(WORKSPACE + "/")
                                    and self._workspace_chain_live(cur))
            if v["deleted"]:
                return cur, chain, self._dead_certain(v)
            if v["kind"] == "symlink" and v["link_target"]:
                if not v.get("linkness_known"):
                    return cur, chain, False
                cur = normpath(v["link_target"], parent_of(cur))
                continue
            if v["kind"] == "file":
                return cur, chain, bool(v.get("linkness_known"))
            return cur, chain, False                 # dir / kind-unknown target
        return cur, chain, False                     # over-length chain

    def _degrade_write(self, chain, provenance, ensure):
        """LAW degrade: a write went through a chain that is not fully SST-known —
        claim no content anywhere; every known node on the chain (and every
        hardgroup peer of a chain node) drops to unknown content so later reads
        are BOT. ensure=True (the write succeeded) additionally records that the
        chain paths exist now; ensure=False (failed redirect) claims no existence.
        Caller bumps fs_clock."""
        for q in chain:
            v = self.fs.get(q)
            peers = set(v["hardgroup"]) if v is not None and v.get("hardgroup") else set()
            if v is None or v["deleted"]:
                if ensure:
                    self.fs[q] = self._vnode(None, UNKNOWN, provenance)
            elif not (v["kind"] == "symlink" and v["link_target"]):
                v["content"] = UNKNOWN
                v["trailing_nl_known"] = False
                v["payload_sha256"] = None
                v["fs_clock"] = self.fs_clock
            self._touch(q, parent_of(q))
            for r in peers:
                w = self.fs.get(r)
                if r != q and w is not None and not w["deleted"]:
                    w["content"] = UNKNOWN
                    w["trailing_nl_known"] = False
                    w["payload_sha256"] = None
                    w["fs_clock"] = self.fs_clock
                    self._touch(r)

    def _parent_entry(self, path, add=None, remove=None):
        """Keep a complete parent listing complete across mutations (edit-replay R6)."""
        par = self.fs.get(parent_of(path))
        if par is not None and par.get("entries_complete"):
            if add is not None:
                par["entries"].add(basename_of(add))
            if remove is not None:
                par["entries"].discard(basename_of(remove))

    def _set_content(self, path, content, trailing_nl_known, provenance=None):
        """Set file content, syncing every hardlink peer (shared inode)."""
        v = self.fs.get(path)
        group = v["hardgroup"] if v is not None and v["hardgroup"] else {path}
        for q in group:
            w = self.fs.get(q)
            if w is None:
                continue
            w["content"] = content
            w["trailing_nl_known"] = trailing_nl_known
            w["payload_sha256"] = _sha(content) if isinstance(content, str) else None
            w["fs_clock"] = self.fs_clock
            if provenance:
                w["provenance"] = provenance

    def state_scope_of(self, path):
        """Meta helper (collection mode): 'created' | 'mutated' | 'native'."""
        v = self.fs.get(self._resolve(normpath(path, self.cwd)))
        if v is None:
            return "native"
        prov = v["provenance"]
        if prov.startswith("redirect@") or prov.startswith(("mut:mkdir", "mut:touch")):
            return "created"
        if prov.startswith("mut:") or v["deleted"]:
            return "mutated"
        return "mutated" if normpath(path, self.cwd) in self.touched else "native"

    # ------------------------------------------------------------- evidence extractor

    def _evidence(self, step):
        """THE mode split (draft §10.1): collection mode reads the stored output;
        sst mode truncates to the OBS_CAP render window first, so no fact past the
        cap can enter belief (DG-4c). Returns (text, capped)."""
        out = step.get("output", "") or ""
        if self.mode == "sst":
            marker = out.rfind("\n...[")
            if marker >= 0 and out.endswith(" more chars]"):
                return out[:marker][:OBS_CAP], True      # already-rendered input
            if len(out) > OBS_CAP:
                return out[:OBS_CAP], True
            return out, False
        return out, len(out) >= STORE_CAP

    # ------------------------------------------------------------- job automaton

    def _fires_due(self, vt):
        """Jobs whose deterministic fire lands in the prologue of step vt (§3.2/§3.3):
        waiting AND launch_vt + K <= vt."""
        return sorted(j for j, job in self.jobs.items()
                      if job["state"] == "waiting" and job["launch_vt"] + job["K"] <= vt)

    def _fire_landing(self, eff):
        """Round-5 F2: the landing-soundness predicate for a job fire — the SAME
        test _pred_redir applies before a determined claim: dst writable, dst not
        a known dir, write chain fully SST-known, AND the final landing's parent
        a known-live dir. A fire whose landing fails this predicate may have
        FAILED in reality (ENOENT / EISDIR / ENOTDIR) or landed at an untracked
        inode — 'fires are deterministic by construction' holds for the
        automaton, not for the filesystem boundary. Returns (final, chain, sound)."""
        dst = normpath(eff["dst"], self.cwd)
        final, chain, fully = self._wchain(dst)
        if self._unwritable(dst):
            return final, chain, False
        rdst = self._resolve(dst)
        dstat, dv = self._stat(rdst)
        if dstat == "alive" and dv is not None and dv["kind"] == "dir":
            return final, chain, False           # append onto a dir: EISDIR
        if not fully:
            return final, chain, False
        pst, pv = self._stat(parent_of(final))
        if pst != "alive" or not pv or pv["kind"] != "dir":
            return final, chain, False           # dead/unknown parent: open fails
        return final, chain, True

    def _fire_due(self, vt):
        """Commit due fires (fold-time): state -> fired, effect applied to the
        overlay; the fs change is attributed to THIS step's delta (its prologue
        fired it). Round-5 F2: the landing routes through the SAME soundness
        predicate as _pred_redir — an unsound landing claims NOTHING (degrade,
        never a minted node): the real append may have failed."""
        for j in self._fires_due(vt):
            job = self.jobs[j]
            job["state"] = "fired"               # the automaton fired either way
            job["deferrals"] = vt - (job["launch_vt"] + job["K"])
            eff = job["effect_parsed"]
            final, chain, sound = self._fire_landing(eff)
            if sound:
                self._apply_redirect(eff, provenance=f"redirect@{vt}")
            else:
                self.fs_clock += 1
                self._degrade_write(chain, f"redirect@{vt}", ensure=False)
                self._mined_alias_invalidate(None)   # bytes may have moved somewhere
                self._revive_uncertain_absences()    # F3: may have created somewhere

    def _spec_fires(self, vt):
        """Speculative fire view for predict(vt, ...): {'fired': set(j),
        'fs': {path: (content_bytes, known)}, 'degraded': set(paths)} — state is
        NOT mutated. Fires due at vt land in that step's prologue, BEFORE the
        recorded command runs (§3.2). Round-5 F2: a fire failing the landing
        predicate marks its whole chain 'degraded' — nothing about those paths
        (existence included) is determined this step; they also stay in 'fs'
        (known False) so every legacy consumer, incl. render_canon's touched
        view, keeps treating them as fire-affected."""
        fired, fs, degraded = set(), {}, set()
        for j in self._fires_due(vt):
            job = self.jobs[j]
            fired.add(j)
            eff = job["effect_parsed"]
            final, chain, sound = self._fire_landing(eff)   # LAW: mirror the fold
            payload = eff["prod"]["payload"] + "\n"
            if not sound:
                degraded.update(chain)
                for q in chain:
                    fs[q] = ("", False)
                continue
            if eff["op"] == ">":
                fs[final] = (payload, True)
                continue
            if final in fs:
                prev, known = fs[final]
            else:
                st, v = self._stat(final)
                if st == "alive" and isinstance(v["content"], str) and v["trailing_nl_known"]:
                    prev, known = v["content"], True
                elif st == "dead":
                    prev, known = "", True       # append onto known-absent creates
                else:
                    prev, known = "", False
            fs[final] = (prev + payload, known)
        return {"fired": fired, "fs": fs, "degraded": degraded}

    def _fire_hot(self, spec, *paths):
        """Round-5 F2 (defensive guard): a fire due THIS step lands (or degrades)
        at these paths in the prologue, BEFORE the command — belief about them is
        one fire stale, so a predictor that does not fold the speculative view
        must go BOT rather than race the fire."""
        if not spec["fs"]:
            return False
        return any(p in spec["fs"] for p in paths)

    def _job_by_cpid(self, cpid):
        for j, job in self.jobs.items():
            if job["cpid"] == cpid:
                return j, job
        return None, None   # never-launched

    def _kill_transition(self, sig, job):
        """The 5-state automaton (draft §5.3, verified semantics). Returns the new
        state or 'ERR' (No such process). TERM on a stopped job does NOT kill it —
        it becomes stopped_pending_term and dies at CONT."""
        s = job["state"] if job else "never"
        alive = s in ("waiting", "stopped", "stopped_pending_term")
        if not alive:
            return "ERR"
        if sig == "TERM":
            return {"waiting": "killed", "stopped": "stopped_pending_term",
                    "stopped_pending_term": "stopped_pending_term"}[s]
        if sig == "STOP":
            return {"waiting": "stopped", "stopped": "stopped",
                    "stopped_pending_term": "stopped_pending_term"}[s]
        if sig == "CONT":
            return {"waiting": "waiting", "stopped": "waiting",
                    "stopped_pending_term": "killed"}[s]
        if sig == "KILL":
            return "killed"
        return s   # -0 liveness probe: no transition

    def _ps_rows(self, spec=None):
        rows = [(1, "S", "init"), (2, "S", "sleep 86400")]
        fired = spec["fired"] if spec else set()
        for j, job in sorted(self.jobs.items()):
            if j in fired or job["state"] in ("fired", "killed"):
                continue
            stat = "S" if job["state"] == "waiting" else "T"
            rows.append((job["cpid"], stat, f"after {j} {job['K']} {job['effect']}"))
        return rows

    # ------------------------------------------------------------- fold

    def fold(self, step):
        """Fold one RENDERED step record {cmd, output, exit, cwd} (meta is never read).
        Raises ParseError on any command outside the universe (totality gate)."""
        vt = self.vt
        self._delta = []
        self._fire_due(vt)                       # prologue fires commit before the command
        p = parse_command(step["cmd"])
        out, capped = self._evidence(step)
        code = step.get("exit", 0)
        if self._cwd_stale and self._cwd_dependent(p):
            # F1 (round-6): the command ran from an unknown dir (docker_env's dead-cwd
            # prologue fallback), so its record does NOT describe self.cwd — mine and
            # mutate nothing; belief holds until the cwd re-anchors on a live dir.
            pass
        else:
            getattr(self, "_fold_" + p["form"])(p, out, capped, code, vt)
        new_cwd = step.get("cwd", self.cwd)      # realized cwd is in the record/render
        if p["form"] == "cd" and code == 0 and normpath(new_cwd) not in self.fs:
            # entailed: the NORMALIZED path resolves to a live dir (the recorded cwd
            # is LOGICAL — possibly '//'-rooted or a symlink alias, F9); the node
            # itself may still be a symlink, so link-ness stays unknown.
            self.fs[normpath(new_cwd)] = self._vnode("dir", UNKNOWN, "image")
        self.cwd = new_cwd
        if self._cwd_stale and self._cwd_anchored():
            self._cwd_stale = False              # F1: a recorded cwd re-anchored live
        self.vt = vt + 1
        return self._delta

    # --- fold: reads (mining) ---

    def _mine_absence(self, verb, patharg, out, code):
        """Learn absence from an error render ONLY when it matches the harvested
        template (conservative: grep exit-1 is a miss on an existing file, etc.).
        F3 (round-6): fail closed on a string-only / malformed template entry."""
        if code == 0 or slashed(patharg):
            return
        text, _ = self._tmpl_entry(verb)
        if text is None:
            return
        try:
            want = text.format(path=patharg)
        except (KeyError, IndexError, ValueError):
            return   # harvested world text with literal braces: never raise, mine nothing
        if not self._dotdot_sound(patharg):
            return   # round-4 class 2: never tombstone a collapsed path through
                     # unknown intermediates (the error may BE the intermediate's)
        if out.strip() == want.strip():
            path = self._resolve(normpath(patharg, self.cwd))
            if path == ELOOP:
                return
            if path not in self.fs:
                # LAW (round-3): a template match proves only "does not RESOLVE" —
                # the path may be a dangling symlink, so link-ness stays unknown
                # UNLESS absence was already lstat-entailed (complete parent
                # listing / reserved job-log namespace), which the mining must
                # not downgrade.
                certain = self._stat(path)[0] == "dead"
                self.fs[path] = self._vnode(None, UNKNOWN, "image", deleted=True,
                                            linkness_known=certain)
            else:
                self.fs[path]["deleted"] = True

    def _fold_cat(self, p, out, capped, code, vt, patharg=None):
        patharg = patharg or p["path"]
        if code != 0:
            self._mine_absence("cat", patharg, out, code)
            return
        if slashed(patharg):
            return   # F3: never attach 'p/' evidence to p
        if not self._dotdot_sound(patharg):
            return   # round-4 class 2: the collapsed path may not be the read one
        given = normpath(patharg, self.cwd)
        path = self._resolve(given)
        if path == ELOOP:
            return
        if path != given:
            self._touch(given, path)    # symlink-mediated read touches link + target (law)
        st, v = self._stat(path)
        if v is None:
            v = self.fs[path] = self._vnode("file", UNKNOWN, "image")
        v["kind"], v["deleted"] = v["kind"] or "file", False
        if isinstance(v["content"], str) and v["trailing_nl_known"]:
            return   # trajectory-created exact bytes: never downgrade to observed
        if capped:
            if not isinstance(v["content"], str):
                v["content"] = PARTIAL
        else:
            self._set_content(path, out, False)   # rendered text; trailing \n unknowable

    def _fold_ls(self, p, out, capped, code, vt, patharg=None):
        patharg = patharg if patharg is not None else p["path"]
        opt = (p.get("opts") or [""])[0] if p.get("opts") else ""
        if patharg and not self._dotdot_sound(patharg):
            return   # round-4 class 2: never attach listing evidence (nor absence —
                     # _mine_absence carries the same guard)
        base = self._resolve(normpath(patharg, self.cwd)) if patharg else self.cwd
        if base == ELOOP:
            return
        if code != 0:
            self._mine_absence("ls", patharg or base, out, code)
            return
        if slashed(patharg):
            return   # F3: never attach 'p/' evidence to p
        v = self.fs.get(base)
        if v is not None and v["kind"] == "file":
            return                                # ls of a file: no listing knowledge
        names = [ln for ln in out.split("\n") if ln] if out else []
        if v is None or v["kind"] is None:
            # S2 conservative boundary (draft §10.2): a kind-UNKNOWN target may be a
            # FILE — busybox/GNU ls of a file renders the path itself (one line equal
            # to the given arg; "/" never appears in a real entry name). Refuse such
            # evidence entirely: never upgrade kind to "dir" from it, record nothing.
            if "l" in (opt if opt.startswith("-") else ""):
                probe = [ln.split(" -> ")[0].split()[-1] for ln in names
                         if ln.split() and not ln.startswith("total")]
            else:
                probe = names
            if any("/" in n for n in probe) or probe == [basename_of(base)]:
                return
        if v is None:
            v = self.fs[base] = self._vnode("dir", UNKNOWN, "image")
        v["kind"], v["deleted"] = "dir", False
        if opt in TIME_FREE_LS and not capped and all(" " not in n for n in names):
            ents = {n for n in names if n not in (".", "..")}
            v["entries"] = set(ents) if v["entries"] is None else v["entries"] | ents
            v["entries_complete"] = "all" if "a" in opt else \
                ("all" if v.get("entries_complete") == "all" else "visible")
            for n in ents:
                child = base.rstrip("/") + "/" + n
                if child not in self.fs:
                    self.fs[child] = self._vnode(None, UNKNOWN, "image")
        elif opt.startswith("-") and "l" in opt and "d" not in opt:
            # F7 (round-3 review): -ld renders the DIR ITSELF (row name == the
            # given arg), never children — the splice is for child rows only.
            # Round-4 class 4: a row is spliced ONLY when its name field is
            # certain — exactly 9 whitespace-separated fields (mode links owner
            # group size month day time|year name). Space-bearing names and
            # device rows ('8, 0' splits the size field) are refused ENTIRELY:
            # never mint a phantom child from ls -l name guessing.
            for ln in out.split("\n")[: None if not capped else -1]:
                if ln.startswith("total") or " -> " in ln:
                    continue
                fields = ln.split()
                if len(fields) != 9:
                    continue
                name = fields[8]
                if name and name not in (".", "..") and "/" not in name \
                        and name != basename_of(base):
                    child = base.rstrip("/") + "/" + name
                    if child not in self.fs:
                        self.fs[child] = self._vnode(None, UNKNOWN, "image")

    def _fold_head(self, p, out, capped, code, vt):
        if code != 0:
            self._mine_absence("head", p["path"], out, code)

    def _fold_tail(self, p, out, capped, code, vt):
        if code != 0:
            self._mine_absence("tail", p["path"], out, code)

    def _fold_grep(self, p, out, capped, code, vt):
        pass   # exit 1 = miss on an existing OR absent file: no safe inference

    def _fold_stat(self, p, out, capped, code, vt):
        if code != 0:
            self._mine_absence("stat", p["path"], out, code)

    def _fold_find(self, p, out, capped, code, vt):
        if code == 0 and not capped and self._dotdot_sound(p["dir"]):
            # round-4 class 2: an unsound '..' collapse mis-keys the replay cache.
            # Round-5 F6: keyed on the RAW command string — a reconstructed
            # unquoted flattening lets two different quoted commands collide.
            self._find_cache[p["raw"]] = (out, self.fs_clock,
                                          normpath(p["dir"], self.cwd))
        kind = {"f": "file", "d": "dir", None: None}[p["type"]]
        for ln in out.split("\n"):
            q = ln.strip()
            if q.startswith("/") and q not in self.fs:
                self.fs[q] = self._vnode(kind, UNKNOWN, "image")

    def _fold_readlink(self, p, out, capped, code, vt):
        if not self._dotdot_sound(p["path"]):
            return   # round-4 class 2: the collapsed path may not be the read one
        path = normpath(p["path"], self.cwd)
        if code == 0 and out and not capped:
            v = self.fs.get(path)
            if v is None:
                v = self.fs[path] = self._vnode("symlink", UNKNOWN, "image")
            v["kind"], v["link_target"], v["deleted"] = "symlink", out.strip(), False
            v["linkness_known"] = True   # readlink success: link-ness explicitly learned

    def _fold_cd(self, p, out, capped, code, vt):
        if code == 0:
            # realized cwd (folded by fold() from the record) is an entailed live dir
            pass

    def _fold_pwd(self, p, out, capped, code, vt):
        pass

    def _fold_uname(self, p, out, capped, code, vt):
        pass

    def _fold_uptime(self, p, out, capped, code, vt):
        pass

    def _fold_sleep(self, p, out, capped, code, vt):
        pass

    def _fold_echo(self, p, out, capped, code, vt):
        pass

    def _fold_ps(self, p, out, capped, code, vt):
        pass   # the job table is the authority; the render is a projection of it

    # --- fold: mutations (guarded by exit==0; a failed mutation changes nothing) ---

    def _fold_mkdir(self, p, out, capped, code, vt):
        if code != 0:
            return
        if not self._dotdot_sound(p["path"]):
            self._forget_path(normpath(p["path"], self.cwd))     # round-4 class 2
            return
        path = normpath(p["path"], self.cwd)
        self.fs_clock += 1
        self.fs[path] = self._vnode("dir", UNKNOWN, f"mut:mkdir@{vt}",
                                    entries=set(), entries_complete="all",
                                    linkness_known=True)
        self._parent_entry(path, add=path)
        self._touch(path, parent_of(path))
        self._delta.append(("created", path, None))
        self._revive_uncertain_absences()        # round-5 F3: a creation event

    def _fold_touch(self, p, out, capped, code, vt):
        if code != 0:
            return
        if not self._dotdot_sound(p["path"]):
            self._forget_path(normpath(p["path"], self.cwd))     # round-4 class 2
            return
        path = self._resolve(normpath(p["path"], self.cwd))
        if path == ELOOP:
            return   # round-4 class 6: belief cycle — claim nothing
        st, _ = self._stat(path)
        if st == "alive":
            self.fs_clock += 1
            self._touch(path)                    # mtime-only; store-time virtualization
            return
        self.fs_clock += 1
        self.fs[path] = self._vnode("file", "", f"mut:touch@{vt}",
                                    trailing_nl_known=True, linkness_known=True)
        self._parent_entry(path, add=path)
        self._touch(path, parent_of(path))
        self._delta.append(("created", path, 0))
        self._revive_uncertain_absences()        # round-5 F3: a creation event

    def _fold_rm(self, p, out, capped, code, vt):
        if code != 0:
            if not p["recursive"] or self._is_absence_template("rm", p["path"], out):
                # a non-recursive failure (Is-a-directory / ENOENT: nothing destroyed)
                # or a template-matched ENOENT recursive failure (nothing existed) is a
                # true no-op — mine absence exactly as before.
                self._mine_absence("rm", p["path"], out, code)
                return
            # F4 (round-6): a recursive rm that failed for ANY other reason destroyed
            # depth-first BEFORE failing (EBUSY on a bind mount gutted /etc; ENOTDIR
            # through a 'link/' arg emptied the resolved dir), so belief about the
            # subtree is stale. Fold conservatively: forget the subtree (the link AND,
            # for a slashed arg, the resolved dir), invalidate aliased content/dir
            # mines, trigger the absence-revival law (inside _forget_path).
            link = normpath(p["path"], self.cwd)
            # resolve the 'link/' arg to its real dir BEFORE forgetting the link node
            # (else _resolve can no longer follow the symlink and misses the dir)
            rdir = self._resolve(link) if slashed(p["path"]) else None
            self._forget_path(link)
            if rdir is not None and rdir not in (link, ELOOP):
                self._forget_path(rdir)
            self._mined_alias_invalidate(None)   # unknown bytes destroyed somewhere
            self._mined_dir_invalidate(None)     # a real dir may have been gutted
            return
        if not self._dotdot_sound(p["path"]):
            self._forget_path(normpath(p["path"], self.cwd))     # round-4 class 2
            self._mined_alias_invalidate(None)   # class 3: unknown bytes destroyed
            self._mined_dir_invalidate(None)     # round-5 F4: maybe a dir died
            return
        path = normpath(p["path"], self.cwd)     # rm operates on the LINK, not its target
        victims = [path]
        if p["recursive"]:
            victims += [q for q in self.fs
                        if q.startswith(path.rstrip("/") + "/") and not self.fs[q]["deleted"]]
        # round-4 class 3: capture alias bases + hardgroup exclusions BEFORE mutating.
        # A recursive rm of anything but a fully-SST-known subtree (an mkdir-created
        # dir tree, or a single linkness-known non-dir) destroys unknown real bytes.
        v0 = self.fs.get(path)
        subtree_known = not p["recursive"] or (
            v0 is not None and v0.get("linkness_known")
            and (v0["kind"] != "dir" or v0["provenance"].startswith("mut:mkdir")))
        bases, uncertain, excl = [], not subtree_known, set(victims)
        # round-5 F4: only a RECURSIVE rm can destroy a real dir (non-recursive
        # rm exits 0 only on non-dirs / links); collect the dir-mine bases too.
        dbases, dunc = [], p["recursive"] and not subtree_known
        for q in victims:
            v = self.fs.get(q)
            if v is not None and v.get("hardgroup"):
                excl |= set(v["hardgroup"])
            applies, basis = self._alias_basis(q, v)
            if applies:
                if basis is None:
                    uncertain = True
                else:
                    bases.append(basis)
            if p["recursive"]:
                dapp, dbasis = self._dir_alias_basis(v)
                if dapp:
                    if dbasis is None:
                        dunc = True
                    else:
                        dbases.append(dbasis)
        self.fs_clock += 1
        for q in sorted(victims):
            v = self.fs.get(q)
            if v is None:
                self.fs[q] = self._vnode(None, UNKNOWN, f"mut:rm@{vt}", deleted=True,
                                         linkness_known=True)   # SST-removed: certain
            else:
                if v["hardgroup"]:
                    v["hardgroup"].discard(q)    # peers keep the shared content
                    v["hardgroup"] = None
                v["deleted"] = True
                v["linkness_known"] = True       # rm removed the PATH: lstat-certain
                v["provenance"] = f"mut:rm@{vt}"
                v["fs_clock"] = self.fs_clock
            self._parent_entry(q, remove=q)
            self._touch(q, parent_of(q))
            self._delta.append(("removed", q))
        self.ws.pop(path, None)
        self._mark_cwd_stale_if_covered([path])  # F1: rm of the cwd (or an ancestor)
        if uncertain:                            # round-4 class 3
            self._mined_alias_invalidate(None, excl)
        else:
            for b in bases:
                self._mined_alias_invalidate(b, excl)
        if p["recursive"]:                       # round-5 F4
            if dunc:
                self._mined_dir_invalidate(None, excl)
            else:
                for b in dbases:
                    self._mined_dir_invalidate(b, excl)

    def _fold_mv(self, p, out, capped, code, vt):
        if code != 0:
            self._mine_absence("mv", p["src"], out, code)
            return
        if not (self._dotdot_sound(p["src"]) and self._dotdot_sound(p["dst"])):
            self._forget_path(normpath(p["src"], self.cwd))      # round-4 class 2
            self._forget_path(normpath(p["dst"], self.cwd))
            self._mined_alias_invalidate(None)                   # class 3
            self._mined_dir_invalidate(None)                     # round-5 F4
            return
        src = normpath(p["src"], self.cwd)
        dst = normpath(p["dst"], self.cwd)
        sv = self.fs.get(src)
        if sv is not None and sv.get("hardgroup") and dst in sv["hardgroup"]:
            # F5 (round-3 review): rename onto the SAME inode is a POSIX no-op —
            # busybox exits 0 and BOTH names persist (GNU errors, exit!=0 path).
            return
        self._mark_cwd_stale_if_covered([src])   # F1: mv of the cwd (or an ancestor)
        rdst = self._resolve(dst)
        if rdst == ELOOP:
            rdst = dst        # round-4 class 6: unresolvable — the conservative
                              # branch below handles the alive-symlink dst shape
        dstat, dv = self._stat(rdst)
        if dstat == "alive" and dv["kind"] == "dir":
            # F1 (round-2 review): POSIX mv into an EXISTING dir targets
            # dst/basename(src) — the dir node itself is never overwritten.
            dst = rdst.rstrip("/") + "/" + basename_of(src)
        elif not (dstat == "dead" or (dstat == "alive" and dv["kind"] == "file")):
            # dst kind/existence unknown: the landing site is ambiguous between
            # dst (rename) and dst/basename(src) (move-into). Fold conservatively.
            self._mv_conservative(src, dst, rdst, vt)
            return
        if dst == src:
            self.mismatches.append((vt, "mv", f"src == dst ({src}) but exit 0"))
            return
        self.fs_clock += 1
        moved = [(src, dst)]
        v = self.fs.get(src)
        if v is not None and v["kind"] == "dir":
            pre = src.rstrip("/") + "/"
            moved += [(q, dst.rstrip("/") + "/" + q[len(pre):])
                      for q in list(self.fs) if q.startswith(pre)]
        # round-4 class 3: aliases of every vacated src path (and of an
        # overwritten dst inode) go stale — capture bases before mutating.
        # Transported content itself stays valid (mv keeps the inode), so the
        # destination paths are excluded from invalidation.
        subtree_unknown = v is not None and v["kind"] == "dir" and not (
            v.get("linkness_known") and v["provenance"].startswith("mut:mkdir"))
        bases, uncertain = [], subtree_unknown
        dbases, dunc = [], subtree_unknown       # round-5 F4: unknown real subdirs vacate
        excl = {b for _, b in moved}
        cand = [(a, self.fs.get(a)) for a, _ in moved]
        ov = self.fs.get(dst)
        if ov is not None and not ov["deleted"]:
            cand.append((dst, ov))               # rename overwrote dst's inode
        for a, av in cand:
            if av is not None and av.get("hardgroup"):
                excl |= set(av["hardgroup"])
            applies, basis = self._alias_basis(a, av)
            if applies:
                if basis is None:
                    uncertain = True
                else:
                    bases.append(basis)
            dapp, dbasis = self._dir_alias_basis(av)   # round-5 F4: vacated dirs
            if dapp:
                if dbasis is None:
                    dunc = True
                else:
                    dbases.append(dbasis)
        for a, b in moved:
            va = self.fs.pop(a, None)
            if va is None:
                va = self._vnode(None, UNKNOWN, "image")   # mv success proves a existed
            self.fs[b] = dict(va, provenance=f"mut:mv@{vt}", fs_clock=self.fs_clock)
            if va.get("hardgroup"):
                va["hardgroup"].discard(a)
                va["hardgroup"].add(b)
            self.fs[a] = self._vnode(None, UNKNOWN, f"mut:mv@{vt}", deleted=True,
                                     linkness_known=True)   # rename vacated the path
            self._touch(a, b, parent_of(a), parent_of(b))
            if a in self.ws:
                self.ws[b] = self.ws.pop(a)
        self._parent_entry(src, remove=src)
        self._parent_entry(dst, add=dst)
        self._delta.append(("moved", src, dst))
        if uncertain:                            # round-4 class 3
            self._mined_alias_invalidate(None, excl)
        else:
            for b in bases:
                self._mined_alias_invalidate(b, excl)
        if dunc:                                 # round-5 F4
            self._mined_dir_invalidate(None, excl)
        else:
            for b in dbases:
                self._mined_dir_invalidate(b, excl)
        self._revive_uncertain_absences()        # round-5 F3: mv-in creates dst

    def _mv_conservative(self, src, dst, rdst, vt):
        """F1: mv succeeded but the dst kind is unknown — the moved node landed at
        dst OR dst/basename(src), undecidable from belief. Tombstone the src side
        (mv success proves it left), transport NOTHING, and shadow the dst path
        itself (exists, kind/content unknown => reads there are BOT). Never writes
        over dst's ancestors or any known-kind node."""
        self.fs_clock += 1
        gone = [src]
        v = self.fs.get(src)
        if v is not None and v["kind"] == "dir":
            pre = src.rstrip("/") + "/"
            gone += [q for q in list(self.fs) if q.startswith(pre)]
        excl = set()                             # round-4 class 3: peers keep bytes
        for a in gone:
            va = self.fs.get(a)
            if va is not None and va.get("hardgroup"):
                excl |= set(va["hardgroup"])
        for a in sorted(gone):
            # round-5 F1: severing the hardgroup here loses inode tracking — the
            # inode is still LIVE behind the uncertain landing site, so every
            # remaining peer degrades to unknown content and is flagged severed
            # (untracked writes through the vacated side can change the shared
            # bytes invisibly from now on).
            self._sever_group(a)
            self.fs[a] = self._vnode(None, UNKNOWN, f"mut:mv@{vt}", deleted=True,
                                     linkness_known=True)   # mv vacated the src side
            self._touch(a, parent_of(a))
            self.ws.pop(a, None)
        for q in {dst, rdst}:
            qv = self.fs.get(q)
            if qv is None or qv["kind"] is None or qv["deleted"]:
                self.fs[q] = self._vnode(None, UNKNOWN, f"mut:mv@{vt}")   # shadow
        self._parent_entry(src, remove=src)
        self._parent_entry(dst, add=dst)         # dst exists after either outcome
        self._touch(dst, parent_of(dst))
        self._delta.append(("moved", src, dst))
        # round-4 class 3: the landing site (and any overwritten inode) is
        # uncertain — every mined-content alias candidate goes stale
        self._mined_alias_invalidate(None, excl)
        self._mined_dir_invalidate(None, excl)   # round-5 F4: maybe a dir vacated
        self._revive_uncertain_absences()        # round-5 F3: dst was created

    def _fold_ln(self, p, out, capped, code, vt):
        if code != 0:
            return
        if not self._dotdot_sound(p["link"]):
            self._forget_path(normpath(p["link"], self.cwd))     # round-4 class 2
            return
        target = p["target"]                     # symlink target stored AS WRITTEN
        link = normpath(p["link"], self.cwd)
        self.fs_clock += 1
        if p["symbolic"]:
            self.fs[link] = self._vnode("symlink", UNKNOWN, f"mut:ln@{vt}",
                                        link_target=target, linkness_known=True)
        elif not self._dotdot_sound(target) \
                or self._resolve(normpath(target, self.cwd)) == ELOOP:
            # round-4 classes 2/6: the aliased inode is uncertain — the link
            # exists (exit 0), nothing else is claimed
            self.fs[link] = self._vnode(None, UNKNOWN, f"mut:ln@{vt}")
        else:
            tpath = self._resolve(normpath(target, self.cwd))
            tv = self.fs.get(tpath)
            if tv is None:
                tv = self.fs[tpath] = self._vnode("file", UNKNOWN, "image")
            group = tv["hardgroup"] or {tpath}
            group.add(link)
            tv["hardgroup"] = group
            # LAW: the new name aliases whatever inode the TARGET names — its
            # link-ness is known only if the target's was (image targets may
            # themselves be symlinks; `ln` dialects differ on what got linked).
            self.fs[link] = dict(tv, provenance=f"mut:ln@{vt}", fs_clock=self.fs_clock,
                                 hardgroup=group,
                                 linkness_known=bool(tv.get("linkness_known")))
            self._touch(tpath)
        self._parent_entry(link, add=link)
        self._touch(link, parent_of(link))
        self._delta.append(("created", link, None))
        self._revive_uncertain_absences()        # round-5 F3: a creation event

    # --- fold: G3 composed forms ---

    def _apply_redirect(self, p, provenance):
        """Shared by the SUCCESSFUL redirect fold and job-effect fires (failures go
        through _fold_redir_fail). Computes the produced content from belief:
        observed-capture (producer output derivable from an earlier render or the
        command text) => content known; blind-capture => UNKNOWN — the honest
        composed-margin surface (draft §6.3). LINK-CONSERVATISM LAW (round-3):
        the write lands where the dst CHAIN resolves — write-through + hardgroup
        sync only when the chain is fully SST-known; otherwise no content claim
        and every known chain node (+ hardgroup peers) degrades to unknown.
        F8: a cat-producer capture inherits the SRC's trailing_nl_known — the
        render never reveals the trailing byte of merely-observed content."""
        dst = normpath(p["dst"], self.cwd)
        prod = p["prod"]
        if prod["kind"] == "echo":
            rendered, render_known, nl_known = prod["payload"], True, True
            desc = "echo"
        else:
            desc = f"{prod['kind']} {prod.get('dir') or prod.get('file')}"
            pred = self._read_predict(
                {"form": "ls", "opts": ["-1"], "path": prod["dir"]}
                if prod["kind"] == "ls" else {"form": "cat", "path": prod["file"]},
                spec=None)
            if pred is not BOT and pred["exit"] == 0:
                rendered, render_known = pred["output"], True
                if prod["kind"] == "cat":            # F8: inherit the src's nl flag
                    sv = self.fs.get(self._resolve(normpath(prod["file"], self.cwd)))
                    nl_known = bool(sv and sv.get("trailing_nl_known"))
                else:
                    nl_known = True                  # ls output is \n-terminated
            else:
                rendered, render_known, nl_known = "", False, False
        text = rendered + ("\n" if rendered and nl_known else "")
        bytes_known = render_known and nl_known
        final, chain, fully = self._wchain(dst)
        self.fs_clock += 1
        if not fully:
            # LAW: landing site uncertain — degrade, claim no content anywhere
            self._degrade_write(chain, provenance, ensure=True)
            # round-4 class 3: bytes changed at an uncertain inode — every
            # mined-content alias candidate goes stale (chain nodes are already
            # UNKNOWN via the degrade)
            self._mined_alias_invalidate(None)
            w = self.ws.setdefault(dst, {"producer": desc, "observed": False,
                                         "appends": 0, "content_lines": None})
            w["observed"] = False
            if p["op"] == ">>":
                w["appends"] += 1
            w["content_lines"] = None
            self._delta.append(("appended" if p["op"] == ">>" else "created", dst, None))
            self._revive_uncertain_absences()    # round-5 F3: a creation event
            return
        st, v = self._stat(final)
        # round-4 class 3: a truncate/append onto an EXISTING inode strands any
        # linkness-unknown alias that mined its pre-write bytes — capture the basis
        pre_basis = v["content"] if st == "alive" and v is not None else False
        pre_excl = ((set(v["hardgroup"]) if st == "alive" and v is not None
                     and v.get("hardgroup") else set()) | {final})
        appended = p["op"] == ">>" and st != "dead"
        if appended:
            obs = render_known
            if st == "alive" and render_known \
                    and isinstance(v["content"], str) and v["trailing_nl_known"]:
                self._set_content(final, v["content"] + text, nl_known, provenance)
            elif st == "alive":
                self._set_content(final, PARTIAL if isinstance(v["content"], str) else UNKNOWN,
                                  False, provenance)
            else:                                        # st unknown: may/may not exist —
                self.fs[final] = self._vnode("file", UNKNOWN, provenance)
                obs = False   # blind-ish capture: prior bytes unknown, never "observed"
            w = self.ws.setdefault(final, {"producer": desc, "observed": obs,
                                           "appends": 0, "content_lines": None})
            w["appends"] += 1
            w["observed"] = w["observed"] and obs
        elif st == "alive" and v is not None and v.get("hardgroup"):
            # '>' TRUNCATES the existing inode — hardgroup peers share the new
            # bytes and the group survives (round-3 V3): sync, never replace.
            self._set_content(final, text if render_known else UNKNOWN,
                              bytes_known, provenance)
            self.ws[final] = {"producer": desc, "observed": render_known, "appends": 0,
                              "content_lines": None}
        else:
            self.fs[final] = self._vnode("file", text if render_known else UNKNOWN,
                                         provenance, trailing_nl_known=bytes_known,
                                         linkness_known=True)
            self._parent_entry(final, add=final)
            self.ws[final] = {"producer": desc, "observed": render_known, "appends": 0,
                              "content_lines": None}
        cur = self.fs[final]["content"]
        self.ws[final]["content_lines"] = cur.count("\n") if isinstance(cur, str) else None
        if pre_basis is not False:               # round-4 class 3 (pre-write alive)
            self._mined_alias_invalidate(
                pre_basis if isinstance(pre_basis, str) else None, pre_excl)
        self._touch(dst, final, parent_of(final))
        nbytes = len(text.encode()) if bytes_known else None
        self._delta.append(("appended" if appended else "created", final, nbytes))
        self._revive_uncertain_absences()        # round-5 F3: a creation event

    def _fold_redir(self, p, out, capped, code, vt):
        if code != 0:
            self._fold_redir_fail(p, vt)
            return
        if slashed(p["dst"]):
            # F3: '<p>/' never opens as a file — exit 0 here is a record anomaly
            self.mismatches.append((vt, "redir", f"slashed dst {p['dst']!r} exit 0"))
            return
        if not self._dotdot_sound(p["dst"]):
            self._forget_path(normpath(p["dst"], self.cwd))      # round-4 class 2
            self._mined_alias_invalidate(None)                   # class 3
            return
        self._apply_redirect(p, provenance=f"redirect@{vt}")

    def _fold_redir_fail(self, p, vt):
        """F3 failure fold (§6.4): exit != 0 is EITHER an open-failure (dst is an
        existing directory / a trailing-slash form — NOTHING was written) OR a
        producer failure (the shell opened/truncated dst before the producer ran,
        so dst holds no new bytes). Claims only what the failure entails; never
        records a ws capture entry. Round-3: the trailing-slash dst never attaches
        evidence to the stripped path, and the truncation claim needs a fully
        SST-known chain — otherwise degrade only (no existence claims either)."""
        if slashed(p["dst"]):
            return   # F3: 'p/' failure (ENOTDIR / not-a-dir open) entails nothing
        if not self._dotdot_sound(p["dst"]):
            # round-4 class 2: a failed open through unsound '..' MAY still have
            # truncated something (shell opens dst before the producer runs) at
            # an uncertain path — forget the textual candidate, claim nothing
            self._forget_path(normpath(p["dst"], self.cwd))
            self._mined_alias_invalidate(None)                   # class 3
            return
        dst = normpath(p["dst"], self.cwd)
        final, chain, fully = self._wchain(dst)
        stt, v = self._stat(final)
        if stt == "alive" and v is not None and v["kind"] == "dir":
            return                               # open failed on the dir: no change
        self.fs_clock += 1
        pstt, pvv = self._stat(parent_of(final))
        if fully and stt == "dead" and pstt == "alive" and pvv \
                and pvv["kind"] == "dir":
            # dst chain known, landing slot known-absent (so not a dir) AND the
            # landing parent known-live (round-5 F2: with a dead/unknown parent
            # the open itself failed ENOENT — NOTHING was created): the open
            # created the final target empty before the producer failed
            self.fs[final] = self._vnode("file", "", f"redirect@{vt}",
                                         trailing_nl_known=True, linkness_known=True)
            self._parent_entry(final, add=final)
            self._delta.append(("created", final, 0))
            self._revive_uncertain_absences()    # round-5 F3: a creation event
        elif fully and stt == "alive" and v["kind"] == "file":
            if p["op"] == ">":
                pre = v["content"]               # round-4 class 3: pre-truncate bytes
                excl = (set(v["hardgroup"]) if v.get("hardgroup") else set()) | {final}
                self._set_content(final, "", True, f"redirect@{vt}")   # truncated
                self._delta.append(("created", final, 0))
                self._mined_alias_invalidate(
                    pre if isinstance(pre, str) else None, excl)
            # '>>' appended nothing: content unchanged
        else:
            # chain/kind/existence unknown: nothing is knowable about where (or
            # whether) bytes/truncation landed — degrade known chain nodes only
            self._degrade_write(chain, f"mut:redir@{vt}", ensure=False)
            self._mined_alias_invalidate(None)                   # round-4 class 3
            self._revive_uncertain_absences()    # round-5 F3: may have created
        if dst in self.ws:
            cur = self.fs.get(dst, {}).get("content")
            self.ws[dst]["content_lines"] = \
                cur.count("\n") if isinstance(cur, str) else None
        self._touch(dst, parent_of(dst))

    def _fold_pipe(self, p, out, capped, code, vt):
        pass   # pipe renders are partial views; conservative — no belief update

    def _fold_cond(self, p, out, capped, code, vt):
        if slashed(p["path"]):
            return   # F3: '[ -e p/ ]' outcomes never transfer to p (ENOTDIR aliasing)
        if not self._dotdot_sound(p["path"]):
            return   # round-4 class 2
        path = self._resolve(normpath(p["path"], self.cwd))
        if path == ELOOP:
            return
        if code == 0:
            v = self.fs.get(path)
            if v is None:
                v = self.fs[path] = self._vnode(None, UNKNOWN, "image")
            v["deleted"] = False
            if p["testop"] == "-f" and v["kind"] is None:
                v["kind"] = "file"
            if p["testop"] == "-d" and v["kind"] is None:
                v["kind"] = "dir"
            r = p["read"]
            if r["form"] == "cat":
                self._fold_cat(r, out, capped, code, vt)
            elif r["form"] == "ls":
                self._fold_ls(r, out, capped, code, vt, patharg=r["path"])
        elif not out and p["testop"] == "-e":
            if path not in self.fs:
                self.fs[path] = self._vnode(None, UNKNOWN, "image", deleted=True)
            else:
                self.fs[path]["deleted"] = True   # [ -e P ] miss entails absence

    # --- fold: process forms ---

    def _fold_after(self, p, out, capped, code, vt):
        if code != 0:
            return   # F4 (round-3 review): a FAILED launch registers no job — it
                     # must never fire (same exit==0 guard as every mutation fold)
        j = p["j"]
        if j in self.jobs:
            self.mismatches.append((vt, "after", f"job index {j} reused"))
            return
        cpid = canonical_pid(j)
        if code == 0 and out.strip() and out.strip() != str(cpid):
            self.mismatches.append((vt, "after", f"echoed pid {out.strip()!r} != {cpid}"))
        self.jobs[j] = {"cpid": cpid, "K": p["K"], "effect": p["effect"],
                        "effect_parsed": p["effect_parsed"], "state": "waiting",
                        "launch_vt": vt, "deferrals": 0}

    def _fold_kill(self, p, out, capped, code, vt):
        j, job = self._job_by_cpid(p["cpid"])
        new = self._kill_transition(p["sig"], job)
        if new == "ERR":
            if code == 0:
                self.mismatches.append((vt, "kill", f"cpid {p['cpid']} dead but exit 0"))
            return
        if code != 0:
            self.mismatches.append((vt, "kill", f"cpid {p['cpid']} alive but exit {code}"))
        job["state"] = new

    # ------------------------------------------------------------- delta channel

    def delta_text(self):
        """Canonical fs-delta of the LAST folded step (prereg §4.2: sorted by primary
        path, <=8 entries + '(+N more)', size field for append checkability where the
        byte count is known). The collector stores this as meta.delta_text (audit
        copy); derived-root builders recompute it via this same re-fold (F8)."""
        if not self._delta:
            return "delta: none"
        ents = []
        for op in sorted(self._delta, key=lambda o: (o[1], o[0])):
            if op[0] == "removed":
                ents.append(f"removed {op[1]}")
            elif op[0] == "moved":
                ents.append(f"moved {op[1]} -> {op[2]}")
            elif op[0] == "created":
                ents.append(f"created {op[1]}" + (f"({op[2]}B)" if op[2] is not None else ""))
            else:
                ents.append(f"appended {op[1]}" + (f"(+{op[2]}B)" if op[2] is not None else ""))
        extra = f" (+{len(ents) - 8} more)" if len(ents) > 8 else ""
        return "delta: " + ", ".join(ents[:8]) + extra

    # =========================================================== predict (SST arm)

    _NO_SPEC = {"fired": frozenset(), "fs": {}, "degraded": frozenset()}

    def _ok(self, output, cwd=None):
        return {"output": output, "exit": 0, "cwd": cwd or self.cwd}

    def _tmpl_entry(self, key):
        """(text, exit) for a harvested error-template entry, or (None, None) when the
        entry is missing, a bare string, or lacks the {text|template, exit} shape.
        F3 (round-6): the canonical harvest (benchmarks/p0/error-templates.json) stores
        {template, exit} per key — shell_state consumes exactly that shape and FAILS
        CLOSED on a string-only entry or a missing exit (a per-dialect exit is never a
        guess: grep ENOENT is 2 on both busybox+GNU, ls ENOENT is 1 busybox / 2 GNU).
        'text' and 'template' are accepted interchangeably as the text key."""
        t = self.error_templates.get(key)
        if not isinstance(t, dict):
            return None, None
        text = t.get("text", t.get("template"))
        if text is None or "exit" not in t:
            return None, None
        return text, t["exit"]

    def _tmpl(self, key, **kw):
        """Predicted error step via the probe-harvested template table; BOT if the
        image's table lacks the key or the entry is not a well-formed {text|template,
        exit} pair (F3: never guess a dialect text OR its exit code).

        P1-review (DG-4b): GNU coreutils quote an error-message filename ONLY when it
        contains a shell-special char (quotearg 'shell-escape-if-needed'), while busybox
        always-quotes — so a determined error prediction for a special-char path is
        dialect-divergent and unknowable from the template alone. BOT any path kwarg
        carrying such a char (rare: ~1/56 recorded errors; the determined/BOT trade always
        favors soundness)."""
        text, code = self._tmpl_entry(key)
        if text is None:
            return BOT
        for pk in ("path", "p", "p2", "dst", "src"):
            v = kw.get(pk)
            if isinstance(v, str) and any(c in _QUOTEARG_SPECIAL for c in v):
                return BOT
        try:
            out = text.format(**kw)
        except (KeyError, IndexError, ValueError):
            return BOT   # harvested world text with literal braces: BOT, never raise
        return {"output": out, "exit": code, "cwd": self.cwd}

    def _is_absence_template(self, verb, patharg, out):
        """Whether `out` is the harvested ENOENT template for `verb` at `patharg` —
        used to tell a true no-op failure (nothing existed) from a partial-destruction
        one (F4: a failed recursive rm)."""
        text, _ = self._tmpl_entry(verb)
        if text is None:
            return False
        try:
            want = text.format(path=patharg)
        except (KeyError, IndexError, ValueError):
            return False
        return out.strip() == want.strip()

    def predict(self, step_index, cmd):
        """Predict the RENDERED step {'output','exit','cwd'} for `cmd` at step_index,
        or BOT (None) where not determined. Pure: never mutates state. Call BEFORE
        fold()ing that step. Raises ParseError outside the universe (totality)."""
        p = parse_command(cmd)
        if self._cwd_stale and self._cwd_dependent(p):
            return BOT     # F1 (round-6): the cwd desynced from docker_env's prologue —
                           # every cwd-dependent surface is BOT until it re-anchors
        spec = self._spec_fires(step_index)
        fn = getattr(self, "_pred_" + p["form"], None)
        if fn is None:
            return BOT
        return fn(p, spec, step_index)

    # --- read core (shared by predict / cond READ / pipe producers / redirect capture)

    def _read_file(self, patharg, spec):
        """('ok', rendered_text, nl_known) | ('dead', None, False) |
        ('bot', None, False) for a cat-class read. nl_known: the full byte
        stream incl. trailing-newline structure is known (round-4 class 1b —
        merely-observed content is the record channel's rstrip of the real
        bytes, so its logical line count is a lower bound only)."""
        if slashed(patharg):
            return "bot", None, False   # F3: 'p/' is never a plain-file read of p
        if not self._dotdot_sound(patharg):
            return "bot", None, False   # round-4 class 2
        given = normpath(patharg, self.cwd)
        path = self._resolve(given)
        if path in spec["fs"]:
            raw, known = spec["fs"][path]
            return ("ok", raw.rstrip("\n"), True) if known else ("bot", None, False)
        st, v = self._stat(path)
        if st == "dead":
            return "dead", None, False
        if st == "alive" and v["kind"] in ("file", None) and isinstance(v["content"], str):
            if self._volatile(path):
                return "bot", None, False   # round-4 class 8: never re-serve /proc,/sys
            return "ok", v["content"].rstrip("\n"), bool(v["trailing_nl_known"])
        return "bot", None, False

    @staticmethod
    def _lines(rendered):
        return rendered.split("\n") if rendered else []

    def _read_predict(self, p, spec):
        spec = spec or self._NO_SPEC
        form = p["form"]
        if form == "ls":
            return self._ls_predict(p.get("opts") or [], p.get("path"), spec)
        status, text, nl_known = self._read_file(p["path"], spec)
        if status == "dead":
            return self._tmpl(form if form != "grep" else "grep", path=p["path"])
        if status == "bot":
            return BOT
        if form == "cat":
            return self._ok(text)
        lines = self._lines(text)
        if form == "head":
            # round-4 class 1c: mirror the record channel's rstrip — a window
            # ending on a blank line joins with a trailing '\n' the record strips
            return self._ok("\n".join(lines[:p["k"]]).rstrip("\n"))
        if form == "tail":
            if not p["k"]:
                return self._ok("")
            if not nl_known:
                return BOT   # round-4 class 1b: rstripped observation places the
                             # window off-by-N when the real file ends in blank lines
            return self._ok("\n".join(lines[-p["k"]:]).rstrip("\n"))
        if form == "grep":
            if "\x00" in text:
                return BOT   # round-4 class 1a: real greps print a dialect-divergent
                             # 'binary file matches' — never replay raw lines
            tok = p["tok"]
            if not tok and not nl_known:
                return BOT   # empty token matches the invisible trailing blank lines
            if p.get("icase"):
                # F2 (round-6): real greps fold bytes/ASCII (C locale); Python
                # str.lower() over-folds Unicode (İ->i̇, K->k, ſ->s, and non-ASCII
                # content like GRÜN), so an icase match is determined ONLY when both
                # the token and the compared text are ASCII (an ASCII fold is byte-
                # identical, so plain .lower() is exact there) — else BOT.
                if not (tok.isascii() and text.isascii()):
                    return BOT
                hits = [ln for ln in lines if tok.lower() in ln.lower()][:8]
            else:
                hits = [ln for ln in lines if tok in ln][:8]
            out = "\n".join(hits).rstrip("\n")   # round-4 class 1c: record rstrip
            return self._ok(out) if hits else {"output": "", "exit": 1, "cwd": self.cwd}
        return BOT

    def _ls_predict(self, opts, patharg, spec):
        opt = opts[0] if opts else ""
        if opt not in TIME_FREE_LS:
            return BOT                     # -l family renders are never predicted (R6)
        if patharg and not self._dotdot_sound(patharg):
            return BOT                     # round-4 class 2
        given = normpath(patharg, self.cwd) if patharg else self.cwd
        path = self._resolve(given)
        if self._volatile(path):
            return BOT                     # round-4 class 8: /proc,/sys listings churn
        if path in spec["degraded"] \
                or any(parent_of(q) == path for q in spec["degraded"]):
            return BOT                     # round-5 F2: an unsound fire may (not)
                                           # have created here this prologue
        st, v = self._stat(path)
        if st == "dead":
            # LAW: ls lstats its arg — it LISTS a dangling symlink, so the error
            # template needs lstat-certain absence.
            if self._dead_certain(v):
                return self._tmpl("ls", path=patharg if patharg else path)
            return BOT
        if st != "alive":
            return BOT
        if v["kind"] == "file":
            if slashed(patharg):
                return BOT       # F3: ls of 'file/' is ENOTDIR, not the path echo
            return self._ok(patharg if patharg else path)
        if v["kind"] != "dir":
            return BOT
        comp = v.get("entries_complete")
        want_all = "a" in opt
        if comp != "all" and not (comp == "visible" and not want_all):
            return BOT
        names = set(v["entries"])
        # speculative fires only create/append (never delete): union their names in
        names |= {basename_of(q) for q in spec["fs"] if parent_of(q) == path}
        if want_all:
            shown = sorted(names | {".", ".."})
        else:
            shown = sorted(n for n in names if not n.startswith("."))
        return self._ok("\n".join(shown))

    # --- per-form predictors

    def _pred_pwd(self, p, spec, vt):
        return self._ok(self.cwd)                                     # R1

    def _pred_cd(self, p, spec, vt):                                  # R2
        # F9 (round-3 review): the realized cwd is LOGICAL (`cd -L`; docker_env
        # records `pwd` output) — never resolve symlinks into it, preserve a
        # leading '//'. Existence/dir-ness is still checked on the resolved path.
        tgt = p["target"]
        if tgt == "":
            return BOT                    # bare cd -> $HOME: image-dependent
        logical = logical_cwd(tgt, self.cwd)
        if tgt in (".", ".."):
            return self._ok("", cwd=logical)   # cwd is live; its parent is a dir
        # round-7 review: bash `/bin/sh` (fedora inner-val, rockylinux final-test) validates
        # every `..` intermediate; a textual collapse through a ghost/file/removed component
        # makes real `cd` FAIL though normpath would succeed. Require the same known-live-dir
        # intermediates every other predictor enforces (the docstring already claimed this).
        if not self._dotdot_sound(tgt):
            return BOT
        phys = self._resolve(normpath(tgt, self.cwd))
        if self._fire_hot(spec, phys):
            return BOT                    # round-5 F2: a fire lands here first
        st, v = self._stat(phys)
        if st == "dead":
            return self._tmpl("cd", path=tgt)   # cd resolves: non-resolution => fail
        if st == "alive" and v["kind"] == "dir":
            return self._ok("", cwd=logical)
        return BOT

    def _pred_cat(self, p, spec, vt):
        return self._read_predict(p, spec)                            # R5

    def _pred_ls(self, p, spec, vt):
        return self._read_predict(p, spec)                            # R6

    def _pred_head(self, p, spec, vt):
        return self._read_predict(p, spec)

    def _pred_tail(self, p, spec, vt):
        return self._read_predict(p, spec)

    def _pred_grep(self, p, spec, vt):
        return self._read_predict(p, spec)

    def _pred_stat(self, p, spec, vt):                                # R8 only
        if not self._dotdot_sound(p["path"]):
            return BOT                     # round-4 class 2
        if self._fire_hot(spec, normpath(p["path"], self.cwd),
                          self._resolve(normpath(p["path"], self.cwd))):
            return BOT                     # round-5 F2: a fire lands here first
        st, sv = self._stat(self._resolve(normpath(p["path"], self.cwd)))
        # LAW: (busybox) stat lstats — it SUCCEEDS on a dangling symlink, so the
        # error template needs lstat-certain absence.
        if st == "dead" and self._dead_certain(sv):
            return self._tmpl("stat", path=p["path"])
        return BOT

    def _pred_find(self, p, spec, vt):                                # R7 replay
        if not self._dotdot_sound(p["dir"]):
            return BOT                     # round-4 class 2
        if self._volatile(normpath(p["dir"], self.cwd)):
            return BOT                     # round-4 class 8: /proc,/sys churn
        key = p["raw"]                     # round-5 F6: raw-string cache key
        hit = self._find_cache.get(key)
        if hit is None:
            return BOT
        out, clock0, d = hit
        if normpath(p["dir"], self.cwd) != d:
            return BOT                     # F5 (round-6): the raw-string cache key is
                                           # cwd-blind — a relative dir denotes a
                                           # DIFFERENT directory after a cd rebind
        pre = d.rstrip("/") + "/"
        if any((q == d or q.startswith(pre)) and c > clock0
               for q, c in self.touched.items()):
            return BOT
        # F11 (round-3 review): a mutated ANCESTOR (rm -r / mv of a dir whose
        # untracked subtree contains d) invalidates the cache even though no
        # touched path sits at/under d itself.
        anc = d
        while anc != "/":
            anc = parent_of(anc)
            if self.touched.get(anc, -1) > clock0:
                return BOT
        if any(q == d or q.startswith(pre) for q in spec["fs"]):
            return BOT
        return self._ok(out)

    def _pred_readlink(self, p, spec, vt):
        # LINK-CONSERVATISM LAW: determined ONLY over SST-known link-ness. A node
        # whose kind came from read evidence (cat => file-or-symlink; ls => dir-or-
        # symlink-to-dir; /etc/mtab is a real-image counterexample) is BOT; a
        # template-mined absence may be a dangling symlink (readlink SUCCEEDS) — BOT.
        if slashed(p["path"]):
            return BOT           # F3: 'p/' forces resolution — never the link itself
        if not self._dotdot_sound(p["path"]):
            return BOT           # round-4 class 2
        path = normpath(p["path"], self.cwd)
        if self._volatile(path):
            return BOT           # round-4 class 8: /proc/self-style targets churn
        if self._fire_hot(spec, path):
            return BOT           # round-5 F2: a fire lands here first
        st, v = self._stat(path)
        if st == "dead":
            if self._dead_certain(v):
                return {"output": "", "exit": 1, "cwd": self.cwd}   # dialect-uniform
            return BOT
        if st == "alive" and v["kind"] == "symlink" and v["link_target"] \
                and v.get("linkness_known"):
            return self._ok(v["link_target"])
        if st == "alive" and v["kind"] in ("file", "dir") and v.get("linkness_known"):
            return {"output": "", "exit": 1, "cwd": self.cwd}
        return BOT

    def _pred_echo(self, p, spec, vt):
        return self._ok(p["text"])                                    # R4

    def _pred_mkdir(self, p, spec, vt):                               # R3
        if not self._dotdot_sound(p["path"]):
            return BOT                           # round-4 class 2
        path = normpath(p["path"], self.cwd)
        if self._unwritable(path):
            return BOT                           # round-4 class 5
        if self._fire_hot(spec, path):
            return BOT                           # round-5 F2: fire lands here first
        st, sv = self._stat(path)
        if st == "alive":
            return self._tmpl("mkdir_exists", path=p["path"])
        pst, pv = self._stat(parent_of(path))
        # LAW: a resolve-mined "dead" may be a DANGLING SYMLINK — mkdir there
        # fails File-exists in reality, so ok('') needs lstat-certain absence.
        if st == "dead" and self._dead_certain(sv) \
                and pst == "alive" and pv and pv["kind"] == "dir":
            return self._ok("")
        return BOT

    def _pred_touch(self, p, spec, vt):
        if not self._dotdot_sound(p["path"]):
            return BOT                           # round-4 class 2
        given = normpath(p["path"], self.cwd)
        path = self._resolve(given)
        if self._unwritable(given, path):
            return BOT                           # round-4 class 5
        if self._fire_hot(spec, given, path):
            return BOT                           # round-5 F2: fire lands here first
        if slashed(p["path"]):
            sv = self.fs.get(path)               # F3: 'p/' needs a known dir
            if sv is None or sv["deleted"] or sv["kind"] != "dir":
                return BOT
        st, v = self._stat(path)
        if st == "alive":
            # round-4 class 4: determined-ok needs PROVEN resolvability — cat/ls
            # evidence ('file'/'dir') or an SST-created node. A kind-unknown alive
            # node (ls-child/find/cond-mined) may be a dangling or looping symlink
            # whose O_CREAT outcome is unknowable.
            if v is not None and v["kind"] in ("file", "dir"):
                return self._ok("")
            return BOT
        pst, pv = self._stat(parent_of(path))
        if pst == "alive" and pv and pv["kind"] == "dir":
            return self._ok("")
        return BOT

    def _pred_rm(self, p, spec, vt):
        if not self._dotdot_sound(p["path"]):
            return BOT                           # round-4 class 2
        path = normpath(p["path"], self.cwd)     # the link itself, never its target
        if self._unwritable(path):
            return BOT                           # round-4 class 5
        if self._fire_hot(spec, path):
            return BOT                           # round-5 F2: fire lands here first
        if p["recursive"] and self._mount_ancestor(path):
            return BOT                           # class 5: EBUSY on the mount row
        if slashed(p["path"]):
            sv = self.fs.get(path)               # F3: 'p/' needs a known dir
            if sv is None or sv["deleted"] or sv["kind"] != "dir":
                return BOT
        st, v = self._stat(path)
        if st == "alive":
            if p["recursive"]:
                return self._ok("")              # -r removes files, dirs and links alike
            k = v["kind"] if v is not None else None
            if k in ("file", "symlink"):
                # cat-evidenced 'file' means file-or-symlink — rm exits 0 either way
                return self._ok("")
            if k == "dir" and v.get("linkness_known"):
                # non-recursive rm of a CERTAIN dir fails: harvested template or
                # BOT. LAW: a symlink-to-dir (linkness unknown) rm's the LINK.
                return self._tmpl("rm_isdir", path=p["path"])
            return BOT   # round-4 class 4: kind-unknown alive (ls-child/find-mined)
                         # may be a real directory — 'Is a directory', exit 1
        if st == "dead":
            # LAW: rm operates on the LINK — a resolve-mined "dead" may be a
            # dangling symlink, which rm removes with exit 0.
            if self._dead_certain(v):
                return self._tmpl("rm", path=p["path"])
        return BOT

    def _pred_mv(self, p, spec, vt):
        # F1 (round-2 review): determined ok('') ONLY when the outcome shape is
        # fully determined — dst a known dir with a known-absent child slot, or
        # dst known-absent (plain rename). Everything else is BOT.
        if not (self._dotdot_sound(p["src"]) and self._dotdot_sound(p["dst"])):
            return BOT                           # round-4 class 2
        src = normpath(p["src"], self.cwd)
        if self._unwritable(src, normpath(p["dst"], self.cwd)):
            return BOT                           # round-4 class 5
        if self._fire_hot(spec, src, normpath(p["dst"], self.cwd)):
            return BOT                           # round-5 F2: fire lands here first
        if self._mount_ancestor(src):
            return BOT                           # class 5: moving a mount's ancestor
        if slashed(p["src"]):
            sv = self.fs.get(self._resolve(src))     # 'p/' names a dir or nothing
            if sv is None or sv["deleted"] or sv["kind"] != "dir":
                return BOT
        st, sv = self._stat(src)
        if st == "dead":
            # LAW: mv lstats the src — a resolve-mined "dead" may be a dangling
            # symlink, which mv renames with exit 0.
            if self._dead_certain(sv):
                return self._tmpl("mv", path=p["src"])
            return BOT
        if st != "alive":
            return BOT
        dst = normpath(p["dst"], self.cwd)
        if dst == src or dst.startswith(src.rstrip("/") + "/"):
            return BOT   # F6 (round-3): src==dst / src-ancestor-of-dst is EINVAL
                         # (dialect-divergent text) or a same-inode corner — never ok
        rdst = self._resolve(dst)
        if self._fire_hot(spec, rdst):
            return BOT                           # round-5 F2: fire lands here first
        dstat, dv = self._stat(rdst)
        if dstat == "alive" and dv["kind"] == "dir":
            child = rdst.rstrip("/") + "/" + basename_of(src)
            if child == src or src.startswith(child.rstrip("/") + "/"):
                return BOT                       # F6 via a dir landing site
            cstat, cv = self._stat(child)
            return self._ok("") if cstat == "dead" and self._dead_certain(cv) \
                else BOT                         # move INTO the dir
        if dstat == "dead" and not slashed(p["dst"]):
            pst, pv = self._stat(parent_of(dst))
            if pst == "alive" and pv and pv["kind"] == "dir":
                return self._ok("")              # rename onto a known-absent dst
        return BOT

    def _pred_ln(self, p, spec, vt):
        if not self._dotdot_sound(p["link"]):
            return BOT                           # round-4 class 2
        link = normpath(p["link"], self.cwd)
        if self._unwritable(link):
            return BOT                           # round-4 class 5
        if self._fire_hot(spec, link):
            return BOT                           # round-5 F2: fire lands here first
        st, lv = self._stat(link)
        if st == "alive":
            return self._tmpl("ln_exists", path=p["link"])
        pst, pv = self._stat(parent_of(link))
        # LAW: ln lstats the link path — a resolve-mined "dead" may be a dangling
        # symlink there, and ln (-s) onto it is File-exists in reality.
        if st != "dead" or not self._dead_certain(lv) \
                or pst != "alive" or not pv or pv["kind"] != "dir":
            return BOT
        if not p["symbolic"]:
            # round-4 class 4: a hard link needs the RESOLVED target to be a
            # known FILE — directories are un-hard-linkable ('Operation not
            # permitted', exit 1) and a kind-unknown target may be one.
            if not self._dotdot_sound(p["target"]):
                return BOT
            tst, tv = self._stat(self._resolve(normpath(p["target"], self.cwd)))
            if tst != "alive" or tv is None or tv["kind"] != "file":
                return BOT
        return self._ok("")

    def _pred_redir(self, p, spec, vt):
        if slashed(p["dst"]):
            return BOT               # F3 (round-3): 'p/' never opens as a file
        if not self._dotdot_sound(p["dst"]):
            return BOT               # round-4 class 2
        dst = normpath(p["dst"], self.cwd)
        if self._unwritable(dst):
            return BOT               # round-4 class 5
        rdst = self._resolve(dst)
        dstat, dv = self._stat(rdst)
        if dstat == "alive" and dv["kind"] == "dir":
            # F3: the open fails on a directory — harvested template or BOT,
            # never determined-ok
            return self._tmpl("redirect_isdir", path=p["dst"])
        final, chain, fully = self._wchain(dst)
        if not fully:
            return BOT               # LAW: landing site (or open outcome) uncertain
        pst, pv = self._stat(parent_of(final))
        if pst != "alive" or not pv or pv["kind"] != "dir":
            return BOT
        if p["prod"]["kind"] == "echo":
            return self._ok("")
        rp = {"form": "ls", "opts": ["-1"], "path": p["prod"]["dir"]} \
            if p["prod"]["kind"] == "ls" else {"form": "cat", "path": p["prod"]["file"]}
        got = self._read_predict(rp, spec)
        if got is BOT:
            return BOT
        if got["exit"] == 0:
            return self._ok("")                  # stdout captured into dst
        return got                               # producer error: stderr shows, its exit

    def _pred_cond(self, p, spec, vt):
        if slashed(p["path"]):
            return BOT           # F3: '[ -op p/ ]' truth never transfers from p
        if not self._dotdot_sound(p["path"]):
            return BOT           # round-4 class 2
        path = self._resolve(normpath(p["path"], self.cwd))
        if self._volatile(path):
            return BOT           # round-4 class 8: /proc,/sys truths churn
        if path in spec["degraded"]:
            return BOT           # round-5 F2: unsound fire — existence unknowable
        if path in spec["fs"]:
            truth = {"-e": True, "-f": True, "-d": False,
                     "-s": len(spec["fs"][path][0]) > 0 if spec["fs"][path][1] else None}[p["testop"]]
        else:
            st, v = self._stat(path)
            if st == "unknown":
                return BOT
            if st == "dead":
                truth = False    # test ops RESOLVE: a dangling link is false for all
            else:
                kind = v["kind"]
                known = bool(v.get("linkness_known"))
                if p["testop"] == "-e":
                    # a resolve-ENDPOINT of kind 'symlink' is an unresolvable link:
                    # -e follows it, outcome unknown
                    truth = None if kind == "symlink" else True
                elif p["testop"] == "-f":
                    # round-4 class 8: cat-success proves READABLE CONTENT, not
                    # regular-file (/dev/null is a chardev) — -f True needs an
                    # SST-created node; ls-dir evidence still yields a sound False.
                    if kind == "dir":
                        truth = False
                    elif kind == "file" and known:
                        truth = True
                    else:
                        truth = None
                elif p["testop"] == "-d":
                    if kind == "dir":
                        truth = True     # ls/dir evidence resolves to a dir either way
                    elif kind == "file":
                        truth = False    # readable content is never a directory
                    else:
                        truth = None
                else:   # -s: a rstripped observation hides a newlines-only file —
                        # empty OBSERVED content proves nothing about real size
                    c = v["content"]
                    if isinstance(c, str) and (len(c) > 0 or v["trailing_nl_known"]):
                        truth = len(c) > 0
                    else:
                        truth = None
        if truth is None:
            return BOT
        if not truth:
            return {"output": "", "exit": 1, "cwd": self.cwd}
        return self._read_predict(p["read"], spec)

    def _prod_nl_known(self, prod, spec):
        """F8/round-4 class 1b: whether the producer's full byte stream (incl.
        trailing-newline structure) is known — ls renders are structural (True);
        a cat producer inherits its source's trailing_nl_known."""
        if prod["kind"] != "cat":
            return True
        spath = self._resolve(normpath(prod["file"], self.cwd))
        fs = (spec or self._NO_SPEC)["fs"]
        if spath in fs:
            return bool(fs[spath][1])
        sv = self.fs.get(spath)
        return bool(sv and sv.get("trailing_nl_known"))

    def _pred_pipe(self, p, spec, vt):
        prod = p["prod"]
        rp = {"form": "ls", "opts": ["-1"], "path": prod["dir"]} \
            if prod["kind"] == "ls" else {"form": "cat", "path": prod["file"]}
        got = self._read_predict(rp, spec)
        if got is BOT:
            return BOT
        filt = p["filt"]
        if got["exit"] != 0:
            # §6.4: producer stderr folds into output; pipeline exit = LAST stage
            code = 1 if filt["kind"] == "grep" else 0   # grep with no input: no match
            return {"output": got["output"], "exit": code, "cwd": self.cwd}
        nl_known = self._prod_nl_known(prod, spec)
        lines = self._lines(got["output"])
        if filt["kind"] == "head":
            return self._ok("\n".join(lines[:filt["k"]]).rstrip("\n"))   # class 1c
        if filt["kind"] == "tail":
            if not filt["k"]:
                return self._ok("")
            if not nl_known:
                return BOT           # round-4 class 1b: window placement unknown
            return self._ok("\n".join(lines[-filt["k"]:]).rstrip("\n"))
        if "\x00" in got["output"]:
            return BOT               # round-4 class 1a: grep binary heuristic
        if not filt["tok"] and not nl_known:
            return BOT               # empty token matches invisible blank tails
        hits = [ln for ln in lines if filt["tok"] in ln][:8]
        return self._ok("\n".join(hits).rstrip("\n")) if hits \
            else {"output": "", "exit": 1, "cwd": self.cwd}

    def _pred_after(self, p, spec, vt):                               # R9
        if p["j"] in self.jobs:
            return BOT
        return self._ok(str(canonical_pid(p["j"])))

    def _pred_kill(self, p, spec, vt):                                # R9
        j, job = self._job_by_cpid(p["cpid"])
        if job is not None and j in spec["fired"]:
            job = dict(job, state="fired")       # fires land before the command runs
        if self._kill_transition(p["sig"], job) == "ERR":
            return self._tmpl("kill", pid=p["cpid"])
        return self._ok("")

    def _pred_ps(self, p, spec, vt):                                  # R9
        return self._ok(render_ps(self._ps_rows(spec)))

    def _pred_uptime(self, p, spec, vt):
        return self._ok(render_uptime(vt))

    def _pred_sleep(self, p, spec, vt):
        return self._ok("")

    def _pred_uname(self, p, spec, vt):
        return BOT   # image identity is never belief-derivable (the honest surface)




# =================================================================== smoke demo

def _demo():
    """Hand-built smoke trajectory (busybox-flavored templates) covering every
    mechanism: mkdir / echo-redirect create+append / created-content readback /
    ls edit-replay / mv transport + absence error / ln -s + readlink + read-through
    + dangling / hard ln surviving rm / cd + pwd / after-launch + STOP/CONT deferral
    + ps renders + fire + job-log readback + kill-miss / cond / pipe / blind- vs
    observed-capture redirect / rm -r workspace. Runs the SAME fold in both
    visibility modes and asserts determined predictions match the records exactly
    and load-bearing BOT steps stay BOT."""
    tmpl = {   # busybox-flavored {text, exit} table (F3: string-only entries fail closed)
        "cat": {"text": "cat: can't open '{path}': No such file or directory", "exit": 1},
        "ls": {"text": "ls: {path}: No such file or directory", "exit": 1},
        "cd": {"text": "/bin/sh: cd: can't cd to {path}", "exit": 2},
        "rm": {"text": "rm: can't remove '{path}': No such file or directory", "exit": 1},
        "mv": {"text": "mv: can't rename '{path}': No such file or directory", "exit": 1},
        "kill": {"text": "sh: can't kill pid {pid}: No such process", "exit": 1},
    }
    eff = "echo gamma_tok >> /tmp/w/task1.log"
    ps_wait = render_ps([(1, "S", "init"), (2, "S", "sleep 86400"),
                         (110, "S", f"after 1 5 {eff}")])
    ps_stop = render_ps([(1, "S", "init"), (2, "S", "sleep 86400"),
                         (110, "T", f"after 1 5 {eff}")])
    ps_gone = render_ps([(1, "S", "init"), (2, "S", "sleep 86400")])
    body = "alpha_token\nbeta_token"
    W = "/tmp/w/d"
    # (cmd, output, exit, cwd, expect) — expect: "det" exact-match, "bot" undetermined
    steps = [
        ("uname -s", "Linux", 0, "/", "bot"),
        (f"mkdir {W}", "", 0, "/", "bot"),
        (f"echo 'alpha_token' > {W}/notes.txt", "", 0, "/", "det"),
        (f"cat {W}/notes.txt", "alpha_token", 0, "/", "det"),
        (f"echo 'beta_token' >> {W}/notes.txt", "", 0, "/", "det"),
        (f"cat {W}/notes.txt", body, 0, "/", "det"),
        (f"ls -1 {W}", "notes.txt", 0, "/", "det"),
        (f"mv {W}/notes.txt {W}/notes.bak", "", 0, "/", "det"),
        (f"cat {W}/notes.bak", body, 0, "/", "det"),
        (f"cat {W}/notes.txt",
         f"cat: can't open '{W}/notes.txt': No such file or directory", 1, "/", "det"),
        (f"ln -s {W}/notes.bak {W}/link", "", 0, "/", "det"),
        (f"readlink {W}/link", f"{W}/notes.bak", 0, "/", "det"),
        (f"cat {W}/link", body, 0, "/", "det"),
        (f"ln {W}/notes.bak {W}/hard", "", 0, "/", "det"),
        (f"rm {W}/notes.bak", "", 0, "/", "det"),
        (f"cat {W}/hard", body, 0, "/", "det"),
        (f"cat {W}/link",
         f"cat: can't open '{W}/link': No such file or directory", 1, "/", "det"),
        (f"cd {W}", "", 0, W, "det"),
        ("pwd", W, 0, W, "det"),
        (f"after 1 5 '{eff}' & echo $!", "110", 0, W, "det"),
        ("/usr/local/bin/tj3-ps -o pid,stat,args", ps_wait, 0, W, "det"),
        ("kill -STOP 110", "", 0, W, "det"),
        ("/usr/local/bin/tj3-ps -o pid,stat,args", ps_stop, 0, W, "det"),
        ("cat /tmp/w/task1.log",
         "cat: can't open '/tmp/w/task1.log': No such file or directory", 1, W, "det"),
        ("kill -CONT 110", "", 0, W, "det"),
        ("cat /tmp/w/task1.log", "gamma_tok", 0, W, "det"),
        ("/usr/local/bin/tj3-ps -o pid,stat,args", ps_gone, 0, W, "det"),
        ("kill -0 110", "sh: can't kill pid 110: No such process", 1, W, "det"),
        (f"[ -f {W}/hard ] && cat {W}/hard", body, 0, W, "det"),
        (f"ls -1 {W} | head -n 2", "hard\nlink", 0, W, "det"),
        ("ls -1 /etc > /tmp/w/blind.lst", "", 0, W, "bot"),
        ("cat /tmp/w/blind.lst", "hosts\npasswd", 0, W, "bot"),
        (f"cat {W}/hard > /tmp/w/obs.txt", "", 0, W, "det"),
        ("cat /tmp/w/obs.txt", body, 0, W, "det"),
        (f"rm -r {W}", "", 0, W, "det"),
    ]
    results = {}
    for mode in ("collection", "sst"):
        st = ShellState(mode=mode, error_templates=tmpl)
        n_det = n_bot = 0
        deltas = []
        for i, (cmd, out, code, cwd, expect) in enumerate(steps):
            pred = st.predict(i, cmd)
            if expect == "det":
                want = {"output": out, "exit": code, "cwd": cwd}
                assert pred == want, \
                    f"[{mode}] step {i} {cmd!r}\n  pred={pred}\n  want={want}"
                n_det += 1
            else:
                assert pred is BOT, f"[{mode}] step {i} {cmd!r}: want BOT, got {pred}"
                n_bot += 1
            st.fold({"cmd": cmd, "output": out, "exit": code, "cwd": cwd})
            deltas.append(st.delta_text())
        assert deltas[2] == "delta: created /tmp/w/d/notes.txt(12B)", deltas[2]
        assert deltas[4] == "delta: appended /tmp/w/d/notes.txt(+11B)", deltas[4]
        assert deltas[7] == "delta: moved /tmp/w/d/notes.txt -> /tmp/w/d/notes.bak", deltas[7]
        assert deltas[30] == "delta: created /tmp/w/blind.lst", deltas[30]
        assert deltas[34] == ("delta: removed /tmp/w/d, removed /tmp/w/d/hard, "
                              "removed /tmp/w/d/link"), deltas[34]
        assert st.jobs[1]["state"] == "fired" and st.jobs[1]["deferrals"] == 1
        assert st.ws["/tmp/w/blind.lst"]["observed"] is False    # blind capture
        assert st.ws["/tmp/w/obs.txt"]["observed"] is True       # observed capture
        assert not st.mismatches, st.mismatches
        results[mode] = (n_det, n_bot)
    assert results["collection"] == results["sst"]

    # 5-state corner: TERM on a stopped job pends; the job dies at CONT (§5.3)
    s2 = ShellState(error_templates=tmpl)
    for i, (cmd, out) in enumerate([
            ("after 2 3 'echo x_tok >> /tmp/w/task2.log' & echo $!", "120"),
            ("kill -STOP 120", ""), ("kill 120", ""), ("kill -CONT 120", "")]):
        assert s2.predict(i, cmd) == {"output": out, "exit": 0, "cwd": "/"}
        s2.fold({"cmd": cmd, "output": out, "exit": 0, "cwd": "/"})
        if cmd == "kill 120":
            assert s2.jobs[2]["state"] == "stopped_pending_term"
    assert s2.jobs[2]["state"] == "killed"
    assert s2.predict(4, "kill -0 120")["exit"] == 1

    # parser totality: everything outside the frozen universe fails loudly
    for bad in ("curl http://x", "ls /etc; pwd", "cat a | grep b | head -n 2",
                "echo hi > /etc/evil", "kill -INT 110", "jobs", "sleep 5",
                "ls -l /etc | head -n 3", "stat -c '%y' /etc/hosts"):
        try:
            parse_command(bad)
        except ParseError:
            pass
        else:
            raise AssertionError(f"parser accepted {bad!r}")

    # exit_cls totality + abort family
    assert [exit_cls(c) for c in (0, 1, 2, 3, 126, 127, 124)] == [0, 1, 2, 1, 126, 127, 124]
    for c in (125, 128, 130, 137, -9):
        try:
            exit_cls(c)
        except ValueError:
            pass
        else:
            raise AssertionError(f"exit_cls classified abort-family {c}")

    n_det, n_bot = results["collection"]
    print(f"shell_state smoke OK: {len(steps)}-step trajectory, both modes — "
          f"{n_det} determined predictions exact-matched, {n_bot} honest BOTs; "
          f"delta channel, 5-state automaton, parser totality, exit_cls all green.")


if __name__ == "__main__":
    _demo()
