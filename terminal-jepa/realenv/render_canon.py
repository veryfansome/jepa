"""Render-side canonical mask layer of dockerfs3 — frozen eval-path code.

Ratified by benchmarks/dockerfs3-design-draft.md §5.5 (mask-placement ruling) and
frozen against benchmarks/dockerfs3-prereg.md §3.2 (the per-field nondeterminism
table), AS AMENDED by the dated §5.5 amendment (2026-07-23): the `-l`-family
dir/file mtime mask MOVED from encode-time (here, conditionally) to STORE-time
(the collector's _V3Session.do). DG-3a diffs the RAW recorded jsonl BEFORE any
encode, so a conditional render-side mask could not make the stored bytes
deterministic — and its touched-set scoping missed the dirs (/, /etc, /tmp)
whose mtimes are set at CONTAINER-CREATION time by unrecorded bootstrap / docker
bind-mounts. This module owns the render-canon rows of that table:

  1. wall clock / load / users / cpu TIME — masked to the fixed tokens the draft
     names (§5.1 uptime: clock -> 00:00:00, users -> 0, load -> 0.00 0.00 0.00;
     §5.4 ps: TIME -> 0:00). In the frozen command universe these fields surface
     only on `uptime` and `tj3-ps` renders;
  2. `-l`-family ls date+time triplets — masked to LS_TIME_TOKEN on EVERY
     long-listing row (runtime mounts, mutated dirs, AND image-constant shipped
     files alike). The collector applies the SAME mask (canon_ls_l_text) store-
     time; canon() re-applies it here as DEFENSE IN DEPTH — on collector-
     canonical data it is a fixed point.

Store-time virtualization (canonical PIDs, uptime elapsed -> vt, the ps dialect
canonicalizer, and now the -l time mask) is the COLLECTOR's job (raw is never
stored un-virtualized); this layer re-masks the render-side residue, so it is a
fixed point on collector-canonical renders.

CONTRACT
  canon(step, state) -> step'
  `step` is a raw step record {cmd, output, exit, cwd, ...}. `state` (the
  COLLECTION-MODE ShellState at that step) is accepted for API compatibility but
  no longer consulted: after the store-time move the `-l` mask is UNCONDITIONAL
  (state-independent), so canon is a pure function of the step. reencode.py /
  mv_encode.py / precompute_baselines call canon on each raw step BEFORE any
  perception render; no meta flag is trusted (F8). Only `output` is ever
  rewritten; when nothing masks, the SAME step object is returned.

INVARIANTS (unit-tested in __main__)
  - TOTAL: never raises — a command outside the frozen universe (or a malformed
    step) passes through unmasked.
  - DETERMINISTIC + IDEMPOTENT: canon(canon(s), s) == canon(s); every mask token
    is a fixed point of its own pattern.
  - STATE-INDEPENDENT: the `-l` mask fires on every long-listing row regardless
    of `state` (state=None is fine); this is what the store-time move buys —
    the render mask no longer depends on a touched-set the tracker could miss.
  - ADVERSARIAL-MARKER SAFE: masking keys on the PARSED COMMAND form, never on
    output content — file content that merely looks like an `ls -l` line, a ps
    table, or an uptime render (e.g. under cat/grep/head) is never touched.

Sha-pinned, version identity (prereg §1): any change to this module requires a
dated prereg amendment + re-baseline. Never a genome chunk; a genome-selected
perception impl cannot skip these masks. Deterministic, stdlib only.
"""

import re

from realenv.shell_state import parse_command

# ------------------------------------------------------------- frozen tokens

CLOCK_TOKEN = "00:00:00"          # wall clock (draft §5.1)
USERS_TOKEN = "0 users"           # users     (draft §5.1)
LOAD_TOKEN = "0.00, 0.00, 0.00"   # load      (draft §5.1)
CPU_TIME_TOKEN = "0:00"           # ps TIME   (draft §5.4)
LS_TIME_TOKEN = "Jan  1 00:00"    # -l-family date+time triplet mask

# ------------------------------------------------------------- field patterns

# ls long-listing structural gate: a mode string opens the line (busybox + GNU;
# optional GNU SELinux '.' / ACL '+' / xattr '@' suffix). Lines that don't look
# like listing rows ('total N', error text) are never rewritten.
_LONG_LINE_RE = re.compile(r"^[-bcdlps][rwxsStT-]{9}[.+@]?\s")

# the date+time triplet of a listing row: 'Jul 20 12:34' | 'Jun 23  2024'.
# First occurrence per row only — the real time field precedes the name field,
# so a date-like FILENAME can never be the first match on a genuine row.
_LS_TIME_RE = re.compile(
    r"(?<=\s)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{1,2}\s+(?:\d{1,2}:\d{2}|\d{4})(?=\s)")

_CLOCK_RE = re.compile(r"(?<![\d:.])\d{1,2}:\d{2}:\d{2}(?![\d:.])")
_USERS_RE = re.compile(r"\b\d+\s+users?\b")
_LOAD_RE = re.compile(r"(load averages?:\s*)\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+")
_PS_TIME_RE = re.compile(r"(?<![\w:])\d+:\d{2}(?::\d{2})?(?![\w:])")


# ------------------------------------------------------------- helpers

def _parse(cmd):
    """parse_command, made total: None outside the frozen universe (pass-through)."""
    try:
        return parse_command(cmd)
    except Exception:
        return None


# ------------------------------------------------------------- per-form masks

def _mask_uptime(out):
    """Wall clock / users / load to the §5.1 fixed tokens. The elapsed field
    ('up <vt>') is store-time-virtualized by the collector and left alone."""
    out = _CLOCK_RE.sub(CLOCK_TOKEN, out, count=1)
    out = _USERS_RE.sub(USERS_TOKEN, out)
    out = _LOAD_RE.sub(lambda m: m.group(1) + LOAD_TOKEN, out)
    return out


def _mask_ps(out):
    """cpu TIME -> 0:00, column-aware: only when the header row declares a TIME
    column (the frozen tj3-ps template '-o pid,stat,args' has none, so canonical
    renders pass through untouched). First time-shaped token per row only."""
    lines = out.split("\n")
    if not lines or "TIME" not in lines[0].split():
        return out
    return "\n".join([lines[0]]
                     + [_PS_TIME_RE.sub(CPU_TIME_TOKEN, ln, count=1)
                        for ln in lines[1:]])


def _mask_listing_times(out):
    """Mask the date+time triplet on EVERY structurally-listing row (the row's
    unit is the render, per prereg §3.2). Non-listing lines ('total N', error
    text, content that doesn't open with a mode string) are never rewritten."""
    masked = []
    for ln in out.split("\n"):
        if _LONG_LINE_RE.match(ln):
            ln = _LS_TIME_RE.sub(LS_TIME_TOKEN, ln, count=1)
        masked.append(ln)
    return "\n".join(masked)


def canon_ls_l_text(output):
    """Shared `-l`-family time-mask — importable by the collector's STORE-time path
    (_V3Session.do) and re-called by canon() at encode-time (defense in depth).
    Replaces the date/time triplet on EVERY long-listing row with LS_TIME_TOKEN.
    Total, deterministic, idempotent (LS_TIME_TOKEN is a fixed point of the pattern);
    the 3-token triplet -> 3-token token substitution preserves the 9-field row shape,
    so the SST's -l child splice (shell_state._fold_ls) reads the same name field."""
    return _mask_listing_times(output)


def _mask_ls(p, out):
    """The -l-family rows: unconditional whole-render date+time mask. Letter ell —
    '-1'/'-a' names-only forms carry no time field and pass through. State-independent:
    the collector already masked this store-time, so on canonical data it is a fixed
    point; the unconditional pass is the defense-in-depth backstop the old touched-set
    scoping lacked (it missed /, /etc, /tmp — the SEVERE-1 leak)."""
    opts = p.get("opts") or []
    opt = opts[0] if opts else ""
    if "l" not in opt:                       # letter ell — '-1' is names-only
        return out
    return canon_ls_l_text(out)


# ------------------------------------------------------------- the contract

def canon(step, state=None):
    """The render-canon mask: step -> step' (prereg §3.2, render-canon rows only).

    Total, deterministic, idempotent. Returns the SAME object when nothing masks
    (byte-for-byte), else a shallow copy with only `output` rewritten. Never mutates
    `step`; never reads meta (F8). `state` is accepted for API compatibility but no
    longer consulted — the -l mask is unconditional after the store-time move."""
    out = step.get("output") if isinstance(step, dict) else None
    if not isinstance(out, str) or not out:
        return step
    p = _parse(step.get("cmd"))
    if p is None:
        return step                          # outside the universe: pass through
    form = p.get("form")
    if form == "uptime":
        new = _mask_uptime(out)
    elif form == "ps":
        new = _mask_ps(out)
    elif form == "ls":
        new = _mask_ls(p, out)
    else:
        return step
    if new == out:
        return step
    masked = dict(step)
    masked["output"] = new
    return masked


# =================================================================== smoke demo

def _demo():
    """Inline smoke: every mask row, both conservative refusals, adversarial
    markers, totality, idempotence, non-mutation — on hand-built renders over a
    real collection-mode ShellState fold."""
    from realenv.shell_state import ShellState, render_ps, render_uptime

    def check(step, state, want_output):
        got = canon(step, state)
        assert got["output"] == want_output, \
            f"{step['cmd']!r}\n  got : {got['output']!r}\n  want: {want_output!r}"
        again = canon(dict(got), state)
        assert again["output"] == got["output"], f"not idempotent: {step['cmd']!r}"
        if got["output"] == step["output"]:
            assert got is step, "unchanged step must be returned as-is"
        else:
            assert step["output"] != got["output"] and "output" in step  # input intact
        return got

    def S(cmd, output, cwd="/", code=0):
        return {"cmd": cmd, "output": output, "exit": code, "cwd": cwd}

    # -- a touched workspace state: mkdir + echo-redirect touch /tmp/w{,/d,...}
    st = ShellState(mode="collection")
    for cmd in ("mkdir /tmp/w/d", "echo 'alpha_tok' > /tmp/w/d/notes.txt"):
        st.fold({"cmd": cmd, "output": "", "exit": 0, "cwd": "/"})

    # row 2: uptime raw -> fixed tokens (clock/users/load; 'up ...' untouched)
    check(S("uptime",
            " 12:34:56 up 3 days, 42 min,  2 users,  load average: 0.15, 0.10, 0.05"),
          st,
          " 00:00:00 up 3 days, 42 min,  0 users,  load average: 0.00, 0.00, 0.00")
    # collector-canonical uptime render is a fixed point (same object back)
    can = S("uptime", render_uptime(7))
    assert canon(can, st) is can

    # row 2: bare tj3-ps (TIME column) masked; frozen template render untouched
    check(S("/usr/local/bin/tj3-ps",
            "  PID USER     TIME   COMMAND\n"
            "    1 root      0:00 sleep 86400\n"
            "  110 root      0:03 after 1 5 echo gamma_tok >> /tmp/w/task1.log"),
          st,
          "  PID USER     TIME   COMMAND\n"
          "    1 root      0:00 sleep 86400\n"
          "  110 root      0:00 after 1 5 echo gamma_tok >> /tmp/w/task1.log")
    can = S("/usr/local/bin/tj3-ps -o pid,stat,args",
            render_ps([(1, "S", "init"), (2, "S", "sleep 86400")]))
    assert canon(can, st) is can

    # row 3: -l-family render of a tracker-touched dir -> whole-render time mask
    check(S("ls -l /tmp/w/d",
            "total 8\n"
            "-rw-r--r--    1 root     root            10 Jul 20 12:01 notes.txt"),
          st,
          "total 8\n"
          "-rw-r--r--    1 root     root            10 Jan  1 00:00 notes.txt")
    check(S("ls -ld d", "drwxr-xr-x 2 root root 4096 Jul 20 12:01 d", cwd="/tmp/w"),
          st, "drwxr-xr-x 2 root root 4096 Jan  1 00:00 d")
    # unconditional (§5.5 amendment): a shipped-file -l row is now masked too — the
    # store-time move dropped the 'leave image-constant mtimes raw' row, so canon is a
    # fixed point on the collector-canonical bytes (which already carry LS_TIME_TOKEN)
    check(S("ls -l /usr/lib",
            "total 24\n-rw-r--r-- 1 root root 1234 Feb  3  2023 libfoo.so"),
          st,
          "total 24\n-rw-r--r-- 1 root root 1234 Jan  1 00:00 libfoo.so")

    # every -l row of an /etc listing is masked (runtime mounts AND shipped files);
    # symlink ' -> target' rows keep their target text, only the time triplet masks
    check(S("ls -la /etc",
            "total 24\n"
            "-rw-r--r--. 1 root root  158 Jun 23  2024 hosts\n"
            "-rw-r--r--  1 root root   13 Jul 20 09:12 hostname\n"
            "-rw-r--r--  1 root root  100 Jul 20 09:12 resolv.conf\n"
            "-rw-r--r--  1 root root 1234 Jan  5  2024 os-release\n"
            "lrwxrwxrwx  1 root root   12 Feb  3  2023 mtab -> /proc/mounts"),
          st,
          "total 24\n"
          "-rw-r--r--. 1 root root  158 Jan  1 00:00 hosts\n"
          "-rw-r--r--  1 root root   13 Jan  1 00:00 hostname\n"
          "-rw-r--r--  1 root root  100 Jan  1 00:00 resolv.conf\n"
          "-rw-r--r--  1 root root 1234 Jan  1 00:00 os-release\n"
          "lrwxrwxrwx  1 root root   12 Jan  1 00:00 mtab -> /proc/mounts")
    check(S("ls -l /etc/resolv.conf",
            "-rw-r--r-- 1 root root 100 Jul 20 09:12 /etc/resolv.conf"),
          st, "-rw-r--r-- 1 root root 100 Jan  1 00:00 /etc/resolv.conf")
    check(S("ls -l hosts", "-rw-r--r-- 1 root root 158 Jul 20 09:12 hosts",
            cwd="/etc"),
          st, "-rw-r--r-- 1 root root 158 Jan  1 00:00 hosts")

    # touched /etc (Tier-S rm) beats the names filter: whole render masked
    st_etc = ShellState(mode="collection")
    st_etc.fold({"cmd": "rm /etc/foo.conf", "output": "", "exit": 0, "cwd": "/"})
    check(S("ls -l /etc",
            "-rw-r--r-- 1 root root 1234 Jan  5  2024 os-release\n"
            "-rw-r--r-- 1 root root  158 Jun 23  2024 hosts"),
          st_etc,
          "-rw-r--r-- 1 root root 1234 Jan  1 00:00 os-release\n"
          "-rw-r--r-- 1 root root  158 Jan  1 00:00 hosts")

    # state-independent: the -l mask fires regardless of the job table / touched set —
    # a due fire, a not-yet-due job, or no state at all, the render masks the same way
    st_due = ShellState(mode="collection")
    for cmd, out in (("after 1 2 'echo x_tok >> /tmp/w/task1.log' & echo $!", "110"),
                     ("pwd", "/")):
        st_due.fold({"cmd": cmd, "output": out, "exit": 0, "cwd": "/"})
    check(S("ls -l /tmp/w",
            "-rw-r--r-- 1 root root 6 Jul 20 12:02 task1.log"),
          st_due, "-rw-r--r-- 1 root root 6 Jan  1 00:00 task1.log")
    st_wait = ShellState(mode="collection")
    st_wait.fold({"cmd": "after 1 2 'echo x_tok >> /tmp/w/task1.log' & echo $!",
                  "output": "110", "exit": 0, "cwd": "/"})
    check(S("ls -l /tmp/w", "-rw-r--r-- 1 root root 6 Jul 20 12:02 task1.log"),
          st_wait, "-rw-r--r-- 1 root root 6 Jan  1 00:00 task1.log")

    # adversarial markers: content that LOOKS like ls -l / ps / uptime under a
    # read verb is never touched (mask keys on the parsed command, not content)
    evil = ("-rw-r--r-- 1 root root 42 Jul 20 12:00 evil\n"
            "12:34:56 up 3 days,  2 users,  load average: 1.00, 2.00, 3.00\n"
            "  PID USER     TIME   COMMAND\n"
            "    7 root      9:59 sh")
    for cmd in ("cat /tmp/w/d/notes.txt",            # touched path, wrong verb
                "grep -F -m 8 root /tmp/w/d/notes.txt",
                "head -n 4 /tmp/w/d/notes.txt",
                "ls -1 /tmp/w/d"):                   # ls, but names-only form
        keep = S(cmd, evil)
        assert canon(keep, st) is keep, f"adversarial mask through {cmd!r}"

    # totality: outside-universe / malformed steps pass through, never raise
    for step in (S("curl -s http://x", evil), S("ls /etc; pwd", evil),
                 S("", "x"), {"cmd": None, "output": "x", "exit": 0, "cwd": "/"},
                 S("uptime", ""), {"cmd": "uptime", "output": None, "exit": 0}):
        assert canon(step, st) is step
    # state=None: every mask still fires (the -l mask is unconditional after the move)
    got = canon(S("uptime", "12:34:56 up 9,  3 users,  load average: 0.50, 0.40, 0.30"),
                None)
    assert got["output"] == "00:00:00 up 9,  0 users,  load average: 0.00, 0.00, 0.00"
    got = canon(S("ls -l /tmp/w/d", "-rw-r--r-- 1 root root 10 Jul 20 12:01 notes.txt"),
                None)
    assert got["output"] == "-rw-r--r-- 1 root root 10 Jan  1 00:00 notes.txt"

    print("render_canon smoke OK: uptime/ps fixed-token masks, unconditional -l "
          "date+time mask (defense in depth for the store-time canon), adversarial-"
          "marker safety, totality, idempotence, state-independence all green.")


if __name__ == "__main__":
    _demo()
