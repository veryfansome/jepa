"""Render-side canonical mask layer of dockerfs3 — frozen eval-path code.

Ratified by benchmarks/dockerfs3-design-draft.md §5.5 (mask-placement ruling) and
frozen against benchmarks/dockerfs3-prereg.md §3.2 (the per-field nondeterminism
table). This module owns EXACTLY the three render-canon rows of that table:

  1. runtime-mount mtimes (resolv.conf, hostname, hosts) — fresh per container
     start, so their time fields are masked wherever an `ls -l`-family render
     shows them (a listing of /etc, or a direct -l render of one of the three);
  2. wall clock / load / users / cpu TIME — masked to the fixed tokens the draft
     names (§5.1 uptime: clock -> 00:00:00, users -> 0, load -> 0.00 0.00 0.00;
     §5.4 ps: TIME -> 0:00). In the frozen command universe these fields surface
     only on `uptime` and `tj3-ps` renders;
  3. parent-dir mtimes of mutated dirs — time fields masked on `-l`-family ls
     renders whose TARGET is tracker-touched (the whole render: the table row's
     unit is the render, not the entry).

Everything else is left byte-for-byte — in particular mtimes/dates of untouched
shipped files (image-constant facts, the "leave raw" row). Store-time
virtualization (canonical PIDs, uptime elapsed -> vt, mutated-path mtimes ->
T+<vt>, the ps dialect canonicalizer) is the COLLECTOR's job (raw is never
stored un-virtualized); this layer re-masks only the measured render-side
residue, so it is a fixed point on collector-canonical renders.

CONTRACT
  canon(step, state) -> step'
  `step` is a raw step record {cmd, output, exit, cwd, ...}; `state` is the
  COLLECTION-MODE realenv.shell_state.ShellState at that step (pre-fold: predict/
  canon before fold, per the tracker's fold-order discipline). reencode.py /
  mv_encode.py / precompute_baselines re-fold shell_state over the raw jsonl per
  sequence and call canon on each step BEFORE any perception render — the
  touched-set is recomputed from the stored records themselves; no meta flag is
  trusted (F8). Only `output` is ever rewritten; when nothing masks, the SAME
  step object is returned (byte-for-byte trivially).

INVARIANTS (unit-tested in __main__)
  - TOTAL: never raises — a command outside the frozen universe (or a malformed
    step) passes through unmasked.
  - DETERMINISTIC + IDEMPOTENT: canon(canon(s, st), st) == canon(s, st); every
    mask token is a fixed point of its own pattern.
  - CONSERVATIVE: when `state` cannot establish touched-ness (state=None, no
    touched view), the -l mask does NOT fire; the state-independent masks
    (uptime / ps / runtime mounts) still apply.
  - ADVERSARIAL-MARKER SAFE: masking keys on the PARSED COMMAND form, never on
    output content — file content that merely looks like an `ls -l` line, a ps
    table, or an uptime render (e.g. under cat/grep/head) is never touched.
  - Fires due at this step land in its prologue before the recorded command
    (draft §3.2), so the touched view unions the speculative-fire paths.

Sha-pinned, version identity (prereg §1): any change to this module requires a
dated prereg amendment + re-baseline. Never a genome chunk; a genome-selected
perception impl cannot skip these masks. Deterministic, stdlib only.
"""

import re

from realenv.shell_state import basename_of, normpath, parent_of, parse_command

# ------------------------------------------------------------- frozen tokens

CLOCK_TOKEN = "00:00:00"          # wall clock (draft §5.1)
USERS_TOKEN = "0 users"           # users     (draft §5.1)
LOAD_TOKEN = "0.00, 0.00, 0.00"   # load      (draft §5.1)
CPU_TIME_TOKEN = "0:00"           # ps TIME   (draft §5.4)
LS_TIME_TOKEN = "Jan  1 00:00"    # -l-family date+time triplet mask

ETC = "/etc"
RUNTIME_MOUNT_PATHS = frozenset({"/etc/resolv.conf", "/etc/hostname", "/etc/hosts"})
RUNTIME_MOUNT_NAMES = frozenset({"resolv.conf", "hostname", "hosts"})

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


def _safe_resolve(state, path):
    """state's symlink resolution, conservatively total (identity on failure)."""
    try:
        return state._resolve(path)
    except Exception:
        return path


def _touched_view(state):
    """The tracker's touched-set at this step, unioned with the paths (and their
    parents) of fires due in THIS step's prologue (draft §3.2: due delayed
    effects commit before the recorded command runs — state.vt is this step's
    index, pre-fold). Empty when state cannot establish touched-ness."""
    if state is None:
        return frozenset()
    touched = set(getattr(state, "touched", None) or ())
    try:
        for q in state._spec_fires(state.vt)["fs"]:
            touched.add(q)
            touched.add(parent_of(q))
    except Exception:
        pass
    return touched


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


def _entry_name(line):
    """The name field of a listing row (last token, symlink ' -> target' shorn).
    Names with spaces don't match — conservative: such a row stays unmasked."""
    body = line.split(" -> ")[0].split()
    return body[-1] if body else ""


def _mask_listing_times(out, names=None):
    """Mask the date+time triplet on long-listing rows; `names` (basenames)
    restricts which rows — None masks every structurally-listing row."""
    masked = []
    for ln in out.split("\n"):
        if _LONG_LINE_RE.match(ln) and \
                (names is None or basename_of(_entry_name(ln)) in names):
            ln = _LS_TIME_RE.sub(LS_TIME_TOKEN, ln, count=1)
        masked.append(ln)
    return "\n".join(masked)


def _mask_ls(p, out, step, state):
    """The -l-family rows: whole-render time mask when the ls TARGET (given or
    symlink-resolved) is tracker-touched or is a runtime mount; the three
    runtime-mount rows only, on a listing of untouched /etc."""
    opts = p.get("opts") or []
    opt = opts[0] if opts else ""
    if "l" not in opt:                       # letter ell — '-1' is names-only
        return out
    cwd = step.get("cwd") or getattr(state, "cwd", "/") or "/"
    given = normpath(p["path"], cwd) if p.get("path") else normpath(cwd)
    targets = {given, _safe_resolve(state, given)}
    touched = _touched_view(state)
    if targets & touched:
        return _mask_listing_times(out)                       # row 3 (touched dir)
    if targets & RUNTIME_MOUNT_PATHS:
        return _mask_listing_times(out)                       # row 1 (mount itself)
    names = set()
    if ETC in targets:
        names |= RUNTIME_MOUNT_NAMES                          # row 1 (in /etc)
    # F14 (round-3 review) — row 3 covers a touched entry rendered via its
    # PARENT's listing too: a mkdir'd/mutated dir (or touched file) shows a fresh
    # mtime on its own row in the parent's -l render; mask those rows by name.
    names |= {basename_of(q) for q in touched if parent_of(q) in targets}
    if names:
        return _mask_listing_times(out, names=names)
    return out


# ------------------------------------------------------------- the contract

def canon(step, state):
    """The render-canon mask: step -> step' (prereg §3.2, render-canon rows only).

    Total, deterministic, idempotent, conservative. Returns the SAME object when
    nothing masks (byte-for-byte), else a shallow copy with only `output`
    rewritten. Never mutates `step`; never reads meta (F8)."""
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
        new = _mask_ls(p, out, step, state)
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
    # untouched shipped dir: byte-for-byte (the 'leave raw' row)
    keep = S("ls -l /usr/lib",
             "total 24\n-rw-r--r-- 1 root root 1234 Feb  3  2023 libfoo.so")
    assert canon(keep, st) is keep

    # row 1: runtime-mount rows masked inside an UNTOUCHED /etc listing only
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
          "-rw-r--r--  1 root root 1234 Jan  5  2024 os-release\n"
          "lrwxrwxrwx  1 root root   12 Feb  3  2023 mtab -> /proc/mounts")
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

    # prologue fires: a job due AT this step touches its dst before the command
    st_due = ShellState(mode="collection")
    for cmd, out in (("after 1 2 'echo x_tok >> /tmp/w/task1.log' & echo $!", "110"),
                     ("pwd", "/")):
        st_due.fold({"cmd": cmd, "output": out, "exit": 0, "cwd": "/"})
    assert st_due.vt == 2 and not st_due.touched      # fire due exactly now
    check(S("ls -l /tmp/w",
            "-rw-r--r-- 1 root root 6 Jul 20 12:02 task1.log"),
          st_due, "-rw-r--r-- 1 root root 6 Jan  1 00:00 task1.log")
    # ...but NOT-yet-due job establishes nothing (conservative)
    st_wait = ShellState(mode="collection")
    st_wait.fold({"cmd": "after 1 2 'echo x_tok >> /tmp/w/task1.log' & echo $!",
                  "output": "110", "exit": 0, "cwd": "/"})
    keep = S("ls -l /tmp/w", "-rw-r--r-- 1 root root 6 Jul 20 12:02 task1.log")
    assert canon(keep, st_wait) is keep

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
    # state=None: state-independent masks fire, touched mask conservatively off
    got = canon(S("uptime", "12:34:56 up 9,  3 users,  load average: 0.50, 0.40, 0.30"),
                None)
    assert got["output"] == "00:00:00 up 9,  0 users,  load average: 0.00, 0.00, 0.00"
    keep = S("ls -l /tmp/w/d", "-rw-r--r-- 1 root root 10 Jul 20 12:01 notes.txt")
    assert canon(keep, None) is keep

    print("render_canon smoke OK: uptime/ps fixed-token masks, touched-dir and "
          "runtime-mount -l masks, prologue-fire touched view, conservative "
          "refusals (untouched dir, not-due job, state=None), adversarial-marker "
          "safety, totality, idempotence, non-mutation all green.")


if __name__ == "__main__":
    _demo()
