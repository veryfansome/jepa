"""The ONE signature/mode/cell labeler for the dockerfs3 (v3.0) command universe.

Frozen eval-path code (sha-pinned, version identity — never a genome chunk). One shared
implementation imported by the collector (writes meta.sig), the harness (re-derives sig
from cmd text and ASSERTS equality, F8), and benchmarks/class_measure.py. v1/v2 roots
keep seq_worldmodel.verb_of (first token) bit-identically — for every v1/v2 command
shape, sig() returns exactly that first token.

Sources of truth (this module implements them verbatim; on conflict THEY win):
  - benchmarks/dockerfs3-prereg.md §4.1 — the G3 BNF + the complete meta.sig vocabulary
    (11 composed families: 6 pipe + 2 redir + 3 cond) + the frozen per-family mode sets
    ("pipe-grep {hit, miss}; cond {hit, miss}; others {ok}").
  - benchmarks/dockerfs3-prereg.md §1 toolset + Annex P0 UD-9 Route B: the ps arm invokes
    the vendored busybox at /usr/local/bin/tj3-ps; sig() normalizes that path to "ps".
  - design draft §9.1: mode is keyed on (exit, output-emptiness) ONLY — recoverable from
    the step record alone (F8); §13.1: "mutation successes are empty-exit-0".
  - dockerfs2 prereg Amendment 3 (the v2 precedent, generalized to all read verbs):
    "exit != 0 or empty output <=> miss".
  - design draft §9.2 / §9.5 (round-6 B1, round-7 M2): the cell pseudo-verb is
    "sig|mode|scope"; created-scope splits on ws_observed into "...|created" /
    "...|created-obs"; cell keys are ATOMIC — sigs themselves contain "|"
    (e.g. "pipe:ls|head"), so no consumer may ever parse a cell key by splitting.

The universe is TOTAL and there is ONE totality authority AND ONE lexer (F2,
round-2 review): membership and structure both come from
realenv.shell_state.parse_command (the prereg §4.1 universe parser). sig() holds
no tokenizer of its own — it calls parse_command, translates its ParseError into
this module's fail-closed ValueError (never a silent fallback — fail-closed even
under python -O, hence no `assert`), and maps the RETURNED PARSE STRUCTURE onto
the frozen label vocabulary. So parse_command and sig() accept/reject the
identical universe by construction — including quoted-space and '\\''-escaped
tokens and every documented exclusion: jobs/fg/bg/wait, kill -INT, `<<<`, `||`,
`;`, `if/then`, depth-2 pipes, REDIR_IN, `cd` in any composed string, `ps` in
pipes, `ls -l`/find producers, and non-workspace redirection targets. (Import
direction: verbsig -> shell_state only; shell_state never imports verbsig back.)

Mode vocabulary (per sig; MODES below is the frozen enumeration):
  read verbs + pipe-grep + cond .... {hit, miss}   hit <=> exit==0 and non-empty
  state verbs (cd, rm, mv, ln, mkdir, touch, kill, sleep) ... {ok, miss}
                                    ok <=> exit==0 and EMPTY (the mutation-success
                                    shape; such cells are the §9.2 rule-2 "ack"
                                    CLASS candidates — ack is a class, not a mode)
  after (bgjob launch) ............ {ok, miss}     ok <=> exit==0 and non-empty
                                    (the recorded launch emits the cpid via `echo $!`)
  pipe head/tail + redir .......... {ok}           constant per the frozen §4.1 mode
                                    sets; anomalies are collection-side G-EM/MutGuard
                                    aborts, never re-labeled here.

Pure python, stdlib only, deterministic.
"""

from realenv import shell_state as _shell_state

# ---------------------------------------------------------------- frozen vocabulary

ATOMIC_VERBS = frozenset({
    # v2's nine
    "uname", "cd", "ls", "cat", "head", "tail", "stat", "find", "grep",
    # v3 additions (prereg §1 toolset)
    "pwd", "echo", "rm", "mv", "ln", "readlink", "mkdir", "touch",
    "ps", "kill", "after", "uptime", "sleep",
})

PIPE_SIGS = ("pipe:ls|head", "pipe:ls|tail", "pipe:ls|grep",
             "pipe:cat|head", "pipe:cat|tail", "pipe:cat|grep")
REDIR_SIGS = ("redir:echo>", "redir:prod>")
COND_SIGS = ("cond:cat", "cond:ls", "cond:head")
COMPOSED_SIGS = PIPE_SIGS + REDIR_SIGS + COND_SIGS  # the 11 measurable families

SIGS = tuple(sorted(ATOMIC_VERBS)) + COMPOSED_SIGS  # the complete meta.sig vocabulary

SCOPES = ("native", "mutated", "created")  # meta.state_scope values (draft §7.5)

_READ_RULE = frozenset({"uname", "ls", "cat", "head", "tail", "stat", "find", "grep",
                        "pwd", "readlink", "uptime", "ps", "echo",
                        "pipe:ls|grep", "pipe:cat|grep",
                        "cond:cat", "cond:ls", "cond:head"})
_STATE_RULE = frozenset({"cd", "rm", "mv", "ln", "mkdir", "touch", "kill", "sleep"})
_LAUNCH_RULE = frozenset({"after"})
_CONST_OK = frozenset({"pipe:ls|head", "pipe:ls|tail", "pipe:cat|head", "pipe:cat|tail",
                       "redir:echo>", "redir:prod>"})

# frozen per-sig mode sets (validated by cell(); mirrors the rule groups above)
MODES = {}
for _s in SIGS:
    if _s in _CONST_OK:
        MODES[_s] = ("ok",)
    elif _s in _READ_RULE:
        MODES[_s] = ("hit", "miss")
    else:  # state + launch
        MODES[_s] = ("ok", "miss")
del _s

# parse-structure -> composed-family label (the 3 cond READ forms)
_COND_FAM = {"cat": "cond:cat", "ls": "cond:ls", "head": "cond:head"}


# ---------------------------------------------------------------- sig

def sig(cmd):
    """Signature label for ANY command in the frozen v3 universe.

    ONE lexer, ONE totality authority (F2, round-2 review): membership AND
    structure both come from shell_state.parse_command — sig() holds no tokenizer
    of its own; it maps the returned parse structure onto the frozen label
    vocabulary, so parse_command accepts a command IFF sig() labels it, by
    construction (quoted-space and '\\''-escaped tokens included).

    Returns the first-token verb for simple commands (bit-identical to v1/v2
    verb_of; /usr/local/bin/tj3-ps is normalized to "ps" by the parser),
    "after"/"kill" for the audited process forms, and one of the 11
    composed-family sigs for G3 commands. Raises the fail-closed ValueError on
    anything outside the universe (never a silent fallback)."""
    if not isinstance(cmd, str):
        # F15 (round-3 review): fail closed on non-string input too — never leak
        # a foreign TypeError out of the one labeler
        raise ValueError(f"verbsig: non-string command (fail-closed): {cmd!r}")
    try:
        p = _shell_state.parse_command(cmd)
    except _shell_state.ParseError as e:
        raise ValueError(
            f"verbsig: out-of-universe command (parse_command rejected): {cmd!r}") from e
    form = p["form"]
    if form == "pipe":
        return f"pipe:{p['prod']['kind']}|{p['filt']['kind']}"
    if form == "redir":
        return "redir:echo>" if p["prod"]["kind"] == "echo" else "redir:prod>"
    if form == "cond":
        return _COND_FAM[p["read"]["form"]]
    if form not in ATOMIC_VERBS:      # fail-closed: a parse form without a label
        raise ValueError(f"verbsig: parse form {form!r} has no sig label: {cmd!r}")
    return form


composed_verb = sig  # the draft §6.2 / prereg §7 name for the same one labeler


# ---------------------------------------------------------------- mode

def mode(verb_or_sig, exit, output_empty):
    """Per-(verb, mode) outcome label from (exit, output-emptiness) ONLY — recoverable
    from the step record alone (F8). See the module docstring for the frozen rule table;
    the read rule is the v2 grep/find precedent generalized (exit!=0 or empty => miss)."""
    v = verb_or_sig
    if v in _CONST_OK:
        return "ok"
    if v in _READ_RULE:
        return "hit" if (exit == 0 and not output_empty) else "miss"
    if v in _STATE_RULE:
        return "ok" if (exit == 0 and output_empty) else "miss"
    if v in _LAUNCH_RULE:
        return "ok" if (exit == 0 and not output_empty) else "miss"
    raise ValueError(f"verbsig: unknown verb/sig for mode(): {verb_or_sig!r}")


# ---------------------------------------------------------------- cell

def cell(sig, mode, state_scope, ws_observed=None):
    """The cell pseudo-verb key "sig|mode|scope" (the classification/fitness unit).

    For state_scope=="created" the unit splits on ws_observed (round-6 B1):
    observed-capture -> "sig|mode|created-obs", blind-capture -> "sig|mode|created";
    ws_observed is REQUIRED there and ignored for native/mutated (those scopes are not
    split). The returned string is an ATOMIC opaque key — sigs contain '|', so it must
    never be parsed by splitting; consumers compare by equality only."""
    if sig not in MODES:
        raise ValueError(f"verbsig: unknown sig for cell(): {sig!r}")
    if mode not in MODES[sig]:
        raise ValueError(f"verbsig: mode {mode!r} not in {MODES[sig]} for sig {sig!r}")
    if state_scope not in SCOPES:
        raise ValueError(f"verbsig: unknown state_scope {state_scope!r} (not in {SCOPES})")
    if state_scope == "created":
        if ws_observed is None:
            raise ValueError("verbsig: created-scope cell requires ws_observed (bool)")
        state_scope = "created-obs" if ws_observed else "created"
    return f"{sig}|{mode}|{state_scope}"
