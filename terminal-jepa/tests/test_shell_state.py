"""Unit tests for the three frozen dockerfs3 eval-path modules (freeze-order step 3):

  realenv/shell_state.py  — the ONE tracker: parser totality over the frozen universe
                            (prereg §4.1), the R1–R9 determined surfaces (draft §10.2),
                            the 5-state job automaton (draft §5.3), the two visibility
                            modes + DG-4c parity (draft §10.1), the canonical delta
                            channel + exit vocabulary (prereg §4.2), per-image error
                            templates (draft §3.5);
  realenv/verbsig.py      — the ONE sig/mode/cell labeler: the complete §4.1 sig
                            vocabulary (22 atomic verbs + 11 composed families), the
                            frozen mode rules, cell pseudo-verb keys incl. the
                            created/created-obs split and key atomicity (draft §9.2/§9.5);
  realenv/render_canon.py — exactly the three render-canon rows of the prereg §3.2
                            per-field nondeterminism table (draft §5.5).

Hand-built trajectories only — no docker, no torch, stdlib-only. Seeds the fuller
tests/test_collect_v3.py battery (prereg §9)."""

import pathlib as _pathlib
import sys as _sys

_ROOT = str(_pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import unittest

from realenv import render_canon as RC
from realenv import shell_state as M
from realenv import verbsig as V

# ---------------------------------------------------------------- shared fixtures

# a busybox-flavored probe-harvested template table (draft §3.5); predict() answers
# error steps ONLY through such a table — never a hand-authored dialect enum.
TMPL = {   # F3 (round-6): the canonical {text, exit} shape — string-only entries fail
           # closed; busybox-flavored exits (cat/ls/rm/mv/kill=1, cd=2)
    "cat": {"text": "cat: can't open '{path}': No such file or directory", "exit": 1},
    "ls": {"text": "ls: {path}: No such file or directory", "exit": 1},
    "cd": {"text": "/bin/sh: cd: can't cd to {path}", "exit": 2},
    "rm": {"text": "rm: can't remove '{path}': No such file or directory", "exit": 1},
    "mv": {"text": "mv: can't rename '{path}': No such file or directory", "exit": 1},
    "kill": {"text": "sh: can't kill pid {pid}: No such process", "exit": 1},
}

PS = "/usr/local/bin/tj3-ps -o pid,stat,args"


def step(cmd, out="", code=0, cwd="/"):
    """A rendered step record — exactly the fields the encoder sees (no meta)."""
    return {"cmd": cmd, "output": out, "exit": code, "cwd": cwd}


def new_state(mode="collection"):
    return M.ShellState(mode=mode, error_templates=TMPL)


def ok(output, cwd="/"):
    return {"output": output, "exit": 0, "cwd": cwd}


EFF = "echo gamma_tok >> /tmp/w/task1.log"


def _traj():
    """The shared 20-step trajectory (parity + determinism): CUD chain, append, ls
    edit-replay, cd/pwd, launch->fire->readback, mv, blind vs observed capture,
    cond, pipe, rm -r, per-image error template. Rows: (cmd, out, exit, post-cwd,
    'det'|'bot')."""
    ps_wait = M.render_ps([(1, "S", "init"), (2, "S", "sleep 86400"),
                           (110, "S", f"after 1 3 {EFF}")])
    body = "alpha_tok\nbeta_tok"
    err = "cat: can't open '/tmp/w/d/m.txt': No such file or directory"
    return [
        ("uname -s", "Linux", 0, "/", "bot"),                       # image identity: never derivable
        ("mkdir /tmp/w/d", "", 0, "/", "bot"),                      # parent listing unknown pre-mkdir
        ("echo 'alpha_tok' > /tmp/w/d/n.txt", "", 0, "/", "det"),
        ("cat /tmp/w/d/n.txt", "alpha_tok", 0, "/", "det"),
        ("echo 'beta_tok' >> /tmp/w/d/n.txt", "", 0, "/", "det"),
        ("ls -1 /tmp/w/d", "n.txt", 0, "/", "det"),
        ("cd /tmp/w/d", "", 0, "/tmp/w/d", "det"),
        ("pwd", "/tmp/w/d", 0, "/tmp/w/d", "det"),
        (f"after 1 3 '{EFF}' & echo $!", "110", 0, "/tmp/w/d", "det"),
        (PS, ps_wait, 0, "/tmp/w/d", "det"),
        ("mv n.txt m.txt", "", 0, "/tmp/w/d", "det"),
        ("cat m.txt", body, 0, "/tmp/w/d", "det"),                  # job 1 fires in this prologue
        ("cat /tmp/w/task1.log", "gamma_tok", 0, "/tmp/w/d", "det"),
        ("ls -1 /etc > /tmp/w/blind.lst", "", 0, "/tmp/w/d", "bot"),
        ("cat /tmp/w/blind.lst", "hosts\npasswd", 0, "/tmp/w/d", "bot"),
        ("[ -f m.txt ] && cat m.txt", body, 0, "/tmp/w/d", "det"),
        ("ls -1 /tmp/w/d | head -n 1", "m.txt", 0, "/tmp/w/d", "det"),
        ("cd ..", "", 0, "/tmp/w", "det"),
        ("rm -r /tmp/w/d", "", 0, "/tmp/w", "det"),
        ("cat /tmp/w/d/m.txt", err, 1, "/tmp/w", "det"),
    ]


def _run(st, rows, check=None):
    """predict-before-fold over rows; asserts det rows exact-match the record and
    bot rows stay BOT (when check is a TestCase). Returns (preds, delta_texts)."""
    preds, deltas = [], []
    for cmd, out, code, cwd, expect in rows:
        pred = st.predict(st.vt, cmd)
        if check is not None:
            if expect == "det":
                check.assertEqual(pred, {"output": out, "exit": code, "cwd": cwd},
                                  f"[{st.mode}] step {st.vt}: {cmd!r}")
            else:
                check.assertIsNone(pred, f"[{st.mode}] step {st.vt}: {cmd!r}")
        preds.append(pred)
        st.fold({"cmd": cmd, "output": out, "exit": code, "cwd": cwd})
        deltas.append(st.delta_text())
    return preds, deltas


def snapshot(st):
    return (st.cwd, st.fs, st.ws, st.jobs, st.touched, st.fs_clock, st.vt,
            st.mismatches)


# ================================================================ parser totality

class TestParserTotality(unittest.TestCase):
    """parse_command is TOTAL over the frozen universe (prereg §4.1) — every BNF
    production parses; anything outside assert-fails (ParseError, a mint gate)."""

    UNIVERSE = [
        # v2 atomic templates
        ("uname", "uname"), ("uname -sr", "uname"),
        ("pwd", "pwd"),
        ("cd", "cd"), ("cd /tmp/w", "cd"), ("cd ..", "cd"),
        ("ls", "ls"), ("ls -1 /etc", "ls"), ("ls -a /etc", "ls"),
        ("ls -la /etc", "ls"), ("ls -l /etc", "ls"),
        ("cat /etc/hosts", "cat"),
        ("head -n 3 /etc/hosts", "head"),
        ("tail -n 5 /etc/hosts", "tail"),
        ("stat -c '%n %s %F %a' /etc/hosts", "stat"),
        ("find /etc -maxdepth 2 -name '*.conf'", "find"),
        ("find /usr -maxdepth 1 -type d -name 'l*'", "find"),
        ("grep -F -m 8 localhost /etc/hosts", "grep"),
        ("grep -F -i -m 8 tok /etc/hosts", "grep"),
        # v3 atomic additions
        ("echo 'hello world'", "echo"),
        ("rm /tmp/w/x", "rm"), ("rm -r /tmp/w/d", "rm"),
        ("mv /tmp/w/a /tmp/w/b", "mv"),
        ("ln -s /tmp/w/a /tmp/w/l", "ln"), ("ln /tmp/w/a /tmp/w/h", "ln"),
        ("readlink /tmp/w/l", "readlink"),
        ("mkdir /tmp/w/d", "mkdir"), ("touch /tmp/w/f", "touch"),
        ("uptime", "uptime"), ("sleep 0", "sleep"), ("sleep 1", "sleep"),
        # audited process forms
        ("kill 110", "kill"), ("kill -STOP 110", "kill"), ("kill -CONT 110", "kill"),
        ("kill -9 110", "kill"), ("kill -0 110", "kill"),
        (PS, "ps"), ("/usr/local/bin/tj3-ps", "ps"),
        ("after 1 5 'echo g_tok >> /tmp/w/task1.log' & echo $!", "after"),
        # G3 PIPE — all 6 families
        ("ls -1 /etc | head -n 3", "pipe"), ("ls -1 /etc | tail -n 2", "pipe"),
        ("ls -1 /etc | grep -F -m 8 conf", "pipe"),
        ("cat /etc/hosts | head -n 1", "pipe"), ("cat /etc/hosts | tail -n 1", "pipe"),
        ("cat /etc/hosts | grep -F -m 8 host", "pipe"),
        # G3 REDIR_W — echo/prod × >/>>
        ("echo 'x_tok' > /tmp/w/f", "redir"), ("echo 'x_tok' >> /tmp/w/f", "redir"),
        ("ls -1 /etc > /tmp/w/f", "redir"), ("cat /etc/hosts >> /tmp/w/f", "redir"),
        # G3 COND — all 4 TESTOPs, all 3 READs
        ("[ -e /tmp/w/f ] && cat /tmp/w/f", "cond"),
        ("[ -f /tmp/w/f ] && head -n 2 /tmp/w/f", "cond"),
        ("[ -d /tmp/w ] && ls -1 /tmp/w", "cond"),
        ("[ -s /tmp/w/f ] && cat /tmp/w/f", "cond"),
    ]

    def test_every_universe_form_parses(self):
        """Every prereg §4.1 production parses to its form — and the ONE labeler
        (verbsig.sig) accepts the identical universe (the two totalities agree)."""
        for cmd, form in self.UNIVERSE:
            p = M.parse_command(cmd)
            self.assertEqual(p["form"], form, f"{cmd!r} parsed as {p['form']!r}")
            V.sig(cmd)   # must not raise: same universe, same totality

    BADS = [
        "",                                            # empty
        "curl -s http://x",                            # unknown verb
        "jobs", "wait", "fg 1",                        # UD-1 exclusions
        "kill -INT 110",                               # -INT is out (UD-1)
        "kill -HUP 110",                               # outside the signal family
        "sleep 5",                                     # foreground sleep is {0,1}
        "ps",                                          # policy invokes the vendored path
        "/usr/local/bin/tj3-ps -ef",                   # non-frozen ps template
        "ls /etc; pwd",                                # ';' excluded
        "cat /a | grep -F -m 8 b | head -n 2",         # depth-2 pipe
        "true || false",                               # '||' excluded
        "cat <<< 'x'",                                 # '<<<' excluded
        "echo 'hi' > /etc/evil",                       # non-workspace redirect
        "ls -1 /etc > /tmp/w/a > /tmp/w/b",            # two operators
        "ls -l /etc | head -n 3",                      # ls -l producer banned
        "find /etc -maxdepth 1 -name x | head -n 2",   # find producer pruned
        "stat -c '%y' /etc/hosts",                     # off-template stat format
        "grep -F -m 3 tok /etc/hosts",                 # off-template grep -m
        "[ -x /tmp/w/f ] && cat /tmp/w/f",             # TESTOP outside {-e,-f,-d,-s}
        "[ -e /tmp/w/a ] && cat /tmp/w/b",             # COND READ path != tested path
        "cd /tmp && ls -1 /tmp",                       # '&&' outside cond/after
        "after 1 5 'echo x_tok > /tmp/w/f'",           # non-canonical after shape
        "cat 'unterminated",                           # unterminated quote
        # S1 review additions — both totality gates must reject all of these
        "echo x_tok > /tmp/w/f",                       # unquoted echo-redirect payload
        "echo 'a' extra > /tmp/w/f",                   # extra token after the payload
        "echo 'x' > /tmp/w",                           # dst == the workspace dir itself
        "echo 'x' > /tmp/w/",                          # dst == the workspace dir (slash)
        "after 1 5 'rm /tmp/w/x' & echo $!",           # non-echo after effect (§3.3)
        "after 1 5 'echo x_tok >> /tmp/w/other.log' & echo $!",   # off-namespace dst
        "after 1 5 'echo x_tok >> /tmp/w/task2.log' & echo $!",   # j/log mismatch
        "uname -x",                                    # option outside UNAME_OPTS
        "uname -s -m",                                 # two option tokens
        "ls -Z /etc",                                  # option outside LS_OPTS
        "ls -lart /etc",                               # option outside LS_OPTS
        "tail -n 5",                                   # missing path (template shape)
        "echo $HOME",                                  # '$' outside the after form
        "echo `id`",                                   # backtick excluded
        "echo a\\b",                                   # backslash excluded (§4.3)
        "cd /etc;pwd",                                 # ';' embedded in a token
        "cat < /tmp/w/f",                              # REDIR_IN excluded
    ]

    def test_out_of_universe_assert_fails(self):
        """Anything outside the frozen universe fails loudly (ParseError IS an
        AssertionError): fail loudly, never guess."""
        self.assertTrue(issubclass(M.ParseError, AssertionError))
        for bad in self.BADS:
            with self.assertRaises(AssertionError, msg=f"parser accepted {bad!r}"):
                M.parse_command(bad)

    def test_exit_cls_vocabulary(self):
        """prereg §4.2: TOTAL specifics-first mapping {0,1,2,126,127}, 124 pilot-only,
        nonzero catch-all -> 1; 125 / >=128 / negative are ABORT family — raise."""
        self.assertEqual([M.exit_cls(c) for c in (0, 1, 2, 3, 99, 126, 127, 124)],
                         [0, 1, 2, 1, 1, 126, 127, 124])
        for c in (125, 128, 130, 137, 143, 148, -9):
            with self.assertRaises(ValueError, msg=f"exit {c} must abort"):
                M.exit_cls(c)


# ================================================================ tracker mechanisms

class TestTrackerFilesystem(unittest.TestCase):
    """The R1–R8 surfaces on hand-built trajectories (draft §10.2)."""

    def test_cud_chain(self):
        """echo> create -> cat readback -> rm -> readback errors via the template."""
        st = new_state()
        self.assertEqual(st.predict(0, "echo 'alpha_tok' > /tmp/w/c.txt"), ok(""))
        st.fold(step("echo 'alpha_tok' > /tmp/w/c.txt"))
        self.assertEqual(st.delta_text(), "delta: created /tmp/w/c.txt(10B)")
        self.assertEqual(st.predict(1, "cat /tmp/w/c.txt"), ok("alpha_tok"))
        st.fold(step("cat /tmp/w/c.txt", "alpha_tok"))
        self.assertEqual(st.predict(2, "rm /tmp/w/c.txt"), ok(""))
        st.fold(step("rm /tmp/w/c.txt"))
        self.assertEqual(st.delta_text(), "delta: removed /tmp/w/c.txt")
        self.assertEqual(
            st.predict(3, "cat /tmp/w/c.txt"),
            {"output": "cat: can't open '/tmp/w/c.txt': No such file or directory",
             "exit": 1, "cwd": "/"})

    def test_mv_displacement(self):
        """After mv, the old path errors and the new path carries the content.
        (F1: the rename is determined only against a KNOWN-absent dst, so the
        workspace listing is observed first.)"""
        st = new_state()
        st.fold(step("echo 'body_tok' > /tmp/w/a.txt"))
        st.fold(step("ls -1 /tmp/w", "a.txt"))               # /tmp/w now complete
        self.assertEqual(st.predict(2, "mv /tmp/w/a.txt /tmp/w/b.txt"), ok(""))
        st.fold(step("mv /tmp/w/a.txt /tmp/w/b.txt"))
        self.assertEqual(st.delta_text(),
                         "delta: moved /tmp/w/a.txt -> /tmp/w/b.txt")
        self.assertEqual(st.predict(3, "cat /tmp/w/b.txt"), ok("body_tok"))
        self.assertEqual(
            st.predict(3, "cat /tmp/w/a.txt"),
            {"output": "cat: can't open '/tmp/w/a.txt': No such file or directory",
             "exit": 1, "cwd": "/"})
        # a second mv of the dead source is a determined error (R8)
        self.assertEqual(
            st.predict(3, "mv /tmp/w/a.txt /tmp/w/c.txt"),
            {"output": "mv: can't rename '/tmp/w/a.txt': No such file or directory",
             "exit": 1, "cwd": "/"})

    def test_symlink_vs_hardlink_after_rm(self):
        """rm of the original: the hardlink keeps the shared content, the symlink
        dangles (reads error); readlink still shows the stored target."""
        st = new_state()
        st.fold(step("mkdir /tmp/w/d"))
        st.fold(step("echo 'body_tok' > /tmp/w/d/f"))
        self.assertEqual(st.predict(2, "ln -s /tmp/w/d/f /tmp/w/d/link"), ok(""))
        st.fold(step("ln -s /tmp/w/d/f /tmp/w/d/link"))
        self.assertEqual(st.predict(3, "readlink /tmp/w/d/link"), ok("/tmp/w/d/f"))
        self.assertEqual(st.predict(3, "cat /tmp/w/d/link"), ok("body_tok"))
        self.assertEqual(st.predict(3, "ln /tmp/w/d/f /tmp/w/d/hard"), ok(""))
        st.fold(step("ln /tmp/w/d/f /tmp/w/d/hard"))
        st.fold(step("rm /tmp/w/d/f"))
        self.assertEqual(st.predict(5, "cat /tmp/w/d/hard"), ok("body_tok"))
        self.assertEqual(
            st.predict(5, "cat /tmp/w/d/link"),
            {"output": "cat: can't open '/tmp/w/d/link': No such file or directory",
             "exit": 1, "cwd": "/"})
        self.assertEqual(st.predict(5, "readlink /tmp/w/d/link"), ok("/tmp/w/d/f"))

    def test_append_ordering_and_observed_append_bot(self):
        """Appends replay in order for trajectory-created files (R5); an append onto
        merely-observed content makes further readbacks BOT (trailing \\n unknowable)."""
        st = new_state()
        st.fold(step("echo 'a1' > /tmp/w/f.txt"))
        st.fold(step("echo 'b2' >> /tmp/w/f.txt"))
        self.assertEqual(st.delta_text(), "delta: appended /tmp/w/f.txt(+3B)")
        st.fold(step("echo 'c3' >> /tmp/w/f.txt"))
        self.assertEqual(st.predict(3, "cat /tmp/w/f.txt"), ok("a1\nb2\nc3"))
        self.assertEqual(st.predict(3, "head -n 2 /tmp/w/f.txt"), ok("a1\nb2"))
        self.assertEqual(st.predict(3, "tail -n 1 /tmp/w/f.txt"), ok("c3"))
        self.assertEqual(st.predict(3, "grep -F -m 8 b2 /tmp/w/f.txt"), ok("b2"))
        self.assertEqual(st.predict(3, "grep -F -m 8 zz /tmp/w/f.txt"),
                         {"output": "", "exit": 1, "cwd": "/"})
        self.assertEqual(st.ws["/tmp/w/f.txt"]["appends"], 2)
        # merely-observed content: readback determined, append poisons it to BOT
        st.fold(step("ls -1 /etc > /tmp/w/b.lst"))
        st.fold(step("cat /tmp/w/b.lst", "hosts\npasswd"))
        self.assertEqual(st.predict(5, "cat /tmp/w/b.lst"), ok("hosts\npasswd"))
        st.fold(step("echo 'z9' >> /tmp/w/b.lst"))
        self.assertIsNone(st.predict(6, "cat /tmp/w/b.lst"))

    def test_blind_vs_observed_capture(self):
        """Blind capture (producer output never observed) -> ws.observed False and
        readback BOT; observed capture -> ws.observed True and readback determined
        (draft §6.3 — the honest composed-margin surface)."""
        st = new_state()
        self.assertIsNone(st.predict(0, "ls -1 /etc > /tmp/w/blind.lst"))
        st.fold(step("ls -1 /etc > /tmp/w/blind.lst"))
        self.assertFalse(st.ws["/tmp/w/blind.lst"]["observed"])
        self.assertIsNone(st.predict(1, "cat /tmp/w/blind.lst"))
        # cat-producer observed capture: content known from the writing command
        st.fold(step("echo 'p_tok' > /tmp/w/src.txt"))
        self.assertEqual(st.predict(2, "cat /tmp/w/src.txt > /tmp/w/obs.txt"), ok(""))
        st.fold(step("cat /tmp/w/src.txt > /tmp/w/obs.txt"))
        self.assertTrue(st.ws["/tmp/w/obs.txt"]["observed"])
        self.assertEqual(st.predict(3, "cat /tmp/w/obs.txt"), ok("p_tok"))
        # ls-producer observed capture: the listing is edit-replay-derivable
        st.fold(step("mkdir /tmp/w/d"))
        st.fold(step("echo 'a_tok' > /tmp/w/d/a.txt"))
        st.fold(step("ls -1 /tmp/w/d > /tmp/w/d.lst"))
        self.assertTrue(st.ws["/tmp/w/d.lst"]["observed"])
        self.assertEqual(st.predict(6, "cat /tmp/w/d.lst"), ok("a.txt"))

    def test_pwd_cd_tracking(self):
        """R1/R2: pwd is always determined; cd tracks entailed targets incl. '..'
        and '/'; dead targets error via the (exit-overriding) template; unknown
        targets and bare cd stay BOT."""
        st = new_state()
        self.assertEqual(st.predict(0, "pwd"), ok("/"))
        self.assertEqual(st.predict(0, "cd /tmp"), ok("", cwd="/tmp"))
        st.fold(step("cd /tmp", cwd="/tmp"))
        self.assertEqual(st.predict(1, "pwd"), ok("/tmp", cwd="/tmp"))
        self.assertEqual(st.predict(1, "cd w"), ok("", cwd="/tmp/w"))   # relative
        st.fold(step("cd w", cwd="/tmp/w"))
        self.assertEqual(st.predict(2, "cd .."), ok("", cwd="/tmp"))
        st.fold(step("cd ..", cwd="/tmp"))
        self.assertEqual(st.predict(3, "cd /"), ok("", cwd="/"))
        st.fold(step("cd /", cwd="/"))
        self.assertEqual(st.predict(4, "cd .."), ok("", cwd="/"))       # .. of / is /
        self.assertIsNone(st.predict(4, "cd"))                          # $HOME: image-dependent
        self.assertIsNone(st.predict(4, "cd /nonexistent"))             # unknown, not entailed
        st.fold(step("mkdir /tmp/w/d"))
        st.fold(step("rm -r /tmp/w/d"))
        self.assertEqual(st.predict(6, "cd /tmp/w/d"),                  # dead: template, exit 2
                         {"output": "/bin/sh: cd: can't cd to /tmp/w/d",
                          "exit": 2, "cwd": "/"})
        st.fold(step("echo 'x_tok' > /tmp/w/f.txt"))
        self.assertIsNone(st.predict(7, "cd /tmp/w/f.txt"))             # a file, not a dir

    def test_workspace_ls_names_only_render(self):
        """R6 edit-replay: names-only ls of a trajectory-created dir is determined
        (sorted, dotfiles only under -a); the -l family is never predicted."""
        st = new_state()
        st.fold(step("mkdir /tmp/w/d"))
        self.assertEqual(st.predict(1, "ls -1 /tmp/w/d"), ok(""))
        st.fold(step("echo 'a_tok' > /tmp/w/d/a.txt"))
        st.fold(step("echo 'b_tok' > /tmp/w/d/b.txt"))
        st.fold(step("echo 'h_tok' > /tmp/w/d/.h.txt"))
        self.assertEqual(st.predict(4, "ls -1 /tmp/w/d"), ok("a.txt\nb.txt"))
        self.assertEqual(st.predict(4, "ls /tmp/w/d"), ok("a.txt\nb.txt"))
        self.assertEqual(st.predict(4, "ls -a /tmp/w/d"),
                         ok(".\n..\n.h.txt\na.txt\nb.txt"))
        self.assertIsNone(st.predict(4, "ls -l /tmp/w/d"))              # -l: ledger-only
        st.fold(step("mv /tmp/w/d/a.txt /tmp/w/d/c.txt"))               # replay tracks edits
        self.assertEqual(st.predict(5, "ls -1 /tmp/w/d"), ok("b.txt\nc.txt"))

    def test_error_templates_per_image(self):
        """R3/R8 speak ONLY through the per-image probe-harvested table: same fold,
        different image dialects -> different renders; missing key -> BOT (never a
        guessed dialect). Absence mining is template-exact and conservative."""
        gnu = {"cat": {"text": "cat: {path}: No such file or directory", "exit": 1}}
        for tmpl, want in (
                (TMPL, "cat: can't open '/tmp/w/g.txt': No such file or directory"),
                (gnu, "cat: /tmp/w/g.txt: No such file or directory")):
            st = M.ShellState(error_templates=tmpl)
            st.fold(step("echo 'x_tok' > /tmp/w/g.txt"))
            st.fold(step("rm /tmp/w/g.txt"))
            self.assertEqual(st.predict(2, "cat /tmp/w/g.txt"),
                             {"output": want, "exit": 1, "cwd": "/"})
        bare = M.ShellState(error_templates={})
        bare.fold(step("echo 'x_tok' > /tmp/w/g.txt"))
        bare.fold(step("rm /tmp/w/g.txt"))
        self.assertIsNone(bare.predict(2, "cat /tmp/w/g.txt"))
        # a template-matching error render teaches absence...
        st = new_state()
        miss = TMPL["cat"]["text"].format(path="/tmp/w/ghost.txt")
        st.fold(step("cat /tmp/w/ghost.txt", miss, 1))
        self.assertEqual(st.predict(1, "cat /tmp/w/ghost.txt"),
                         {"output": miss, "exit": 1, "cwd": "/"})
        # ...but a non-matching error text mines nothing (conservative)
        st2 = new_state()
        st2.fold(step("cat /tmp/w/ghost.txt",
                      "cat: /tmp/w/ghost.txt: Permission denied", 1))
        self.assertIsNone(st2.predict(1, "cat /tmp/w/ghost.txt"))


# ================================================================ job automaton

class TestJobAutomaton(unittest.TestCase):
    """R9: the 5-state machine {waiting, stopped, stopped_pending_term, fired,
    killed} (draft §5.3) with canonical PIDs and canonical ps renders."""

    def test_launch_fire_readback(self):
        """launch -> waiting (ps shows S) -> deterministic fire in the due step's
        prologue -> job-log readback with the known payload."""
        st = new_state()
        launch = "after 1 2 'echo gamma_tok >> /tmp/w/task1.log' & echo $!"
        self.assertEqual(st.predict(0, launch), ok("110"))   # canonical pid 100+10j
        st.fold(step(launch, "110"))
        self.assertEqual(st.jobs[1]["state"], "waiting")
        wait_render = M.render_ps(
            [(1, "S", "init"), (2, "S", "sleep 86400"),
             (110, "S", "after 1 2 echo gamma_tok >> /tmp/w/task1.log")])
        self.assertEqual(st.predict(1, PS), ok(wait_render))
        # pre-fire: the reserved job-log namespace is entailed-absent
        self.assertEqual(
            st.predict(1, "cat /tmp/w/task1.log"),
            {"output": TMPL["cat"]["text"].format(path="/tmp/w/task1.log"),
             "exit": 1, "cwd": "/"})
        st.fold(step(PS, wait_render))
        # due at vt=2: the fire lands in this step's PROLOGUE, before the command
        gone = M.render_ps([(1, "S", "init"), (2, "S", "sleep 86400")])
        self.assertEqual(st.predict(2, PS), ok(gone))        # speculative fire view
        self.assertEqual(st.predict(2, "cat /tmp/w/task1.log"), ok("gamma_tok"))
        st.fold(step("cat /tmp/w/task1.log", "gamma_tok"))
        self.assertEqual(st.jobs[1]["state"], "fired")
        self.assertEqual(st.jobs[1]["deferrals"], 0)
        self.assertEqual(st.delta_text(), "delta: created /tmp/w/task1.log(10B)")
        self.assertEqual(st.predict(3, PS), ok(gone))
        self.assertEqual(st.mismatches, [])

    def test_kill_term_and_never_launched(self):
        """TERM on waiting kills; a killed cpid and a never-launched cpid are both
        'No such process' (a cpid absent from the table is never-launched)."""
        st = new_state()
        st.fold(step("after 1 9 'echo g_tok >> /tmp/w/task1.log' & echo $!", "110"))
        self.assertEqual(st.predict(1, "kill 110"), ok(""))
        st.fold(step("kill 110"))
        self.assertEqual(st.jobs[1]["state"], "killed")
        dead = {"output": "sh: can't kill pid 110: No such process",
                "exit": 1, "cwd": "/"}
        self.assertEqual(st.predict(2, "kill 110"), dead)
        self.assertEqual(st.predict(2, "kill -0 110"), dead)
        self.assertEqual(st.predict(2, "kill 990"),
                         {"output": "sh: can't kill pid 990: No such process",
                          "exit": 1, "cwd": "/"})
        # a killed job never fires: its log path stays entailed-absent forever
        self.assertEqual(st.predict(2, "cat /tmp/w/task1.log")["exit"], 1)

    def test_stop_defers_fire_cont_resumes(self):
        """STOP freezes the countdown (no fire while stopped; ps shows T); CONT
        resumes; the late fire records its deferral count."""
        st = new_state()
        st.fold(step("after 1 2 'echo gamma_tok >> /tmp/w/task1.log' & echo $!", "110"))
        self.assertEqual(st.predict(1, "kill -STOP 110"), ok(""))
        st.fold(step("kill -STOP 110"))
        self.assertEqual(st.jobs[1]["state"], "stopped")
        # vt=2 would be the fire step, but a stopped job cannot fire
        miss = TMPL["cat"]["text"].format(path="/tmp/w/task1.log")
        self.assertEqual(st.predict(2, "cat /tmp/w/task1.log"),
                         {"output": miss, "exit": 1, "cwd": "/"})
        st.fold(step("cat /tmp/w/task1.log", miss, 1))
        stopped_render = M.render_ps(
            [(1, "S", "init"), (2, "S", "sleep 86400"),
             (110, "T", "after 1 2 echo gamma_tok >> /tmp/w/task1.log")])
        self.assertEqual(st.predict(3, PS), ok(stopped_render))
        st.fold(step(PS, stopped_render))
        self.assertEqual(st.predict(4, "kill -CONT 110"), ok(""))
        st.fold(step("kill -CONT 110"))
        self.assertEqual(st.jobs[1]["state"], "waiting")
        self.assertEqual(st.predict(5, "cat /tmp/w/task1.log"), ok("gamma_tok"))
        st.fold(step("cat /tmp/w/task1.log", "gamma_tok"))
        self.assertEqual(st.jobs[1]["state"], "fired")
        self.assertEqual(st.jobs[1]["deferrals"], 3)         # due at 2, fired at 5
        self.assertEqual(st.mismatches, [])

    def test_term_on_stopped_pends_until_cont(self):
        """The §5.3 corner: TERM on a stopped job does NOT kill it — it pends
        (still alive, ps shows T, kill -0 succeeds) and dies exactly at CONT."""
        st = new_state()
        st.fold(step("after 2 9 'echo x_tok >> /tmp/w/task2.log' & echo $!", "120"))
        st.fold(step("kill -STOP 120"))
        self.assertEqual(st.predict(st.vt, "kill 120"), ok(""))
        st.fold(step("kill 120"))
        self.assertEqual(st.jobs[2]["state"], "stopped_pending_term")
        pend_render = M.render_ps(
            [(1, "S", "init"), (2, "S", "sleep 86400"),
             (120, "T", "after 2 9 echo x_tok >> /tmp/w/task2.log")])
        self.assertEqual(st.predict(st.vt, PS), ok(pend_render))
        self.assertEqual(st.predict(st.vt, "kill -0 120"), ok(""))
        st.fold(step("kill -STOP 120"))                      # STOP keeps it pending
        self.assertEqual(st.jobs[2]["state"], "stopped_pending_term")
        self.assertEqual(st.predict(st.vt, "kill -CONT 120"), ok(""))
        st.fold(step("kill -CONT 120"))                      # the deferred TERM lands
        self.assertEqual(st.jobs[2]["state"], "killed")
        self.assertEqual(st.predict(st.vt, "kill -0 120"),
                         {"output": "sh: can't kill pid 120: No such process",
                          "exit": 1, "cwd": "/"})
        self.assertEqual(st.predict(st.vt, "cat /tmp/w/task2.log")["exit"], 1)

    def test_kill9_on_stopped(self):
        """KILL is unconditional: it takes down even a stopped job."""
        st = new_state()
        st.fold(step("after 1 9 'echo g_tok >> /tmp/w/task1.log' & echo $!", "110"))
        st.fold(step("kill -STOP 110"))
        self.assertEqual(st.predict(2, "kill -9 110"), ok(""))
        st.fold(step("kill -9 110"))
        self.assertEqual(st.jobs[1]["state"], "killed")
        gone = M.render_ps([(1, "S", "init"), (2, "S", "sleep 86400")])
        self.assertEqual(st.predict(3, PS), ok(gone))

    def test_kill0_liveness_probe(self):
        """-0 probes without transitioning: exit 0 on a live job, no state change."""
        st = new_state()
        st.fold(step("after 1 9 'echo g_tok >> /tmp/w/task1.log' & echo $!", "110"))
        self.assertEqual(st.predict(1, "kill -0 110"), ok(""))
        st.fold(step("kill -0 110"))
        self.assertEqual(st.jobs[1]["state"], "waiting")


# ================================================================ review-fix regressions

class TestListingEvidenceGuard(unittest.TestCase):
    """S2: `ls` of a kind-UNKNOWN path must never be folded as a directory listing
    when the output carries the ls-of-a-file signature (names containing '/', or
    exactly the target's basename) — no kind upgrade, no fs keys, no entries."""

    def test_ls_of_file_absolute_form(self):
        """The reviewer's exact repro: after a parent listing, `ls -1` of the
        kind-unknown FILE /etc/hosts must not corrupt belief — cd stays BOT and
        the concatenated fs key never exists."""
        st = new_state()
        st.fold(step("ls -1 /etc", "hosts\npasswd"))
        self.assertIsNone(st.fs["/etc/hosts"]["kind"])       # known, kind unknown
        st.fold(step("ls -1 /etc/hosts", "/etc/hosts"))      # a FILE: ls renders the path
        self.assertIsNone(st.predict(2, "cd /etc/hosts"), "cd into a file went determined")
        self.assertNotIn("/etc/hosts//etc/hosts", st.fs)     # corrupted key must not exist
        self.assertIsNone(st.fs["/etc/hosts"]["kind"])       # never upgraded to dir
        self.assertIsNone(st.fs["/etc/hosts"]["entries"])    # no listing knowledge
        self.assertIsNone(st.predict(2, "[ -d /etc/hosts ] && ls -1 /etc/hosts"))
        self.assertIsNone(st.predict(2, "ls -1 /etc/hosts")) # no edit-replay either

    def test_ls_of_file_relative_form(self):
        """The relative form (`ls hosts` from /etc) renders the basename — same
        refusal."""
        st = new_state()
        st.fold(step("ls -1 /etc", "hosts\npasswd"))
        st.fold(step("cd /etc", "", 0, "/etc"))
        st.fold(step("ls hosts", "hosts", 0, "/etc"))
        self.assertIsNone(st.fs["/etc/hosts"]["kind"])
        self.assertIsNone(st.predict(3, "cd /etc/hosts"))

    def test_ls_l_of_file_refused(self):
        """The -l form of the same trap: the row's name field equals the target
        (or carries '/'), so a kind-unknown target folds nothing."""
        st = new_state()
        st.fold(step("ls -1 /etc", "hosts\npasswd"))
        st.fold(step("ls -l /etc/hosts",
                     "-rw-r--r-- 1 root root 158 Jan  1 00:00 /etc/hosts"))
        self.assertIsNone(st.fs["/etc/hosts"]["kind"])
        self.assertNotIn("/etc/hosts//etc/hosts", st.fs)

    def test_genuine_dir_listing_still_folds(self):
        """Clean listing evidence for a kind-unknown target keeps folding (no '/'
        in names, not the basename echo): kind -> dir, entries recorded."""
        st = new_state()
        st.fold(step("ls -1 /etc", "hosts\npasswd"))
        st.fold(step("ls -1 /etc/apk", "keys\nrepositories"))
        self.assertEqual(st.fs["/etc/apk"]["kind"], "dir")
        self.assertEqual(st.fs["/etc/apk"]["entries"], {"keys", "repositories"})


class TestPredictSoundnessFixes(unittest.TestCase):
    """M1/M2/M3: determined-surface soundness fixes from the spec review."""

    def test_rm_of_known_dir_without_r(self):
        """M1: non-recursive rm of a known dir is never determined-ok — the
        harvested rm_isdir template if present, else BOT."""
        st = new_state()                                     # TMPL lacks rm_isdir
        st.fold(step("mkdir /tmp/w/d"))
        self.assertIsNone(st.predict(1, "rm /tmp/w/d"))
        with_isdir = dict(TMPL, rm_isdir={"text": "rm: '{path}': Is a directory",
                                          "exit": 1})
        st2 = M.ShellState(error_templates=with_isdir)
        st2.fold(step("mkdir /tmp/w/d"))
        self.assertEqual(st2.predict(1, "rm /tmp/w/d"),
                         {"output": "rm: '/tmp/w/d': Is a directory",
                          "exit": 1, "cwd": "/"})
        self.assertEqual(st2.predict(1, "rm -r /tmp/w/d"), ok(""))   # -r stays ok
        st2.fold(step("echo 'x_tok' > /tmp/w/f"))
        self.assertEqual(st2.predict(2, "rm /tmp/w/f"), ok(""))      # file rm stays ok

    def test_brace_bearing_templates_never_raise(self):
        """M2: probe-harvested templates are untrusted world text — literal braces
        must yield BOT (predict) / no mining (fold), never a raise."""
        evil = {"cat": {"text": "cat: {path}: No such file {or} directory",  # KeyError
                        "exit": 1},
                "rm": {"text": "rm: {path}: cannot remove {", "exit": 1}}    # ValueError
        st = M.ShellState(error_templates=evil)
        st.fold(step("echo 'x_tok' > /tmp/w/f"))
        st.fold(step("rm /tmp/w/f"))
        self.assertIsNone(st.predict(2, "cat /tmp/w/f"))     # KeyError -> BOT
        self.assertIsNone(st.predict(2, "rm /tmp/w/f"))      # ValueError -> BOT
        st2 = M.ShellState(error_templates=evil)             # fold-side mining path
        st2.fold(step("cat /tmp/w/ghost", "cat: /tmp/w/ghost: No such file", 1))
        self.assertIsNone(st2.predict(1, "cat /tmp/w/ghost"))

    def test_append_onto_unknown_existence_sets_blind(self):
        """M3: '>>' onto a workspace file of UNKNOWN prior existence is a blind-ish
        capture — ws.observed False (feeds the created/created-obs split), and the
        readback stays BOT."""
        st = new_state()
        self.assertEqual(st.predict(0, "echo 'x_tok' >> /tmp/w/u.txt"), ok(""))
        st.fold(step("echo 'x_tok' >> /tmp/w/u.txt"))
        self.assertFalse(st.ws["/tmp/w/u.txt"]["observed"])
        self.assertIsNone(st.predict(1, "cat /tmp/w/u.txt"))
        # contrast: '>' truncate-create of the same unknown path IS observed
        st2 = new_state()
        st2.fold(step("echo 'y_tok' > /tmp/w/u.txt"))
        self.assertTrue(st2.ws["/tmp/w/u.txt"]["observed"])


class TestMvIntoExistingDir(unittest.TestCase):
    """Round-2 F1: POSIX `mv SRC DST` with DST an existing directory moves SRC
    INTO it (DST/basename(SRC)); the dir node is never overwritten. Both repro
    trajectories from the round-2 review, asserting the REAL shell sides."""

    def test_mv_file_into_created_subdir(self):
        """Reviewer repro 1: mkdir d; create f.txt; mv f.txt d — then cat/ls of d
        must behave like the real shell (d stays a directory holding f.txt)."""
        st = new_state()
        st.fold(step("mkdir /tmp/w/d"))
        st.fold(step("echo 'body_tok' > /tmp/w/f.txt"))
        # dst is a known dir with a known-absent child slot -> determined ok
        self.assertEqual(st.predict(2, "mv /tmp/w/f.txt /tmp/w/d"), ok(""))
        st.fold(step("mv /tmp/w/f.txt /tmp/w/d"))
        self.assertEqual(st.delta_text(),
                         "delta: moved /tmp/w/f.txt -> /tmp/w/d/f.txt")
        self.assertEqual(st.fs["/tmp/w/d"]["kind"], "dir")   # dir node untouched
        # real shell: `cat /tmp/w/d` errors Is-a-directory -> never determined-ok
        self.assertIsNone(st.predict(3, "cat /tmp/w/d"))
        # real shell: the listing shows the moved-in name
        self.assertEqual(st.predict(3, "ls -1 /tmp/w/d"), ok("f.txt"))
        # the content rode along to the child path; the old path is dead
        self.assertEqual(st.predict(3, "cat /tmp/w/d/f.txt"), ok("body_tok"))
        self.assertEqual(
            st.predict(3, "cat /tmp/w/f.txt"),
            {"output": "cat: can't open '/tmp/w/f.txt': No such file or directory",
             "exit": 1, "cwd": "/"})

    def test_mv_unobserved_file_into_workspace_root(self):
        """Reviewer repro 2 (the sanctioned Tier-S op `mv <f> /tmp/w/`): the
        workspace dir node survives; the name lands as a child; the rest of the
        trajectory keeps workspace coverage."""
        st = new_state()
        self.assertIsNone(st.predict(0, "mv /etc/motd /tmp/w/"))   # src unobserved
        st.fold(step("mv /etc/motd /tmp/w/"))
        self.assertEqual(st.fs["/tmp/w"]["kind"], "dir")     # root not poisoned
        self.assertFalse(st.fs["/tmp/w"]["deleted"])
        self.assertIn("/tmp/w/motd", st.fs)
        self.assertTrue(st.fs["/etc/motd"]["deleted"])
        self.assertEqual(st.predict(1, "cd /tmp/w"), ok("", cwd="/tmp/w"))
        self.assertEqual(st.predict(1, "touch /tmp/w/t"), ok(""))
        self.assertIsNone(st.predict(1, "[ -d /tmp/w ] && ls -1 /tmp/w"))
        # (entries honestly BOT: an unobserved name just moved in)
        self.assertIsNone(st.predict(1, "cat /tmp/w/motd"))  # content unobserved

    def test_mv_onto_kind_unknown_dst_is_conservative(self):
        """dst of unknown kind/existence: BOT predict; fold tombstones src,
        transports nothing, and shadows dst (reads there BOT) — dst could be an
        unobserved directory, so neither landing site may be claimed."""
        st = new_state()
        st.fold(step("echo 'body_tok' > /tmp/w/a.txt"))
        self.assertIsNone(st.predict(1, "mv /tmp/w/a.txt /tmp/w/b.txt"))
        st.fold(step("mv /tmp/w/a.txt /tmp/w/b.txt"))
        self.assertIsNone(st.predict(2, "cat /tmp/w/b.txt"))         # shadowed
        self.assertIsNone(st.predict(2, "[ -f /tmp/w/b.txt ] && cat /tmp/w/b.txt"))
        self.assertEqual(st.predict(2, "cat /tmp/w/a.txt")["exit"], 1)  # tombstone
        self.assertEqual(st.fs["/tmp/w"]["kind"], "dir")


class TestRedirDirAndTrailingSlash(unittest.TestCase):
    """Round-2 F3: redirect onto a known dir and reads of a known FILE via a
    trailing-slash path are never determined-ok."""

    def test_redirect_onto_known_dir_predict(self):
        """`echo 'x' > <known dir>`: harvested redirect_isdir template if present,
        else BOT — never the determined mutation-ack."""
        st = new_state()                                     # TMPL lacks the key
        st.fold(step("mkdir /tmp/w/d"))
        self.assertIsNone(st.predict(1, "echo 'x_tok' > /tmp/w/d"))
        with_t = dict(TMPL, redirect_isdir={
            "text": "sh: can't create {path}: Is a directory", "exit": 1})
        st2 = M.ShellState(error_templates=with_t)
        st2.fold(step("mkdir /tmp/w/d"))
        self.assertEqual(st2.predict(1, "echo 'x_tok' > /tmp/w/d"),
                         {"output": "sh: can't create /tmp/w/d: Is a directory",
                          "exit": 1, "cwd": "/"})
        # a plain file dst keeps the determined surface
        st2.fold(step("echo 'y_tok' > /tmp/w/f"))
        self.assertEqual(st2.predict(2, "echo 'z_tok' > /tmp/w/f"), ok(""))

    def test_redirect_onto_known_dir_fold_records_failure_honestly(self):
        """The failed open writes nothing: dir node intact, no ws entry, no delta."""
        st = new_state()
        st.fold(step("mkdir /tmp/w/d"))
        st.fold(step("echo 'x_tok' > /tmp/w/d",
                     "sh: can't create /tmp/w/d: Is a directory", 1))
        self.assertEqual(st.delta_text(), "delta: none")
        self.assertEqual(st.fs["/tmp/w/d"]["kind"], "dir")
        self.assertEqual(st.fs["/tmp/w/d"]["entries"], set())
        self.assertNotIn("/tmp/w/d", st.ws)
        self.assertEqual(st.predict(2, "ls -1 /tmp/w/d"), ok(""))  # still edit-replays

    def test_trailing_slash_reads_of_known_file(self):
        """`cat f.txt/` / `ls -1 f.txt/` on a known FILE: real shell ENOTDIR —
        normpath must not silently answer with the file's content/path echo."""
        st = new_state()
        st.fold(step("echo 'body_tok' > /tmp/w/f.txt"))
        self.assertEqual(st.predict(1, "cat /tmp/w/f.txt"), ok("body_tok"))
        self.assertIsNone(st.predict(1, "cat /tmp/w/f.txt/"))
        self.assertIsNone(st.predict(1, "ls -1 /tmp/w/f.txt/"))
        self.assertIsNone(st.predict(1, "head -n 1 /tmp/w/f.txt/"))
        self.assertIsNone(st.predict(1, "[ -e /tmp/w/f.txt/ ] && cat /tmp/w/f.txt/"))
        self.assertIsNone(st.predict(1, "touch /tmp/w/f.txt/"))
        self.assertIsNone(st.predict(1, "rm /tmp/w/f.txt/"))

    def test_trailing_slash_fold_evidence_stays_conservative(self):
        """A slash-form miss must not tombstone the (existing) file, and a
        slash-form render must not be attached to the stripped path."""
        st = new_state()
        st.fold(step("echo 'body_tok' > /tmp/w/f.txt"))
        st.fold(step("[ -e /tmp/w/f.txt/ ] && cat /tmp/w/f.txt/", "", 1))
        self.assertFalse(st.fs["/tmp/w/f.txt"]["deleted"])
        self.assertEqual(st.predict(2, "cat /tmp/w/f.txt"), ok("body_tok"))
        st.fold(step("cat /tmp/w/f.txt/",
                     "cat: read error: Not a directory", 1))
        self.assertFalse(st.fs["/tmp/w/f.txt"]["deleted"])
        self.assertEqual(st.predict(3, "cat /tmp/w/f.txt"), ok("body_tok"))

    def test_trailing_slash_on_known_dir_keeps_coverage(self):
        """`ls -1 d/` of a known dir is real-shell-identical to `ls -1 d` — the
        determined listing survives the F3 guards."""
        st = new_state()
        st.fold(step("mkdir /tmp/w/d"))
        st.fold(step("echo 'a_tok' > /tmp/w/d/a.txt"))
        self.assertEqual(st.predict(2, "ls -1 /tmp/w/d/"), ok("a.txt"))
        self.assertEqual(st.predict(2, "rm -r /tmp/w/d/"), ok(""))


class TestFindReplay(unittest.TestCase):
    """M5(a) — R7: find replay requires the identical command, an uncapped prior
    render, and a subtree untouched since (incl. the speculative-fire guard)."""

    CMD = "find /etc -maxdepth 2 -name '*.conf'"

    def test_replay_identical_command(self):
        st = new_state()
        out = "/etc/a.conf\n/etc/d/b.conf"
        self.assertIsNone(st.predict(0, self.CMD))           # no prior render
        st.fold(step(self.CMD, out))
        self.assertEqual(st.predict(1, self.CMD), ok(out))   # identical replay
        # any command drift (glob, depth, type) is a different key -> BOT
        self.assertIsNone(st.predict(1, "find /etc -maxdepth 2 -name '*.cfg'"))
        self.assertIsNone(st.predict(1, "find /etc -maxdepth 1 -name '*.conf'"))
        self.assertIsNone(st.predict(1, "find /etc -maxdepth 2 -type f -name '*.conf'"))

    def test_staleness_precondition(self):
        """A mutation UNDER the find dir invalidates the cache; an unrelated
        mutation elsewhere does not."""
        st = new_state()
        st.fold(step(self.CMD, "/etc/a.conf"))
        st.fold(step("echo 'x_tok' > /tmp/w/f"))             # unrelated touch
        self.assertEqual(st.predict(2, self.CMD), ok("/etc/a.conf"))
        st.fold(step("rm /etc/a.conf"))                      # touches under /etc
        self.assertIsNone(st.predict(3, self.CMD))

    def test_spec_fire_guard(self):
        """A job fire due in THIS step's prologue writes under the find dir -> the
        replay is refused (the cached render predates the fire)."""
        st = new_state()
        cmd = "find /tmp/w -maxdepth 1 -name 'task*'"
        st.fold(step(cmd, ""))
        st.fold(step("after 1 1 'echo x_tok >> /tmp/w/task1.log' & echo $!", "110"))
        self.assertIsNone(st.predict(2, cmd))                # due now: guard fires
        # the SAME step with no fire due would have replayed
        st2 = new_state()
        st2.fold(step(cmd, ""))
        self.assertEqual(st2.predict(1, cmd), ok(""))

    def test_capped_render_never_cached(self):
        """sst mode: an over-cap find render is capped evidence — never cached,
        so the replay stays BOT (uncapped-prior precondition)."""
        big = "\n".join(f"/etc/conf{i:04d}.conf" for i in range(200))
        self.assertGreater(len(big), M.OBS_CAP)
        sst = new_state("sst")
        sst.fold(step("find /etc -maxdepth 1 -name '*.conf'", big))
        self.assertIsNone(sst.predict(1, "find /etc -maxdepth 1 -name '*.conf'"))


class TestStateScopeOf(unittest.TestCase):
    """M5(b): state_scope_of — the collection-mode meta labeler feeding cell scope."""

    def test_created_mutated_native(self):
        st = new_state()
        st.fold(step("echo 'x_tok' > /tmp/w/f.txt"))         # redirect@ -> created
        st.fold(step("mkdir /tmp/w/d"))                      # mut:mkdir -> created
        st.fold(step("touch /tmp/w/t"))                      # mut:touch -> created
        for p in ("/tmp/w/f.txt", "/tmp/w/d", "/tmp/w/t"):
            self.assertEqual(st.state_scope_of(p), "created", p)
        st.fold(step("mv /tmp/w/f.txt /tmp/w/g.txt"))
        self.assertEqual(st.state_scope_of("/tmp/w/g.txt"), "mutated")   # mut:mv
        self.assertEqual(st.state_scope_of("/tmp/w/f.txt"), "mutated")   # tombstone
        st.fold(step("rm /etc/foo.conf"))                    # Tier-S mutation
        self.assertEqual(st.state_scope_of("/etc/foo.conf"), "mutated")
        self.assertEqual(st.state_scope_of("/tmp/w"), "mutated")  # touched parent
        self.assertEqual(st.state_scope_of("/etc/hosts"), "native")  # untracked
        self.assertEqual(st.state_scope_of("/usr/lib"), "native")

    def test_relative_and_symlink_resolution(self):
        st = new_state()
        st.fold(step("echo 'x_tok' > /tmp/w/f.txt"))
        st.fold(step("cd /tmp/w", "", 0, "/tmp/w"))
        self.assertEqual(st.state_scope_of("f.txt"), "created")      # cwd-relative
        st.fold(step("ln -s /tmp/w/f.txt /tmp/w/l", "", 0, "/tmp/w"))
        self.assertEqual(st.state_scope_of("l"), "created")          # resolves to f.txt


class TestDeltaTruncation(unittest.TestCase):
    """M5(c): the canonical delta line caps at 8 entries + '(+N more)' (prereg §4.2)."""

    def test_over_eight_entries(self):
        st = new_state()
        st.fold(step("mkdir /tmp/w/d"))
        for i in range(10):
            st.fold(step(f"echo 'x_tok' > /tmp/w/d/f{i}.txt"))
        st.fold(step("rm -r /tmp/w/d"))                      # 11 removals
        txt = st.delta_text()
        self.assertEqual(txt.count("removed "), 8)
        self.assertTrue(txt.endswith(" (+3 more)"), txt)
        # exactly eight entries: no suffix
        st2 = new_state()
        st2.fold(step("mkdir /tmp/w/e"))
        for i in range(7):
            st2.fold(step(f"echo 'x_tok' > /tmp/w/e/f{i}.txt"))
        st2.fold(step("rm -r /tmp/w/e"))                     # 8 removals
        self.assertNotIn("more)", st2.delta_text())


class TestMismatchAudit(unittest.TestCase):
    """M5(d): the collection-mode audit trail actually records mismatches (the
    other suites only ever assert mismatches == [])."""

    def test_positive_mismatch_records(self):
        st = new_state()
        st.fold(step("after 1 5 'echo x_tok >> /tmp/w/task1.log' & echo $!", "110"))
        self.assertEqual(st.mismatches, [])
        # echoed-pid mismatch: canonical pid of job 2 is 120, record says 999
        st.fold(step("after 2 5 'echo y_tok >> /tmp/w/task2.log' & echo $!", "999"))
        self.assertEqual(len(st.mismatches), 1)
        self.assertEqual(st.mismatches[0][1], "after")
        self.assertIn("echoed pid", st.mismatches[0][2])
        # job-index reuse
        st.fold(step("after 1 4 'echo z_tok >> /tmp/w/task1.log' & echo $!", "110"))
        self.assertEqual(len(st.mismatches), 2)
        self.assertIn("reused", st.mismatches[1][2])
        # kill exit-mismatch: alive cpid recorded with nonzero exit
        st.fold(step("kill 110", "", 1))
        self.assertEqual(len(st.mismatches), 3)
        self.assertIn("alive but exit 1", st.mismatches[2][2])
        # kill exit-mismatch: dead/never-launched cpid recorded with exit 0
        st.fold(step("kill 990", "", 0))
        self.assertEqual(len(st.mismatches), 4)
        self.assertIn("dead but exit 0", st.mismatches[3][2])


class TestUniverseAgreement(unittest.TestCase):
    """M5(e)/S1: the BIDIRECTIONAL totality property — parse_command accepts a
    command IFF verbsig.sig accepts it, over the enumerated universe, the
    enumerated exclusions, and a mixed fuzz battery (one universe, one gate)."""

    FUZZ = [
        # in-universe variants
        "uname", "uname -a", "uname -sm", "uname -sr", "ls", "ls -lt /usr/lib",
        "ls -lS /etc", "ls -i /etc", "ls -R /usr", "ls -1a /tmp/w", "ls -a1 /tmp/w",
        "cd", "cd ..", "cd /", "sleep 0", "sleep 1", "kill 0", "kill -9 110",
        "head -n 20 /etc/passwd", "tail -n 12 /var/log/dpkg.log",
        "grep -F -i -m 8 tok /etc/hosts", "grep -F -m 8 'a;b' /etc/hosts",
        "find /usr -maxdepth 3 -type d -name 'lib*'",
        "echo 'x_tok' >> /tmp/w/f", "cat /tmp/w/f >> /tmp/w/g",
        "ls -1 /tmp/w > /tmp/w/self.lst",
        "[ -s /tmp/w/f ] && head -n 2 /tmp/w/f",
        "after 3 8 'echo m_tok >> /tmp/w/task3.log' & echo $!",
        "/usr/local/bin/tj3-ps", "/usr/local/bin/tj3-ps -o pid,stat,args",
        "echo bare_tok", "echo 'two words'",
        # out-of-universe near-misses
        "sleep 2", "sleep 5", "ps", "ps -ef", "ps aux", "/usr/local/bin/tj3-ps -ef",
        "uname -p", "uname -a -s", "ls -Z /etc", "ls -lart /etc", "ls -l -a /etc",
        "echo x_tok > /tmp/w/f", "echo 'a' b > /tmp/w/f", "echo 'x' > /tmp/w",
        "echo '' > /tmp/w/f", "echo 'x' > '/tmp/w/f'",
        "after 1 5 'rm /tmp/w/x' & echo $!",
        "after 1 5 'echo x_tok > /tmp/w/task1.log' & echo $!",
        "after 1 5 'echo x_tok >> /tmp/w/task2.log' & echo $!",
        "after 1 5 'echo x_tok >> /tmp/w/task1.log'",
        "kill", "kill -HUP 110", "kill -TERM abc", "kill 110 120",
        "grep -F -m 3 tok /etc/hosts", "grep -m 8 tok /etc/hosts",
        "stat -c '%y' /etc/hosts", "stat /etc/hosts",
        "head -n x /etc/hosts", "tail -n 5", "cat", "cat a b",
        "find /etc -name '*.conf'", "touch /tmp/w/a /tmp/w/b", "mv /tmp/w/a",
        "ln -s /tmp/w/a", "readlink", "mkdir", "pwd extra", "uptime -p",
        "[ -f /tmp/w/f ] && tail -n 2 /tmp/w/f", "[ -f /tmp/w/a ]",
        "cd /etc;pwd", "cat /etc/hosts | grep -F -m 8 x | head -n 1",
        "echo $HOME", "echo `id`", "echo a\\b", "cat < /tmp/w/f",
        "cat <<< 'x'", "true || false", "sleep 1 &", "wait", "jobs",
    ]

    @staticmethod
    def _sq(text):
        """The collector's single-quote escape: ' -> '\\'' (mirrors _sq/_lex)."""
        return "'" + text.replace("'", "'\\''") + "'"

    @classmethod
    def _quote_fuzz(cls):
        """Round-2 F2 generators: quoted-space and embedded-quote ('\\'') tokens
        in every family slot — the 14-split shapes the old sig lexer rejected."""
        cmds = []
        qtok = cls._sq("a'b")
        qtok2 = cls._sq("tok'n roll")
        for path in ("/tmp/w/a b", "/tmp/w/it's.txt", "/tmp/my dir/f g",
                     "/etc/o'clock d"):
            q = cls._sq(path)
            cmds += [
                f"cat {q}",
                f"ls -1 {q}",
                f"head -n 2 {q}",
                f"stat -c '%n %s %F %a' {q}",
                f"cat {q} | head -n 2",
                f"cat {q} | grep -F -m 8 'two words'",
                f"ls -1 {q} | tail -n 3",
                f"ls -1 {q} | grep -F -m 8 {qtok}",
                f"cat {q} > /tmp/w/f",
                f"cat {q} >> /tmp/w/f",
                f"[ -e {q} ] && cat {q}",
                f"[ -f {q} ] && head -n 2 {q}",
                f"[ -d {q} ] && ls -1 {q}",
                f"grep -F -m 8 {qtok2} {q}",
            ]
        cmds += [
            "echo " + cls._sq("pay'load x") + " > /tmp/w/f",
            "echo " + cls._sq("two words") + " >> /tmp/w/f",
            "find /etc -maxdepth 2 -name " + cls._sq("*'*"),
            "find /usr -maxdepth 1 -type f -name " + cls._sq("a b*"),
            "echo 'x' > '/tmp/w/a b'",       # quoted dst: both gates must reject
            "cat 'a b' 'c d'",               # two quoted args: both must reject
        ]
        return cmds

    def _accepts(self, fn, err, cmd):
        try:
            fn(cmd)
            return True
        except err:
            return False

    def test_parse_iff_sig(self):
        cmds = ([c for c, _ in TestParserTotality.UNIVERSE]
                + TestParserTotality.BADS + self.FUZZ + self._quote_fuzz())
        for cmd in cmds:
            p_ok = self._accepts(M.parse_command, M.ParseError, cmd)
            s_ok = self._accepts(V.sig, ValueError, cmd)
            self.assertEqual(
                p_ok, s_ok,
                f"totality split on {cmd!r}: parse_command={p_ok} sig={s_ok}")

    def test_quoted_tokens_accepted_and_labeled(self):
        """The round-2 review's 14 splits: parse_command admits these, so sig()
        must label them (one lexer, one totality)."""
        for cmd, want in [
                ("cat '/tmp/w/it'\\''s.txt'", "cat"),
                ("cat '/tmp/w/a b'", "cat"),
                ("cat '/tmp/w/a b' | head -n 2", "pipe:cat|head"),
                ("ls -1 '/tmp/my dir' | head -n 3", "pipe:ls|head"),
                ("ls -1 /etc | grep -F -m 8 'a'\\''b'", "pipe:ls|grep"),
                ("cat '/tmp/w/a b' > /tmp/w/f", "redir:prod>"),
                ("echo 'a'\\''b' > /tmp/w/f", "redir:echo>"),
                ("[ -e '/tmp/w/a b' ] && cat '/tmp/w/a b'", "cond:cat"),
                ("[ -f '/tmp/w/a b' ] && head -n 2 '/tmp/w/a b'", "cond:head"),
                ("grep -F -m 8 'two words' /etc/hosts", "grep"),
                ("find /etc -maxdepth 2 -name '*'\\''*'", "find"),
                ("stat -c '%n %s %F %a' '/tmp/w/a'\\''b'", "stat"),
        ]:
            self.assertEqual(V.sig(cmd), want, cmd)
            M.parse_command(cmd)   # and the one authority admits it


# ================================================================ parity + determinism

class TestVisibilityAndDeterminism(unittest.TestCase):
    def test_modes_agree_on_identical_inputs(self):
        """DG-4c: the collection-mode and render-parity (sst) folds share transition
        code — on identical sub-cap inputs, predictions AND end state are identical."""
        rows = _traj()
        results = {}
        for mode in ("collection", "sst"):
            st = new_state(mode=mode)
            preds, deltas = _run(st, rows, check=self)
            self.assertEqual(st.mismatches, [], f"[{mode}] audit mismatches")
            results[mode] = (preds, deltas, snapshot(st))
        c, s = results["collection"], results["sst"]
        self.assertEqual(c[0], s[0], "mode-divergent predictions")
        self.assertEqual(c[1], s[1], "mode-divergent delta channel")
        self.assertEqual(c[2], s[2], "mode-divergent end state")

    def test_two_folds_identical(self):
        """Determinism: two folds of the same trajectory are identical — predictions,
        delta texts, and the full end state (no wall-clock, no randomness)."""
        rows = _traj()
        a, b = new_state(), new_state()
        preds_a, deltas_a = _run(a, rows)
        preds_b, deltas_b = _run(b, rows)
        self.assertEqual(preds_a, preds_b)
        self.assertEqual(deltas_a, deltas_b)
        self.assertEqual(snapshot(a), snapshot(b))

    def test_sst_mode_never_mines_past_render_cap(self):
        """The one sanctioned mode difference (draft §10.1): sst-mode evidence is
        truncated to the OBS_CAP render window before mining, so an over-cap render
        yields no belief (readback BOT) where collection mode keeps the full text."""
        big = "\n".join(f"line_{i:04d}" for i in range(300))   # 2999 chars > OBS_CAP
        self.assertGreater(len(big), M.OBS_CAP)
        coll, sst = new_state("collection"), new_state("sst")
        coll.fold(step("cat /etc/big.conf", big))
        sst.fold(step("cat /etc/big.conf", big))
        self.assertEqual(coll.predict(1, "cat /etc/big.conf"), ok(big))
        self.assertIsNone(sst.predict(1, "cat /etc/big.conf"))
        # an already-rendered record (truncation marker) is likewise capped
        sst2 = new_state("sst")
        sst2.fold(step("cat /etc/big.conf", big[:M.OBS_CAP] + "\n...[1399 more chars]"))
        self.assertIsNone(sst2.predict(1, "cat /etc/big.conf"))


# ================================================================ verbsig

class TestVerbSig(unittest.TestCase):
    """The ONE sig/mode/cell labeler (prereg §4.1 vocabulary, draft §9.1–§9.5)."""

    ATOMIC_EXAMPLES = {
        "uname": "uname -s",
        "cd": "cd /tmp/w",
        "ls": "ls -la /etc",
        "cat": "cat /etc/hosts",
        "head": "head -n 3 /etc/hosts",
        "tail": "tail -n 5 /etc/hosts",
        "stat": "stat -c '%n %s %F %a' /etc/hosts",
        "find": "find /etc -maxdepth 2 -name '*.conf'",
        "grep": "grep -F -m 8 localhost /etc/hosts",
        "pwd": "pwd",
        "echo": "echo 'hello_tok'",
        "rm": "rm -r /tmp/w/d",
        "mv": "mv /tmp/w/a /tmp/w/b",
        "ln": "ln -s /tmp/w/a /tmp/w/l",
        "readlink": "readlink /tmp/w/l",
        "mkdir": "mkdir /tmp/w/d",
        "touch": "touch /tmp/w/f",
        "ps": PS,                        # UD-9 Route B: vendored path -> "ps"
        "kill": "kill -STOP 110",
        "after": "after 1 5 'echo g_tok >> /tmp/w/task1.log' & echo $!",
        "uptime": "uptime",
        "sleep": "sleep 1",
    }

    COMPOSED_EXAMPLES = {
        "pipe:ls|head": "ls -1 /etc | head -n 3",
        "pipe:ls|tail": "ls -1 /etc | tail -n 2",
        "pipe:ls|grep": "ls -1 /etc | grep -F -m 8 conf",
        "pipe:cat|head": "cat /etc/hosts | head -n 1",
        "pipe:cat|tail": "cat /etc/hosts | tail -n 1",
        "pipe:cat|grep": "cat /etc/hosts | grep -F -m 8 host",
        "redir:echo>": "echo 'x_tok' >> /tmp/w/f.txt",
        "redir:prod>": "ls -1 /etc > /tmp/w/f.txt",
        "cond:cat": "[ -e /tmp/w/f ] && cat /tmp/w/f",
        "cond:ls": "[ -d /tmp/w ] && ls -1 /tmp/w",
        "cond:head": "[ -s /tmp/w/f ] && head -n 4 /tmp/w/f",
    }

    def test_every_atomic_verb_labels(self):
        """All 22 atomic verbs label as their first-token verb (v1/v2 verb_of
        continuity; the vendored tj3-ps path normalizes to 'ps')."""
        self.assertEqual(set(self.ATOMIC_EXAMPLES), set(V.ATOMIC_VERBS))
        for verb, cmd in self.ATOMIC_EXAMPLES.items():
            self.assertEqual(V.sig(cmd), verb, cmd)
        self.assertIs(V.composed_verb, V.sig)   # the draft §6.2 name, same labeler

    def test_all_composed_families_label(self):
        """All 11 composed families (6 pipe + 2 redir + 3 cond) label correctly."""
        self.assertEqual(set(self.COMPOSED_EXAMPLES), set(V.COMPOSED_SIGS))
        for fam, cmd in self.COMPOSED_EXAMPLES.items():
            self.assertEqual(V.sig(cmd), fam, cmd)
        # both operators and both producers collapse into the two redir families
        self.assertEqual(V.sig("echo 'x_tok' > /tmp/w/f.txt"), "redir:echo>")
        self.assertEqual(V.sig("cat /etc/hosts >> /tmp/w/f.txt"), "redir:prod>")

    def test_sig_vocabulary_complete(self):
        """SIGS is exactly the prereg §4.1 vocabulary: 22 atomic + 11 composed."""
        self.assertEqual(len(V.SIGS), 33)
        self.assertEqual(set(V.SIGS), set(V.ATOMIC_VERBS) | set(V.COMPOSED_SIGS))

    def test_out_of_universe_raises(self):
        """sig() is fail-closed (ValueError) on every documented exclusion."""
        bads = [
            "", "pwd\npwd",                                # empty / embedded newline
            "kill -INT 110",                               # UD-1
            "kill 110 120",
            "jobs", "wait", "fg 1", "if true",
            "cat /a | head -n 2 | tail -n 1",              # depth-2 pipe
            "true || false", "ls /etc; pwd",               # || and ;
            "cat <<< 'x'", "cat < /tmp/w/f",               # <<< / REDIR_IN
            "echo 'x' > /etc/evil",                        # non-workspace target
            "ls -l /etc | head -n 2",                      # ls -l producer
            "find /etc -maxdepth 1 -name x | head -n 2",   # find producer
            "/usr/local/bin/tj3-ps | head -n 2",           # ps in pipes
            "[ -e /a ] && cd /a",                          # cd in a composed string
            "[ -x /a ] && cat /a",                         # bad TESTOP
            "[ -e /a ] && cat /b",                         # path mismatch
            "echo $HOME",                                  # $ outside the after form
            "echo `id`", "echo a\\b",                      # backtick / backslash
            "after 1 5 echo x_tok & echo $!",              # non-canonical after
            "cat 'unterminated",
        ]
        for bad in bads:
            with self.assertRaises(ValueError, msg=f"sig accepted {bad!r}"):
                V.sig(bad)

    def test_mode_rules(self):
        """The frozen (exit, output-emptiness) rules: read hit<=>exit0+nonempty;
        state ok<=>exit0+EMPTY (the mutation-ack shape); after ok<=>exit0+nonempty
        (echoed cpid); pipe head/tail + redir constant ok."""
        # read verbs (+ pipe-grep + cond): grep hit / miss both ways
        self.assertEqual(V.mode("grep", 0, False), "hit")
        self.assertEqual(V.mode("grep", 1, True), "miss")
        self.assertEqual(V.mode("cat", 0, True), "miss")     # empty read = miss
        self.assertEqual(V.mode("cat", 1, False), "miss")    # read-verb miss on exit!=0
        self.assertEqual(V.mode("pipe:cat|grep", 0, False), "hit")
        self.assertEqual(V.mode("pipe:ls|grep", 1, True), "miss")
        self.assertEqual(V.mode("cond:head", 0, False), "hit")
        self.assertEqual(V.mode("cond:cat", 1, True), "miss")
        # state verbs: mutation ack = empty exit-0; anything else is miss
        for v in ("rm", "mv", "ln", "mkdir", "touch", "cd", "kill", "sleep"):
            self.assertEqual(V.mode(v, 0, True), "ok", v)
            self.assertEqual(V.mode(v, 1, False), "miss", v)
        self.assertEqual(V.mode("mv", 0, False), "miss")     # unexpected output
        # launch: the recorded step echoes the cpid, so ok is NON-empty
        self.assertEqual(V.mode("after", 0, False), "ok")
        self.assertEqual(V.mode("after", 0, True), "miss")
        # constant-ok families never re-label anomalies
        self.assertEqual(V.mode("pipe:ls|head", 1, False), "ok")
        self.assertEqual(V.mode("redir:echo>", 0, True), "ok")
        with self.assertRaises(ValueError):
            V.mode("nonesuch", 0, True)

    def test_mode_sets_frozen(self):
        """MODES matches the prereg §4.1 per-family sets verbatim: pipe-grep {hit,
        miss}; cond {hit, miss}; other composed {ok}; read verbs {hit, miss};
        state + launch {ok, miss}."""
        reads = {"uname", "ls", "cat", "head", "tail", "stat", "find", "grep",
                 "pwd", "readlink", "uptime", "ps", "echo",
                 "pipe:ls|grep", "pipe:cat|grep", "cond:cat", "cond:ls", "cond:head"}
        const = {"pipe:ls|head", "pipe:ls|tail", "pipe:cat|head", "pipe:cat|tail",
                 "redir:echo>", "redir:prod>"}
        want = {s: (("ok",) if s in const else
                    ("hit", "miss") if s in reads else ("ok", "miss"))
                for s in V.SIGS}
        self.assertEqual(V.MODES, want)

    def test_cell_keys_and_created_split(self):
        """The cell pseudo-verb 'sig|mode|scope'; created scope splits on
        ws_observed into created / created-obs (round-6 B1); ws_observed is
        REQUIRED there and ignored for native/mutated."""
        self.assertEqual(V.cell("cat", "hit", "native"), "cat|hit|native")
        self.assertEqual(V.cell("grep", "miss", "native"), "grep|miss|native")
        self.assertEqual(V.cell("rm", "ok", "mutated"), "rm|ok|mutated")
        self.assertEqual(V.cell("cat", "hit", "created", ws_observed=False),
                         "cat|hit|created")
        self.assertEqual(V.cell("cat", "hit", "created", ws_observed=True),
                         "cat|hit|created-obs")
        with self.assertRaises(ValueError):
            V.cell("cat", "hit", "created")                  # ws_observed required
        self.assertEqual(V.cell("cat", "hit", "native", ws_observed=True),
                         "cat|hit|native")                   # not split: ignored
        self.assertEqual(V.cell("rm", "ok", "mutated", ws_observed=False),
                         "rm|ok|mutated")

    def test_cell_atomicity_and_rejects(self):
        """Cell keys are ATOMIC opaque strings: sigs contain '|', so splitting
        misparses — only equality against a re-derived key is safe (round-7 M2)."""
        key = V.cell("pipe:ls|head", "ok", "created", ws_observed=True)
        self.assertEqual(key, "pipe:ls|head|ok|created-obs")
        self.assertNotEqual(key.split("|")[0], "pipe:ls|head")   # naive split misparses
        self.assertEqual(V.cell("pipe:ls|head", "ok", "created", ws_observed=True),
                         key)                                    # opaque-key round-trip
        for bad in (("nonesuch", "ok", "native"),
                    ("pipe:ls|head", "hit", "native"),   # mode outside the family set
                    ("cat", "ok", "native"),             # read verb has no 'ok'
                    ("cat", "hit", "elsewhere")):        # unknown scope
            with self.assertRaises(ValueError, msg=repr(bad)):
                V.cell(*bad)


# ================================================================ render_canon

def _touched_ws_state():
    """A collection-mode state whose tracker touched /tmp/w, /tmp/w/d, and the
    notes.txt file (mkdir + echo-redirect)."""
    st = M.ShellState(mode="collection")
    st.fold(step("mkdir /tmp/w/d"))
    st.fold(step("echo 'alpha_tok' > /tmp/w/d/notes.txt"))
    return st


class TestRenderCanon(unittest.TestCase):
    """Exactly the three render-canon rows of the prereg §3.2 table (draft §5.5),
    conservative and total; everything else passes through byte-for-byte."""

    def test_uptime_row_masks(self):
        """Row 2 (uptime): wall clock / users / load -> fixed tokens; the
        collector-canonical render is a fixed point (same object back)."""
        st = _touched_ws_state()
        got = RC.canon(step(
            "uptime",
            " 12:34:56 up 3 days, 42 min,  2 users,  load average: 0.15, 0.10, 0.05"),
            st)
        self.assertEqual(
            got["output"],
            " 00:00:00 up 3 days, 42 min,  0 users,  load average: 0.00, 0.00, 0.00")
        can = step("uptime", M.render_uptime(7))
        self.assertIs(RC.canon(can, st), can)

    def test_ps_time_row_masks(self):
        """Row 2 (ps): cpu TIME -> 0:00, only when the header declares a TIME
        column; the frozen tj3-ps template render passes through untouched."""
        st = _touched_ws_state()
        got = RC.canon(step(
            "/usr/local/bin/tj3-ps",
            "  PID USER     TIME   COMMAND\n"
            "    1 root      0:00 sleep 86400\n"
            "  110 root      0:03 after 1 5 echo g_tok >> /tmp/w/task1.log"), st)
        self.assertEqual(
            got["output"],
            "  PID USER     TIME   COMMAND\n"
            "    1 root      0:00 sleep 86400\n"
            "  110 root      0:00 after 1 5 echo g_tok >> /tmp/w/task1.log")
        can = step(PS, M.render_ps([(1, "S", "init"), (2, "S", "sleep 86400")]))
        self.assertIs(RC.canon(can, st), can)

    def test_touched_dir_l_mask(self):
        """Row 3: a -l-family render whose TARGET is tracker-touched gets the
        whole-render date+time mask (the row's unit is the render)."""
        st = _touched_ws_state()
        got = RC.canon(step(
            "ls -l /tmp/w/d",
            "total 8\n"
            "-rw-r--r--    1 root     root            10 Jul 20 12:01 notes.txt"), st)
        self.assertEqual(
            got["output"],
            "total 8\n"
            "-rw-r--r--    1 root     root            10 Jan  1 00:00 notes.txt")
        rel = RC.canon(step("ls -ld d",
                            "drwxr-xr-x 2 root root 4096 Jul 20 12:01 d",
                            cwd="/tmp/w"), st)
        self.assertEqual(rel["output"], "drwxr-xr-x 2 root root 4096 Jan  1 00:00 d")

    def test_untouched_mtimes_pass_through(self):
        """The 'leave raw' row: mtimes of untouched shipped files are image-constant
        facts — byte-for-byte, same object back."""
        st = _touched_ws_state()
        keep = step("ls -l /usr/lib",
                    "total 24\n-rw-r--r-- 1 root root 1234 Feb  3  2023 libfoo.so")
        self.assertIs(RC.canon(keep, st), keep)

    def test_runtime_mount_rows(self):
        """Row 1: the three runtime-mount rows mask inside an untouched /etc
        listing (other rows keep their dates); a direct -l of a mount masks; a
        touched /etc masks the whole render instead."""
        st = _touched_ws_state()
        got = RC.canon(step(
            "ls -la /etc",
            "total 24\n"
            "-rw-r--r--. 1 root root  158 Jun 23  2024 hosts\n"
            "-rw-r--r--  1 root root   13 Jul 20 09:12 hostname\n"
            "-rw-r--r--  1 root root  100 Jul 20 09:12 resolv.conf\n"
            "-rw-r--r--  1 root root 1234 Jan  5  2024 os-release\n"
            "lrwxrwxrwx  1 root root   12 Feb  3  2023 mtab -> /proc/mounts"), st)
        self.assertEqual(
            got["output"],
            "total 24\n"
            "-rw-r--r--. 1 root root  158 Jan  1 00:00 hosts\n"
            "-rw-r--r--  1 root root   13 Jan  1 00:00 hostname\n"
            "-rw-r--r--  1 root root  100 Jan  1 00:00 resolv.conf\n"
            "-rw-r--r--  1 root root 1234 Jan  5  2024 os-release\n"
            "lrwxrwxrwx  1 root root   12 Feb  3  2023 mtab -> /proc/mounts")
        direct = RC.canon(step(
            "ls -l /etc/resolv.conf",
            "-rw-r--r-- 1 root root 100 Jul 20 09:12 /etc/resolv.conf"), st)
        self.assertEqual(direct["output"],
                         "-rw-r--r-- 1 root root 100 Jan  1 00:00 /etc/resolv.conf")
        rel = RC.canon(step("ls -l hosts",
                            "-rw-r--r-- 1 root root 158 Jul 20 09:12 hosts",
                            cwd="/etc"), st)
        self.assertEqual(rel["output"],
                         "-rw-r--r-- 1 root root 158 Jan  1 00:00 hosts")
        # a mutation under /etc beats the names filter: whole render masked
        st_etc = M.ShellState(mode="collection")
        st_etc.fold(step("rm /etc/foo.conf"))
        got = RC.canon(step(
            "ls -l /etc",
            "-rw-r--r-- 1 root root 1234 Jan  5  2024 os-release"), st_etc)
        self.assertEqual(got["output"],
                         "-rw-r--r-- 1 root root 1234 Jan  1 00:00 os-release")

    def test_prologue_fire_touched_view(self):
        """A job due at THIS step fires in its prologue, so its dst counts as
        touched for the -l mask; a not-yet-due job establishes nothing."""
        st_due = M.ShellState(mode="collection")
        st_due.fold(step("after 1 2 'echo x_tok >> /tmp/w/task1.log' & echo $!", "110"))
        st_due.fold(step("pwd", "/"))
        self.assertEqual(st_due.vt, 2)                       # fire due exactly now
        self.assertEqual(st_due.touched, {})                 # ...but not yet folded
        got = RC.canon(step("ls -l /tmp/w",
                            "-rw-r--r-- 1 root root 6 Jul 20 12:02 task1.log"), st_due)
        self.assertEqual(got["output"],
                         "-rw-r--r-- 1 root root 6 Jan  1 00:00 task1.log")
        st_wait = M.ShellState(mode="collection")
        st_wait.fold(step("after 1 2 'echo x_tok >> /tmp/w/task1.log' & echo $!", "110"))
        keep = step("ls -l /tmp/w", "-rw-r--r-- 1 root root 6 Jul 20 12:02 task1.log")
        self.assertIs(RC.canon(keep, st_wait), keep)

    def test_adversarial_marker_content_not_masked(self):
        """Masking keys on the PARSED COMMAND, never on output content: file content
        that merely LOOKS like ls -l / uptime / ps under a read verb is untouched."""
        st = _touched_ws_state()
        evil = ("-rw-r--r-- 1 root root 42 Jul 20 12:00 evil\n"
                "12:34:56 up 3 days,  2 users,  load average: 1.00, 2.00, 3.00\n"
                "  PID USER     TIME   COMMAND\n"
                "    7 root      9:59 sh")
        for cmd in ("cat /tmp/w/d/notes.txt",               # touched path, wrong verb
                    "grep -F -m 8 root /tmp/w/d/notes.txt",
                    "head -n 4 /tmp/w/d/notes.txt",
                    "ls -1 /tmp/w/d"):                      # ls, but names-only form
            keep = step(cmd, evil)
            self.assertIs(RC.canon(keep, st), keep, f"masked through {cmd!r}")

    def test_conservative_when_unknown_and_total(self):
        """TOTAL and conservative: out-of-universe / malformed steps pass through
        unmasked (never raise); with state=None the state-independent masks still
        fire but the touched -l mask does not."""
        st = _touched_ws_state()
        evil = "-rw-r--r-- 1 root root 42 Jul 20 12:00 x"
        for s in (step("curl -s http://x", evil),
                  step("ls /etc; pwd", evil),
                  step("", "x"),
                  {"cmd": None, "output": "x", "exit": 0, "cwd": "/"},
                  step("uptime", ""),
                  {"cmd": "uptime", "output": None, "exit": 0}):
            self.assertIs(RC.canon(s, st), s)
        got = RC.canon(step(
            "uptime", "12:34:56 up 9,  3 users,  load average: 0.50, 0.40, 0.30"),
            None)
        self.assertEqual(got["output"],
                         "00:00:00 up 9,  0 users,  load average: 0.00, 0.00, 0.00")
        keep = step("ls -l /tmp/w/d", "-rw-r--r-- 1 root root 10 Jul 20 12:01 notes.txt")
        self.assertIs(RC.canon(keep, None), keep)

    def test_idempotence_and_non_mutation(self):
        """canon(canon(s)) == canon(s) on every masked form; the input step is never
        mutated; a masked step comes back as a copy with only output rewritten."""
        st = _touched_ws_state()
        cases = [
            step("uptime",
                 " 12:34:56 up 3 days,  2 users,  load average: 0.15, 0.10, 0.05"),
            step("/usr/local/bin/tj3-ps",
                 "  PID USER     TIME   COMMAND\n    7 root      9:59 sh"),
            step("ls -l /tmp/w/d",
                 "total 8\n-rw-r--r-- 1 root root 10 Jul 20 12:01 notes.txt"),
            step("ls -la /etc",
                 "-rw-r--r-- 1 root root 158 Jun 23  2024 hosts"),
        ]
        for s in cases:
            before = dict(s)
            got = RC.canon(s, st)
            self.assertEqual(s, before, f"input mutated: {s['cmd']!r}")
            self.assertIsNot(got, s, f"expected a masked copy: {s['cmd']!r}")
            self.assertNotEqual(got["output"], s["output"])
            again = RC.canon(dict(got), st)
            self.assertEqual(again["output"], got["output"],
                             f"not idempotent: {s['cmd']!r}")


if __name__ == "__main__":
    unittest.main()
