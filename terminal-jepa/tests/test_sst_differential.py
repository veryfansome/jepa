"""DG-4a standing differential harness: the SST vs the REAL shell (round-3 method,
institutionalized). Runs the round-3 reviewer's edge-trajectory battery on BOTH
mint families — alpine:latest (busybox ash) and debian:stable-slim (dash + GNU
coreutils) — folds the real records through realenv.shell_state.ShellState, and
asserts the GOLDEN RULE at every step:

    predict(t, cmd) is either BOT or EXACTLY equal to the real rendered step.

Rows additionally pin expected coverage ("det" must stay determined — catches
coverage collapse; "bot" must stay BOT — catches over-claiming). Error templates
are probe-harvested per image inside the container (draft §3.5), exactly like the
collector — never hand-authored.

Docker-required classes skip cleanly (@unittest.skipUnless) when docker or an
image is unavailable. The fold-level ports of the round-3 repros that need no
shell (V9 failed-launch, V13 payload-law parser rejections, V16 render_canon
parent-listing residue) run unconditionally.

Trajectory index (round-3 review V1–V17 + battery §1–§18):
  V1/§1 redirect > through symlink      V2/§2 append >> through symlink
  V3/§3 > onto hardlink peer            §4 > through DANGLING workspace symlink
  V4/§5 mv onto own hardlink            V5/§6 mv dir onto/into itself (EINVAL)
  §7 mv trailing slashes                V6/§12 ls -ld self-row (phantom child)
  V7/§9 cd through symlinked dir        V15/§10 cd // (logical cwd)
  V8/§15 dotfile absence entailment     V10/§14 cat-producer trailing-nl capture
  V11/§11 dangling image symlink        V12 image symlink observed via cat
  V14/§13 find replay vs rm -r ancestor V17 trailing-slash redirect dst
  §17 test -f through a link            §18 mv onto symlink-to-file

Round-4 review rows (classes 1–8; TestRound4FoldPorts is the docker-free half,
the test_r4_* trajectories the live half): grep binary heuristic; tail/head over
rstripped observations; '..' traversal (predict + mine-absence + cd-logical);
alias staleness after rm/mv/truncate; kind-unknown rm / hard-ln-to-dir / ls -l
space-row phantoms; bind-mount + /proc//sys//dev writability; symlink-cycle hop
cap; universe seam (unquoted quotes/globs/tilde); cond -f on cat-only evidence
and /proc//sys volatility. The mint-scale gate lives in test_sst_mint_replay.py.

Round-5 review rows (findings F1–F6; TestRound5FoldPorts is the docker-free
half, the test_r5_* trajectories the live half): severed-hardgroup peer
degradation (conservative mv); the job-fire failure channel (unsound landings
claim nothing — fire after rm -r of the workspace / through a link with a
missing parent / onto a dir); the absence-revival law (uncertain tombstones
dropped on any creation event); dir-mine staleness (listings/kind at
linkness-unknown nodes vs rm -r/mv of the real dir); the leading-dash parser
seam; the raw-string find-replay cache key.
"""

import pathlib as _pathlib
import subprocess
import sys as _sys
import unittest

_ROOT = str(_pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from realenv import render_canon as RC
from realenv import shell_state as M
from realenv.shell_state import BOT, ShellState

# busybox-ash, dash, AND bash /bin/sh — the last two (fedora inner-val, rockylinux final-test)
# validate every `..` intermediate, so a textual-collapse cd bug diverges only here (round-7).
IMAGES = ("alpine:latest", "debian:stable-slim", "fedora:latest", "rockylinux:9")


def _docker_ready():
    try:
        if subprocess.run(["docker", "info"], capture_output=True,
                          timeout=30).returncode != 0:
            return False
        return all(subprocess.run(["docker", "image", "inspect", img],
                                  capture_output=True, timeout=30).returncode == 0
                   for img in IMAGES)
    except Exception:
        return False


HAVE_DOCKER = _docker_ready()


def step(cmd, out="", code=0, cwd="/"):
    return {"cmd": cmd, "output": out, "exit": code, "cwd": cwd}


# ================================================================ fold-level ports
# (no shell needed: the automaton is collector-driven, the parser is pure)

TMPL_BB = {   # busybox-flavored {text, exit} table (F3: string-only entries fail closed)
    "cat": {"text": "cat: can't open '{path}': No such file or directory", "exit": 1},
    "ls": {"text": "ls: {path}: No such file or directory", "exit": 1},
    "cd": {"text": "/bin/sh: cd: can't cd to {path}", "exit": 2},
    "rm": {"text": "rm: can't remove '{path}': No such file or directory", "exit": 1},
    "mv": {"text": "mv: can't rename '{path}': No such file or directory", "exit": 1},
    "kill": {"text": "sh: can't kill pid {pid}: No such process", "exit": 1},
}


class TestFoldLevelPorts(unittest.TestCase):
    """Round-3 repros that need no real shell."""

    def test_v9_failed_after_launch_registers_no_job(self):
        """F4: an exit!=0 launch must never enter the job table nor ever fire."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("after 1 2 'echo g_tok >> /tmp/w/task1.log' & echo $!",
                     "sh: after: not found", 127))
        self.assertEqual(st.jobs, {})
        st.fold(step("pwd", "/"))
        st.fold(step("pwd", "/"))
        # the reserved job-log namespace stays entailed-absent forever
        self.assertEqual(
            st.predict(3, "cat /tmp/w/task1.log"),
            {"output": TMPL_BB["cat"]["text"].format(path="/tmp/w/task1.log"),
             "exit": 1, "cwd": "/"})

    def test_v13_payload_charset_law(self):
        """F13: backslash / '$' / backtick inside quoted payloads and leading-dash
        payloads are out of the universe — echo, echo-redirect, and after TOKs."""
        bads = [
            "echo '-n'", "echo '-e'", "echo -n hi", "echo 'a\\nb'",
            "echo 'a\\nb' > /tmp/w/e", "echo '-n' > /tmp/w/f",
            "echo '$HOME'", "echo '`id`'", "echo '$X' > /tmp/w/f",
            "after 1 2 'echo $HOME >> /tmp/w/task1.log' & echo $!",
            "after 1 2 'echo `id` >> /tmp/w/task1.log' & echo $!",
            "after 1 2 'echo a\\nb >> /tmp/w/task1.log' & echo $!",
            "after 1 2 'echo -n >> /tmp/w/task1.log' & echo $!",
        ]
        for bad in bads:
            with self.assertRaises(M.ParseError, msg=f"parser accepted {bad!r}"):
                M.parse_command(bad)
        # literal-safe payloads stay in-universe (incl. non-leading dashes)
        for good in ("echo 'hi -n there'", "echo 'a b' > /tmp/w/f",
                     "after 1 2 'echo x_tok >> /tmp/w/task1.log' & echo $!"):
            M.parse_command(good)

    def test_f12_wsf_fence_on_normalized_dst(self):
        """F12: normpath escapes of /tmp/w/ are out of the universe."""
        for bad in ("echo 'x' > /tmp/w/../../etc/profile",
                    "echo 'x' > /tmp/w/..",
                    "echo 'x' > /tmp/w/d/..",
                    "echo 'x' > /tmp/w/.",
                    "ls -1 /etc > /tmp/w/../evil"):
            with self.assertRaises(M.ParseError, msg=f"parser accepted {bad!r}"):
                M.parse_command(bad)
        for good in ("echo 'x' > /tmp/w/./f", "echo 'x' > /tmp/w//f",
                     "echo 'x' > /tmp/w/d/../f"):
            M.parse_command(good)

    def test_v16_parent_listing_masks_touched_child_row(self):
        """F14 (render_canon): a touched dir's own mtime row inside its PARENT's
        -l listing is masked; untouched sibling rows keep their dates."""
        st = ShellState(mode="collection")
        st.fold(step("mkdir /tmp/w/d"))                      # touches /tmp/w too
        s = step("ls -l /tmp",
                 "drwxr-xr-x 3 root root 4096 Jul 23 09:35 w\n"
                 "-rw-r--r-- 1 root root   10 Feb  3  2023 other")
        got = RC.canon(s, st)
        self.assertIsNot(got, s)
        self.assertEqual(got["output"],
                         "drwxr-xr-x 3 root root 4096 Jan  1 00:00 w\n"
                         "-rw-r--r-- 1 root root   10 Feb  3  2023 other")
        # a touched FILE's row via the parent listing masks likewise
        st2 = ShellState(mode="collection")
        st2.fold(step("echo 'x_tok' > /tmp/w/f"))
        st2.fold(step("touch /tmp/w/f"))                     # mtime-only touch
        s2 = step("ls -l /tmp/w", "-rw-r--r-- 1 root root 6 Jul 23 09:40 f")
        self.assertEqual(RC.canon(s2, st2)["output"],
                         "-rw-r--r-- 1 root root 6 Jan  1 00:00 f")

    def test_law_write_through_image_link_degrades(self):
        """LINK-CONSERVATISM LAW: a redirect through an SST symlink onto an
        image-inherited node has an uncertain landing site — BOT ack, and the
        fold degrades every known chain node (stale content is never served);
        the link itself survives. Both visibility modes stay in lockstep."""
        rows = [step("cat /etc/hosts", "127.0.0.1 localhost"),
                step("ln -s /etc/hosts /tmp/w/lh"),
                step("echo 'x' > /tmp/w/lh"),
                step("echo 'y' >> /tmp/w/lh")]
        snaps = []
        for mode in ("collection", "sst"):
            st = ShellState(mode=mode, error_templates=TMPL_BB)
            st.fold(rows[0])
            self.assertEqual(st.predict(1, "cat /etc/hosts"),
                             {"output": "127.0.0.1 localhost", "exit": 0, "cwd": "/"})
            st.fold(rows[1])
            self.assertIsNone(st.predict(2, "echo 'x' > /tmp/w/lh"))
            st.fold(rows[2])
            self.assertIsNone(st.predict(3, "cat /etc/hosts"))     # degraded
            self.assertIsNone(st.predict(3, "cat /tmp/w/lh"))
            self.assertEqual(st.predict(3, "readlink /tmp/w/lh"),  # link survives
                             {"output": "/etc/hosts", "exit": 0, "cwd": "/"})
            st.fold(rows[3])
            snaps.append((st.fs, st.ws, st.touched, st.cwd))
        self.assertEqual(snaps[0], snaps[1], "mode divergence on the degrade path")

    def test_slashed_dst_success_record_is_mismatch_only(self):
        """F3: an exit-0 record with a trailing-slash redirect dst is physically
        impossible — logged as a mismatch, no belief written."""
        st = ShellState(error_templates=TMPL_BB)
        st.fold(step("echo 'old' > /tmp/w/f"))
        st.fold(step("echo 'x' > /tmp/w/f/", "", 0))
        self.assertTrue(st.mismatches)
        self.assertEqual(st.predict(2, "cat /tmp/w/f"),
                         {"output": "old", "exit": 0, "cwd": "/"})

    def test_f15_verbsig_fail_closed_on_non_string(self):
        from realenv import verbsig as V
        for bad in (None, 7, b"pwd", ["pwd"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                V.sig(bad)


class TestRound4FoldPorts(unittest.TestCase):
    """Round-4 review repros that need no real shell (classes 1–8)."""

    def test_r4c1_grep_binary_content_is_bot(self):
        """Class 1a: NUL-bearing content is never grep-replayed (real greps print
        a dialect-divergent 'binary file matches' line)."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("cat /opt/bin1", "TZif2\x00\x00\x00data"))
        self.assertIsNone(st.predict(1, "grep -F -m 8 TZif2 /opt/bin1"))
        self.assertIsNone(st.predict(1, "cat /opt/bin1 | grep -F -m 8 TZif2"))
        # cat replay of the same bytes stays determined (no heuristic on cat)
        self.assertEqual(st.predict(1, "cat /opt/bin1"),
                         {"output": "TZif2\x00\x00\x00data", "exit": 0, "cwd": "/"})

    def test_r4c1_tail_needs_trailing_nl_known(self):
        """Class 1b: the record channel rstrips trailing newlines, so a merely-
        observed file may have MORE logical lines than the render shows — tail
        windows are BOT; head windows stay sound and mirror the rstrip."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("cat /opt/obs", "line1\n\nline3"))
        self.assertIsNone(st.predict(1, "tail -n 2 /opt/obs"))
        self.assertIsNone(st.predict(1, "cat /opt/obs | tail -n 1"))
        self.assertEqual(st.predict(1, "tail -n 0 /opt/obs"),
                         {"output": "", "exit": 0, "cwd": "/"})     # well-defined
        # class 1c: a head window ending on the blank line 2 joins with a
        # trailing '\n' the record strips — the prediction must strip it too
        self.assertEqual(st.predict(1, "head -n 2 /opt/obs"),
                         {"output": "line1", "exit": 0, "cwd": "/"})
        self.assertEqual(st.predict(1, "head -n 9 /opt/obs"),
                         {"output": "line1\n\nline3", "exit": 0, "cwd": "/"})
        # SST-created content has known bytes: tail stays determined
        st2 = ShellState(mode="sst", error_templates=TMPL_BB)
        st2.fold(step("echo 'a_tok' > /tmp/w/t"))
        st2.fold(step("echo 'b_tok' >> /tmp/w/t"))
        self.assertEqual(st2.predict(2, "tail -n 1 /tmp/w/t"),
                         {"output": "b_tok", "exit": 0, "cwd": "/"})

    def test_r4c2_dotdot_mine_absence_never_corrupts(self):
        """Class 2: a failed read through unknown '..' intermediates must not
        tombstone the textually-collapsed path (pure-fold port of the review's
        belief-corruption repro)."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("echo 'x_tok' > /tmp/w/bf"))
        st.fold(step("cat /tmp/w/nope/../bf",
                     TMPL_BB["cat"]["text"].format(path="/tmp/w/nope/../bf"), 1))
        # the SST-created file still reads back correctly
        self.assertEqual(st.predict(2, "cat /tmp/w/bf"),
                         {"output": "x_tok", "exit": 0, "cwd": "/"})
        # predict through unknown intermediates is BOT both ways
        self.assertIsNone(st.predict(2, "cat /tmp/w/nope/../bf"))
        self.assertIsNone(st.predict(2, "touch /tmp/w/gone/../bf"))
        self.assertIsNone(st.predict(2, "echo 'y' > /tmp/w/gone2/../bf2"))
        # ...but a known-live linkness-known intermediate keeps it determined
        st.fold(step("mkdir /tmp/w/d8"))
        self.assertEqual(st.predict(3, "cat /tmp/w/d8/../bf"),
                         {"output": "x_tok", "exit": 0, "cwd": "/"})
        # round-7: cd '..' through an unknown intermediate is BOT (bash /bin/sh —
        # fedora/rockylinux — validates each component; only known-live dirs stay determined)
        self.assertIsNone(st.predict(3, "cd /tmp/w/gone/.."))
        st.fold(step("mkdir /tmp/w/dd9"))
        self.assertEqual(st.predict(4, "cd /tmp/w/dd9/.."),
                         {"output": "", "exit": 0, "cwd": "/tmp/w"})

    def test_r4c3_alias_staleness_invalidates_mined_content(self):
        """Class 3: content mined at a linkness-unknown node goes BOT after a
        mutation destroys a node with EQUAL known bytes (a true alias must share
        bytes — equality is the necessary condition)."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("mkdir /tmp/w/m"))
        st.fold(step("echo 'out_tok' > /tmp/w/out"))
        st.fold(step("ln -s /tmp/w/out /tmp/w/m/abs"))
        st.fold(step("mv /tmp/w/m /tmp/w/n"))               # conservative fold
        self.assertIsNone(st.predict(4, "cat /tmp/w/n/abs"))
        st.fold(step("cat /tmp/w/n/abs", "out_tok"))        # mined, linkness unknown
        self.assertEqual(st.predict(5, "cat /tmp/w/n/abs"),
                         {"output": "out_tok", "exit": 0, "cwd": "/"})
        st.fold(step("rm /tmp/w/out"))
        self.assertIsNone(st.predict(6, "cat /tmp/w/n/abs"),
                          "stale alias content served after rm of equal bytes")
        # unrelated mined content with DIFFERENT bytes survives an exact-known rm
        st2 = ShellState(mode="sst", error_templates=TMPL_BB)
        st2.fold(step("cat /opt/other", "other_tok"))
        st2.fold(step("echo 'mine_tok' > /tmp/w/f"))
        st2.fold(step("rm /tmp/w/f"))
        self.assertEqual(st2.predict(3, "cat /opt/other"),
                         {"output": "other_tok", "exit": 0, "cwd": "/"})
        # ...but an UNCERTAIN-bytes destruction (never-read node) invalidates all
        st2.fold(step("rm /opt/unread"))
        self.assertIsNone(st2.predict(4, "cat /opt/other"))

    def test_r4c4_kind_unknown_mutation_acks_are_bot(self):
        """Class 4: rm on a kind-unknown alive node (ls-child / find-mined) and
        hard ln onto a non-file target are never determined-ok."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("ls -1 /opt/box", "nf\nsub"))
        self.assertIsNone(st.predict(1, "rm /opt/box/sub"))     # may be a dir
        self.assertIsNone(st.predict(1, "touch /opt/box/sub"))  # may be a loop/dangler
        self.assertEqual(st.predict(1, "rm -r /opt/box/sub"),   # -r removes either
                         {"output": "", "exit": 0, "cwd": "/"})
        st.fold(step("find /opt/fx -maxdepth 1 -name 'dd'", "/opt/fx/dd"))
        self.assertIsNone(st.predict(2, "rm /opt/fx/dd"))
        # hard ln to an SST-created DIRECTORY: never ok
        st.fold(step("mkdir /tmp/w/hd"))
        st.fold(step("mkdir /tmp/w/hp"))
        self.assertIsNone(st.predict(4, "ln /tmp/w/hd /tmp/w/hp/h"))
        # hard ln to an SST-created FILE stays determined
        st.fold(step("echo 'x_tok' > /tmp/w/hp/f"))
        self.assertEqual(st.predict(5, "ln /tmp/w/hp/f /tmp/w/hp/h"),
                         {"output": "", "exit": 0, "cwd": "/"})

    def test_r4c4_ls_l_splice_refuses_uncertain_rows(self):
        """Class 4: ls -l rows whose name field is not certain (space-bearing
        names, device rows) are refused entirely — no phantom children."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("ls -l /opt/sp",
                     "total 4\n"
                     "-rw-r--r-- 1 root root 2 Jul 23 09:00 a b\n"        # space name
                     "brw-rw---- 1 root disk 8, 0 Jul 23 09:00 sda\n"     # device row
                     "-rw-r--r-- 1 root root 2 Jul 23 09:00 clean"))
        self.assertNotIn("/opt/sp/b", st.fs)         # no phantom from 'a b'
        self.assertNotIn("/opt/sp/a", st.fs)
        self.assertNotIn("/opt/sp/sda", st.fs)       # device size field split
        self.assertIn("/opt/sp/clean", st.fs)        # certain 9-field row spliced
        self.assertIsNone(st.predict(1, "rm /opt/sp/b"))
        self.assertIsNone(st.predict(1, "mv /opt/sp/b /opt/sp/c"))

    def test_r4c5_unwritable_mutations_are_bot(self):
        """Class 5: mutations on runtime bind-mounts or under /proc //sys //dev
        are BOT even when belief knows the node; reads are unaffected."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("cat /etc/hosts", "127.0.0.1 localhost"))
        self.assertEqual(st.predict(1, "cat /etc/hosts"),       # read unaffected
                         {"output": "127.0.0.1 localhost", "exit": 0, "cwd": "/"})
        for cmd in ("rm /etc/hosts", "mv /etc/hosts /tmp/w/hh",
                    "touch /etc/hosts", "touch /proc/zzz9", "mkdir /proc/r4",
                    "rm -r /etc", "mv /etc /tmp/w/etc2"):
            self.assertIsNone(st.predict(1, cmd), cmd)
        # a workspace-escaping redirect dst is already OUT of the universe (F12)
        with self.assertRaises(M.ParseError):
            M.parse_command("echo 'x' > /tmp/w/../../proc/x")

    def test_r4c6_symlink_cycles_hop_capped(self):
        """Class 6: SST-known symlink cycles exceed the _resolve hop cap and go
        BOT everywhere (real shells raise ELOOP, exit 1)."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("ln -s /tmp/w/cy2 /tmp/w/cy1"))
        st.fold(step("ln -s /tmp/w/cy1 /tmp/w/cy2"))
        st.fold(step("ln -s /tmp/w/self /tmp/w/self"))
        for cmd in ("touch /tmp/w/cy1", "cat /tmp/w/cy1", "touch /tmp/w/self",
                    "echo 'x' > /tmp/w/cy1", "[ -f /tmp/w/cy1 ] && cat /tmp/w/cy1",
                    "mv /tmp/w/cy1 /tmp/w/other", "stat -c '%n %s %F %a' /tmp/w/cy1"):
            self.assertIsNone(st.predict(3, cmd), cmd)
        # rm of a cycle MEMBER operates on the link itself: still determined
        self.assertEqual(st.predict(3, "rm /tmp/w/cy1"),
                         {"output": "", "exit": 0, "cwd": "/"})

    def test_r4c7_universe_seam_shell_expansion_chars(self):
        """Class 7: unquoted double quotes, glob characters and tilde leave the
        universe; quoted globs and the real /usr/bin/[ path stay in."""
        for bad in ('cat "/tmp/w/qf"', "cat /tmp/w/zz*", "ls -1 /tmp/w/q?",
                    "cat /tmp/w/f[12]", "cd ~", "cat ~/notes", "echo ~",
                    "rm /tmp/w/*", "find /opt -maxdepth 1 -name *.conf",
                    "after 1 2 'echo x* >> /tmp/w/task1.log' & echo $!",
                    "after 1 2 'echo a;b >> /tmp/w/task1.log' & echo $!"):
            with self.assertRaises(M.ParseError, msg=f"parser accepted {bad!r}"):
                M.parse_command(bad)
        for good in ("find /opt -maxdepth 1 -name '*.conf'",
                     "grep -F -m 8 'a*b' /etc/hosts",
                     "grep -F -m 8 'a;b' /etc/hosts",
                     "cat /usr/bin/[",
                     "stat -c '%n %s %F %a' /usr/bin/[",
                     "echo 'glob *? safe when quoted' > /tmp/w/f",
                     "[ -f /tmp/w/f ] && cat /tmp/w/f"):
            M.parse_command(good)
        # verbsig stays bidirectionally total with the parser on the seam
        from realenv import verbsig as V
        for cmd in ('cat "/tmp/w/qf"', "cat /tmp/w/zz*", "cat /usr/bin/["):
            p_ok = s_ok = True
            try:
                M.parse_command(cmd)
            except M.ParseError:
                p_ok = False
            try:
                V.sig(cmd)
            except ValueError:
                s_ok = False
            self.assertEqual(p_ok, s_ok, cmd)

    def test_r4c8_read_evidence_honesty(self):
        """Class 8: cat-success proves readable content, not regular-file; and
        observed content under /proc //sys is never re-served."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("cat /dev/null", ""))
        self.assertIsNone(st.predict(1, "[ -f /dev/null ] && cat /dev/null"))
        st.fold(step("cat /proc/uptime", "12.34 45.67"))
        self.assertIsNone(st.predict(2, "cat /proc/uptime"))    # volatile
        self.assertIsNone(st.predict(2, "grep -F -m 8 4 /proc/uptime"))
        st.fold(step("ls -1 /proc", "1\nuptime\nversion"))
        self.assertIsNone(st.predict(3, "ls -1 /proc"))         # listings churn
        # SST-created nodes keep full cond determinism
        st.fold(step("echo 'x_tok' > /tmp/w/cf"))
        self.assertEqual(st.predict(4, "[ -f /tmp/w/cf ] && cat /tmp/w/cf"),
                         {"output": "x_tok", "exit": 0, "cwd": "/"})
        st.fold(step("mkdir /tmp/w/cd8"))
        self.assertEqual(st.predict(5, "[ -d /tmp/w/cd8 ] && ls -1 /tmp/w/cd8"),
                         {"output": "", "exit": 0, "cwd": "/"})
        # -d False on cat-evidence stays sound (readable content is never a dir)
        st.fold(step("cat /opt/known", "k_tok"))
        self.assertEqual(st.predict(6, "[ -d /opt/known ] && ls -1 /opt/known"),
                         {"output": "", "exit": 1, "cwd": "/"})
        self.assertIsNone(st.predict(6, "[ -f /opt/known ] && cat /opt/known"))


class TestRound5FoldPorts(unittest.TestCase):
    """Round-5 review repros that need no real shell (findings F1–F6)."""

    def test_f1_severed_hardgroup_peer_degrades(self):
        """F1: a conservative mv that severs a hardgroup degrades every peer to
        unknown content AT SEVERING TIME — a later append through the vacated
        side can change the shared inode invisibly, so the peer's bytes must
        never be served again."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("mkdir /tmp/w/d2"))
        st.fold(step("echo 'seed_tok' > /tmp/w/d2/f"))
        st.fold(step("ln /tmp/w/d2/f /tmp/w/peer"))
        self.assertEqual(st.predict(3, "cat /tmp/w/peer"),
                         {"output": "seed_tok", "exit": 0, "cwd": "/"})
        st.fold(step("mv /tmp/w/d2 /tmp/w/e2"))              # dst unknown: conservative
        self.assertIsNone(st.predict(4, "cat /tmp/w/peer"),
                          "severed peer served stale bytes")
        st.fold(step("echo 'add_tok' >> /tmp/w/e2/f"))       # append via vacated side
        self.assertIsNone(st.predict(5, "cat /tmp/w/peer"))
        self.assertIsNone(st.predict(5, "cat /tmp/w/e2/f"))
        # the severed flag outlives a re-established tracked write: a later
        # uncertain invalidation must still reach the peer (no linkness exemption)
        st.fold(step("echo 'fresh_tok' > /tmp/w/peer"))
        self.assertEqual(st.predict(6, "cat /tmp/w/peer"),
                         {"output": "fresh_tok", "exit": 0, "cwd": "/"})
        st.fold(step("echo 'more_tok' >> /tmp/w/e2/f"))      # unsound chain again
        self.assertIsNone(st.predict(7, "cat /tmp/w/peer"),
                          "severed node kept the linkness_known exemption")
        # the src side stays a certain tombstone (template-exact miss)
        self.assertEqual(st.predict(7, "cat /tmp/w/d2/f")["exit"], 1)

    def test_f2_fire_after_workspace_removal_claims_nothing(self):
        """F2 (B3 shape): a fire whose landing parent is tombstoned FAILED in
        reality — the fold must not mint the log node, and the reserved-namespace
        entailment dies with the workspace."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("after 1 2 'echo z_tok >> /tmp/w/task1.log' & echo $!", "110"))
        st.fold(step("rm -r /tmp/w"))
        st.fold(step("pwd", "/"))                            # fire due: unsound landing
        self.assertEqual(st.jobs[1]["state"], "fired")       # the automaton fired
        self.assertIsNone(st.predict(3, "cat /tmp/w/task1.log"),
                          "minted content from a failed fire")
        self.assertIsNone(st.predict(3, "[ -e /tmp/w/task1.log ] && cat /tmp/w/task1.log"))

    def test_f2_fire_through_link_with_missing_parent(self):
        """F2 (B2 shape): the fire lands through an SST-known link at a path whose
        parent is unknown — the real append may fail ENOENT, so nothing may be
        claimed at the target (no minted node, touch stays BOT)."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("after 1 2 'echo x_tok >> /tmp/w/task1.log' & echo $!", "110"))
        st.fold(step("ln -s /tmp/w/u/deep /tmp/w/task1.log"))
        st.fold(step("pwd", "/"))                            # fire due: unsound landing
        self.assertIsNone(st.predict(3, "touch /tmp/w/u/deep"),
                          "determined-ok on a path a failed fire never created")
        self.assertIsNone(st.predict(3, "cat /tmp/w/task1.log"))
        # the link itself survives the degrade
        self.assertEqual(st.predict(3, "readlink /tmp/w/task1.log"),
                         {"output": "/tmp/w/u/deep", "exit": 0, "cwd": "/"})

    def test_f2_fire_onto_dir_keeps_entries(self):
        """F2 (B4 shape): a fire onto a dir at the log path fails EISDIR — the
        degrade claims nothing, and the dir's tracked (empty) listing survives."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("after 1 2 'echo d_tok >> /tmp/w/task1.log' & echo $!", "110"))
        st.fold(step("mkdir /tmp/w/task1.log"))
        st.fold(step("pwd", "/"))                            # fire due: EISDIR
        self.assertIsNone(st.predict(3, "cat /tmp/w/task1.log"))
        self.assertEqual(st.predict(3, "ls -1 /tmp/w/task1.log"),
                         {"output": "", "exit": 0, "cwd": "/"})

    def test_f3_absence_revival_on_creation(self):
        """F3: an uncertain (template-mined) tombstone is dropped by ANY later
        successful creation event — the path may be a dangling symlink the
        creation just re-targeted; certain (SST-performed) deadness survives."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("cat /opt/e1dl",
                     TMPL_BB["cat"]["text"].format(path="/opt/e1dl"), 1))
        self.assertEqual(st.predict(1, "cat /opt/e1dl")["exit"], 1)   # still valid
        st.fold(step("echo 'pay_tok' > /tmp/w/pp"))          # a creation event
        self.assertIsNone(st.predict(2, "cat /opt/e1dl"),
                          "stale uncertain absence served after a creation")
        # certain deadness is unaffected by creations
        st.fold(step("rm /tmp/w/pp"))
        st.fold(step("touch /tmp/w/other"))
        self.assertEqual(st.predict(4, "cat /tmp/w/pp")["exit"], 1)

    def test_f4_dir_mine_staleness(self):
        """F4: entries/kind mined at a linkness-unknown node (possible image
        symlink-to-dir) are invalidated when a real dir is destroyed with an
        uncertain listing (rm -r of an unknown node) — the E2 fold port."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("ls -a /opt/e2ld", ".\n..\nk1\nk2"))
        self.assertEqual(st.predict(1, "ls -a /opt/e2ld"),
                         {"output": ".\n..\nk1\nk2", "exit": 0, "cwd": "/"})
        st.fold(step("rm -r /opt/e2real"))                   # may BE that dir
        self.assertIsNone(st.predict(2, "ls -a /opt/e2ld"),
                          "stale mined listing served after the real dir died")
        self.assertIsNone(st.predict(2, "[ -d /opt/e2ld ] && ls -1 /opt/e2ld"))
        # SST-created dirs keep their facts (linkness_known exemption)
        st2 = ShellState(mode="sst", error_templates=TMPL_BB)
        st2.fold(step("mkdir /tmp/w/keep"))
        st2.fold(step("rm -r /opt/unknown"))
        self.assertEqual(st2.predict(2, "ls -1 /tmp/w/keep"),
                         {"output": "", "exit": 0, "cwd": "/"})

    def test_f5_leading_dash_seam(self):
        """F5: leading-dash tokens in RELATIVE path positions and the grep TOK
        position leave the universe (real tools parse them as options); absolute
        paths, './-x' forms and the frozen option positions stay in."""
        for bad in ("touch '-x'", "touch -x", "grep -F -m 8 '-v' /tmp/w/gd",
                    "cat '-f'", "rm '-r'", "mkdir '-p'", "cd -",
                    "mv '-a' /tmp/w/b", "mv /tmp/w/a '-b'",
                    "ln -s /tmp/w/a '-l'", "ln '-t' /tmp/w/l",
                    "readlink '-f'", "head -n 2 '-'", "stat -c '%n %s %F %a' '-'",
                    "find '-d' -maxdepth 1 -name 'x'",
                    "[ -e '-x' ] && cat '-x'",
                    "ls -1 '-x' | head -n 2", "cat '-x' | tail -n 1",
                    "cat '-x' >> /tmp/w/f"):
            with self.assertRaises(M.ParseError, msg=f"parser accepted {bad!r}"):
                M.parse_command(bad)
        for good in ("touch '/tmp/w/-x'", "touch './-y'", "cat '/tmp/w/-f'",
                     "rm '/tmp/w/-r'", "grep -F -m 8 'a-b' /tmp/w/gd",
                     "grep -F -m 8 'has -v inside' /tmp/w/gd",
                     "find /opt -maxdepth 1 -name '-x'",     # -name operand: exempt
                     "mv '/tmp/w/-a' /tmp/w/b", "cd ./-d"):
            M.parse_command(good)
        # verbsig stays bidirectionally total with the parser on the seam
        from realenv import verbsig as V
        for cmd in ("touch '-x'", "grep -F -m 8 '-v' /tmp/w/gd", "touch './-y'"):
            p_ok = s_ok = True
            try:
                M.parse_command(cmd)
            except M.ParseError:
                p_ok = False
            try:
                V.sig(cmd)
            except ValueError:
                s_ok = False
            self.assertEqual(p_ok, s_ok, cmd)

    def test_f6_find_cache_raw_key(self):
        """F6: the R7 replay cache keys on the RAW command string — the quoted
        flattening of two DIFFERENT commands must never collide."""
        cmd_y = "find /tmp/w/A -maxdepth 1 -name 'B -maxdepth 1 -name z'"
        cmd_x = "find '/tmp/w/A -maxdepth 1 -name B' -maxdepth 1 -name 'z'"
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step(cmd_y, ""))
        self.assertEqual(st.predict(1, cmd_y),               # identical raw: replay
                         {"output": "", "exit": 0, "cwd": "/"})
        self.assertIsNone(st.predict(1, cmd_x),
                          "cache-key injection: replayed the wrong command")


class TestRound6FoldPorts(unittest.TestCase):
    """Round-6 live-battery repros that need no real shell (findings F1–F5)."""

    def test_f1_stale_cwd_latch(self):
        """F1: once a tombstone covers the tracked cwd, docker_env's `cd <cwd>
        2>/dev/null` prologue falls back to the exec start dir — every cwd-dependent
        surface (pwd, cd relative AND absolute, bare/relative resolution) is BOT
        until a recorded cwd re-anchors on a live dir. Absolute-arg commands hold."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("mkdir /tmp/w/hp"))
        st.fold(step("cd /tmp/w/hp", "", 0, "/tmp/w/hp"))
        self.assertEqual(st.predict(2, "pwd"),
                         {"output": "/tmp/w/hp", "exit": 0, "cwd": "/tmp/w/hp"})
        st.fold(step("rm -r /tmp/w/hp", "", 0, "/tmp/w/hp"))     # tombstones the cwd
        for cwd_dep in ("pwd", "ls -1", "cd /tmp/w/hp", "cd ..", "cat notes",
                        "ls -1 sub"):
            self.assertIsNone(st.predict(3, cwd_dep),
                              f"cwd-dependent {cwd_dep!r} not BOT at a dead cwd")
        # absolute-arg commands are UNAFFECTED by the stale cwd
        st.fold(step("echo 'z_tok' > /tmp/w/abs", "", 0, "/tmp/w/hp"))
        self.assertEqual(st.predict(4, "cat /tmp/w/abs"),
                         {"output": "z_tok", "exit": 0, "cwd": "/tmp/w/hp"})
        # re-anchor: re-creating the cwd path makes the prologue's cd succeed again
        st.fold(step("mkdir /tmp/w/hp", "", 0, "/tmp/w/hp"))
        self.assertEqual(st.predict(5, "pwd"),
                         {"output": "/tmp/w/hp", "exit": 0, "cwd": "/tmp/w/hp"})
        self.assertEqual(st.predict(5, "cd /tmp/w/hp"),
                         {"output": "", "exit": 0, "cwd": "/tmp/w/hp"})

    def test_f1_stale_cwd_via_mv_and_ancestor(self):
        """F1: an mv of the cwd, and an rm -r of an ANCESTOR of the cwd, both latch
        the stale-cwd guard the same way (the whole subtree tombstones)."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("mkdir /tmp/w/a"))
        st.fold(step("cd /tmp/w/a", "", 0, "/tmp/w/a"))
        st.fold(step("mv /tmp/w/a /tmp/w/b", "", 0, "/tmp/w/a"))  # cwd vacated
        self.assertIsNone(st.predict(3, "pwd"))
        st2 = ShellState(mode="sst", error_templates=TMPL_BB)
        st2.fold(step("mkdir /tmp/w/hp"))
        st2.fold(step("mkdir /tmp/w/hp/sub"))
        st2.fold(step("cd /tmp/w/hp/sub", "", 0, "/tmp/w/hp/sub"))
        st2.fold(step("rm -r /tmp/w/hp", "", 0, "/tmp/w/hp/sub"))  # ancestor of cwd
        self.assertIsNone(st2.predict(4, "pwd"))
        self.assertIsNone(st2.predict(4, "ls -1"))

    def test_f2_grep_icase_ascii_only(self):
        """F2: real greps fold bytes/ASCII (C locale) — an icase match is determined
        only when the token AND the compared text are ASCII; Python str.lower
        over-folds Unicode. Case-sensitive grep is unaffected."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("echo 'seed_tok' > /tmp/w/s"))
        self.assertEqual(st.predict(1, "grep -F -i -m 8 'SEED_TOK' /tmp/w/s"),
                         {"output": "seed_tok", "exit": 0, "cwd": "/"})
        self.assertIsNone(st.predict(1, "grep -F -i -m 8 'SÉED' /tmp/w/s"),
                          "non-ASCII token icase determined")
        st.fold(step("echo 'GRÜN_tok' > /tmp/w/u"))
        self.assertIsNone(st.predict(2, "grep -F -i -m 8 'grün_tok' /tmp/w/u"))
        self.assertIsNone(st.predict(2, "grep -F -i -m 8 'B_TOK' /tmp/w/u"),
                          "ASCII token icase determined against non-ASCII content")
        # case-SENSITIVE grep over exactly-known content stays determined
        self.assertEqual(st.predict(2, "grep -F -m 8 'GRÜN' /tmp/w/u"),
                         {"output": "GRÜN_tok", "exit": 0, "cwd": "/"})

    def test_f3_template_exit_contract(self):
        """F3: shell_state consumes the {template|text, exit} shape and FAILS CLOSED
        on a string-only entry or a missing exit — never a guessed dialect code. A
        well-formed per-dialect exit is honored (GNU grep ENOENT=2)."""
        for tbl in ({"cat": "cat: {path}: No such file or directory"},   # string-only
                    {"cat": {"text": "cat: {path}: No such file or directory"}}):  # no exit
            st = ShellState(mode="sst", error_templates=tbl)
            st.fold(step("echo 'x' > /tmp/w/f"))
            st.fold(step("rm /tmp/w/f"))
            self.assertIsNone(st.predict(2, "cat /tmp/w/f"),
                              "served a determined error from a malformed entry")
        # {template, exit} with the GNU grep-ENOENT exit of 2, via an entailed-absent path
        gnu = {"grep": {"template": "grep: {path}: No such file or directory", "exit": 2}}
        st3 = ShellState(mode="sst", error_templates=gnu)
        st3.fold(step("mkdir /tmp/w/dd"))
        st3.fold(step("ls -a /tmp/w/dd", ".\n.."))          # entails the child slot absent
        self.assertEqual(st3.predict(2, "grep -F -m 8 tok /tmp/w/dd/miss"),
                         {"output": "grep: /tmp/w/dd/miss: No such file or directory",
                          "exit": 2, "cwd": "/"})

    def test_f4_failed_recursive_rm_folds_conservatively(self):
        """F4: a recursive rm that failed for a non-ENOENT reason gutted the tree
        depth-first before failing — belief about the subtree is stale. A template-
        matched ENOENT and any non-recursive failure remain true no-ops."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("mkdir /tmp/w/g"))
        st.fold(step("echo 'seed_tok' > /tmp/w/g/f"))
        self.assertEqual(st.predict(2, "cat /tmp/w/g/f"),
                         {"output": "seed_tok", "exit": 0, "cwd": "/"})
        st.fold(step("rm -r /tmp/w/g",
                     "rm: cannot remove '/tmp/w/g/f': Device or resource busy", 1))
        self.assertIsNone(st.predict(3, "cat /tmp/w/g/f"),
                          "served stale bytes after a partial-destruction rm -r")
        self.assertIsNone(st.predict(3, "ls -1 /tmp/w/g"))
        # a template-matched ENOENT recursive rm is a true no-op (mines absence only)
        st2 = ShellState(mode="sst", error_templates=TMPL_BB)
        st2.fold(step("echo 'x' > /tmp/w/keep"))
        st2.fold(step("rm -r /tmp/w/gone",
                      TMPL_BB["rm"]["text"].format(path="/tmp/w/gone"), 1))
        self.assertEqual(st2.predict(2, "cat /tmp/w/keep"),
                         {"output": "x", "exit": 0, "cwd": "/"})
        # a non-recursive failure never triggers the conservative forget
        st3 = ShellState(mode="sst", error_templates=TMPL_BB)
        st3.fold(step("mkdir /tmp/w/d3"))
        st3.fold(step("echo 'k_tok' > /tmp/w/d3/f"))
        st3.fold(step("rm /tmp/w/d3", "rm: '/tmp/w/d3': Is a directory", 1))
        self.assertEqual(st3.predict(3, "cat /tmp/w/d3/f"),
                         {"output": "k_tok", "exit": 0, "cwd": "/"})
        # slashed symlink arg: the failure resolves through the link and empties the
        # real dir — the fold forgets the RESOLVED dir, not just the link
        st4 = ShellState(mode="sst", error_templates=TMPL_BB)
        st4.fold(step("mkdir /tmp/w/g4"))
        st4.fold(step("echo 'seed_tok' > /tmp/w/g4/f"))
        st4.fold(step("ln -s /tmp/w/g4 /tmp/w/ld4"))
        st4.fold(step("rm -r /tmp/w/ld4/",
                      "rm: cannot remove '/tmp/w/ld4/': Not a directory", 1))
        self.assertIsNone(st4.predict(4, "cat /tmp/w/g4/f"),
                          "slashed-symlink rm -r left the resolved dir's bytes live")
        self.assertIsNone(st4.predict(4, "ls -1 /tmp/w/g4"))

    def test_f5_relative_find_replay_rebinds_on_cd(self):
        """F5: the R7 find-replay cache keys on the raw string, but a relative dir
        resolves against the CURRENT cwd — the replay is guarded on
        normpath(dir, self.cwd) == the cached resolved dir."""
        st = ShellState(mode="sst", error_templates=TMPL_BB)
        st.fold(step("cd /opt/fA", "", 0, "/opt/fA"))
        st.fold(step("find sub -maxdepth 1 -name '*.zz'", "sub/hit.zz", 0, "/opt/fA"))
        self.assertEqual(st.predict(2, "find sub -maxdepth 1 -name '*.zz'"),
                         {"output": "sub/hit.zz", "exit": 0, "cwd": "/opt/fA"})
        st.fold(step("cd /opt/fB", "", 0, "/opt/fB"))
        self.assertIsNone(st.predict(3, "find sub -maxdepth 1 -name '*.zz'"),
                          "replayed context-A's find output after a cd rebind")


# ================================================================ docker battery

def _harvest(box):
    """Probe-harvest the per-image error-template table inside the container
    (draft §3.5) — the same records the collector would mine, never hand-authored."""
    t = {}
    a, b = "/tmp/w/__nopeA__", "/tmp/w/__nopeB__"
    # F3 (round-6): carry the REAL per-dialect exit with each template — the canonical
    # {text, exit} shape (benchmarks/p0/error-templates.json). ls ENOENT is 1 on
    # busybox / 2 on GNU; cat/rm/mv ENOENT are 1 on both; grep ENOENT is 2 on both.
    for key, cmd, path in (("cat", f"cat {a}", a),
                           ("ls", f"ls {a}", a),
                           ("rm", f"rm {a}", a),
                           ("mv", f"mv {a} {b}", a)):
        r = box.run(cmd)
        if r["exit"] != 0 and path in r["output"]:
            t[key] = {"text": r["output"].replace(path, "{path}"), "exit": r["exit"]}
    r = box.run(f"cd {a}")
    if r["exit"] != 0 and a in r["output"]:
        t["cd"] = {"text": r["output"].replace(a, "{path}"), "exit": r["exit"]}
    return t


@unittest.skipUnless(HAVE_DOCKER,
                     "docker (or alpine:latest / debian:stable-slim) unavailable")
class TestDifferentialGoldenRule(unittest.TestCase):
    """The GOLDEN RULE on real shells: predict() is BOT or exactly reality —
    every trajectory runs on BOTH images."""

    boxes = None

    @classmethod
    def setUpClass(cls):
        from realenv.docker_env import DockerBox
        cls.boxes, cls.templates = {}, {}
        cls.det = cls.total = 0                       # determined-coverage tally
        for img in IMAGES:
            box = DockerBox(img, mem="256m")
            cls.boxes[img] = box
            box._exec("mkdir -p /tmp/w /opt")
            # image-space fixtures for V10/V11 (bootstrap, unrecorded)
            box._exec("printf 'obs' > /opt/nonl; ln -s /opt/missing11 /opt/dangling11")
            cls.templates[img] = _harvest(box)

    @classmethod
    def tearDownClass(cls):
        for box in (cls.boxes or {}).values():
            try:
                box.close()
            except Exception:
                pass
        if cls.total:
            print(f"\n[differential] determined coverage: {cls.det}/{cls.total} "
                  f"steps determined (both images, golden rule held on all)")

    # ------------------------------------------------------------ runner

    def _fresh(self, img):
        box = self.boxes[img]
        box._exec("rm -rf /tmp/w && mkdir -p /tmp/w")
        box.cwd = "/"
        return box, ShellState(mode="collection", error_templates=self.templates[img])

    def _run(self, box, st, rows):
        """rows: (cmd, expect) with expect in {'det','bot',None}; every step
        asserts the golden rule; 'det'/'bot' additionally pin coverage."""
        for cmd, expect in rows:
            pred = st.predict(st.vt, cmd)
            r = box.run(cmd)
            real = {"output": r["output"], "exit": r["exit"], "cwd": r["cwd"]}
            if pred is not BOT:
                self.assertEqual(
                    pred, real,
                    f"[{box.image}] GOLDEN-RULE violation at step {st.vt}: {cmd!r}")
            if expect == "det":
                self.assertIsNotNone(
                    pred, f"[{box.image}] coverage collapse (BOT) at {cmd!r}")
            elif expect == "bot":
                self.assertIsNone(
                    pred, f"[{box.image}] expected BOT at {cmd!r}, got {pred}")
            type(self).total += 1
            if pred is not BOT:
                type(self).det += 1
            st.fold({"cmd": cmd, **real})

    def _both(self, rows_fn):
        for img in IMAGES:
            box, st = self._fresh(img)
            self._run(box, st, rows_fn(img))

    # ------------------------------------------------------------ trajectories

    def test_v1_v2_redirect_through_symlink(self):
        self._both(lambda img: [
            ("echo 'old' > /tmp/w/t1", "det"),
            ("ln -s /tmp/w/t1 /tmp/w/l1", "bot"),        # link path never observed
            ("echo 'new' > /tmp/w/l1", "det"),           # chain fully SST-known
            ("cat /tmp/w/t1", "det"),                    # bytes went THROUGH the link
            ("readlink /tmp/w/l1", "det"),               # the link survived the write
            ("cat /tmp/w/l1", "det"),
            ("echo 'add' >> /tmp/w/l1", "det"),          # V2: append through the link
            ("cat /tmp/w/t1", "det"),
            ("cat /tmp/w/l1", "det"),
        ])

    def test_v3_redirect_onto_hardlink_peer(self):
        self._both(lambda img: [
            ("mkdir /tmp/w/d3", "bot"),                  # complete dir: ln stays det
            ("echo 'old' > /tmp/w/d3/f3", "det"),
            ("ln /tmp/w/d3/f3 /tmp/w/d3/h3", "det"),
            ("echo 'new' > /tmp/w/d3/h3", "det"),        # truncate = same inode
            ("cat /tmp/w/d3/f3", "det"),                 # peer sees the new bytes
            ("echo 'p_tok' >> /tmp/w/d3/f3", "det"),
            ("cat /tmp/w/d3/h3", "det"),
        ])

    def test_s4_redirect_through_dangling_workspace_symlink(self):
        self._both(lambda img: [
            ("ln -s /tmp/w/tgt4 /tmp/w/dl4", "bot"),
            ("echo 'x_tok' > /tmp/w/dl4", "det"),        # creates the TARGET
            ("cat /tmp/w/tgt4", "det"),
            ("readlink /tmp/w/dl4", "det"),              # link survives
            ("cat /tmp/w/dl4", "det"),
        ])

    def test_v4_mv_onto_own_hardlink(self):
        # busybox: rename(same inode) = rc 0 NO-OP, both names persist;
        # GNU: rc 1 error — the fold is conservative on both.
        self._both(lambda img: [
            ("mkdir /tmp/w/d4", "bot"),
            ("echo 'x_tok' > /tmp/w/d4/f4", "det"),
            ("ln /tmp/w/d4/f4 /tmp/w/d4/h4", "det"),
            ("mv /tmp/w/d4/f4 /tmp/w/d4/h4", "bot"),     # dialect-divergent outcome
            ("cat /tmp/w/d4/f4", "det"),                 # src must NOT be tombstoned
            ("cat /tmp/w/d4/h4", "det"),
        ])

    def test_v5_mv_dir_onto_into_itself(self):
        self._both(lambda img: [
            ("mkdir /tmp/w/d5", "bot"),
            ("mv /tmp/w/d5 /tmp/w/d5", "bot"),           # F6: EINVAL family
            ("mv /tmp/w/d5 /tmp/w/d5/sub", "bot"),
            ("ls -1 /tmp/w/d5", "det"),                  # dir intact and empty
        ])

    def test_s7_mv_trailing_slashes(self):
        self._both(lambda img: [
            ("echo 'x_tok' > /tmp/w/f7", "det"),
            ("mv /tmp/w/f7 /tmp/w/g7/", "bot"),          # g7 absent: fails, f7 intact
            ("cat /tmp/w/f7", "det"),
            ("mkdir /tmp/w/d7", "bot"),
            ("mv /tmp/w/d7/ /tmp/w/e7", "bot"),          # slashed src, unknown dst
        ])

    def test_v6_ls_ld_self_row(self):
        self._both(lambda img: [
            ("mkdir /tmp/w/d6", "bot"),
            ("cd /tmp/w", "det"),
            ("ls -ld d6", "bot"),                        # renders the dir ITSELF
            ("rm /tmp/w/d6/d6", "det"),                  # no phantom: entailed-absent
            ("mv /tmp/w/d6/d6 /tmp/w/z6", "det"),        # both via harvested templates
            ("cd /tmp/w/d6/sub", "det"),                 # entailed-dead cd: template
        ])

    def test_v7_cd_through_symlinked_dir(self):
        self._both(lambda img: [
            ("mkdir /tmp/w/d7", "bot"),
            ("ln -s /tmp/w/d7 /tmp/w/ld7", "bot"),
            ("cd /tmp/w/ld7", "det"),                    # F9: cwd is LOGICAL
            ("pwd", "det"),
            ("cd ..", "det"),                            # textual: back to /tmp/w
            ("pwd", "det"),
        ])

    def test_v15_cd_double_slash_logical(self):
        self._both(lambda img: [
            ("cd //", "det"),                            # pwd prints '//'
            ("pwd", "det"),
            ("cd //tmp", "det"),                         # '//' root is sticky
            ("pwd", "det"),
            ("cd /tmp//w", "det"),                       # internal '//' collapses
            ("pwd", "det"),
        ])

    def test_v8_dotfile_absence_entailment(self):
        self._both(lambda img: [
            ("ls -1 /root", "bot"),
            ("cat /root/.bashrc", "bot"),                # F2: dotfiles never entailed
            ("cat /root/__nodot__", "det"),              # non-dot entailment stays
        ])

    def test_v10_cat_producer_trailing_newline(self):
        self._both(lambda img: [
            ("cat /opt/nonl", "bot"),                    # first observation
            ("cat /opt/nonl > /tmp/w/f10", "det"),
            ("cat /tmp/w/f10", "det"),                   # render-known either way
            ("echo 'x_tok' >> /tmp/w/f10", "det"),       # ack only
            ("cat /tmp/w/f10", "bot"),                   # F8: 'obsx' vs 'obs\\nx'
        ])

    def test_v11_dangling_image_symlink(self):
        self._both(lambda img: [
            ("cat /opt/dangling11", "bot"),              # template-matched miss
            ("cat /opt/dangling11", None),               # replay may or may not fire
            ("readlink /opt/dangling11", "bot"),         # real: target, exit 0
            ("ln -s /elsewhere /opt/dangling11", "bot"), # real: File exists, exit 1
            ("rm /opt/dangling11", "det"),               # the FOLDED readlink taught
                                                         # the link: rm of a known
                                                         # symlink is determined-ok
        ])

    def test_v12_image_symlink_observed_via_cat(self):
        self._both(lambda img: [
            ("cat /etc/os-release", "bot"),
            ("readlink /etc/os-release", "bot"),         # link-ness never learned
            ("cat /etc/hostname", "bot"),
            ("readlink /etc/hostname", "bot"),
        ])

    def test_v14_find_replay_vs_removed_ancestor(self):
        def rows(img):
            box = self.boxes[img]
            box._exec("mkdir -p /tmp/w/s14a/aa")         # bootstrap, unrecorded
            return [
                ("find /tmp/w/s14a/aa -maxdepth 1 -name '*.zz'", "bot"),
                ("find /tmp/w/s14a/aa -maxdepth 1 -name '*.zz'", "det"),   # replay
                ("rm -r /tmp/w/s14a", "bot"),
                ("find /tmp/w/s14a/aa -maxdepth 1 -name '*.zz'", "bot"),   # F11
            ]
        self._both(rows)

    def test_v17_trailing_slash_redirect_dst(self):
        self._both(lambda img: [
            ("echo 'old' > /tmp/w/f17", "det"),
            ("echo 'x' > /tmp/w/f17/", "bot"),           # fails; f17 untouched
            ("cat /tmp/w/f17", "det"),                   # still 'old'
            ("echo 'y' >> /tmp/w/f17/", "bot"),
            ("cat /tmp/w/f17", "det"),
        ])

    def test_s15_workspace_dotfiles(self):
        self._both(lambda img: [
            ("mkdir /tmp/w/s15", "bot"),
            ("echo 'hidden' > /tmp/w/s15/.h", "det"),
            ("ls -1 /tmp/w/s15", "det"),                 # dotfile hidden
            ("ls -a /tmp/w/s15", "det"),
            ("cat /tmp/w/s15/.h", "det"),
        ])

    def test_s17_test_f_through_link(self):
        self._both(lambda img: [
            ("echo 'body' > /tmp/w/t17", "det"),
            ("ln -s /tmp/w/t17 /tmp/w/l17", "bot"),
            ("[ -f /tmp/w/l17 ] && cat /tmp/w/l17", "det"),
            ("readlink /tmp/w/l17", "det"),
        ])

    def test_s18_mv_onto_symlink_to_file(self):
        self._both(lambda img: [
            ("echo 'tgt' > /tmp/w/t18", "det"),
            ("ln -s /tmp/w/t18 /tmp/w/l18", "bot"),
            ("echo 'src' > /tmp/w/s18", "det"),
            ("mv /tmp/w/s18 /tmp/w/l18", "bot"),         # replaces the LINK
            ("cat /tmp/w/t18", "det"),                   # target untouched
            ("readlink /tmp/w/l18", "det"),              # now a plain file: exit 1
            ("cat /tmp/w/l18", "det"),
        ])

    def test_happy_path_cud_chain(self):
        """The determined R2–R8 backbone on real shells: create/read/append/
        list/mv/link/rm with template-exact misses (det-coverage anchor)."""
        self._both(lambda img: [
            ("mkdir /tmp/w/hp", "bot"),
            ("echo 'alpha_tok' > /tmp/w/hp/n.txt", "det"),
            ("cat /tmp/w/hp/n.txt", "det"),
            ("echo 'beta_tok' >> /tmp/w/hp/n.txt", "det"),
            ("cat /tmp/w/hp/n.txt", "det"),
            ("ls -1 /tmp/w/hp", "det"),
            ("cd /tmp/w/hp", "det"),
            ("pwd", "det"),
            ("mv n.txt m.txt", "det"),
            ("cat m.txt", "det"),
            ("cat n.txt", "det"),                        # template-exact miss
            ("ln /tmp/w/hp/m.txt /tmp/w/hp/h", "det"),
            ("rm /tmp/w/hp/m.txt", "det"),
            ("cat /tmp/w/hp/h", "det"),                  # hardlink kept the bytes
            ("readlink /tmp/w/hp/h", "det"),             # SST-created file: exit 1
            ("cd ..", "det"),
            ("rm -r /tmp/w/hp", "det"),
            ("cat /tmp/w/hp/h", "det"),                  # template-exact miss
        ])

    # ------------------------------------------------ round-4 battery rows

    def test_r4_deep_chain(self):
        """3-hop symlink chains: read/append/rm/readlink through the chain and
        touch-creation through a dangling chain (round-4 verified-sound edges)."""
        self._both(lambda img: [
            ("echo 'body_tok' > /tmp/w/f", "det"),
            ("ln -s /tmp/w/f /tmp/w/l1", "bot"),
            ("ln -s /tmp/w/l1 /tmp/w/l2", "bot"),
            ("ln -s /tmp/w/l2 /tmp/w/l3", "bot"),
            ("cat /tmp/w/l3", "det"),
            ("readlink /tmp/w/l3", "det"),
            ("echo 'x2_tok' >> /tmp/w/l3", "det"),
            ("cat /tmp/w/f", "det"),
            ("rm /tmp/w/l1", "det"),
            ("cat /tmp/w/l3", "det"),                    # template-exact miss
            ("readlink /tmp/w/l2", "det"),
            ("touch /tmp/w/l3", "det"),                  # O_CREAT re-creates l1
            ("cat /tmp/w/l3", "det"),
        ])

    def test_r4_hard_group(self):
        """3-member hardlink group spanning mv/append/rm/truncate (round-4)."""
        self._both(lambda img: [
            ("mkdir /tmp/w/g", "bot"),
            ("echo 'seed_tok' > /tmp/w/g/f", "det"),
            ("ln /tmp/w/g/f /tmp/w/g/h1", "det"),
            ("ln /tmp/w/g/h1 /tmp/w/g/h2", "det"),
            ("mv /tmp/w/g/h1 /tmp/w/g/r1", "det"),
            ("cat /tmp/w/g/r1", "det"),
            ("echo 'add_tok' >> /tmp/w/g/r1", "det"),
            ("cat /tmp/w/g/f", "det"),
            ("rm /tmp/w/g/f", "det"),
            ("cat /tmp/w/g/h2", "det"),
            ("echo 'trunc_tok' > /tmp/w/g/h2", "det"),
            ("cat /tmp/w/g/r1", "det"),
            ("ls -1 /tmp/w/g", "det"),
            ("rm -r /tmp/w/g", "det"),
            ("cat /tmp/w/g/r1", "det"),                  # template-exact miss
        ])

    def test_r4_alias_staleness(self):
        """Round-4 class 3 live: content folded onto a linkness-unknown alias path
        must go BOT after rm of the node holding the same bytes."""
        self._both(lambda img: [
            ("mkdir /tmp/w/m", "bot"),
            ("echo 'out_tok' > /tmp/w/out", "det"),
            ("ln -s /tmp/w/out /tmp/w/m/abs", "det"),
            ("ln -s ../out /tmp/w/m/rel", "det"),
            ("cat /tmp/w/m/abs", "det"),
            ("cat /tmp/w/m/rel", "det"),
            ("mv /tmp/w/m /tmp/w/n", "bot"),             # conservative fold
            ("cat /tmp/w/n/abs", "bot"),                 # folds mined content
            ("cat /tmp/w/n/rel", "bot"),
            ("readlink /tmp/w/n/rel", "bot"),
            ("rm /tmp/w/out", "det"),
            ("cat /tmp/w/n/abs", "bot"),                 # THE class-3 fix: stale
        ])

    def test_r4_boundary_dotdot(self):
        """Round-4 class 2 live: '..' through a missing intermediate is ENOENT in
        reality (BOT), never corrupts belief about the collapsed path, and stays
        determined through a known-live dir; cd remains logical."""
        self._both(lambda img: [
            ("echo 'x_tok' > /tmp/w/bf", "det"),
            ("cat /tmp/w/nope/../bf", "bot"),            # real ENOENT
            ("cat /tmp/w/bf", "det"),                    # belief NOT tombstoned
            ("echo 'y_tok' > /tmp/w/nope2/../bf2", "bot"),
            ("cd /tmp/w/gone/..", "bot"),                # round-7: bash validates '..'
            ("touch /tmp/w/gone3/../bf", "bot"),
            ("mkdir /tmp/w/d8", "bot"),
            ("cat /tmp/w/d8/../bf", "det"),              # known-live intermediate
        ])

    def test_r4_kindless_and_phantom(self):
        """Round-4 class 4 live: kind-unknown mined nodes never get determined
        rm acks; ls -l space-rows never mint phantom children."""
        def rows(img):
            box = self.boxes[img]
            box._exec("mkdir -p /opt/r4box/sub /opt/r4fx/dd /opt/r4sp && "
                      "echo native > /opt/r4box/nf && echo x > '/opt/r4sp/a b'")
            return [
                ("ls -1 /opt/r4box", "bot"),
                ("rm /opt/r4box/sub", "bot"),            # real: Is a directory
                ("find /opt/r4fx -maxdepth 1 -name 'dd'", "bot"),
                ("rm /opt/r4fx/dd", "bot"),              # real: Is a directory
                ("ls -l /opt/r4sp", "bot"),
                ("rm /opt/r4sp/b", "bot"),               # real ENOENT (no phantom)
            ]
        self._both(rows)

    def test_r4_hard_ln_dir_and_cycles(self):
        """Round-4 classes 4/6 live: hard ln onto a dir fails; symlink cycles
        are ELOOP — all BOT."""
        self._both(lambda img: [
            ("mkdir /tmp/w/hd", "bot"),
            ("mkdir /tmp/w/hp", "bot"),
            ("ln /tmp/w/hd /tmp/w/hp/h", "bot"),         # real exit 1
            ("ln -s /tmp/w/cy2 /tmp/w/cy1", "bot"),
            ("ln -s /tmp/w/cy1 /tmp/w/cy2", "bot"),
            ("touch /tmp/w/cy1", "bot"),                 # real ELOOP exit 1
            ("ln -s /tmp/w/self /tmp/w/self", "bot"),
            ("touch /tmp/w/self", "bot"),                # real ELOOP exit 1
        ])

    def test_r4_writability_and_volatile(self):
        """Round-4 classes 5/8 live: bind-mount rm (EBUSY) and /proc mutations
        are BOT; /proc content is never re-served; cat-only -f evidence is BOT."""
        self._both(lambda img: [
            ("cat /etc/hosts", "bot"),
            ("rm /etc/hosts", "bot"),                    # real: Resource busy
            ("cd /proc", "bot"),
            ("touch /proc/zzz9", "bot"),                 # real: read-only fs
            ("cat /proc/uptime", "bot"),
            ("cat /proc/uptime", "bot"),                 # volatile: never re-served
            ("cat /dev/null", "bot"),
            ("[ -f /dev/null ] && cat /dev/null", "bot"),   # real exit 1 (chardev)
        ])

    def test_r4_sound_edges(self):
        """Round-4 verified-sound det anchors: head/tail 0-and-k windows, grep -m,
        cond -s on known and empty files, pipes, dotfile-free ls -a."""
        self._both(lambda img: [
            ("mkdir /tmp/w/se", "bot"),
            ("echo 'l1_tok' > /tmp/w/se/t", "det"),
            ("echo 'l2_tok' >> /tmp/w/se/t", "det"),
            ("echo 'l3_zz' >> /tmp/w/se/t", "det"),
            ("head -n 0 /tmp/w/se/t", "det"),
            ("tail -n 0 /tmp/w/se/t", "det"),
            ("head -n 2 /tmp/w/se/t", "det"),
            ("tail -n 1 /tmp/w/se/t", "det"),
            ("grep -F -m 8 tok /tmp/w/se/t", "det"),
            ("grep -F -i -m 8 TOK /tmp/w/se/t", "det"),
            ("[ -s /tmp/w/se/t ] && cat /tmp/w/se/t", "det"),
            ("touch /tmp/w/se/e0", "det"),
            ("[ -s /tmp/w/se/e0 ] && cat /tmp/w/se/e0", "det"),
            ("ls -a /tmp/w/se", "det"),
            ("cat /tmp/w/se/t | grep -F -m 8 l", "det"),
            ("ls -1 /tmp/w/se | tail -n 2", "det"),
            ("cat /tmp/w/se/t", "det"),
        ])

    # ------------------------------------------------ round-5 battery rows

    def test_r5_severed_hardgroup(self):
        """Round-5 F1 live (M2): a conservative mv severs the hardgroup; an
        append through the vacated side changes the shared inode — the peer's
        bytes must be BOT, never the pre-severing content."""
        self._both(lambda img: [
            ("mkdir /tmp/w/d2", "bot"),
            ("echo 'seed_tok' > /tmp/w/d2/f", "det"),
            ("ln /tmp/w/d2/f /tmp/w/peer", None),
            ("mv /tmp/w/d2 /tmp/w/e2", "bot"),           # dst unknown: conservative
            ("echo 'add_tok' >> /tmp/w/e2/f", None),
            ("cat /tmp/w/peer", "bot"),                  # real: seed_tok\nadd_tok
            ("cat /tmp/w/d2/f", "det"),                  # certain tombstone miss
        ])

    def test_r5_absence_revival(self):
        """Round-5 F3 live (E1 order B): absence mined at a dangling image link
        must not outlive a creation that makes the link resolve."""
        def rows(img):
            box = self.boxes[img]
            box._exec("rm -rf /opt/e1dl /opt/e1missing; "
                      "ln -s /opt/e1missing /opt/e1dl")   # bootstrap, unrecorded
            return [
                ("cat /opt/e1dl", "bot"),                # absence mined at the LINK
                ("echo 'pay_tok' > /tmp/w/pp", "det"),   # creation drops it (F3)
                ("mv /tmp/w/pp /opt/e1missing", "bot"),
                ("cat /opt/e1dl", "bot"),                # real: pay_tok, exit 0
            ]
        self._both(rows)

    def test_r5_dir_alias_staleness(self):
        """Round-5 F4 live (E2): a listing mined through an unknown symlink-to-dir
        dies with the real dir (rm -r and the mv variant)."""
        def rows(img):
            box = self.boxes[img]
            box._exec("rm -rf /opt/e2real /opt/e2ld /opt/e2moved; "
                      "mkdir -p /opt/e2real; touch /opt/e2real/k1 /opt/e2real/k2; "
                      "ln -s /opt/e2real /opt/e2ld")      # bootstrap, unrecorded
            return [
                ("ls -a /opt/e2ld", "bot"),
                ("ls -a /opt/e2ld", "det"),              # replay while the dir is real
                ("rm -r /opt/e2real", "bot"),
                ("[ -d /opt/e2ld ] && ls -1 /opt/e2ld", "bot"),   # kind invalidated
                ("ls -a /opt/e2ld", "bot"),              # real: error / bare name
                                                         # (the folded miss may then
                                                         # legitimately re-mine absence)
            ]
        self._both(rows)
        def rows_mv(img):
            box = self.boxes[img]
            box._exec("rm -rf /opt/e2real /opt/e2ld /opt/e2moved; "
                      "mkdir -p /opt/e2real; touch /opt/e2real/k1 /opt/e2real/k2; "
                      "ln -s /opt/e2real /opt/e2ld")
            return [
                ("ls -a /opt/e2ld", "bot"),
                ("mv /opt/e2real /opt/e2moved", "bot"),
                ("ls -a /opt/e2ld", "bot"),              # real: error / bare name
            ]
        self._both(rows_mv)

    def test_r5_find_cache_raw_key(self):
        """Round-5 F6 live (Q2): two in-universe find commands whose UNQUOTED
        flattenings collide must not share a replay cache entry."""
        def rows(img):
            box = self.boxes[img]
            box._exec("rm -rf /tmp/w/A && mkdir -p /tmp/w/A && touch /tmp/w/A/B")
            return [
                ("find /tmp/w/A -maxdepth 1 -name 'B -maxdepth 1 -name z'", "bot"),
                ("find '/tmp/w/A -maxdepth 1 -name B' -maxdepth 1 -name 'z'", "bot"),
            ]
        self._both(rows)

    def test_f14_parent_listing_mask_on_real_render(self):
        """F14 against the REAL -l render shapes of both images: the workspace
        row in /tmp's listing masks after a workspace mutation."""
        for img in IMAGES:
            box, st = self._fresh(img)
            self._run(box, st, [("mkdir /tmp/w/dm", "bot")])
            r = box.run("ls -l /tmp")
            rec = {"cmd": "ls -l /tmp", "output": r["output"],
                   "exit": r["exit"], "cwd": r["cwd"]}
            got = RC.canon(rec, st)
            w_rows = [ln for ln in got["output"].split("\n")
                      if ln.split() and ln.split()[-1] == "w"]
            self.assertTrue(w_rows, f"[{img}] no 'w' row in: {got['output']!r}")
            for ln in w_rows:
                self.assertIn(RC.LS_TIME_TOKEN, ln,
                              f"[{img}] unmasked touched-child row: {ln!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
