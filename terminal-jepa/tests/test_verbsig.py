"""Unit smoke for realenv/verbsig.py — the frozen v3 sig/mode/cell labeler.

Covers: every atomic verb, every one of the 11 composed families, the audited process
forms, v1/v2 first-token bit-identity (vs seq_worldmodel.verb_of), the documented
universe exclusions (fail-closed), the frozen mode rules, and cell-key construction
incl. the created/created-obs split. Run: uv run python -m unittest tests.test_verbsig
"""

import unittest

from realenv import verbsig as V


class TestSig(unittest.TestCase):
    def test_every_atomic_verb(self):
        cases = {
            "uname -a": "uname", "cd /etc": "cd", "ls -la /etc": "ls",
            "cat /etc/os-release": "cat", "head -n 5 /etc/passwd": "head",
            "tail -n 3 /etc/passwd": "tail", "stat -c '%n %s %F %a' /etc/hosts": "stat",
            "find /etc -maxdepth 2 -type f -name 'os-*'": "find",
            "grep -F -m 8 'localhost' /etc/hosts": "grep",
            "grep -F -i -m 8 'tok' /etc/hosts": "grep",  # v2 hit-arm -i variant
            "pwd": "pwd", "echo tok_bare": "echo", "rm /tmp/w/notes.txt": "rm",
            "mv /etc/foo.conf /etc/foo.conf.bak": "mv",
            "ln -s /etc/hosts /tmp/w/l1": "ln", "ln /etc/hosts /tmp/w/h1": "ln",
            "readlink /tmp/w/l1": "readlink", "mkdir /tmp/w/d1": "mkdir",
            "touch /tmp/w/t1": "touch",
            "/usr/local/bin/tj3-ps": "ps",       # bare 'ps' is OUT (Annex P0 UD-9d)
            "/usr/local/bin/tj3-ps -o pid,stat,args": "ps",
            "kill 110": "kill", "kill -STOP 110": "kill", "kill -CONT 110": "kill",
            "kill -9 110": "kill", "kill -0 110": "kill", "uptime": "uptime",
            "sleep 1": "sleep",
            "after 1 3 'echo tok_a >> /tmp/w/task1.log' & echo $!": "after",
        }
        for cmd, want in cases.items():
            self.assertEqual(V.sig(cmd), want, cmd)
        # every atomic verb exercised
        self.assertEqual(set(cases.values()), set(V.ATOMIC_VERBS))

    def test_every_composed_family(self):
        cases = {
            "ls -1 /etc | head -n 4": "pipe:ls|head",
            "ls -1 /usr/lib | tail -n 6": "pipe:ls|tail",
            "ls -1 /usr/lib | grep -F -m 8 'so'": "pipe:ls|grep",
            "cat /etc/hosts | head -n 2": "pipe:cat|head",
            "cat /etc/passwd | tail -n 3": "pipe:cat|tail",
            "cat /etc/hosts | grep -F -m 8 localhost": "pipe:cat|grep",
            "echo 'tok_alpha' > /tmp/w/notes.txt": "redir:echo>",
            "echo 'tok_alpha' >> /tmp/w/notes.txt": "redir:echo>",
            "ls -1 /usr/lib > /tmp/w/libs.txt": "redir:prod>",
            "cat /etc/os-release >> /tmp/w/os.txt": "redir:prod>",
            "[ -f /tmp/w/notes.txt ] && cat /tmp/w/notes.txt": "cond:cat",
            "[ -e /etc ] && ls -1 /etc": "cond:ls",
            "[ -d /etc ] && ls -1 /etc": "cond:ls",
            "[ -s /tmp/w/a.txt ] && head -n 3 /tmp/w/a.txt": "cond:head",
        }
        for cmd, want in cases.items():
            self.assertEqual(V.sig(cmd), want, cmd)
        self.assertEqual(set(cases.values()), set(V.COMPOSED_SIGS))
        self.assertEqual(len(V.COMPOSED_SIGS), 11)

    def test_v1_v2_first_token_bit_identity(self):
        from realenv import seq_worldmodel as M
        v2_shapes = [
            "uname -a", "uname", "cd ..", "cd /", "cd /etc", "ls", "ls -la /etc",
            "ls -lt /usr/lib", "cat /etc/os-release", "head -n 5 /etc/passwd",
            "tail -n 12 /var/log/dpkg.log", "stat -c '%n %s %F %a' /etc/hosts",
            "find /etc -maxdepth 2 -type f -name 'os-*'",
            "grep -F -m 8 'localhost' /etc/hosts",
        ]
        for cmd in v2_shapes:
            self.assertEqual(V.sig(cmd), M.verb_of(cmd), cmd)

    def test_out_of_universe_raises(self):
        bad = [
            "", "  ", "wget http://x", "jobs", "fg", "bg %1", "wait",  # UD-1 exclusions
            "kill -INT 110", "kill -TERM abc", "kill",                 # kill family
            "ps", "ps -ef",                                            # bare ps (UD-9d)
            "/usr/local/bin/tj3-ps -ef",                               # non-frozen template
            "sleep 5",                                                 # sleep is {0,1}
            "ps | grep sleep",                                         # ps in pipes
            "ls -l /etc | head -n 2",                                  # ls -l producer
            "find /etc | head -n 2",                                   # find producer
            "cat /etc/hosts | head -n 2 | tail -n 1",                  # depth-2
            "cat /etc/hosts | wc -l",                                  # count filter
            "cat a; ls", "cat a || ls", "cat a && ls",                 # ; || bare-&&
            "grep -F x <<< y", "cat < /etc/hosts",                     # <<< REDIR_IN
            "[ -x /bin/sh ] && cat /bin/sh",                           # TESTOP not frozen
            "[ -f /tmp/w/a ] && grep -F x /tmp/w/a",                   # READ not frozen
            "[ -f /tmp/w/a ] && cat /tmp/w/b",                         # P mismatch
            "[ -f /tmp/w/a ]",                                         # bare test
            "echo 'x' > /etc/hosts",                                   # non-WSF target
            "echo x > /tmp/w/a.txt",                                   # unquoted payload
            "cd /etc && ls",                                           # cd in composed
            "ls -1 /etc | head -n 2 > /tmp/w/x",                       # two operators
            "echo `id`", "echo $HOME", "echo 'a", "sleep 1 &",         # subst/quotes/&
            "after 1 3 'echo x >> /tmp/w/t.log'",                      # non-canonical after
            "cat a\nls",
        ]
        for cmd in bad:
            with self.assertRaises(ValueError, msg=cmd):
                V.sig(cmd)

    def test_composed_verb_alias(self):
        self.assertIs(V.composed_verb, V.sig)


class TestMode(unittest.TestCase):
    def test_read_rule_generalized_grep_miss(self):
        for v in ("ls", "cat", "grep", "find", "readlink", "ps", "pipe:cat|grep",
                  "cond:ls"):
            self.assertEqual(V.mode(v, 0, False), "hit", v)
            self.assertEqual(V.mode(v, 0, True), "miss", v)   # empty => miss (v2 rule)
            self.assertEqual(V.mode(v, 1, False), "miss", v)  # exit!=0 => miss
            self.assertEqual(V.mode(v, 2, True), "miss", v)

    def test_state_rule_mutation_ack_shape(self):
        for v in ("cd", "rm", "mv", "ln", "mkdir", "touch", "kill", "sleep"):
            self.assertEqual(V.mode(v, 0, True), "ok", v)     # exit 0 + empty = success
            self.assertEqual(V.mode(v, 1, False), "miss", v)  # error text folded in
            self.assertEqual(V.mode(v, 0, False), "miss", v)  # anomalous => excluded side

    def test_launch_rule(self):
        self.assertEqual(V.mode("after", 0, False), "ok")  # cpid echoed
        self.assertEqual(V.mode("after", 0, True), "miss")
        self.assertEqual(V.mode("after", 1, False), "miss")

    def test_const_ok_families(self):
        for s in ("pipe:ls|head", "pipe:ls|tail", "pipe:cat|head", "pipe:cat|tail",
                  "redir:echo>", "redir:prod>"):
            for e, emp in ((0, False), (0, True), (1, False)):
                self.assertEqual(V.mode(s, e, emp), "ok", s)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            V.mode("wget", 0, False)

    def test_modes_table_total(self):
        self.assertEqual(set(V.MODES), set(V.SIGS))
        for s, ms in V.MODES.items():
            self.assertIn(ms, (("ok",), ("hit", "miss"), ("ok", "miss")), s)


class TestCell(unittest.TestCase):
    def test_basic_keys(self):
        self.assertEqual(V.cell("cat", "hit", "native"), "cat|hit|native")
        self.assertEqual(V.cell("rm", "ok", "mutated"), "rm|ok|mutated")
        self.assertEqual(V.cell("pipe:ls|head", "ok", "native"), "pipe:ls|head|ok|native")

    def test_created_split_on_ws_observed(self):
        self.assertEqual(V.cell("cat", "hit", "created", ws_observed=True),
                         "cat|hit|created-obs")
        self.assertEqual(V.cell("cat", "hit", "created", ws_observed=False),
                         "cat|hit|created")
        with self.assertRaises(ValueError):
            V.cell("cat", "hit", "created")  # ws_observed required

    def test_ws_observed_ignored_off_created(self):
        self.assertEqual(V.cell("cat", "hit", "native", ws_observed=True),
                         "cat|hit|native")

    def test_validation(self):
        with self.assertRaises(ValueError):
            V.cell("wget", "hit", "native")
        with self.assertRaises(ValueError):
            V.cell("redir:echo>", "hit", "created", ws_observed=True)  # mode not in set
        with self.assertRaises(ValueError):
            V.cell("cat", "ok", "native")  # read verbs have no "ok"
        with self.assertRaises(ValueError):
            V.cell("cat", "hit", "workspace")  # unknown scope


if __name__ == "__main__":
    unittest.main()
