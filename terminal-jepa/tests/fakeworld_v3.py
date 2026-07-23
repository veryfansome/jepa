"""A faithful in-memory Linux-ish world for the v3 collector smoke (docker-free).

Independent of the SST: models an image fs + a Tier-W workspace, all v3 mutations,
composition, and just enough of the process arm (after registers a job effect; the
fire-script prologue writes the task-log; kill/ps are stored from the SST so their
FakeWorld outputs are ignored). Reads/mutations return REAL bytes so the fresh-SST
replay genuinely cross-checks collector <-> tracker semantics.
"""

import fnmatch
import re

from realenv.shell_state import (basename_of, logical_cwd, normpath, parent_of)

_QRE = re.compile(r"'((?:[^']|'\\'')*)'")


def _unq(tok):
    if len(tok) >= 2 and tok[0] == "'" and tok[-1] == "'":
        return tok[1:-1].replace("'\\''", "'")
    return tok.replace("'\\''", "'")


class FakeWorld:
    # per-instantiation counter: the ls -l mtime VARIES across instances so two same-seed
    # mints produce DIFFERENT raw `-l` bytes — the collector's store-time canonicalizer is
    # then the only thing that makes them byte-identical, so the docker-free determinism
    # test genuinely exercises the SEVERE-1 fix (a real container-creation mtime differs
    # between two back-to-back mints exactly this way).
    _n_instances = 0

    def __init__(self, image, error_templates):
        FakeWorld._n_instances += 1
        self._mtime_nonce = FakeWorld._n_instances
        self.image = image
        self.tmpl = error_templates
        self.cwd = "/"
        # regular files: path -> content str ; dirs: set ; symlinks: path -> target
        self.content = {}
        self.dirset = set()
        self.symlink = {}
        self.groups = {}          # hardlink groups: path -> set(paths sharing bytes)
        self.jobs = {}            # j -> (token, logpath)
        self._v2_state = None     # populated by collect_docker._v2_probe (cache slot)
        self._seed_image()

    # ---- image seed ----

    def _seed_image(self):
        # a small, deterministic, dotfile-free image tree (content is arbitrary — the SST
        # BOTs image reads, so only listings/stability are cross-checked)
        dirs = ["/etc", "/etc/ssl", "/etc/conf.d", "/usr", "/usr/lib", "/usr/bin",
                "/usr/share", "/var", "/var/log", "/tmp"]
        files = {
            "/etc/os-release": "NAME=fake\nVERSION_ID=1\nID=fake\nPRETTY_NAME=Fake\n" * 3,
            "/etc/hostname": "fakebox\n",
            "/etc/hosts": "127.0.0.1 localhost\n::1 localhost\n" * 4,
            "/etc/issue": "Fake Linux \\n \\l\n",
            "/etc/profile": "export PATH=/usr/bin\numask 022\n" * 5,
            "/etc/shells": "/bin/sh\n/bin/ash\n" * 3,
            "/etc/conf.d/net.conf": "listen 80\nserver main\nworker 4\n" * 6,
            "/etc/ssl/openssl.cnf": "ssl on\nmodule kernel\ncache 1\n" * 8,
            "/usr/lib/libfoo.so": "binary_lib_content\ncharset utf8\n" * 10,
            "/usr/lib/os-release": "usr copy\nID=fake\n" * 4,
            "/usr/bin/tool.sh": "#!/bin/sh\necho tool\nalias x=y\n" * 5,
            "/usr/share/readme.txt": "readme root daemon nologin localhost\n" * 7,
            "/var/log/boot.log": "boot ok\nmount root\nsystemd start\n" * 9,
        }
        self.dirset = set(dirs) | {"/", "/proc", "/sys", "/dev"}
        # model /proc's VOLATILE PID children (live pids that churn between reads): they must
        # NEVER reach a recorded byte — `-R` is dropped and /proc//sys//dev are excluded from
        # every ls/find/cd target pool (SEVERE-2). test_no_volatile_fs_in_recorded_output
        # asserts no /proc/<pid> literal and no recursive listing survives.
        for pid in ("1", "204", "216"):
            self.dirset.add("/proc/" + pid)
        self.content = dict(files)

    def image_dirs_files(self):
        return sorted(self.dirset), sorted(self.content)

    # ---- path helpers ----

    def _resolve(self, path, hops=0):
        if hops > 40:
            return path
        if path in self.symlink:
            return self._resolve(normpath(self.symlink[path], parent_of(path)), hops + 1)
        return path

    def _is_dir(self, path):
        return self._resolve(path) in self.dirset

    def _is_file(self, path):
        return self._resolve(path) in self.content

    def _exists(self, path):
        r = self._resolve(path)
        return r in self.content or r in self.dirset or path in self.symlink

    def _children(self, d):
        d = d.rstrip("/") or "/"
        pref = "/" if d == "/" else d + "/"
        names = set()
        for p in list(self.content) + list(self.dirset) + list(self.symlink):
            if p.startswith(pref):
                rest = p[len(pref):]
                if rest and "/" not in rest:
                    names.add(rest)
        return names

    # ---- error rendering (matches the SST's templates by construction) ----

    def _err(self, key, path=None, pid=None):
        e = self.tmpl.get(key)
        if not e:
            return f"{key}: error", 1
        t = e["template"]
        try:
            out = t.format(path=path, pid=pid)
        except (KeyError, IndexError, ValueError):
            out = t
        return out, e["exit"]

    # ---- cp_in / _exec (bootstrap + v2 probe) ----

    def cp_in(self, src, dst):
        if dst.endswith("busybox-tj3"):
            return True
        # workspace seed: copy image file bytes to the ws path
        if src in self.content:
            self.content[dst] = self.content[src]
            self.dirset.add(parent_of(dst))
            return True
        self.content.setdefault(dst, "seed\n")
        return True

    def _exec(self, script, extra_timeout=0):
        s = script
        if s.startswith("for t in head tail stat find grep"):
            return "head\ntail\nstat\nfind\ngrep\n", "", 0
        if s.startswith("for g in "):
            d = re.search(r"find '([^']*)' -maxdepth 3", s).group(1)
            lines = []
            for g in _GLOBS:
                hit = self._find(d, 3, None, g)
                if hit:
                    lines.append(f"{g}|{hit[0]}")
            return ("\n".join(lines) + "\n") if lines else "", "", 0
        paths = [p for p in re.findall(r"'([^']*)'", s) if p.startswith("/")]
        if s.startswith(("stat -c '%s %n' ", "wc -c ")):
            return "".join(f"{len(self.content[p])} {p}\n"
                           for p in paths if p in self.content), "", 0
        if s.startswith("wc -l "):
            return "".join(f"{self.content[p].count(chr(10))} {p}\n"
                           for p in paths if p in self.content), "", 0
        # bootstrap scripts
        if "echo ok" in s:
            return "ok\n", "", 0
        if "readlink -f /bin/sh" in s:
            return "/bin/sh\n", "", 0
        return "", "", 0

    def _find(self, base, md, ty, glob):
        base = base.rstrip("/") or "/"
        pref = "/" if base == "/" else base + "/"
        hits = []
        pool = ([p for p in self.dirset] if ty != "f" else []) \
            + ([p for p in self.content] if ty != "d" else [])
        for p in sorted(pool):
            if base != "/" and not p.startswith(pref):
                continue
            if base == "/" and p == "/":
                continue
            rel = p[len(pref):] if p.startswith(pref) else p
            if rel.count("/") + 1 <= md and fnmatch.fnmatch(basename_of(p), glob):
                hits.append(p)
        return hits

    # ---- the run() contract ----

    def run(self, cmd, prologue="", extra_timeout=0):
        self._apply_prologue(prologue)
        c = cmd.strip()
        if c == "cd" or (c.startswith("cd ") and ";" not in c and "&&" not in c):
            return self._cd(cmd, c)
        out, code = self._dispatch(c)
        return {"cmd": cmd, "output": out.rstrip("\n"), "exit": code, "cwd": self.cwd,
                "dur_ms": 0}

    def _apply_prologue(self, prologue):
        # fire-scripts write `echo go > /tmp/.tj/g<j>`: fire job j (append token to log)
        for j in re.findall(r"/tmp/\.tj/g(\d+)", prologue or ""):
            j = int(j)
            if j in self.jobs:
                tok, log = self.jobs.pop(j)
                self.content[log] = self.content.get(log, "") + tok + "\n"
                self.dirset.add(parent_of(log))

    def _cd(self, cmd, c):
        tgt = c[3:].strip() or "/"
        phys = self._resolve(normpath(tgt, self.cwd))
        if phys in self.dirset:
            self.cwd = logical_cwd(tgt, self.cwd)
            return {"cmd": cmd, "output": "", "exit": 0, "cwd": self.cwd, "dur_ms": 0}
        out, code = self._err("cd", path=tgt)
        return {"cmd": cmd, "output": out, "exit": code, "cwd": self.cwd, "dur_ms": 0}

    def _dispatch(self, c):
        toks = _lex(c)
        v = toks[0]
        if v == "after":
            return self._after(c, toks)
        if "|" in toks:
            return self._pipe(c, toks)
        if ">" in toks or ">>" in toks:
            return self._redir(c, toks)
        if v == "[":
            return self._cond(c, toks)
        if v == "uname":
            return "Linux fakebox 6.1.0 aarch64 GNU/Linux", 0
        if v == "pwd":
            return self.cwd, 0
        if v == "uptime":
            return "up", 0
        if v == "sleep":
            return "", 0
        if v == "echo":
            return " ".join(_unq(t) for t in toks[1:]), 0
        if v.endswith("tj3-ps"):
            return "", 0                    # stored from SST
        if v == "kill":
            return "", 0                    # stored from SST
        return self._simple(c, toks, v)

    def _read_content(self, path):
        r = self._resolve(path)
        if r in self.content:
            return self.content[r]
        return None

    def _simple(self, c, toks, v):
        if v == "cat":
            p = normpath(_unq(toks[-1]), self.cwd)
            data = self._read_content(p)
            if data is None:
                return self._err("cat", path=_unq(toks[-1]))
            return data, 0
        if v == "ls":
            return self._ls(toks)
        if v in ("head", "tail"):
            k = int(toks[2]); p = normpath(_unq(toks[3]), self.cwd)
            data = self._read_content(p)
            if data is None:
                return self._err(v, path=_unq(toks[3]))
            lines = data.split("\n")
            if lines and lines[-1] == "":
                lines = lines[:-1]
            keep = lines[:k] if v == "head" else lines[-k:]
            return "\n".join(keep), 0
        if v == "stat":
            p = normpath(_unq(toks[-1]), self.cwd)
            r = self._resolve(p)
            if r in self.content:
                return f"{r} {len(self.content[r])} regular file 644", 0
            if r in self.dirset:
                return f"{r} 4096 directory 755", 0
            return self._err("stat", path=_unq(toks[-1]))
        if v == "grep":
            # grep -F [-i] -m 8 TOK PATH
            i = 2 if toks[1] == "-F" and toks[2] != "-i" else 3
            tok = _unq(toks[i + 2]); p = normpath(_unq(toks[i + 3]), self.cwd)
            data = self._read_content(p)
            if data is None:
                return self._err("grep", path=_unq(toks[i + 3]))
            hits = [ln for ln in data.split("\n") if tok in ln][:8]
            return ("\n".join(hits), 0) if hits else ("", 1)
        if v == "find":
            base = _unq(toks[1]); md = int(toks[toks.index("-maxdepth") + 1])
            ty = toks[toks.index("-type") + 1] if "-type" in toks else None
            glob = _unq(toks[-1])
            return "\n".join(self._find(normpath(base, self.cwd), md, ty, glob)), 0
        if v == "readlink":
            p = normpath(_unq(toks[-1]), self.cwd)
            if p in self.symlink:
                return self.symlink[p], 0
            return "", 1
        if v == "mkdir":
            p = normpath(_unq(toks[-1]), self.cwd)
            self.dirset.add(p)
            return "", 0
        if v == "touch":
            p = self._resolve(normpath(_unq(toks[-1]), self.cwd))
            if not self._exists(p):
                self.content[p] = ""
            return "", 0
        if v == "rm":
            return self._rm(toks)
        if v == "mv":
            return self._mv(toks)
        if v == "ln":
            return self._ln(toks)
        return f"sh: {v}: not found", 127

    def _ls(self, toks):
        opt = next((t for t in toks[1:] if t.startswith("-")), "")
        pathtoks = [t for t in toks[1:] if not t.startswith("-")]
        base = normpath(_unq(pathtoks[0]), self.cwd) if pathtoks else self.cwd
        r = self._resolve(base)
        if r in self.content:
            return (_unq(pathtoks[0]) if pathtoks else base), 0    # ls of a file: echoes arg
        if r not in self.dirset:
            return self._err("ls", path=(_unq(pathtoks[0]) if pathtoks else base))
        names = self._children(r)
        want_all = "a" in opt
        if want_all:
            shown = sorted(names | {".", ".."})
        else:
            shown = sorted(n for n in names if not n.startswith("."))
        if "l" in opt and opt != "-1":
            mt = self._ls_mtime()      # per-instance VARYING triplet (see _mtime_nonce)
            rows = [f"-rw-r--r-- 1 root root 10 {mt} {n}" for n in shown
                    if n not in (".", "..")]
            return "total %d\n" % len(rows) + "\n".join(rows), 0
        return "\n".join(shown), 0

    _MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
               "Oct", "Nov", "Dec")

    def _ls_mtime(self):
        """A VALID but per-instance-VARYING ls -l date+time triplet ('Mon DD HH:MM').
        Container-creation mtimes differ across mints exactly this way, so two same-seed
        mints only agree after the collector's store-time canon_ls_l_text mask."""
        n = self._mtime_nonce
        return f"{self._MONTHS[n % 12]} {1 + n % 28:2d} {n % 24:02d}:{(n * 7) % 60:02d}"

    def _rm(self, toks):
        rec = toks[1] == "-r"
        p = normpath(_unq(toks[-1]), self.cwd)
        r = self._resolve(p) if p in self.symlink else p
        target = p if p in self.symlink else r
        if not self._exists(p):
            return self._err("rm", path=_unq(toks[-1]))
        if target in self.dirset and not rec:
            return self._err("rm_isdir", path=_unq(toks[-1]))
        # recursive: remove subtree
        victims = [target]
        if rec:
            pref = target.rstrip("/") + "/"
            victims += [q for q in list(self.content) + list(self.dirset)
                        if q.startswith(pref)]
        for q in victims:
            self._unlink(q)
        return "", 0

    def _unlink(self, q):
        self.content.pop(q, None)
        self.dirset.discard(q)
        self.symlink.pop(q, None)
        grp = self.groups.pop(q, None)
        if grp:
            grp.discard(q)
            for m in grp:
                self.groups[m] = grp

    def _mv(self, toks):
        src = normpath(_unq(toks[1]), self.cwd)
        dst = normpath(_unq(toks[2]), self.cwd)
        if not self._exists(src):
            return self._err("mv", path=_unq(toks[1]))
        rdst = self._resolve(dst)
        if rdst in self.dirset:
            dst = rdst.rstrip("/") + "/" + basename_of(src)
        # move the node (and, for a dir, its subtree)
        moving = [(src, dst)]
        if src in self.dirset:
            pref = src.rstrip("/") + "/"
            moving += [(q, dst.rstrip("/") + "/" + q[len(pref):])
                       for q in list(self.content) + list(self.dirset) + list(self.symlink)
                       if q.startswith(pref)]
        for a, b in moving:
            if a in self.content:
                self.content[b] = self.content.pop(a)
            elif a in self.dirset:
                self.dirset.discard(a); self.dirset.add(b)
            elif a in self.symlink:
                self.symlink[b] = self.symlink.pop(a)
        return "", 0

    def _ln(self, toks):
        sym = toks[1] == "-s"
        rest = toks[2:] if sym else toks[1:]
        target = _unq(rest[0]); link = normpath(_unq(rest[1]), self.cwd)
        if self._exists(link):
            return self._err("ln_exists" if "ln_exists" in self.tmpl else "cat",
                             path=_unq(rest[1]))
        if sym:
            self.symlink[link] = target
        else:
            tp = self._resolve(normpath(target, self.cwd))
            if tp not in self.content:
                return "ln: hard link: not a file", 1
            self.content[link] = self.content[tp]
            grp = self.groups.get(tp) or {tp}
            grp.add(link)
            for m in grp:
                self.groups[m] = grp
        return "", 0

    def _redir(self, c, toks):
        op = ">>" if ">>" in toks else ">"
        oi = toks.index(op)
        dst = normpath(_unq(toks[oi + 1]), self.cwd)
        left = toks[:oi]
        if left[0] == "echo":
            payload = _unq(left[1])
            text = payload + "\n"
        else:
            out, code = self._simple(" ".join(left), left, left[0])
            if code != 0:
                return out, code            # producer failed; stderr shows (folded)
            text = out + ("\n" if out else "")
        r = self._resolve(dst)
        if op == ">>" and r in self.content:
            self.content[r] = self.content[r] + text
        else:
            self.content[r] = text
        self.dirset.add(parent_of(r))
        return "", 0

    def _pipe(self, c, toks):
        i = toks.index("|")
        pout, pcode = self._simple(" ".join(toks[:i]), toks[:i], toks[0])
        filt = toks[i + 1:]
        fk = filt[0]
        if pcode != 0:
            return pout, (1 if fk == "grep" else 0)
        lines = pout.split("\n") if pout else []
        if fk in ("head", "tail"):
            k = int(filt[2])
            keep = lines[:k] if fk == "head" else lines[-k:]
            return "\n".join(keep), 0
        tok = _unq(filt[filt.index("8") + 1])
        hits = [ln for ln in lines if tok in ln][:8]
        return ("\n".join(hits), 0) if hits else ("", 1)

    def _cond(self, c, toks):
        # [ TESTOP P ] && READ P
        testop = toks[1]; p = normpath(_unq(toks[2]), self.cwd)
        rb = toks[toks.index("&&") + 1:]
        r = self._resolve(p)
        truth = {"-e": self._exists(p), "-f": r in self.content,
                 "-d": r in self.dirset,
                 "-s": r in self.content and len(self.content[r]) > 0}[testop]
        if not truth:
            return "", 1
        return self._simple(" ".join(rb), rb, rb[0])

    def _after(self, c, toks):
        # after j K 'echo TOK >> /tmp/w/task<j>.log' & echo $!
        j = int(toks[1])
        eff = _QRE.search(c).group(1)
        etoks = eff.split()
        tok = etoks[1]; log = etoks[3]
        self.jobs[j] = (tok, log)
        return str(5000 + j), 0            # a "real" pid; do() virtualizes to canonical


_GLOBS = ("*.conf", "*.d", "*.so*", "*.sh", "*.cfg", "*.txt", "*.py", "*.list",
          "*.service", "lib*", "*ssl*", "*.pem", "*.crt", "*rc", "*.gz", "*.h",
          "*.json", "*.ini", "os-release", "*.cnf")


def _lex(cmd):
    """Single-quote-aware tokenizer for FakeWorld dispatch (mirrors _sq embedding)."""
    toks, i, n = [], 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "'":
            j = i + 1
            while j < n:
                if cmd[j] == "'" and cmd[j:j + 4] != "'\\''":
                    j += 1
                    break
                j += 4 if cmd[j:j + 4] == "'\\''" else 1
            toks.append(cmd[i:j])
            i = j
            continue
        j = i
        while j < n and cmd[j] not in " \t":
            j += 1
        toks.append(cmd[i:j])
        i = j
    return toks
