"""Diverse real command sampler (Phase R1, rebuilt 2026-07-16 after the review found the
first cut was fixed 12-step scripts replicated 120x -> effective eval n~=12).

A stateful generator: it tracks the files/dirs it has created and samples variable-length
sessions from a broad pool of parameterized command templates across many tools, in random
order with random parameters — so trajectories are genuinely diverse (large effective-n)
and a held-out TOOL appears in many distinct contexts (a real transfer test, not one
replicated script). Templates reference tracked paths so most commands are valid; natural
failures (missing files, empty grep, bad git state) still occur.
"""

import random

WORDS = ["alpha", "config", "server", "cache", "token", "buffer", "index", "query",
         "result", "handler", "session", "payload", "worker", "socket", "stream",
         "module", "vector", "matrix", "record", "packet", "kernel", "daemon", "proxy"]
EXTS = ["txt", "md", "py", "json", "log", "csv", "ini", "sh", "yaml"]
DIRS = ["src", "docs", "data", "lib", "tests", "build", "bin", "conf", "tmp", "assets"]


def _nm(rng):
    n = rng.choice(WORDS)
    return n + (str(rng.randint(0, 999)) if rng.random() < 0.5 else "")


class WS:
    """Tracks created paths so templates can reference real files/dirs."""
    def __init__(self):
        self.files, self.dirs = [], ["."]

    def add_file(self, p):
        if p not in self.files:
            self.files.append(p)

    def add_dir(self, p):
        if p not in self.dirs:
            self.dirs.append(p)

    def rm(self, p):
        if p in self.files:
            self.files.remove(p)


def _f(ws, rng):
    return rng.choice(ws.files) if ws.files else None


def _d(ws, rng):
    return rng.choice(ws.dirs) if ws.dirs else "."


def _newfile(ws, rng):
    d = _d(ws, rng)
    p = (f"{d}/" if d != "." else "") + f"{_nm(rng)}.{rng.choice(EXTS)}"
    return p


# Each template: (needs, fn). needs in {"", "file", "dir"} gates applicability.
# fn(ws, rng) -> command string, and mutates ws to reflect the effect.

def _t_mkdir(ws, rng):
    d = f"{rng.choice(DIRS)}{rng.randint(0,9)}"
    ws.add_dir(d)
    return f"mkdir -p {d}/{rng.choice(DIRS)}" if rng.random() < 0.3 else f"mkdir -p {d}"

def _t_write(ws, rng):
    p = _newfile(ws, rng); ws.add_file(p)
    body = " ".join(_nm(rng) for _ in range(rng.randint(1, 4)))
    return f"printf '%s\\n%s\\n' '{body}' '{_nm(rng)} {rng.randint(0,99)}' > {p}"

def _t_pyscript(ws, rng):
    p = f"{_nm(rng)}.py"; ws.add_file(p)
    n = rng.randint(2, 8)
    kind = rng.choice([f"print(sum(range({n})))", f"print([i*i for i in range({n})])",
                       f"import json; print(json.dumps({{'n': {n}}}))",
                       f"print('{_nm(rng)}'.upper())"])
    return f"printf '{kind}\\n' > {p}"

def _t_append(ws, rng):
    p = _f(ws, rng)
    return f"printf '# {_nm(rng)}\\n' >> {p}"

def _t_cat(ws, rng):
    return f"cat {_f(ws, rng)}"

def _t_head(ws, rng):
    return f"{rng.choice(['head','tail'])} -{rng.randint(1,3)} {_f(ws, rng)}"

def _t_ls(ws, rng):
    return rng.choice(["ls", "ls -la", "ls -la " + _d(ws, rng), "ls -R"])

def _t_wc(ws, rng):
    return f"wc {rng.choice(['-l','-w','-c'])} {_f(ws, rng)}"

def _t_cp(ws, rng):
    s = _f(ws, rng); d = _newfile(ws, rng); ws.add_file(d)
    return f"cp {s} {d}"

def _t_mv(ws, rng):
    s = _f(ws, rng); d = _newfile(ws, rng); ws.rm(s); ws.add_file(d)
    return f"mv {s} {d}"

def _t_rm(ws, rng):
    p = _f(ws, rng); ws.rm(p)
    return f"rm {p}"

def _t_grep(ws, rng):
    return f"grep {rng.choice(WORDS)} {_f(ws, rng)}"

def _t_sort(ws, rng):
    return rng.choice([f"sort {_f(ws, rng)}", f"sort -n {_f(ws, rng)} | head -3"])

def _t_find(ws, rng):
    return rng.choice([f"find . -name '*.{rng.choice(EXTS)}'", "find . -type f", "find . -type d"])

def _t_stat(ws, rng):
    return f"stat {_f(ws, rng)}"

def _t_py(ws, rng):
    p = _f(ws, rng)
    return rng.choice([f"python3 {p}" if p and p.endswith('.py') else "python3 -c 'print(1/0)'",
                       "python3 -c 'import os; print(len(os.listdir(\".\")))'",
                       f"python3 -c 'print(2**{rng.randint(2,10)})'"])

def _t_missing(ws, rng):  # natural failure
    return f"cat {_d(ws, rng)}/missing_{_nm(rng)}.txt"

# held-out tools ----------------------------------------------------------------

def _t_git(ws, rng):
    return rng.choice(["git init", "git status", "git log", "git log --oneline",
                       f"git add {_f(ws, rng)}", "git commit -m 'wip'", "git diff",
                       f"git checkout no-branch-{_nm(rng)}", "git push", "git branch"])

def _t_sed(ws, rng):
    return f"sed 's/{rng.choice(WORDS)}/X/g' {_f(ws, rng)}"

def _t_awk(ws, rng):
    return f"awk '{{print NF, $0}}' {_f(ws, rng)}"

def _t_du(ws, rng):
    return rng.choice(["du -sh .", f"du -sh {_d(ws, rng)}", "du -a . | sort -n | tail -3"])

def _t_diff(ws, rng):
    a, b = _f(ws, rng), _f(ws, rng)
    return f"diff {a} {b}"

def _t_tar(ws, rng):
    return rng.choice([f"tar cf arch_{_nm(rng)}.tar {_f(ws, rng)}", "tar tf nonexistent.tar"])


TEMPLATES = {
    # universal (train + val)
    "mkdir": ("", _t_mkdir), "write": ("", _t_write), "pyscript": ("", _t_pyscript),
    "ls": ("", _t_ls), "find": ("", _t_find), "missing": ("dir", _t_missing),
    "append": ("file", _t_append), "cat": ("file", _t_cat), "head": ("file", _t_head),
    "wc": ("file", _t_wc), "cp": ("file", _t_cp), "mv": ("file", _t_mv),
    "rm": ("file", _t_rm), "grep": ("file", _t_grep), "sort": ("file", _t_sort),
    "stat": ("file", _t_stat), "py": ("", _t_py),
    # held-out tools (only in the split that includes them)
    "git": ("", _t_git), "sed": ("file", _t_sed), "awk": ("file", _t_awk),
    "du": ("", _t_du), "diff": ("file", _t_diff), "tar": ("file", _t_tar),
}
UNIVERSAL = {"mkdir", "write", "pyscript", "ls", "find", "missing", "append", "cat",
             "head", "wc", "cp", "mv", "rm", "grep", "sort", "stat", "py"}
HELD_OUT_TOOLS = {"git", "sed", "awk", "du", "diff", "tar"}


def gen_session(rng, allowed, length):
    """A diverse variable-length real session over the allowed template set. Ensures a
    file exists before file-needing templates; interleaves tools randomly."""
    ws = WS()
    names = sorted(allowed)
    cmds = []
    # seed with a create so file-needing templates are usable
    cmds.append(_t_write(ws, rng))
    for _ in range(length - 1):
        for _try in range(8):
            name = rng.choice(names)
            needs, fn = TEMPLATES[name]
            if needs == "file" and not ws.files:
                continue
            cmds.append(fn(ws, rng))
            break
    return cmds


def sample_session(rng, allowed_tools, min_len=8, max_len=22):
    """Pick an allowed template set and realize a diverse session. allowed_tools may add
    held-out tools to the universal set."""
    allowed = set(UNIVERSAL) | set(allowed_tools or [])
    length = rng.randint(min_len, max_len)
    # a loose "task" tag = the dominant held-out tool present, else 'mixed'
    tag = "+".join(sorted(set(allowed_tools or []))) or "universal"
    return tag, gen_session(rng, allowed, length), sorted(allowed)
