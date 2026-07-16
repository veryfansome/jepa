"""Real task programs (Phase R1): each yields a sequence of genuine shell commands that
do real work in the session workspace, producing real observations (real ls -la
timestamps, git hashes, python tracebacks, byte content, tool errors). Tasks are tagged
by the tools they exercise so datagen can hold whole tools out for transfer evaluation.
Intentional failures are included — real agents see errors, and error semantics are real
dynamics the world model must learn.
"""

import random

WORDS = ["alpha", "config", "server", "cache", "token", "buffer", "index", "query",
         "result", "handler", "session", "payload", "worker", "socket", "stream"]
EXTS = ["txt", "md", "py", "json", "log", "csv"]
DIRS = ["src", "docs", "data", "lib", "tests", "build"]


def _name(rng):
    return rng.choice(WORDS)


def task_files(rng):
    """Create / list / copy / move / remove files and dirs — core fs manipulation."""
    d, f, ext = rng.choice(DIRS), _name(rng), rng.choice(EXTS)
    yield f"mkdir -p {d}/{_name(rng)}"
    yield f"printf '%s\\n%s\\n' '{_name(rng)} {_name(rng)}' '{_name(rng)}' > {d}/{f}.{ext}"
    yield f"ls -la {d}"
    yield f"cat {d}/{f}.{ext}"
    yield f"cp {d}/{f}.{ext} {d}/{f}.bak"
    yield f"mv {d}/{f}.bak {d}/{f}.old"
    yield f"wc -l {d}/{f}.{ext}"
    yield f"cat {d}/missing_{_name(rng)}.{ext}"           # real ENOENT error
    yield f"rm {d}/{f}.old"
    yield "ls -laR ."


def task_text(rng):
    """Text processing: grep / sed / sort / uniq / head — the tools an agent uses to
    read and transform files."""
    f = f"{_name(rng)}.txt"
    lines = [f"{rng.randint(1,99)} {rng.choice(WORDS)} {rng.choice(WORDS)}" for _ in range(6)]
    yield "printf '%s\\n' " + " ".join(f"'{ln}'" for ln in lines) + f" > {f}"
    yield f"cat {f}"
    kw = rng.choice(WORDS)
    yield f"grep {kw} {f}"                                # may match nothing (exit 1)
    yield f"sort {f}"
    yield f"sort -n {f} | head -3"
    yield f"sed 's/{rng.choice(WORDS)}/REDACTED/g' {f}"
    yield f"wc -w {f}"
    yield f"grep -c {kw} {f}"


def task_git(rng):
    """A real git workflow: init, stage, commit, inspect — outputs carry real hashes,
    branch names, and status formatting (heavy nuisance, real semantics). Includes real
    FAILURES (status before init, log before commit, checkout of a missing branch, push
    with no remote) so success-prediction transfer to this held-out tool is testable."""
    f = f"{_name(rng)}.py"
    yield "git status"                                   # fails: not a git repo yet
    yield "git init"
    yield "git log"                                      # fails: no commits yet
    yield f"printf 'def {_name(rng)}():\\n    return {rng.randint(0,9)}\\n' > {f}"
    yield "git status"
    yield f"git add {f}"
    yield "git commit -m 'add module'"
    yield f"git checkout no-such-branch-{_name(rng)}"    # fails: pathspec not found
    yield f"printf '\\n# {_name(rng)}\\n' >> {f}"
    yield "git diff"
    yield "git log --oneline"
    yield "git push"                                     # fails: no configured remote


def task_python(rng):
    """Write and run a small python script — real stdout, real tracebacks on error."""
    f = f"{_name(rng)}.py"
    n = rng.randint(2, 6)
    yield f"printf 'print(sum(range({n})))\\n' > {f}"
    yield f"python3 {f}"
    yield f"printf 'import json\\nprint(json.dumps({{\"k\": {rng.randint(0,9)}}}))\\n' > {f}"
    yield f"python3 {f}"
    yield "python3 -c 'print(1/0)'"                       # real ZeroDivisionError traceback
    yield f"python3 -c 'import os; print(os.listdir(\".\"))'"


# Registry: task -> tools it exercises (for held-out-tool splits).
TASKS = {
    "files": (task_files, {"mkdir", "ls", "cat", "cp", "mv", "wc", "rm", "printf"}),
    "text": (task_text, {"grep", "sed", "sort", "uniq", "head", "wc", "printf"}),
    "git": (task_git, {"git", "printf"}),
    "python": (task_python, {"python3", "printf"}),
}


def sample_task(rng, allowed=None):
    """Pick a task by name (optionally restricted to an allowed set) and realize it."""
    names = sorted(allowed) if allowed else sorted(TASKS)
    name = rng.choice(names)
    fn, tools = TASKS[name]
    return name, list(fn(rng)), tools
