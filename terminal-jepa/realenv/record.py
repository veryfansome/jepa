"""Real shell trajectory recorder (Phase R1).

The open-world replacement for the synthetic env: runs ACTUAL shell commands in an
isolated per-trajectory workspace and logs the real terminal transcript plus a
filesystem snapshot. Observations are genuine tool output with genuine irreducible
nuisance (timestamps, git hashes, byte content, OS ordering) — the JEPA regime that
finding 24 says our synthetic clean-serialization lacked. Ground truth for evaluation
comes from real filesystem diffs + exit codes, NOT a hand-coded ontology.

Safety: every command runs with cwd confined to a fresh temp workspace; `cd` is tracked
in-process and cannot escape the workspace; a per-command timeout bounds runaways. Task
programs (realenv/tasks.py) emit only curated, network-free commands on relative paths.
This is deliberately not a general shell — it is a controlled recorder for data.
"""

import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile


class Session:
    """One trajectory's isolated workspace. run(cmd) executes a real command and
    returns the observation dict; snapshot() reads the true fs state for eval labels."""

    def __init__(self, sandbox_root, timeout=10):
        pathlib.Path(sandbox_root).mkdir(parents=True, exist_ok=True)
        self.workspace = pathlib.Path(tempfile.mkdtemp(prefix="tj-", dir=sandbox_root)).resolve()
        self.timeout = timeout
        self.cwd = self.workspace  # persistent cwd within the sandbox
        # Deterministic-ish, pager-free, quiet env so transcripts are clean of $HOME leaks.
        self.env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
            "HOME": str(self.workspace), "PAGER": "cat", "GIT_PAGER": "cat",
            "GIT_CONFIG_GLOBAL": str(self.workspace / ".gitconfig"),
            "GIT_AUTHOR_NAME": "tj", "GIT_AUTHOR_EMAIL": "tj@sandbox",
            "GIT_COMMITTER_NAME": "tj", "GIT_COMMITTER_EMAIL": "tj@sandbox",
            "LC_ALL": "C", "TERM": "dumb", "NO_COLOR": "1",
        }

    def _resolve_cd(self, target):
        """Resolve a cd target, refusing to escape the workspace."""
        dest = (self.cwd / target).resolve() if not os.path.isabs(target) else pathlib.Path(target).resolve()
        try:
            dest.relative_to(self.workspace)
        except ValueError:
            return None  # escape attempt — reject
        return dest if dest.is_dir() else None

    def run(self, cmd):
        """Execute one real command string. Returns {cmd, stdout, stderr, exit,
        cwd_rel}. `cd DIR` is handled in-process so cwd persists across steps."""
        cwd_rel = str(self.cwd.relative_to(self.workspace)) or "."
        stripped = cmd.strip()
        if stripped.startswith("cd ") and "&&" not in stripped and ";" not in stripped:
            target = stripped[3:].strip() or "."
            dest = self._resolve_cd(target)
            if dest is None:
                return {"cmd": cmd, "stdout": "", "stderr": f"cd: {target}: no such directory",
                        "exit": 1, "cwd_rel": cwd_rel}
            self.cwd = dest
            return {"cmd": cmd, "stdout": "", "stderr": "", "exit": 0,
                    "cwd_rel": str(self.cwd.relative_to(self.workspace)) or "."}
        try:
            p = subprocess.run(cmd, shell=True, cwd=str(self.cwd), env=self.env,
                               capture_output=True, text=True, timeout=self.timeout)
            out, err, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            out, err, code = "", "command timed out", 124
        except Exception as e:  # noqa: BLE001 — record any executor failure as an obs
            out, err, code = "", f"executor error: {e}", 125
        # Normalize the random temp workspace path to a stable token: otherwise it is a
        # per-trajectory-CONSTANT string leaking into output (git init, errors) — exactly
        # the slow-feature distractor (banner) that findings 2/12 warned about. Real
        # nuisance (hashes, timestamps, ordering) is deliberately kept.
        out = out.replace(str(self.workspace), "/work")
        err = err.replace(str(self.workspace), "/work")
        return {"cmd": cmd, "stdout": out, "stderr": err, "exit": code, "cwd_rel": cwd_rel}

    def snapshot(self, include_git=True):
        """True fs state of the workspace: {relpath: 'd' | 8-hex content hash}. Used for
        eval labels (fs diff between steps), never shown to the model. include_git=True
        (default) counts .git mutations so git add/commit register as state-changing —
        WITHOUT it, state-change on git was a degenerate printf-only label (2026-07-16
        review). The volatile .git/index.lock/objects churn is real dynamics."""
        state = {}
        for dirpath, dirnames, filenames in os.walk(self.workspace):
            if not include_git:
                dirnames[:] = [d for d in dirnames if d != ".git"]
            for d in dirnames:
                rel = str((pathlib.Path(dirpath) / d).relative_to(self.workspace))
                state[rel] = "d"
            for f in filenames:
                fp = pathlib.Path(dirpath) / f
                rel = str(fp.relative_to(self.workspace))
                try:
                    state[rel] = hashlib.sha1(fp.read_bytes()).hexdigest()[:8]
                except OSError:
                    state[rel] = "err"
        return state

    def close(self):
        shutil.rmtree(self.workspace, ignore_errors=True)


def fs_diff(before, after):
    """Symbolic outcome label from two snapshots: created / deleted / modified paths.
    This is the real-world, open-vocabulary ground truth — derived from the actual fs,
    not a fixed 301-slot ontology."""
    b, a = set(before), set(after)
    created = sorted(a - b)
    deleted = sorted(b - a)
    modified = sorted(p for p in (a & b) if before[p] != after[p])
    return {"created": created, "deleted": deleted, "modified": modified,
            "n_changed": len(created) + len(deleted) + len(modified)}
