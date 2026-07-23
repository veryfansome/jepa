"""Docker-backed shell recorder (Phase R2, 2026-07-16 redesign).

Real Linux filesystems from real Docker images (distros + app images) as the substrate:
common paths (/etc, /var, /usr), real system identity (uname, /etc/os-release), real tool
output. A DockerBox runs each command via its own `docker exec` (subprocess captures
stdout/stderr/exit cleanly — no sentinel/echo/buffering games) while tracking cwd in
Python so `cd` persists across a SEQUENCE (uname -> cd /etc -> ls -> cat os-release ...).
The world model must predict later observations from the accumulated history — build a
picture of what system it is on and where things are, then predict `ls`/`cat`/`cd` on
paths it has not directly seen this episode.

Safety: containers are ephemeral (--rm --network none), read-only exploration of throwaway
images; each exec has a timeout; nothing touches the host.
"""

import re
import subprocess
import time


def image_present(image):
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def pull(image):
    return subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True).returncode == 0


def _q(s):
    """POSIX single-quote for safe embedding in sh -c."""
    return "'" + str(s).replace("'", "'\\''") + "'"


def hostname_for(image):
    """Review-B F4: fixed, image-derived container hostname (docker's default is the 12-hex
    container ID — a fresh per-run nonce that leaks into /etc/hostname, hostname(1), \\h
    prompts, etc., breaking byte-reproducibility of observations)."""
    h = re.sub(r"[^a-zA-Z0-9-]+", "-", "box-" + str(image)).strip("-")
    return h[:63] or "box"


class DockerBox:
    def __init__(self, image, mem="512m", cmd_timeout=8, hostname=None,
                 init=False, label=None):
        """v3 (dockerfs3-prereg §7): opt-in `init=True` runs tini as PID 1 and `label`
        stamps `tj3-mint=<seed>` for the orphan sweep. Defaults keep v1/v2 launches
        bit-identical."""
        self.image = image
        self.cmd_timeout = cmd_timeout
        self.cwd = "/"
        argv = ["docker", "run", "-d", "--rm", "--network", "none",
                "--hostname", hostname or hostname_for(image), "-m", mem, "--cpus", "1"]
        if init:
            argv.append("--init")
        if label:
            argv += ["--label", label]
        argv += [image, "sleep", "86400"]
        self.cid = subprocess.run(argv, capture_output=True, text=True).stdout.strip()
        if not self.cid:
            raise RuntimeError(f"could not start container for {image}")

    def _exec(self, script, extra_timeout=0):
        """Run a script in the container; returns (stdout, stderr, returncode) as STRINGS.
        Review-B F1: capture BYTES and decode with errors="replace" — binary content becomes
        replacement-char mojibake (as a real terminal shows), never a host-side
        "executor error: 'utf-8' codec ..." artifact in an observation.
        v3: `extra_timeout` extends the budget for sleep- and barrier-bearing steps so a
        legitimate barrier wait can't surface as a spurious 124 (prereg §7)."""
        try:
            r = subprocess.run(["docker", "exec", self.cid, "/bin/sh", "-c", script],
                               capture_output=True, timeout=self.cmd_timeout + extra_timeout)
            return (r.stdout.decode("utf-8", errors="replace"),
                    r.stderr.decode("utf-8", errors="replace"), r.returncode)
        except subprocess.TimeoutExpired:
            return "", "command timed out", 124
        except Exception as e:  # noqa: BLE001
            return "", f"executor error: {e}", 125

    def cp_in(self, src, dst):
        """Copy a host file into the container (bootstrap artifacts: the `after` helper,
        the UD-9 tj3-ps busybox). Returns True on success."""
        return subprocess.run(["docker", "cp", "-q", src, f"{self.cid}:{dst}"],
                              capture_output=True).returncode == 0

    def run(self, cmd, prologue="", extra_timeout=0):
        """Run one command in the tracked cwd. `cd` is resolved in-container so it persists
        (relative paths, .., symlinks all handled by the shell). Returns
        {cmd, output(stdout+stderr), exit, cwd, dur_ms}.

        v3 prologue-injection seam (prereg §7, draft §3.2): `prologue` is collector-composed
        UNRECORDED scaffolding (fire-scripts for due jobs, post-signal barriers) prepended
        inside the same `sh -c` BEFORE the cwd prologue — on the cd branch too, so a job due
        at a cd step fires on time. The recorded command string is `cmd` alone; recorded ≡
        executed applies to it only. `dur_ms` is host-side instrumentation: the collector
        strips it from the recorded step and routes it to the timing side-channel — it must
        never enter a render (draft §3.4)."""
        pre = f"{prologue}; " if prologue else ""
        c = cmd.strip()
        t0 = time.monotonic()
        if c == "cd" or c.startswith("cd ") and ";" not in c and "&&" not in c:
            target = c[3:].strip() or "$HOME"
            out, err, code = self._exec(
                f"{pre}cd {_q(self.cwd)} 2>/dev/null && cd {target} && pwd", extra_timeout)
            dur = int((time.monotonic() - t0) * 1000)
            if code == 0 and out.strip():
                self.cwd = out.strip()
                return {"cmd": cmd, "output": "", "exit": 0, "cwd": self.cwd, "dur_ms": dur}
            return {"cmd": cmd, "output": (err or out).strip(), "exit": code or 1,
                    "cwd": self.cwd, "dur_ms": dur}
        out, err, code = self._exec(f"{pre}cd {_q(self.cwd)} 2>/dev/null; {cmd}", extra_timeout)
        dur = int((time.monotonic() - t0) * 1000)
        return {"cmd": cmd, "output": (out + err).rstrip("\n"), "exit": code, "cwd": self.cwd,
                "dur_ms": dur}

    def enumerate(self, roots=("/etc", "/var", "/usr", "/bin", "/sbin", "/root", "/home",
                               "/opt", "/lib", "/srv"), max_per=6000):
        rs = " ".join(roots)
        d, _, _ = self._exec(f"find {rs} -maxdepth 5 -type d 2>/dev/null | head -{max_per}")
        f, _, _ = self._exec(f"find {rs} -maxdepth 5 -type f 2>/dev/null | head -{max_per}")
        return [x for x in d.split("\n") if x], [x for x in f.split("\n") if x]

    def tool_help(self, tool):
        out, err, code = self._exec(f"{tool} --help 2>&1 | head -100")
        if code != 0 or not out:
            out, _, _ = self._exec(f"busybox {tool} --help 2>&1 | head -100")
        return out

    def system_id(self):
        out, _, _ = self._exec("uname -a; cat /etc/os-release 2>/dev/null | head -2")
        return out.strip()

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True)
