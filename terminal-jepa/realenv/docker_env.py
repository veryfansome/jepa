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

import subprocess


def image_present(image):
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def pull(image):
    return subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True).returncode == 0


def _q(s):
    """POSIX single-quote for safe embedding in sh -c."""
    return "'" + str(s).replace("'", "'\\''") + "'"


class DockerBox:
    def __init__(self, image, mem="512m", cmd_timeout=8):
        self.image = image
        self.cmd_timeout = cmd_timeout
        self.cwd = "/"
        self.cid = subprocess.run(
            ["docker", "run", "-d", "--rm", "--network", "none", "-m", mem, "--cpus", "1",
             image, "sleep", "7200"],
            capture_output=True, text=True).stdout.strip()
        if not self.cid:
            raise RuntimeError(f"could not start container for {image}")

    def _exec(self, script):
        try:
            r = subprocess.run(["docker", "exec", self.cid, "/bin/sh", "-c", script],
                               capture_output=True, text=True, timeout=self.cmd_timeout)
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return "", "command timed out", 124
        except Exception as e:  # noqa: BLE001
            return "", f"executor error: {e}", 125

    def run(self, cmd):
        """Run one command in the tracked cwd. `cd` is resolved in-container so it persists
        (relative paths, .., symlinks all handled by the shell). Returns
        {cmd, output(stdout+stderr), exit, cwd}."""
        c = cmd.strip()
        if c == "cd" or c.startswith("cd ") and ";" not in c and "&&" not in c:
            target = c[3:].strip() or "$HOME"
            out, err, code = self._exec(f"cd {_q(self.cwd)} 2>/dev/null && cd {target} && pwd")
            if code == 0 and out.strip():
                self.cwd = out.strip()
                return {"cmd": cmd, "output": "", "exit": 0, "cwd": self.cwd}
            return {"cmd": cmd, "output": (err or out).strip(), "exit": code or 1, "cwd": self.cwd}
        out, err, code = self._exec(f"cd {_q(self.cwd)} 2>/dev/null; {cmd}")
        return {"cmd": cmd, "output": (out + err).rstrip("\n"), "exit": code, "cwd": self.cwd}

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
