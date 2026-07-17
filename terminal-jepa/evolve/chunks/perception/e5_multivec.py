"""perception recipe for the MULTI-VECTOR stream: e5-base-v2 (the champion encoder) with a
structured K=4 segment render per observation, alongside the standard e5 single-vector recipe
(so the single z_obs/z_cmd — the target/eval space — are IDENTICAL to enc_e5_base / the
data/dockerfs-e5 root).

Segments (up to K, empty ones dropped -> masked out in the stream):
  seg 0: "passage: cwd=<cwd> exit=<exit>"                (state/status channel, always present)
  seg 1..3: the observation's non-empty output lines split into 3 contiguous strips of roughly
            equal line count, each "passage: "-prefixed and char-capped. A short output (<=
            a few lines) yields fewer segments; a cd (no output) yields only seg 0.
"""

from evolve.chunks.perception.baseline import OBS_CAP
from evolve.chunks.perception.enc_e5_base import MODEL, render_obs, render_cmd, pool  # noqa: F401 (single-vector recipe unchanged)

K = 4
SEG_CAP = 800  # chars per output strip (3 strips cover OBS_CAP=1600 with headroom)


def render_obs_multi(step):
    segs = [f"passage: cwd={step.get('cwd', '/')} exit={step.get('exit', 0)}"]
    out = (step.get("output", "") or "")[:OBS_CAP]
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if lines:
        n_strips = min(3, len(lines))
        per = (len(lines) + n_strips - 1) // n_strips
        for i in range(n_strips):
            strip = "\n".join(lines[i * per:(i + 1) * per])[:SEG_CAP]
            if strip:
                segs.append("passage: " + strip)
    return segs[:K]
