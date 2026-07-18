"""perception variant: drop the exit code from the render (mild de-lexicalization; R3 lens),
ModernBERT + mean pool."""
from evolve.chunks.perception.baseline import MODEL, render_cmd, pool, OBS_CAP
def render_obs(step):
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"cwd={step.get('cwd','/')}\n{out}"
