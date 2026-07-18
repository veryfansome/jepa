"""perception TIER-2 encoder swap: microsoft/codebert-base, a code-aware 768-d encoder (shell
output/config files are code-ish). Mean pool."""
from evolve.chunks.perception.baseline import render_obs, render_cmd, pool
MODEL = "microsoft/codebert-base"
