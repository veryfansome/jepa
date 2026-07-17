"""perception TIER-1 (foveal render on the champion e5 encoder): intfloat/e5-base-v2 + mean-pool
(the recorded champion recipe), with a biologically-motivated FOVEAL compression of `ls` output.

Motivation (efficient coding / foveation): the frozen encoder truncates at max_length=256 tokens —
a hard bandwidth bottleneck, like the optic nerve (~100x fewer axons than photoreceptors). On this
corpus 15.9% of renders hit that cap and the truncated set is entirely the content verbs (cat/ls).
In a long-format `ls -l` line each entry costs ~20 tokens, but the only discriminative payload — the
FILENAME — is 1-2 tokens at the end; the leading columns (permission bits, link count, uid/gid
`root root`, block-aligned size) are near-constant redundancy and the date field is the COLLECTION
wall-clock (identical within a run — pure nuisance). Barlow's redundancy-reduction says a fixed
channel should not spend capacity on predictable, constant signal; foveated vision allocates the
fovea to the informative region. So for `ls` observations we collapse each long-format line to
`<type-char> <name>` (symlink `name -> target` preserved) and drop the `total N` header. This lets
the full SET OF ENTRIES survive the 256-token cap (a `ls -l /` render drops 371 -> 53 tokens, ~7x;
truncated `ls` steps fall ~67-71%), and removes the constant timestamp/metadata tokens that would
otherwise inflate same-verb foil similarity in the retrieval eval.

STRICTLY verb-gated: only observations whose command verb is `ls` are transformed; cat/uname/cd and
plain `ls` output are passed through byte-identical, so the champion behaviour on every other verb is
preserved exactly and any regression risk is confined to `ls`. Deterministic, data-independent, no
learned params. e5 is 768-d so mean-pool needs no adapter. render_cmd keeps e5's required
'passage: ' prefix.
"""
import re

MODEL = "intfloat/e5-base-v2"
OBS_CAP = 1600

# A long-format ls line: TYPE + 9 mode chars (+ optional acl/xattr flag), link count, owner, group,
# size (a number, human like '4.0K', OR a 'major, minor' device pair), then the 3-field date
# (month day time-or-year), then the NAME (which may contain spaces or a ' -> target' for symlinks).
_LS_LINE = re.compile(
    r'^([bcdlpsD\-])[rwxsStTlL.\-]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+'
    r'(?:\d+,\s+\d+|\S+)\s+\S+\s+\S+\s+\S+\s+(.+?)\s*$')
_TOTAL = re.compile(r'^total\s+\d+\s*$')
# 1-char type marker kept as the cheap dir/file/symlink cue; regular files ('-') get no prefix.
_TYPEMARK = {'d': 'd ', 'l': 'l ', '-': '', 'b': 'b ', 'c': 'c ', 'p': 'p ', 's': 's ', 'D': 'D '}


def _fovea_line(ln):
    if _TOTAL.match(ln):
        return None  # block-count header: pure noise, drop it
    m = _LS_LINE.match(ln)
    if not m:
        return ln  # blank lines, `-R` section headers (`./etc:`), non-long-format: keep as-is
    return _TYPEMARK.get(m.group(1), m.group(1) + ' ') + m.group(2)


def _foveate(out):
    kept = (_fovea_line(l) for l in out.split("\n"))
    return "\n".join(k for k in kept if k is not None)


def render_obs(step):
    out = step.get("output", "") or ""
    cmd = step.get("cmd", "") or ""
    # Verb-gate: only foveate directory listings; never touch cat/uname/cd (or any other) output.
    if cmd.split()[0:1] == ["ls"]:
        out = _foveate(out)
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"


def render_cmd(step):
    return "passage: " + step["cmd"]


def pool(h, mask):
    # e5 standard usage: masked mean over tokens. e5-base-v2 is 768-d, so no adapter is needed.
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)

