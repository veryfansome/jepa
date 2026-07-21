"""perception TIER-1 (R12, v2-world render revision): e5-base-v2 champion recipe with an
EXACT-TOKENIZER WINDOW-FIT HEAD+TAIL policy for `cat` observations only — every other step's
render is BYTE-IDENTICAL to enc_e5_base (the dockerfs2-e5 incumbent root).

WHY (the v2 diagnosis): fitness now includes tail (+.347, lowest wm .717), grep-hit (+.326,
within-traj baseline binds at .513) and cat (+.293, cross-image retrieve-by-cmd .530). The
incumbent render head-truncates: reencode.py encodes with max_length=256, so a `cat` render's
visible span is the first ~600 chars — measured on dockerfs2, 46.6% of cat observations
token-overflow and everything past the window is invisible. Three consequences the revision
attacks: (1) file TAILS — the observation channel v2 added — never enter obs space through cat
steps, so the corpus statistics binding file identity -> end-of-file content are simply absent
from the history/target distribution; (2) ~8% of grep-hit matches over previously-cat'ed files
lie beyond the visible window (346/4412 measured), capping in-context match prediction; (3)
cross-image cat pairs that differ only past the head look identical — measured on long
multi-image cat outputs, near-identical (>0.95 8-gram Jaccard) pairs drop from 75.4% (head
window) to 71.3% (70/30 head+tail window), i.e. the tail slice restores system-discriminative
content that weakens the strong rbc baseline differentially, while the WM retains in-context
system evidence (uname/os-release) to exploit it.

WHY THE RECORDED TRAPS DON'T RECUR: (a) unlike R8 kmv, no ls sketch — ls (+.661, the best
margin) and find/cd/grep/head/tail/stat/uname renders are byte-identical, so their embeddings,
the foil geometry of every other verb, and the R11-sensitive cd/cwd channel are untouched;
(b) unlike e5-large, the encoder and the space are unchanged — only token-overflowing cat
renders move; (c) the overflow test uses the REAL e5 tokenizer (exact, not kmv's char
estimator), which also means binary cats (huge char counts that collapse to a few UNK tokens,
e.g. 65536 chars -> 82 tokens) never trigger the policy and stay byte-identical — the change
lands precisely on long TEXT files; (d) the split is head-dominant (70/30), keeping the
predictable file-identity head that the WM already exploits (cat wm .823) while the kmv cat
policy gave 40% to the tail. The fitted render is guaranteed <= 256 tokens, so the tail slice
and the skipped-lines marker (a visible length signal the incumbent's phantom char-1600 marker
never delivered) are actually encoded.

Deterministic (no RNG; the HF fast tokenizer is deterministic), leak-free (render_obs reads
only its own step's cmd/output/cwd/exit; render_cmd only step['cmd'] — the cmd precedes its
obs, so verb dispatch is causal), no learned state; e5-base-v2 is 768-d so mean-pool needs no
adapter. The tokenizer is lazily loaded once per process (reencode.py loads the same model's
tokenizer anyway); a calibrated char-budget fallback keeps the module functional if
transformers is unavailable at render time.
"""

NAME = "r12_tailwindow_catfit"
DESCRIPTION = (
    "e5-base-v2 champion render, except token-overflowing `cat` observations get an exact-"
    "tokenizer-fitted 70/30 head+tail slice (skipped-lines marker in-window), putting file "
    "tails, hidden grep matches and end-of-file system-discriminative content into the "
    "encoded obs space; all other steps byte-identical to enc_e5_base."
)

MODEL = "intfloat/e5-base-v2"
OBS_CAP = 1600          # incumbent char cap, kept for the byte-identical path
WINDOW = 256            # reencode.py's encode window (truncation=True, max_length=256)
HEAD_FRAC = 0.70        # head-dominant split: identity head stays, tail slice is the addition

_TOK = None
_TOK_FAILED = False


def _ntok(text):
    """Exact e5 token count incl. special tokens; -1 if the tokenizer is unavailable."""
    global _TOK, _TOK_FAILED
    if _TOK is None and not _TOK_FAILED:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(MODEL)
        except Exception:
            _TOK_FAILED = True
    if _TOK is None:
        return -1
    return len(_TOK(text, add_special_tokens=True)["input_ids"])


def _fits(text):
    n = _ntok(text)
    if n < 0:
        # fallback calibration: p05 chars/token on this corpus is ~1.75 (R8 measurement),
        # so <= 1.75*WINDOW chars can only rarely overflow; be conservative at 450 chars.
        return len(text) <= 450
    return n <= WINDOW


def _incumbent(step):
    """enc_e5_base render, byte-for-byte (the hedge: unchanged steps re-encode identically)."""
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\\n{out}"


def _assemble(prefix, lines, h, t):
    n = len(lines)
    skip = n - h - t
    if skip <= 0:
        return prefix + "\\n".join(lines)
    marker = f"\\n...[{skip} of {n} lines skipped]...\\n"
    tail = "\\n".join(lines[n - t:]) if t else ""
    return prefix + "\\n".join(lines[:h]) + marker + tail


def _headtail_fit(prefix, out):
    """Largest 70/30 head+tail line slice of the FULL output whose render fits the window.

    Token count is (near-)monotone in kept-line count -> binary search on k, with every
    candidate verified by the real tokenizer; char-level fallback for the degenerate
    single-enormous-line case. Always returns a render that fits (hard floor guaranteed)."""
    lines = out.rstrip("\\n").split("\\n")
    n = len(lines)
    lo, hi, best = 1, n - 1, None       # k = total kept lines; k == n never fits (it overflowed)
    while lo <= hi:
        k = (lo + hi) // 2
        h = max(1, int(round(k * HEAD_FRAC)))
        t = k - h
        cand = _assemble(prefix, lines, h, t)
        if _fits(cand):
            best, lo = cand, k + 1
        else:
            hi = k - 1
    if best is not None:
        return best
    # degenerate: even one line overflows (minified/soup line) -> char-level head+tail
    head_c, tail_c = 900, 380
    while head_c >= 16:
        cand = (prefix + out[:head_c]
                + f"\\n...[{max(0, len(out) - head_c - tail_c)} chars skipped]...\\n"
                + out[-tail_c:])
        if _fits(cand):
            return cand
        head_c, tail_c = int(head_c * 0.7), int(tail_c * 0.7)
    return prefix + out[:200]           # hard floor: always <= window


def render_obs(step):
    r = _incumbent(step)
    toks = (step.get("cmd", "") or "").split()
    if not toks or toks[0] != "cat":
        return r                        # non-cat: byte-identical, no tokenizer call needed
    out = step.get("output", "") or ""
    if not out.strip() or _fits(r):
        return r                        # short/binary(UNK-collapsed) cat: byte-identical
    prefix = f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\\n"
    return _headtail_fit(prefix, out)


def render_cmd(step):
    # champion command render, unchanged (e5's required prefix; reads only the action)
    return "passage: " + step["cmd"]


def pool(h, mask):
    # e5 standard masked mean-pool (unchanged; 768-d, no adapter)
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)
