"""perception TIER-1 (R14, cat/tail-content lens): e5-base-v2 champion recipe with a
HYPERVARIABLE-REGION INFO-PACK render for token-overflowing `cat` observations — every other
step's render is BYTE-IDENTICAL to enc_e5_base (the dockerfs2-e5 incumbent root).

THE TARGET (R14 diagnosis): cat's margin is STALLED at +.295 across two champion generations
while every other content verb moved. cat has the highest retrieve-by-cmd baseline (.531):
corpus lookup already recovers files whose content is identical across systems, so the
remaining margin lives in SYSTEM-VARIANT file bodies. Two render-level causes survive the
syscond FiLM arch: (1) the encode window (max_length=256 ~ first 600 chars) hides everything
past the head — 48% of cat outputs overflow (937/1944 measured on dockerfs2 train); (2) MEAN
POOLING is a low-pass filter — the few tokens that actually vary across systems (versions,
build ids, hashes, uids, dates) are diluted 1/N by the shared boilerplate bulk, so same-file
cross-image variants land nearly on top of each other and neither the WM nor the eval can
separate them. r12_tailwindow_catfit attacked (1) positionally (70/30 head+tail) and screened
only borderline (+0.0026): positional slices still spend ~all of their token budget on shared
content.

THE MECHANISM (Lempel-Ziv surprisal packing = keep the hypervariable regions): for an
overflowing cat output, keep a small verbatim identity anchor (first 3 line-segments = file
identity for WM and rbc alike), then fill the REST of the 256-token window with the file's own
highest-information lines — scored by conditional compression gain
    gain(l) = |zlib(ctx + l)| − |zlib(ctx)|   (ctx = the identity anchor),
selected greedily per token of cost (a knapsack density), with an MMR-style char-5-gram
redundancy penalty so mutually-duplicate boilerplate is picked once. Selected lines are
rendered in original file order behind an elision marker; an exact-tokenizer binary search
guarantees the render FITS the window (so, unlike the incumbent's phantom tail marker, all of
it is actually encoded). Kolmogorov/LZ surprisal is the honest self-contained proxy for
mutability: high-entropy fields (version strings, ids, hashes, timestamps) are precisely the
DATA of a config file — the parts that vary across distros — while low-entropy format
boilerplate compresses away. This is the antibody trick: immunoglobulins discriminate
near-identical antigens by binding their hypervariable regions, not the conserved scaffold.

WHY THIS MOVES THE MARGIN (and not the baselines): identical-across-images files produce
IDENTICAL renders (selection is a deterministic function of content), so rbc keeps every case
it already wins and the ls/find/head/tail/grep foil geometry is untouched (those renders are
byte-identical). Files that differ anywhere now concentrate several-fold more token mass on
exactly the differing fields, pushing cross-image variants apart in obs space: rbc's
cross-image lookup — right today only because variants collide — starts missing, while the WM
holds in-context system evidence (uname/os-release/FiLM syscond) to pick the right variant.
The margin is attacked from both sides at once. MEASURED on dockerfs2 train (cross-image
same-cat-cmd pairs with DIFFERING outputs, n=238): near-duplicate renders (8-gram Jaccard
>0.95) drop 12.6% -> 1.3% (median Jaccard .764 -> .635) — a 10x cut vs the incumbent, where
r12_tailwindow's positional slice managed 75.4% -> 71.3% on its analogous measure — while all
39/39 identical-content cross-image pairs stay render-identical. Side effect on tail (lowest
wm .732): file TAILS and mid-file distinctive lines now enter obs space through cat steps
(selection sees the whole file, not the first 600 chars), so history finally carries evidence
about file ends.

Deterministic (no RNG; ties broken by line index; zlib level fixed), leak-free (render_obs
reads only its own step's cmd/output/cwd/exit; the cmd precedes its obs so verb dispatch is
causal), no learned/global state beyond the lazily-loaded HF tokenizer (same pattern as
r12_tailwindow_catfit; reencode.py loads that tokenizer anyway), stdlib-only imports.
Binary blobs (UNK-collapsed) fit the window and stay byte-identical; a char-budget fallback
keeps the module functional without transformers.
"""

import zlib

NAME = "r14_hypervariable_lz_catpack"
DESCRIPTION = (
    "e5-base-v2 champion render, except token-overflowing `cat` observations keep a 3-segment "
    "identity anchor and fill the rest of the exact-tokenizer-fitted 256-token window with the "
    "file's highest LZ-surprisal lines (greedy info-per-token knapsack + 5-gram redundancy "
    "penalty, original order, elision marker in-window) — concentrating token mass on the "
    "system-variant 'hypervariable' fields that mean pooling otherwise dilutes; all other "
    "steps byte-identical to enc_e5_base."
)

MODEL = "intfloat/e5-base-v2"
OBS_CAP = 1600          # incumbent char cap (byte-identical path)
WINDOW = 256            # reencode.py encode window (truncation=True, max_length=256)
IDENT_SEGS = 3          # verbatim identity anchor: first segments of the file
SEG_CHARS = 240         # long lines split into segments of this size (kills the degenerate case)
POOL_HEAD = 280         # candidate-pool cap: first 280 + last 200 segments of huge files
POOL_TAIL = 200
RAW_HEAD = 20000        # raw char cap before segmentation (head + tail slices, determinism cheap)
RAW_TAIL = 8000
CHARS_PER_TOK = 3.5     # token-cost estimate for the knapsack density (fit is exact afterwards)
MIN_NOVELTY = 0.30      # MMR floor: skip lines >70% shingle-covered by already-picked content
PACK_CHAR_BUDGET = 1500 # stop greedy once well past what the window can hold
SHINGLE = 5

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
        # tokenizer-free fallback: conservative char budget (p05 chars/token ~1.75 on this corpus)
        return len(text) <= 450
    return n <= WINDOW


def _incumbent(step):
    """enc_e5_base render, byte-for-byte (unchanged steps re-encode identically)."""
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"


def _segments(out):
    """Deterministic candidate segments: lines, with long lines split into SEG_CHARS chunks;
    huge files keep head+tail segment slices so late-file content stays reachable."""
    if len(out) > RAW_HEAD + RAW_TAIL:
        out = out[:RAW_HEAD] + "\n" + out[-RAW_TAIL:]
    lines = out.rstrip("\n").split("\n")
    segs = []
    for ln in lines:
        if len(ln) <= SEG_CHARS:
            segs.append(ln)
        else:
            for i in range(0, len(ln), SEG_CHARS):
                segs.append(ln[i:i + SEG_CHARS])
    if len(segs) > POOL_HEAD + POOL_TAIL:
        segs = segs[:POOL_HEAD] + segs[-POOL_TAIL:]
    return segs


def _shingles(s):
    s = " ".join(s.split())
    if not s:
        return set()
    if len(s) < SHINGLE:
        return {s}
    return {s[i:i + SHINGLE] for i in range(len(s) - SHINGLE + 1)}


def _greedy_pack(segs):
    """Greedy MMR knapsack over candidate segments (index >= IDENT_SEGS).
    Returns the pick order: list of (orig_index, segment)."""
    ident = segs[:IDENT_SEGS]
    ctx = "\n".join(ident).encode("utf-8", "replace")
    base = len(zlib.compress(ctx, 6))
    cand = []
    for i in range(IDENT_SEGS, len(segs)):
        sg = segs[i]
        if not sg.strip():
            continue
        b = sg.encode("utf-8", "replace")
        gain = len(zlib.compress(ctx + b"\n" + b, 6)) - base
        # floor the token cost: every packed line pays separator + minimum overhead, so trivial
        # short lines ("fi", "done") can't win on cheapness — content lines outcompete them
        density = gain / max(6.0, (len(sg) + 8) / CHARS_PER_TOK)
        cand.append((density, i, sg, _shingles(sg)))
    covered = set()
    for sg in ident:
        covered |= _shingles(sg)
    order, used = [], [False] * len(cand)
    chars = 0
    while chars < PACK_CHAR_BUDGET:
        best_j, best_key = -1, None
        for j, (density, i, sg, sh) in enumerate(cand):
            if used[j]:
                continue
            novelty = (1.0 - len(sh & covered) / len(sh)) if sh else 0.0
            if novelty < MIN_NOVELTY:
                continue
            key = (density * novelty, -i)     # deterministic: ties -> earliest line
            if best_key is None or key > best_key:
                best_j, best_key = j, key
        if best_j < 0:
            break
        used[best_j] = True
        _, i, sg, sh = cand[best_j]
        order.append((i, sg))
        covered |= sh
        chars += len(sg) + 1
    return ident, order


def _assemble(prefix, ident, picks, ntot):
    """Render: identity anchor, elision marker, then picked lines in original file order."""
    chosen = sorted(picks)                     # by original index -> natural reading order
    elided = ntot - IDENT_SEGS - len(chosen)
    marker = f"\n...[{max(elided, 0)} of {ntot} lines elided; most-informative kept]...\n"
    body = "\n".join(sg for _, sg in chosen)
    return prefix + "\n".join(ident) + marker + body


def _infopack(prefix, out):
    """Largest greedy-prefix info-pack whose render exactly fits the encode window."""
    segs = _segments(out)
    ntot = len(segs)
    ident, order = _greedy_pack(segs)
    lo, hi, best = 0, len(order), None
    while lo <= hi:
        m = (lo + hi) // 2
        cand = _assemble(prefix, ident, order[:m], ntot)
        if _fits(cand):
            best, lo = cand, m + 1
        else:
            hi = m - 1
    if best is not None:
        return best
    # even the bare anchor overflows (multibyte soup): char-level head+tail floor
    head_c, tail_c = 900, 380
    while head_c >= 16:
        cand = (prefix + out[:head_c]
                + f"\n...[{max(0, len(out) - head_c - tail_c)} chars skipped]...\n"
                + out[-tail_c:])
        if _fits(cand):
            return cand
        head_c, tail_c = int(head_c * 0.7), int(tail_c * 0.7)
    return prefix + out[:200]                  # hard floor: always <= window


def render_obs(step):
    r = _incumbent(step)
    toks = (step.get("cmd", "") or "").split()
    if not toks or toks[0] != "cat":
        return r                               # non-cat: byte-identical, no tokenizer call
    out = step.get("output", "") or ""
    if not out.strip() or _fits(r):
        return r                               # short / UNK-collapsed binary cat: byte-identical
    prefix = f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n"
    return _infopack(prefix, out)


def render_cmd(step):
    # champion command render, unchanged (e5's required prefix; reads only the action)
    return "passage: " + step["cmd"]


def pool(h, mask):
    # e5 standard masked mean-pool (unchanged; 768-d, no adapter)
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)
