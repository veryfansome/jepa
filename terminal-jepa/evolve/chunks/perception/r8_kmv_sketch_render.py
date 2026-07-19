"""perception TIER-1 (serialization under the token bottleneck): intfloat/e5-base-v2 + the champion
prefixes/pooling (enc_e5_base), with a KMV-SKETCH TRUNCATION POLICY replacing head truncation.

The untried lever: WHAT survives the encoder's 256-token cap. Measured on this corpus with the e5
tokenizer (chars/token p50 = 2.2-2.7, p05 = 1.75), the incumbent's 1600-char head-cut is an
illusion — the tokenizer silently keeps only the FIRST ~550-600 chars, and 29% of cat and 15% of
ls observations (exactly the content verbs fitness is scored on) lose everything after that point.
For ls that surviving prefix is the ALPHABETICAL head of the listing, which is dominated by
early-alphabet entries shared across distros (bin, boot, dev, etc ...), so truncated listings from
different systems look artificially alike: the true obs embedding loses its system-discriminative
tail, same-verb foils crowd in, and a world model that correctly knows WHICH system it is on
cannot be rewarded for it. The no-knowledge baselines lose far less from this, so restoring
discriminative content to the true embedding is a differential lever for the WM.

Mechanism (bottom-k min-wise sketch; Broder 1997, K-minimum-values distinct-value sketches): when
a render would overflow the token window, KEEP THE LINES WITH THE SMALLEST MD5 HASH — for ls,
hashing the FILENAME field only (long-format metadata stripped from the key, so the same entry on
two systems hashes identically even when its size/date/links differ) — until a calibrated
token-budget estimate is filled; render the kept lines in their original order plus an omission
marker. Bottom-k selection by a fixed content hash is the classic KMV sketch: two systems' sketches
co-retain exactly the shared entries (co-retention agreement 1.00 measured on cross-system /etc
listings), so embedding similarity between truncated listings tracks the TRUE overlap of the full
listings instead of the overlap of their alphabetical prefixes — and for `ls -R` the sketch becomes
a whole-filesystem fingerprint rather than the first two directories. cat output is a sequential
document, not a set, so it gets a HEAD+TAIL policy (60/40): identity signals concentrate at the
top (shebang, comments, package names) and end of files, the middle of long files is bulk. The
token budget uses a field-structural estimator (fields + punctuation + digits/1.8 + subword
overflow, with a shredding branch for >14-char unbroken alpha runs like base64/PGP) calibrated on
the corpus: 93% of sketched renders fit 256 real tokens exactly (worst case 276 = marker clipped),
mean fill 202/256. Renders that fit the window are BYTE-IDENTICAL to enc_e5_base (~85% of steps;
uname/cd always), so regression risk is confined to the truncated content-verb steps the policy
targets. Deterministic (fixed md5, no RNG, no corpus-fitted state), leak-free (render_obs reads
only its own step; render_cmd only step['cmd']), no learned params; e5 is 768-d so mean-pool needs
no adapter.
"""
import hashlib
import re

NAME = "r8_kmv_sketch_render"
DESCRIPTION = (
    "e5-base-v2 with KMV (bottom-k min-hash) sketch truncation: under the 256-token cap, keep the "
    "smallest-hash ls entries (keyed by filename, metadata-invariant) and head+tail of cat output, "
    "so truncated observation embeddings track true content overlap instead of alphabetical prefixes."
)

MODEL = "intfloat/e5-base-v2"
OBS_CAP = 1600          # incumbent char cap, kept for the byte-identical under-budget path
TOKEN_BUDGET = 230      # estimated content tokens (~256 minus header ~22 and marker slack)
_MARGIN = 1.3           # safety factor on the token estimator (calibrated: 7% mild overflow, max +20)

# long-format `ls -l` line -> capture the NAME field (incl. symlink 'name -> target'); the hash key
# must ignore per-system metadata (size/date/links/owner) so shared entries co-retain across systems.
_LS_LONG = re.compile(
    r'^([bcdlpsD\-])[rwxsStTlL.\-]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+'
    r'(?:\d+,\s+\d+|\S+)\s+\S+\s+\S+\s+\S+\s+(.+?)\s*$')


def _hash_key(line):
    m = _LS_LONG.match(line)
    return m.group(2) if m else line


def _est_field(f):
    """WordPiece token estimate for one whitespace field (calibrated on-corpus vs the e5 tokenizer)."""
    a = sum(c.isalpha() for c in f)
    d = sum(c.isdigit() for c in f)
    p = len(f) - a - d
    if a > 14:  # unbroken alpha run: not a dictionary word (base64/hash/PGP) -> shredded ~2.2 chars/tok
        return p + d / 1.8 + a / 2.2
    return p + d / 1.8 + (1.0 if a else 0.0) + max(0.0, a - 4) / 4.0


def _est(line):
    return 1.0 + _MARGIN * sum(_est_field(f) for f in line.split())


def _sketch_ls(out, budget):
    """Bottom-k min-hash (KMV) line sketch: keep smallest-md5-of-filename lines up to the token
    budget, rendered in original order + omission marker. Deterministic; same entry -> same hash
    on every system, so two systems' sketches co-retain exactly their shared entries."""
    lines = out.split("\n")
    n = len(lines)
    order = sorted(range(n), key=lambda i: (
        hashlib.md5(_hash_key(lines[i]).encode("utf-8", "replace")).hexdigest(), i))
    keep, used = [], 0.0
    for i in order:
        c = _est(lines[i])
        if used + c > budget:
            continue  # line too big for remaining budget; smaller-hash lines may still fit
        keep.append(i)
        used += c
    if not keep:  # degenerate: a single enormous line
        return out[:400] + "\n...[line truncated]"
    keep.sort()
    return "\n".join(lines[i] for i in keep) + f"\n...[{n - len(keep)}/{n} lines]"


def _head_tail(out, budget):
    """Sequential-document policy for cat (and any future verb): 60% of the budget from the head,
    the rest from the tail, with an omission marker between."""
    lines = out.split("\n")
    n = len(lines)
    head_budget = 0.6 * budget
    head, used = [], 0.0
    for ln in lines:
        c = _est(ln)
        if used + c > head_budget:
            break
        head.append(ln)
        used += c
    tail_budget = budget - used
    tail, used_t = [], 0.0
    for ln in reversed(lines[len(head):]):
        c = _est(ln)
        if used_t + c > tail_budget:
            break
        tail.append(ln)
        used_t += c
    tail.reverse()
    if not head and not tail:  # degenerate: a single enormous line
        return out[:400] + "\n...[line truncated]"
    omitted = n - len(head) - len(tail)
    return "\n".join(head) + f"\n...[{omitted} lines omitted]...\n" + "\n".join(tail)


def render_obs(step):
    out = step.get("output", "") or ""
    total = sum(_est(l) for l in out.split("\n")) if out else 0.0
    if total > TOKEN_BUDGET:
        if (step.get("cmd", "") or "").split()[0:1] == ["ls"]:
            out = _sketch_ls(out, TOKEN_BUDGET)
        else:
            out = _head_tail(out, TOKEN_BUDGET)
    elif len(out) > OBS_CAP:
        # unreachable in practice (>1600 chars always over budget) but kept for exact incumbent parity
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"


def render_cmd(step):
    # champion command render, unchanged (e5's required prefix; reads only the action)
    return "passage: " + step["cmd"]


def pool(h, mask):
    # e5 standard masked mean-pool; e5-base-v2 is 768-d so no adapter is needed
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)
