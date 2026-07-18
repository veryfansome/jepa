"""stream + perception (R7): ROLE-CANONICAL structured multi-vector observation readout.

This delivers BOTH halves of the multi-vector contract in one module:
  * a perception render `render_obs_multi(step) -> list[str]` (+ `K`, `MODEL`, `pool`,
    `render_obs`, `render_cmd`) consumed by evolve/mv_encode.py to build a data root whose
    caches carry per-step "z_obs_multi" [n,K,D] + "obs_valid" [n,K]; the single-vector
    z_obs/z_cmd (the FIXED target/eval space) are inherited byte-identically from enc_e5_base;
  * a stream `collate/extract_cmd_pred/flatten_predictions/leakage_ok` that lays each step out
    as [cmd_i, obs_i^0 .. obs_i^{K-1}] (stride 1+K) and reads the structured segments, while the
    per-step TARGET stays the single standardized mean-pooled z_obs (baselines/foils/eval fixed).

WHY THIS AND NOT THE LINE-STRIPS (mv_obs_k4, which scored below its matched control):
  The K=4 line-strip render split output at arbitrary line-INDEX boundaries, so segment k of one
  observation and segment k of another held unrelated entries (no cross-observation alignment for
  the stream's per-slot positional read), and each strip was still a mean-pool over its third of
  the raw, metadata-heavy lines. Here each of the K segments is keyed to a FIXED semantic ROLE,
  identical across every observation, and the ls name-content is CANONICALIZED (nuisance columns
  stripped, entries sorted) so the discriminative payload gets its own un-diluted vector:
    seg 0  STATE   : "passage: cwd=<cwd> exit=<exit>"                (always present)
    seg 1  NAMES   : ls  -> first half of foveated+sorted entry NAMES (metadata stripped)
                     cat -> the file's identifying HEAD lines (e.g. os-release ID line)
                     other-> first content strip
    seg 2  BODY    : ls  -> second half of the sorted entry NAMES
                     cat -> the file BODY continuation
                     other-> second content strip
    seg 3  SIG     : a compact STRUCTURAL type-signature line, content-addressable by
                     command-type + directory shape: "verb=<v> n=<bucket> types=<d:.. f:.. l:..>
                     ext=<top extensions>" — a slot stable across observations of the same
                     command family that directly separates ls-vs-cat and directory shape.
  Empty roles are dropped (a cd with no output yields only seg 0) and masked out via key_pad.

FIXED across streams (so fitness stays the honest content-verb margin): the per-step target is the
single standardized mean-pooled z_obs, the eval space, the foils, and every baseline are unchanged
— only what the predictor READS changes.

Causality: identical stream order (cmd_i before its obs segments; steps in order), so a causal
arch mask gives cmd_t access only to segments of obs_<t. leakage_ok corrupts ALL K segments of
obs_3 and requires cmd predictions at steps <= 3 to be bit-stable.
"""

import re

import torch

from realenv import seq_worldmodel as M
# Inherit the champion single-vector e5 recipe verbatim: MODEL, render_obs, render_cmd, pool are
# BIT-IDENTICAL to enc_e5_base, so z_obs/z_cmd (the FIXED target/eval space) match data/dockerfs-e5.
from evolve.chunks.perception.enc_e5_base import MODEL, render_obs, render_cmd, pool  # noqa: F401
from evolve.chunks.perception.baseline import OBS_CAP

NAME = "r7_role_multivec"
DESCRIPTION = ("Role-canonical structured multi-vector observation stream: "
               "[cmd, state, names, body, type-signature] per step (stride 5); "
               "targets/eval/baselines unchanged (single-vector space).")

# ---------------------------------------------------------------------------
# Perception half: render_obs_multi (+ K). Consumed by evolve/mv_encode.py.
# ---------------------------------------------------------------------------
K = 4
STRIDE = 1 + K
SEG_CAP = 800  # chars per content segment

# Long-format ls line parser (reused convention from r6_foveal_ls_names): pull the TYPE char and
# the NAME, discarding the constant permission/link/owner/group/size/date nuisance columns.
_LS_LINE = re.compile(
    r'^([bcdlpsD\-])[rwxsStTlL.\-]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+'
    r'(?:\d+,\s+\d+|\S+)\s+\S+\s+\S+\s+\S+\s+(.+?)\s*$')
_TOTAL = re.compile(r'^total\s+\d+\s*$')
_SECTION = re.compile(r'^\S.*:\s*$')  # `-R` section header like `./etc:`


def _verb(step):
    return (step.get("cmd", "") or "").split()[0:1] or [""]


def _ls_entries(out):
    """Return (names, type_marks) with metadata stripped. Falls back to raw tokens for
    short-format (`ls -1`) or busybox output where the long-format regex does not match."""
    names, types = [], []
    for ln in out.split("\n"):
        s = ln.rstrip()
        if not s or _TOTAL.match(s) or _SECTION.match(s):
            continue
        m = _LS_LINE.match(s)
        if m:
            names.append(m.group(2).strip())
            types.append(m.group(1))
        else:
            # short-format: line may hold one or several space-separated names; keep tokens
            for tok in s.split():
                names.append(tok)
                types.append('-')
    return names, types


_EXT = re.compile(r'\.([A-Za-z0-9]{1,6})$')


def _bucket(n):
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 8:
        return "3-8"
    if n <= 24:
        return "9-24"
    if n <= 64:
        return "25-64"
    return "65+"


def _signature(verb, names, types):
    """A compact, order-invariant structural fingerprint keyed by command family + dir shape."""
    from collections import Counter
    tc = Counter(types)
    tstr = " ".join(f"{t}:{tc[t]}" for t in sorted(tc)) if tc else "-"
    exts = Counter()
    for nm in names:
        base = nm.split(" -> ")[0]  # symlink: signature the link name, not the target
        m = _EXT.search(base)
        exts[m.group(1).lower() if m else "_"] += 1
    estr = " ".join(f"{e}:{exts[e]}" for e, _ in exts.most_common(6))
    return f"verb={verb} n={_bucket(len(names))} types={tstr} ext={estr}"


def render_obs_multi(step):
    """K role-canonical segments; empty roles dropped (masked out downstream). seg0 always present."""
    cwd = step.get("cwd", "/")
    exit_ = step.get("exit", 0)
    verb = _verb(step)[0]
    out = (step.get("output", "") or "")[:OBS_CAP]

    segs = [f"passage: cwd={cwd} exit={exit_}"]  # seg 0: STATE (always present)

    if verb == "ls":
        names, types = _ls_entries(out)
        if names:
            order = sorted(range(len(names)), key=lambda i: names[i])  # canonical order-invariant
            names = [names[i] for i in order]
            types = [types[i] for i in order]
            half = (len(names) + 1) // 2
            seg1 = "\n".join(names[:half])[:SEG_CAP]
            seg2 = "\n".join(names[half:])[:SEG_CAP]
            if seg1:
                segs.append("passage: " + seg1)   # seg 1: NAMES (first half)
            if seg2:
                segs.append("passage: " + seg2)   # seg 2: NAMES (second half)
            # seg 3: structural SIGNATURE (only if room; role slot index kept stable below)
            sig = _signature(verb, names, types)
            segs.append("passage: " + sig)
    else:
        # cat / uname / config: identifying head + body continuation, then a light signature.
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if lines:
            head = "\n".join(lines[: max(1, len(lines) // 2)])[:SEG_CAP]
            body = "\n".join(lines[max(1, len(lines) // 2):])[:SEG_CAP]
            if head:
                segs.append("passage: " + head)   # seg 1: HEAD / identifying content
            if body:
                segs.append("passage: " + body)   # seg 2: BODY continuation
            segs.append(f"passage: verb={verb} nlines={_bucket(len(lines))}")  # seg 3: SIG

    return segs[:K]


# ---------------------------------------------------------------------------
# Stream half: collate / extract_cmd_pred / flatten_predictions / leakage_ok.
# ---------------------------------------------------------------------------
def collate(batch, device):
    maxn = max(s["z_obs"].shape[0] for s in batch)
    L = STRIDE * maxn
    B = len(batch)
    tok = torch.zeros(B, L, M.D)
    types = torch.zeros(B, L, dtype=torch.long)          # {0=cmd, 1=obs} — arch has Embedding(2,d)
    key_pad = torch.ones(B, L, dtype=torch.bool)         # True = pad
    tgt = torch.zeros(B, maxn, M.D)                       # FIXED single-vector target
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
    for bi, s in enumerate(batch):
        n = s["z_obs"].shape[0]
        zm, valid = s["z_obs_multi"], s["obs_valid"]
        for i in range(n):
            base = STRIDE * i
            tok[bi, base] = s["z_cmd"][i]
            types[bi, base] = 0
            key_pad[bi, base] = False
            for k in range(K):
                if valid[i, k]:
                    tok[bi, base + 1 + k] = zm[i, k]
                    key_pad[bi, base + 1 + k] = False
                types[bi, base + 1 + k] = 1
            tgt[bi, i] = s["z_obs"][i]
            cmd_mask[bi, i] = True
    return {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
            "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::STRIDE]


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    """Mirrors seq_worldmodel.flatten_predictions (step order; prev/true from the single z_obs)."""
    net.eval()
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, h = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = pred[:, 0::STRIDE].cpu()
        cmd_h = h[:, 0::STRIDE].cpu()
        for bi, s in enumerate(chunk):
            n = s["z_obs"].shape[0]
            for t in range(n):
                preds.append(cmd_pred[bi, t]); hids.append(cmd_h[bi, t])
                trues.append(s["z_obs"][t])
                prevs.append(s["z_obs"][t - 1] if t > 0 else torch.zeros(M.D))
                cmds.append(s["cmds"][t]); imgs.append(s["image"])
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs,
            "verbs": [M.verb_of(c) for c in cmds]}


@torch.no_grad()
def leakage_ok(net, device):
    """Corrupt ALL K segments of obs_3; cmd predictions at steps 0..3 must not move (causal)."""
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, M.D), "z_cmd": torch.randn(6, M.D),
            "z_obs_multi": torch.randn(6, K, M.D), "obs_valid": torch.ones(6, K, dtype=torch.bool),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::STRIDE].clone().cpu()
    b1 = collate(seq, device)
    base = STRIDE * 3
    b1["tok"][0, base + 1:base + 1 + K] = torch.randn(K, M.D, device=device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::STRIDE].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())

