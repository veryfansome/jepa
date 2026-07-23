"""FROZEN four-axis measured-classification protocol for dockerfs3 (freeze-order step 6).

The honesty core: given a collected v3 root (raw per-image jsonl), it MEASURES — never
assumes — a class for every (sig, mode, state_scope[, ws_observed]) CELL, by computing the
four pre-registered axes on RENDERED text and applying the §9.2 ordered precedence. It emits
the machine-readable `dockerfs3-classes.json` shape that `bench_versions.resolve()` loads and
asserts against summary.json (§9.5). Deterministic, seed-free (foil sampling is pinned to
seed 0, the v2 protocol constant), fail-closed (a cell under the coverage floor is
`under-floor` — excluded and reported, never guessed).

SOURCES OF TRUTH (this tool implements them; on conflict THEY win):
  benchmarks/dockerfs3-prereg.md §4.5 / design-draft §9 — the unit, the four axes, and the
    §9.2 SIX classes + ordered precedence (semi-echo, ack, echo/const, sim, noisy-excluded,
    content) with the created-scope ws_observed sub-cell split (round-6 B1).
  realenv/verbsig.py — the ONE sig/mode/cell labeler (F8: sig+mode recomputed from the record).
  realenv/shell_state.py — the SST (axis-3 determined/BOT surface + exact-match; mode="sst"
    is render parity, mode="collection" recomputes ws_observed's ws.observed flag).
  realenv/render_canon.py — the pre-perception render mask (axes are on post-canon text).
  benchmarks/axis1_measure.py — the FROZEN v2 axis-1 protocol this generalizes from per-verb
    to per-cell (retrieve_by_cmd command-only predictor + same-cell foil top-1, e5 space).

THE FOUR MEASURED AXES (§9.1), all per cell:
  axis-1  no-history predictability — top-1 of a command-only predictor (retrieve_by_cmd:
          nearest CROSS-TRAJECTORY fit command's observation) scored against same-cell foils.
          θ1 = 0.59 (v2 midpoint; recalibrated by amendment on pilot data).
  axis-2  cmd<->obs rendered-text containment (own step), prior 0.656. Two columns: OUTPUT-body
          containment + a render-prefix (resulting cwd) containment so cwd-echo channels (cd)
          are measurable; the rule keys on their max.
  axis-2' history containment (DG-6 derivability / workspace-echo-loop): does the observation
          transit an EARLIER command's text OR a prior rendered observation in the trajectory.
  axis-3  SST exact-match rate = fold the cell's steps through ShellState(mode="sst") and count
          the fraction where predict() EXACTLY equals the recorded observation, BOT counted as
          mismatch (the all-steps denominator, §9.1). >=0.90 => sim. Determined-coverage and
          accuracy-on-determined are reported as separate ledger columns.
  axis-4  replay stability — cross-replay field-flip rate from a twin mint (0 for a
          deterministic mint by construction; the ceiling C keys the noisy-excluded rule).
          Without a --twin-root, C defaults to 1.0 (deterministic-by-construction) and the
          protocol still exists / runs.

§9.2 ORDERED PRECEDENCE (borderline => excluded side; first matching rule wins):
  1. axis-2 stat >= thresh OR axis-2' stat >= thresh          -> semi-echo   (excluded)
  2. success-mode near-dup >= 0.95 AND median output <= 5 ch  -> ack         (excluded)
  3. axis-1 top-1 >= theta1                                   -> echo/const  (excluded)
  4. axis-3 exact-match >= 0.90                               -> sim         (excluded)
  5. axis-4 ceiling C < 0.80 (post-masking)                   -> noisy-excluded
  6. else                                                     -> content     (enters fitness)

  uv run python -m benchmarks.class_measure --root <v3-root> [--out classes.json]
  uv run python -m benchmarks.class_measure --smoke        # tiny synthetic-root unit smoke
"""

import argparse
import collections
import hashlib
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from realenv import render_canon, verbsig
from realenv import seq_worldmodel as M
from realenv import shell_state as S

TOOL_VERSION = "v3-4axis-1"

# ---------------------------------------------------------------- frozen defaults
# Every threshold is the pre-registered PRIOR; the pilot freezes the final numbers by dated
# amendment (§2). They are surfaced as CLI overrides so the recalibration is a parameter sweep,
# never a code edit — but the DEFAULTS below are the committed priors.
DEFAULTS = {
    "coverage_floor": 30,     # G-COV: >=30 cross-image pairs per cell (else battery)
    "min_images": 2,          # a "cross-image pair" needs the cell to span >=2 images
    "axis1_theta1": 0.59,     # §9.1 v2 midpoint
    "axis2_thresh": 0.656,    # §9.1 v2 prior
    "axis2p_thresh": 0.656,   # axis-2' shares the axis-2 prior until recalibrated (§2 item 3)
    "axis3_sim": 0.90,        # §9.2 rule 4
    "axis4_content": 0.80,    # §9.2 rule 5 content-eligible floor
    "axis4_report": 0.50,     # 0.50..0.80 => reported-only
    "ack_dup": 0.95,          # §9.2 rule 2 success-mode near-dup
    "ack_maxchars": 5,        # §9.2 rule 2 median OUTPUT-field chars (field-scoped)
}

CLASSES = ("content", "semi-echo", "ack", "echo/const", "sim", "noisy-excluded", "under-floor")

_TOK_RE = re.compile(r"[A-Za-z0-9_./:@%+=,-]+")


# ================================================================ loading / cells

def load_root(root, splits=None):
    """Read a v3 root's per-image sequences. Returns a list of seq dicts, each
    {image, seq_idx, split, steps:[{cmd,output,exit,cwd,meta}]}. `splits` defaults to
    whichever of train/val the root actually carries (an ablate root has train only, F6)."""
    root = pathlib.Path(root)
    if splits is None:
        splits = [s for s in ("train", "val") if (root / f"{s}.jsonl").exists()]
    if not splits:
        raise FileNotFoundError(f"no train.jsonl / val.jsonl under {root}")
    seqs = []
    for split in splits:
        path = root / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path) as fh:
            for si, line in enumerate(fh):
                sq = json.loads(line)
                sq.setdefault("split", split)
                sq.setdefault("seq_idx", si)
                seqs.append(sq)
    return seqs, list(splits)


def _output_empty(step):
    return not step.get("output")


def _canon_output(step):
    """The post-render_canon, OBS_CAP-truncated observation body (render parity)."""
    try:
        c = render_canon.canon(step)
    except Exception:               # noqa: BLE001 — canon is total; never let it abort a run
        c = step
    out = c.get("output", "") or ""
    if len(out) > M.OBS_CAP:
        out = out[:M.OBS_CAP]
    return out


def _created_target(parsed, cwd):
    """The path a created-scope classification keys on (the read/written path), cwd-resolved."""
    f = parsed.get("form")
    if f == "redir":
        return S.normpath(parsed["dst"], cwd)
    for k in ("path", "file", "dir"):
        if parsed.get(k):
            return S.normpath(parsed[k], cwd)
    if f == "cond":
        return S.normpath(parsed["read"]["path"], cwd)
    return None


def cell_rows(seqs, error_templates_for, diagnostics):
    """Fold every trajectory once (collection-mode SST for ws_observed + an sst-mode SST for
    the axis-3 determined/exact surface) and return a flat per-step table. sig and mode are
    RECOMPUTED from the record (F8, authoritative); state_scope is read from the collection
    tracker's meta (the collection-mode authority); ws_observed is recomputed from the SST's
    ws.observed flag for readback steps (the write step carries it in meta). Steps whose command
    is out-of-universe are counted in diagnostics and dropped (fail-closed: never guessed)."""
    rows = []
    seq_id = 0
    for sq in seqs:
        image = sq.get("image", "?")
        steps = sq.get("steps", [])
        tmpl = error_templates_for(image) if error_templates_for else {}
        col = S.ShellState(mode="collection", error_templates=tmpl)
        sst = S.ShellState(mode="sst", error_templates=tmpl)
        for si, step in enumerate(steps):
            cmd = step.get("cmd", "")
            meta = step.get("meta", {}) or {}
            try:
                sig = verbsig.sig(cmd)
            except ValueError as e:
                diagnostics["out_of_universe"] += 1
                if len(diagnostics["out_of_universe_ex"]) < 8:
                    diagnostics["out_of_universe_ex"].append((image, cmd, str(e)[:80]))
                # still advance the trackers so downstream steps stay consistent
                _safe_fold(col, step); _safe_fold(sst, step)
                continue
            mode = verbsig.mode(sig, step.get("exit", 0), _output_empty(step))
            scope = meta.get("state_scope", "native")
            if scope not in verbsig.SCOPES:
                diagnostics["bad_state_scope"] += 1
                scope = "native"
            # ws_observed: created-scope only; the write step carries meta.ws_observed, a
            # readback recomputes it from the ws.observed flag the earlier write fold set.
            ws_obs = None
            if scope == "created":
                ws_obs = _ws_observed(step, meta, col)
            # axis-3 determined/exact BEFORE folding this step into the sst tracker
            det, exact = _axis3_step(sst, si, step)
            rows.append({
                "image": image, "seq_id": seq_id, "step": si, "cmd": cmd,
                "sig": sig, "mode": mode, "scope": scope, "ws_observed": ws_obs,
                "output_body": _canon_output(step), "cwd": step.get("cwd", "/"),
                "exit": step.get("exit", 0), "hit": meta.get("hit"),
                "determined": det, "exact": exact,
            })
            _safe_fold(col, step); _safe_fold(sst, step)
        seq_id += 1
    return rows


def _safe_fold(st, step):
    rec = {"cmd": step.get("cmd", ""), "output": step.get("output", "") or "",
           "exit": step.get("exit", 0), "cwd": step.get("cwd", "/")}
    try:
        st.fold(rec)
    except S.ParseError:
        st.vt += 1          # keep the virtual clock aligned with the record's step index


def _ws_observed(step, meta, col):
    """created-scope ws_observed (§6.3): the created file's content transited a prior COMMAND's
    text OR a prior RENDERED OBSERVATION. The write step caches it in meta; a readback consults
    the collection SST's ws.observed flag (set by the earlier write fold). Default False (blind):
    a mis-labeled blind cell still faces axis-2' at the slice level, so blind is the safe side."""
    if "ws_observed" in meta:
        return bool(meta["ws_observed"])
    try:
        parsed = S.parse_command(step.get("cmd", ""))
    except S.ParseError:
        return False
    tgt = _created_target(parsed, col.cwd)
    if tgt is None:
        return False
    key = col._resolve(S.normpath(tgt, col.cwd))
    entry = col.ws.get(key) or col.ws.get(tgt)
    return bool(entry.get("observed")) if entry else False


def _axis3_step(sst, si, step):
    """(determined, exact) for one step under the sst-mode tracker, BEFORE folding it.
    exact-match mirrors the golden-rule mint cross-check: a non-BOT predict() dict compared
    field-for-field to the recorded {output(post-canon), exit, cwd}. BOT => (False, False)."""
    try:
        pred = sst.predict(sst.vt, step.get("cmd", ""))
    except S.ParseError:
        return False, False
    if pred is S.BOT:
        return False, False
    try:
        rec_out = render_canon.canon(step).get("output", "") or ""
    except Exception:               # noqa: BLE001
        rec_out = step.get("output", "") or ""
    rec = {"output": rec_out, "exit": step.get("exit", 0), "cwd": step.get("cwd", "/")}
    return True, (pred == rec)


# ================================================================ text containment

def _tok(text):
    return set(_TOK_RE.findall(text or ""))


def _containment(a_tokens, b_tokens):
    """Directional token-set containment |a & b| / |a| — the fraction of a's tokens present in
    b. 0.0 when a is empty (nothing to be echoed)."""
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def axis2_step(row):
    """(output_containment, prefix_containment) for one step. output: how much of the OUTPUT
    body is echoed from the command; prefix: how much of the resulting cwd is echoed (catches
    cd's `cwd=<target>`). The rule keys on the max of the two."""
    cmd_tok = _tok(row["cmd"])
    out_c = _containment(_tok(row["output_body"]), cmd_tok)
    pre_c = _containment(_tok(row["cwd"]), cmd_tok)
    return out_c, pre_c


def axis2p_seq(seq_rows):
    """History containment per step: |output tokens present in ANY earlier command OR earlier
    observation| / |output tokens|. Detects the workspace echo-loop and derivable readbacks."""
    prior = set()
    vals = []
    for r in seq_rows:
        vals.append(_containment(_tok(r["output_body"]), prior))
        prior |= _tok(r["cmd"])
        prior |= _tok(r["output_body"])
    return vals


# ================================================================ axis-1 (e5 space)

def _fake_embedder(dim=128):
    """Deterministic hashing bag-of-tokens embedder — no network, cross-run stable. Identical
    texts map to identical vectors (so retrieve_by_cmd resolves exact command matches). Used by
    the unit smoke and any --encoder fake run; NOT the frozen production encoder."""
    def embed(texts):
        out = torch.zeros(len(texts), dim)
        for i, t in enumerate(texts):
            for tokn in _TOK_RE.findall(t or "") or [""]:
                h = int(hashlib.md5(tokn.encode("utf-8", "ignore")).hexdigest(), 16)
                out[i, h % dim] += 1.0
                out[i, (h // dim) % dim] += 0.5
            n = out[i].norm()
            if n > 0:
                out[i] /= n
        return out
    return embed


def _e5_embedder():
    """The frozen production encoder: enc_e5_base renders, max_length=256, mean-pool (the
    axis1_measure v2 protocol). Lazy — only imported/downloaded when actually used."""
    from transformers import AutoModel, AutoTokenizer
    from evolve.chunks.perception import enc_e5_base as PERC
    device = M.pick_device()
    tok = AutoTokenizer.from_pretrained(PERC.MODEL)
    model = AutoModel.from_pretrained(PERC.MODEL).to(device).eval()

    @torch.no_grad()
    def embed(texts, bs=96):
        out = torch.zeros(len(texts), M.D)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            e = tok([texts[j] for j in b], return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
            e = {k: v.to(device) for k, v in e.items()}
            h = model(**e).last_hidden_state
            p = PERC.pool(h, e["attention_mask"]).float().cpu()
            for k, j in enumerate(b):
                out[j] = p[k]
        return out
    return embed


def _render_obs_row(row):
    """The axis-1 observation render (seq_worldmodel.render_obs parity, on post-canon output)."""
    return f"cwd={row['cwd']} exit={row['exit']}\n{row['output_body']}"


def axis1_predictions(rows, embed):
    """retrieve_by_cmd (the command-only, no-history predictor): each step's predicted obs is
    the observation of the nearest CROSS-TRAJECTORY fit command in e5 space. Standardized with
    the whole-root stats (a single corpus; there is no held-out corpus at class-measure time —
    all rendered steps are the retrieval pool). Returns z_true[N,D], z_pred[N,D]."""
    obs_texts = [_render_obs_row(r) for r in rows]
    cmd_texts = [r["cmd"] for r in rows]
    z_obs = embed(obs_texts)
    z_cmd = embed(cmd_texts)
    mo = z_obs.mean(0, keepdim=True); so = z_obs.std(0, keepdim=True).clamp(min=1e-6)
    mc = z_cmd.mean(0, keepdim=True); sc = z_cmd.std(0, keepdim=True).clamp(min=1e-6)
    z_obs = (z_obs - mo) / so
    z_cmd = (z_cmd - mc) / sc
    seqid = torch.tensor([r["seq_id"] for r in rows])
    N = z_cmd.shape[0]
    pred = torch.zeros_like(z_obs)
    for i in range(0, N, 256):
        sl = slice(i, min(i + 256, N))
        d = torch.cdist(z_cmd[sl], z_cmd)                       # [b, N]
        same = seqid[sl].unsqueeze(1) == seqid.unsqueeze(0)     # mask own trajectory (no-history)
        d = d.masked_fill(same, float("inf"))
        pred[sl] = z_obs[d.argmin(1)]
    return z_obs, pred


def axis1_cell_top1(z_true, z_pred, idxs):
    """Same-cell foil top-1 for a cell's step indices (the v2 per_verb_breakdown statistic,
    pooled by cell). Seed 0, 4 rounds, 63 foils — the frozen axis1_measure protocol."""
    ii = torch.tensor(idxs)
    sub_true = z_true[ii]; sub_pred = z_pred[ii]
    labels = ["c"] * len(idxs)
    return M.retrieval(sub_pred, sub_true, labels, seed=0)["top1_sameverb"]


# ================================================================ axis-4 (twin)

def axis4_flips(rows, twin_root, error_templates_for, diagnostics):
    """Cross-replay field-flip rate per cell from a twin mint (§9.1 axis-4). Aligns steps by
    (image, seq_idx, step) and compares {output(post-canon), exit, cwd}; flip_rate = fraction of
    aligned steps that differ. Returns {cell_key: (flip_rate, n_aligned)} or {} when no twin
    (C defaults to 1.0 downstream — deterministic-by-construction)."""
    if not twin_root:
        return {}
    twin_seqs, _ = load_root(twin_root)
    twin = {}
    for sq in twin_seqs:
        for si, step in enumerate(sq.get("steps", [])):
            twin[(sq.get("image"), sq.get("seq_idx"), si)] = step
    by_cell = collections.defaultdict(lambda: [0, 0])          # cell -> [flips, aligned]
    # rebuild cell keys the same way the main table does (cheap: reuse the row objects)
    for r in rows:
        key = (r["image"], _seq_idx_of(r), r["step"])
        tw = twin.get(key)
        if tw is None:
            continue
        cell = r["_cell"]
        a = (r["output_body"], r["cwd"], r["exit"])
        tb = (_canon_output(tw), tw.get("cwd", "/"), tw.get("exit", 0))
        by_cell[cell][1] += 1
        if a != tb:
            by_cell[cell][0] += 1
    return {c: (f / n if n else 0.0, n) for c, (f, n) in by_cell.items()}


def _seq_idx_of(row):
    return row.get("_seq_idx", row["seq_id"])


# ================================================================ precedence

def classify(stats, thr):
    """§9.2 ordered precedence on a cell's measured statistics. Returns (class, rule_no).
    A rule whose statistic is unavailable (None) cannot fire — measurement gaps never
    manufacture an exclusion, and never a content guess either."""
    a2 = stats["axis2_stat"]
    a2p = stats["axis2p_stat"]
    if (a2 is not None and a2 >= thr["axis2_thresh"]) or \
       (a2p is not None and a2p >= thr["axis2p_thresh"]):
        return "semi-echo", 1
    dup, med = stats["ack_dup"], stats["ack_median_chars"]
    if dup is not None and med is not None and dup >= thr["ack_dup"] and med <= thr["ack_maxchars"]:
        return "ack", 2
    a1 = stats["axis1_top1"]
    if a1 is not None and a1 >= thr["axis1_theta1"]:
        return "echo/const", 3
    if stats["axis3_exact"] is not None and stats["axis3_exact"] >= thr["axis3_sim"]:
        return "sim", 4
    c = stats["axis4_ceiling"]
    if c is not None and c < thr["axis4_content"]:
        return "noisy-excluded", 5
    return "content", 6


# ================================================================ orchestration

def measure_root(root, splits=None, *, embedder=None, encoder="e5", twin_root=None,
                 error_templates_for="default", thresholds=None):
    """Compute the frozen four-axis classification over a v3 root. Returns the classes.json
    dict. `embedder` (callable list[str]->[N,D] tensor) overrides `encoder` ('e5'|'fake'|'none').
    `error_templates_for` is a callable image->{sst templates}; 'default' uses the P0 harvest via
    collect_docker.sst_error_templates; None disables error templates (SST BOTs error surfaces)."""
    thr = dict(DEFAULTS)
    thr.update(thresholds or {})
    diagnostics = {"out_of_universe": 0, "out_of_universe_ex": [], "bad_state_scope": 0}

    if error_templates_for == "default":
        error_templates_for = _default_templates()
    seqs, splits = load_root(root, splits)
    rows = cell_rows(seqs, error_templates_for, diagnostics)

    # attach the cell key + a seq_idx alias for twin alignment
    for r in rows:
        r["_cell"] = verbsig.cell(r["sig"], r["mode"], r["scope"],
                                  ws_observed=r["ws_observed"] if r["scope"] == "created" else None)
    # map seq_id -> the source sequence's seq_idx (for twin alignment)
    sid2idx = {}
    sid = 0
    for sq in seqs:
        sid2idx[sid] = sq.get("seq_idx"); sid += 1
    for r in rows:
        r["_seq_idx"] = sid2idx.get(r["seq_id"], r["seq_id"])

    # group by cell
    cells = collections.defaultdict(list)
    for i, r in enumerate(rows):
        cells[r["_cell"]].append(i)

    # axis-1 (global embed once), only if an encoder is available
    z_true = z_pred = None
    if embedder is None and encoder != "none":
        embedder = _e5_embedder() if encoder == "e5" else _fake_embedder()
    if embedder is not None and rows:
        z_true, z_pred = axis1_predictions(rows, embedder)

    # axis-2' per trajectory
    by_seq = collections.defaultdict(list)
    for r in rows:
        by_seq[r["seq_id"]].append(r)
    for sid, srows in by_seq.items():
        srows.sort(key=lambda r: r["step"])
        for r, v in zip(srows, axis2p_seq(srows)):
            r["_axis2p"] = v

    # axis-4 (twin)
    flips = axis4_flips(rows, twin_root, error_templates_for, diagnostics)

    out_rows = []
    n_measured = n_under = 0
    for cell, idxs in sorted(cells.items()):
        crows = [rows[i] for i in idxs]
        n = len(crows)
        n_images = len({r["image"] for r in crows})
        coverage_pairs = n if n_images >= thr["min_images"] else 0
        under = coverage_pairs < thr["coverage_floor"]
        r0 = crows[0]

        # ---- always-cheap text axes (computed even for under-floor, informative) ----
        a2_out = [axis2_step(r)[0] for r in crows]
        a2_pre = [axis2_step(r)[1] for r in crows]
        a2_stat = _mean([max(o, p) for o, p in zip(a2_out, a2_pre)])
        a2p_vals = [r.get("_axis2p", 0.0) for r in crows]
        a2p_stat = _mean(a2p_vals)
        # axis-3
        n_det = sum(1 for r in crows if r["determined"])
        n_exact = sum(1 for r in crows if r["exact"])
        axis3_exact = n_exact / n
        det_cov = n_det / n
        acc_det = (n_exact / n_det) if n_det else None
        # ack (success/hit-mode outputs; the cell's mode already fixes ok/hit vs miss)
        outs = [r["output_body"] for r in crows]
        top = collections.Counter(outs).most_common(1)
        ack_dup = (top[0][1] / n) if top else None
        ack_med = int(statistics.median([len(o) for o in outs])) if outs else None
        # axis-4
        flip_rate, n_aligned = flips.get(cell, (None, 0))
        ceiling = (1.0 - flip_rate) if flip_rate is not None else (1.0 if twin_root else 1.0)

        row = {
            "cell": cell, "sig": r0["sig"], "mode": r0["mode"], "state_scope": r0["scope"],
            "ws_observed": r0["ws_observed"], "n": n, "n_images": n_images,
            "coverage_pairs": coverage_pairs, "under_floor": under,
            "axis2": {"output_containment": round(_mean(a2_out), 4),
                      "prefix_containment": round(_mean(a2_pre), 4),
                      "stat": round(a2_stat, 4)},
            "axis2p": {"history_containment": round(a2p_stat, 4), "stat": round(a2p_stat, 4)},
            "axis3": {"exact_match": round(axis3_exact, 4),
                      "determined_coverage": round(det_cov, 4),
                      "accuracy_on_determined": round(acc_det, 4) if acc_det is not None else None,
                      "n_determined": n_det},
            "axis4": {"ceiling": round(ceiling, 4) if ceiling is not None else None,
                      "flip_rate": round(flip_rate, 4) if flip_rate is not None else None,
                      "n_aligned": n_aligned, "measured": bool(twin_root)},
            "ack": {"success_dup": round(ack_dup, 4) if ack_dup is not None else None,
                    "median_output_chars": ack_med},
        }

        if under:
            # FAIL-CLOSED: never classify a sub-floor cell from its own thin statistics.
            row["axis1"] = {"top1": None, "n": n}
            row["class"] = "under-floor"
            row["precedence_rule"] = None
            row["tracker_top1"] = None
            n_under += 1
        else:
            axis1_top1 = None
            if z_true is not None:
                axis1_top1 = round(axis1_cell_top1(z_true, z_pred, idxs), 4)
            row["axis1"] = {"top1": axis1_top1, "n": n}
            stats = {"axis2_stat": a2_stat, "axis2p_stat": a2p_stat,
                     "ack_dup": ack_dup, "ack_median_chars": ack_med,
                     "axis1_top1": axis1_top1, "axis3_exact": axis3_exact,
                     "axis4_ceiling": ceiling}
            klass, rule = classify(stats, thr)
            row["class"] = klass
            row["precedence_rule"] = rule
            # the tracker same-verb-foil top-1 is a STANDING credit-ledger column on content cells
            row["tracker_top1"] = round(axis3_exact, 4) if klass == "content" else None
            n_measured += 1
        out_rows.append(row)

    return {
        "tool": "class_measure", "version": TOOL_VERSION,
        "root": str(root), "splits": splits, "encoder": (encoder if embedder is None else "custom"),
        "twin_root": str(twin_root) if twin_root else None,
        "thresholds": thr,
        "n_cells_total": len(out_rows),
        "n_cells_measured": n_measured, "n_cells_under_floor": n_under,
        "n_steps": len(rows),
        "class_counts": dict(collections.Counter(r["class"] for r in out_rows)),
        "rows": out_rows,
        "diagnostics": diagnostics,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _default_templates():
    from realenv.collect_docker import sst_error_templates
    return sst_error_templates


# ================================================================ unit smoke

def _synthetic_root(dirpath):
    """A tiny hand-built v3 root exercising every precedence rule. Two 'images' so cells can
    clear a lowered coverage floor; the smoke uses coverage_floor=2, min_images=2."""
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    def step(cmd, output, exit, cwd, **meta):
        return {"cmd": cmd, "output": output, "exit": exit, "cwd": cwd, "meta": meta}

    def echo_seq(payload, fname):
        # echo-loop: write a mined payload, read it back (observed-capture => semi-echo via axis-2')
        return [
            step(f"echo '{payload}' > /tmp/w/{fname}", "", 0, "/", arm="m_redirect",
                 role="intervene", state_scope="created", ws_target=True, ws_observed=True),
            step(f"cat /tmp/w/{fname}", payload, 0, "/", arm="m_redirect",
                 role="revisit", state_scope="created", ws_target=True),
        ]

    seqs = []
    for img in ("img-a:latest", "img-b:latest"):
        for k in range(3):
            steps = []
            steps += [step("cd /etc", "", 0, "/etc", state_scope="native")]          # echo (axis-2 prefix)
            steps += [step("pwd", "/etc", 0, "/etc", state_scope="native")]           # SST-determined (excluded)
            steps += [step(f"mkdir /tmp/w/d{k}", "", 0, "/etc",                        # ack (empty ok)
                           state_scope="created", ws_target=True, ws_observed=False)]
            steps += [step("cat /etc/hostname", f"host-{img}-{k}", 0, "/etc",          # content (unpredictable)
                           state_scope="native")]
            steps += echo_seq(f"secret{k}", f"f{k}.txt")                              # semi-echo readback
            if img == "img-a:latest" and k == 0:
                # a cell present in ONE image / ONE step only => must land under-floor (fail-closed)
                steps += [step("uname -m", "aarch64", 0, "/etc", state_scope="native")]
            seqs.append({"image": img, "seq_idx": k, "steps": steps})

    with open(dirpath / "train.jsonl", "w") as fh:
        for sq in seqs:
            fh.write(json.dumps(sq) + "\n")
    return dirpath


_EXCLUDED = {"semi-echo", "ack", "echo/const", "sim", "noisy-excluded"}


def _smoke():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="class_measure_smoke_")
    root = _synthetic_root(pathlib.Path(tmp) / "root")
    thr = {"coverage_floor": 2, "min_images": 2}
    res = measure_root(root, embedder=_fake_embedder(), error_templates_for=None, thresholds=thr)

    # ---- schema / determinism invariants ----
    assert res["n_steps"] > 0
    assert res["n_cells_total"] == len(res["rows"])
    assert set(r["class"] for r in res["rows"]) <= set(CLASSES)
    for r in res["rows"]:                        # every row carries the full column set
        for col in ("cell", "sig", "mode", "state_scope", "class", "n", "coverage_pairs",
                    "under_floor", "axis1", "axis2", "axis2p", "axis3", "axis4", "ack"):
            assert col in r, (col, r)
    res2 = measure_root(root, embedder=_fake_embedder(), error_templates_for=None, thresholds=thr)
    assert json.dumps(res["rows"]) == json.dumps(res2["rows"]), "non-deterministic output"

    by_cell = {r["cell"]: r for r in res["rows"]}
    # axis-3 measures the SST determined+exact surface: pwd is fully determined & exact...
    pwd = by_cell["pwd|hit|native"]
    assert pwd["axis3"]["exact_match"] >= 0.90 and pwd["axis3"]["determined_coverage"] == 1.0, pwd
    assert pwd["class"] in _EXCLUDED, pwd                 # ...so it never enters fitness
    # observed-capture readback -> semi-echo via axis-2' history containment (rule 1)
    rb = by_cell["cat|hit|created-obs"]
    assert rb["class"] == "semi-echo" and rb["precedence_rule"] == 1, rb
    assert rb["axis2p"]["history_containment"] >= 0.656, rb
    # mkdir empty-ok -> ack (rule 2: 100% dup, median 0 chars)
    mk = by_cell["mkdir|ok|created"]
    assert mk["class"] == "ack" and mk["precedence_rule"] == 2, mk
    # cd -> excluded (axis-2 prefix containment of the echoed cwd)
    assert by_cell["cd|ok|native"]["class"] in _EXCLUDED
    # cat /etc/hostname -> content (unpredictable image file: all axes low)
    ch = by_cell["cat|hit|native"]
    assert ch["class"] == "content" and ch["precedence_rule"] == 6, ch
    # the single-image uname cell -> under-floor, fail-closed (no class guessed, axis1 null)
    uf = by_cell["uname|hit|native"]
    assert uf["class"] == "under-floor" and uf["under_floor"] and uf["axis1"]["top1"] is None, uf

    # ---- direct precedence unit: the sim branch (rule 4) fires when 1-3 don't ----
    base = {"axis2_stat": 0.1, "axis2p_stat": 0.1, "ack_dup": 0.2, "ack_median_chars": 40,
            "axis1_top1": 0.2, "axis3_exact": 0.95, "axis4_ceiling": 1.0}
    assert classify(base, DEFAULTS) == ("sim", 4)
    assert classify({**base, "axis1_top1": 0.8}, DEFAULTS) == ("echo/const", 3)
    assert classify({**base, "axis3_exact": 0.1}, DEFAULTS) == ("content", 6)
    assert classify({**base, "axis3_exact": 0.1, "axis4_ceiling": 0.4}, DEFAULTS) \
        == ("noisy-excluded", 5)
    assert classify({**base, "axis2_stat": 0.9}, DEFAULTS) == ("semi-echo", 1)

    print(json.dumps(res, indent=1))
    print(f"\n[smoke] root={root}")
    print(f"[smoke] steps={res['n_steps']} cells={res['n_cells_total']} "
          f"measured={res['n_cells_measured']} under_floor={res['n_cells_under_floor']}")
    print(f"[smoke] class_counts={res['class_counts']}")
    print("[smoke] OK — schema, determinism, all 6 precedence rules, fail-closed under-floor")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", default=None, help="comma list; default = whatever the root has")
    ap.add_argument("--twin-root", default=None, help="a twin mint for axis-4 flip rates")
    ap.add_argument("--encoder", choices=("e5", "fake", "none"), default="e5")
    ap.add_argument("--coverage-floor", type=int, default=DEFAULTS["coverage_floor"])
    ap.add_argument("--smoke", action="store_true", help="run the synthetic-root unit smoke")
    for k in ("axis1_theta1", "axis2_thresh", "axis2p_thresh", "axis3_sim",
              "axis4_content", "ack_dup"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None)
    args = ap.parse_args(argv)

    if args.smoke:
        return _smoke()
    if not args.root:
        ap.error("--root is required (or use --smoke)")
    thresholds = {"coverage_floor": args.coverage_floor}
    for k in ("axis1_theta1", "axis2_thresh", "axis2p_thresh", "axis3_sim",
              "axis4_content", "ack_dup"):
        v = getattr(args, k)
        if v is not None:
            thresholds[k] = v
    splits = args.splits.split(",") if args.splits else None
    res = measure_root(args.root, splits=splits, encoder=args.encoder,
                       twin_root=args.twin_root, thresholds=thresholds)
    txt = json.dumps(res, indent=1)
    if args.out:
        pathlib.Path(args.out).write_text(txt)
    print(txt)
    return res


if __name__ == "__main__":
    main()
