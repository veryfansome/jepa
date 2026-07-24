"""score_genome — the trustworthy fitness. Assemble a genome, train the world model with the
genome's objective, HARD-FILTER on the per-genome no-leakage + calibration guards, then score
the content-verb MARGIN on the inner-val (held-out) split. Reuses the validated R4 eval from
realenv/seq_worldmodel.py so a fitness number means exactly what the R4 headline meant.

Fitness = mean over seeds of
    content_top1(WM) - max(content_top1(retrieve_by_cmd), content_top1(no_history), content_top1(copy_prev))
on ls+cat (content) verbs of the inner-val images. Never touches final-test.
"""

import hashlib
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # terminal-jepa root

import torch

from realenv import seq_worldmodel as M
from evolve import genome as G
from evolve.splits import split_val
from evolve import bench_versions as BV

D = M.D
CHANCE_SLACK = 0.05  # predict-mean top-1 must stay below this (chance ~1/64=0.016)
BASE_CACHE = pathlib.Path(__file__).resolve().parent / "archive" / "base_cache.json"
_ARTIFACT_SHA_CACHE = {}

# S2 (prereg §4.3 / G-COV): per-split content-cell coverage floor. A content cell with fewer
# than this many SURVIVING content steps in the CURRENT eval split is under-powered — demoted
# out of the pooled fitness margin into the §9.3 report battery (see _demote_lowcov). Pinned
# scoring-time floor; the frozen class table is never touched.
_V3_GCOV_FLOOR = 500

# MINT-CONFIRMED θ1 ejection (prereg §4.9, dated 2026-07-23): the class table froze head/tail/stat
# native as content on PILOT axis-1 (< θ1=0.59); the 12-image mint re-measurement revealed their
# true no-history predictability crosses θ1 (head 0.717 / tail 0.710 / stat 0.786) — a fixed read
# of a fixed file on a fixed image is a CONSTANT that retrieve-by-command banks via image-
# memorization, not cross-image world knowledge. The θ1 gate excludes them (echo/const). Scoring-
# layer demotion (mirrors _demote_lowcov); the frozen classes.json is untouched. The v3.0 fitness
# surface is the 5 cells whose mint axis-1 < 0.59 (ls/cat native+mutated, find native) — the
# mutated-scope cells being the v3 value-add (lowest retrievability, highest world knowledge).
_V3_THETA1_DEMOTED = frozenset({"head|hit|native", "tail|hit|native", "stat|hit|native"})


def _demote_theta1(verbs, content, demoted=_V3_THETA1_DEMOTED):
    """§4.9 mint-confirmed θ1 echo/const demotion. Rewrite every step of a pinned mint-demoted
    content cell to the excluded "<cell>-echoconst" pseudo-verb — the cell is command-retrievable
    (image-memorization) at mint scale, so it leaves the pooled margin for the report battery.
    Frozen classes.json untouched. Returns (new_verbs, hit) where hit = the demoted cells present."""
    hit = {c for c in content if c in demoted}
    if hit:
        verbs = [(v + "-echoconst") if v in hit else v for v in verbs]
    return verbs, sorted(hit)


def _demote_lowcov(verbs, content, floor=_V3_GCOV_FLOOR):
    """S2/G-COV per-split coverage-demotion (prereg §4.3). Given the per-step cell pseudo-verbs
    and the frozen content-cell set, count each content cell's surviving steps IN THIS split and
    rewrite EVERY step of any content cell below `floor` to the excluded "<cell>-lowcov" pseudo-
    verb — generalizing the -echo/-miss rewrites, so an under-covered cell leaves the fitness
    pool (its own excluded verb group) and lands in the report battery. Per-split by construction
    (a cell content in inner may be low-cov in final). The frozen classes.json is untouched.
    Returns (new_verbs, lowcov) where lowcov = {demoted_cell: pre-demotion step count}."""
    from collections import Counter
    counts = Counter(v for v in verbs if v in content)
    lowcov = {c: n for c, n in counts.items() if n < floor}
    if lowcov:
        verbs = [(v + "-lowcov") if v in lowcov else v for v in verbs]
    return verbs, lowcov


def _cached_encode(data_root, split, model, device):
    """Harness-owned wrapper around M.cached_encode (§13.2 layering: the fail-closed gate consults
    evolve-side concepts — bench_versions + cache_meta.json — so realenv stays evolve-free). On a
    v3-policy root it REQUIRES the root-level cache_meta.json + perception stamp to match
    expectations and RAISES otherwise (a stamp-less/stale v3 cache can never be scored); v1/v2 roots
    pass straight through untouched (bit-identical). ALL harness cached_encode calls route here."""
    if BV.is_v3_policy(data_root):
        BV.require_v3_cache(data_root)
    return M.cached_encode(data_root, split, model, device)


def _strip_target_only(seqs):
    """§8.2 strip seam. The two places genome stream code receives seq dicts (stream.collate,
    stream.flatten_predictions) and the batcher's fit all receive a per-sequence shallow COPY with
    the target-only keys (exit_cls, z_delta) REMOVED — so the v3 aux channels are structurally
    invisible to genome code. Identity pass-through (returns the SAME list) when no seq carries
    those keys: v1/v2 = zero-cost, bit-identical."""
    keys = ("exit_cls", "z_delta")
    if not any(k in s for s in seqs for k in keys):
        return seqs
    return [{k: v for k, v in s.items() if k not in keys} for s in seqs]


def _subsample_per_image(seqs, n_per_image, seed):
    """§11.5 train-side seeded per-image subsample to n_per_image seqs/image (the ablate comparison
    arm: the champion trained on the FULL root subsampled, matched to the ablate train size).
    Deterministic in `seed`; preserves original order within each image."""
    by_img = {}
    for i, s in enumerate(seqs):
        by_img.setdefault(s["image"], []).append(i)
    keep = set()
    for img in sorted(by_img):
        idxs = by_img[img]
        if len(idxs) <= n_per_image:
            keep.update(idxs)
        else:
            keep.update(random.Random(f"subsample:{seed}:{img}").sample(idxs, n_per_image))
    return [s for i, s in enumerate(seqs) if i in keep]


def _root_artifact_sha(root):
    """sha256 over summary.json + the emb-seq caches of a root (§13.2 base_cache keying). Memoized
    per path (the caches are large). Called only for v3-policy roots — v1/v2 keys never reach it."""
    root = str(root)
    if root in _ARTIFACT_SHA_CACHE:
        return _ARTIFACT_SHA_CACHE[root]
    h = hashlib.sha256()
    rp = pathlib.Path(root)
    for name in ["summary.json"] + sorted(p.name for p in rp.glob("emb-seq-*.pt")):
        p = rp / name
        if p.exists():
            h.update(name.encode())
            h.update(p.read_bytes())
    sha = h.hexdigest()[:16]
    _ARTIFACT_SHA_CACHE[root] = sha
    return sha


def _base_cache_key(data_root, split, seed, steps, spec, arms, val_root, stats_root, train_desc):
    """base_cache.json key. v1/v2 roots keep the EXACT historical format (`root|split|seed|steps`
    (+`|v2`)) so cached entries stay valid and scoring is bit-identical. v3-policy roots use the
    extended key (§13.2): arm set + classes_sha + root artifact sha + train-set descriptor +
    val/stats-root artifact shas (when they differ) — the baselines are fit-/val-/stats-dependent,
    so a re-mint into the same path, or a subsampled arm, can never be served a stale baseline max.
    Built now even though the v3 baseline arms (within_traj_mut/sst/sst_composite) join `arms`
    later, so keying is correct the moment they land."""
    if not BV.is_v3_policy(data_root):
        vtag = "" if not spec.get("within_traj_in_max") else "|v2"
        return f"{data_root}|{split}|{seed}|{steps}{vtag}"
    parts = [data_root, split, str(seed), str(steps),
             "arms=" + ",".join(sorted(arms)),
             "classes=" + (BV.classes_sha(data_root) or "none"),
             "art=" + _root_artifact_sha(data_root),
             "train=" + train_desc]
    if val_root and val_root != data_root:
        parts.append("val=" + _root_artifact_sha(val_root))
    if stats_root and stats_root != data_root:
        parts.append("stats=" + _root_artifact_sha(stats_root))
    return "|".join(parts)


def _v3_cell_verbs(evalset, raw_seqs, spec):
    """v3 per-step cell pseudo-verb rewrite (§9.5) + the Finding-1 axis-2' echo purge (§4.6).

    Returns (verbs, forced_foils, diag). `verbs[i]` is the ATOMIC cell key "sig|mode|scope[-obs]"
    for step i (flatten order == _data_tensors' true/prev order), so `content_retrieval` and the
    same-verb foil pools pool by CELL with no retrieval-code change (design §13.3). sig is
    RE-DERIVED from cmd text via the frozen labeler and F8-ASSERTED against meta.sig; mode/scope/
    ws_observed follow the frozen §9.5 rule. A content-cell step whose per-step history-containment
    axis-2' >= the frozen purge threshold is rewritten to "<cell>-echo" (leaving the content pool) —
    a per-STEP purge, zero command-echo in content by construction. forced_foils [N,8] long holds
    the pre-mutation twin's FULL-array index in slot 0 (meta.pre_obs_step), -1 elsewhere (§8.1).

    Reuses `class_measure.axis2p_seq` (through-mutation echo propagation) and `_canon_output`
    verbatim (design: "reuse class_measure.axis2p_seq's logic"). Lazy imports keep v1/v2 untouched."""
    from realenv import verbsig
    from benchmarks import class_measure as CM
    content = spec["content"]
    thresh = spec["axis2p_purge_thresh"]
    if len(raw_seqs) != len(evalset):
        raise ValueError(f"v3 raw/eval seq count mismatch: {len(raw_seqs)} != {len(evalset)}")
    verbs, forced_rows = [], []
    n_total = purged = oou = 0
    for sq, raw in zip(evalset, raw_seqs):
        n = sq["z_obs"].shape[0]
        steps = raw.get("steps", [])
        if len(steps) != n:
            raise ValueError(f"v3 raw/eval step count mismatch (image {sq.get('image')}): "
                             f"{len(steps)} != {n}")
        seq_start = n_total
        cells, rows = [], []
        for t, step in enumerate(steps):
            cmd = step.get("cmd", "")
            if cmd != sq["cmds"][t]:
                raise ValueError(f"v3 raw/eval cmd misalignment (image {sq.get('image')} step {t}): "
                                 f"{cmd!r} != {sq['cmds'][t]!r}")
            meta = step.get("meta", {}) or {}
            try:
                sig = verbsig.sig(cmd)
                if meta.get("sig") is not None and meta["sig"] != sig:
                    raise ValueError(f"F8 sig mismatch: recomputed {sig!r} != meta.sig "
                                     f"{meta['sig']!r} for {cmd!r} — data-integrity fault")
                mode = verbsig.mode(sig, step.get("exit", 0), not step.get("output"))
                scope = meta.get("state_scope", "native")
                if scope not in verbsig.SCOPES:
                    scope = "native"
                ws_obs = bool(meta.get("ws_observed")) if scope == "created" else None
                cell = verbsig.cell(sig, mode, scope, ws_observed=ws_obs)
            except ValueError as e:
                if "F8 sig mismatch" in str(e):
                    raise
                cell = "__oou__"      # out-of-universe -> excluded pseudo-verb (never content)
                oou += 1
            cells.append(cell)
            # F2: an out-of-universe step contributes NO tokens to the axis-2' prior — mirror
            # class_measure.cell_rows, which `continue`s on oou before the prior accumulates.
            # (An oou step is never a content cell, so its own a2p is irrelevant; but its tokens
            # must not shift a LATER content step's history-containment away from the frozen table.)
            # At the mint the totality-asserted policy yields oou==0; this keeps eval==measurement
            # bit-exact even off the mint path.
            if cell == "__oou__":
                rows.append({"cmd": "", "output_body": ""})
            else:
                rows.append({"cmd": cmd, "output_body": CM._canon_output(step)})
        a2p = CM.axis2p_seq(rows)     # through-mutation history containment per step
        for t in range(n):
            cell = cells[t]
            if cell in content and a2p[t] >= thresh:
                cell = cell + "-echo"          # per-step echo purge (§4.6 Finding-1)
                purged += 1
            verbs.append(cell)
            pos = (steps[t].get("meta", {}) or {}).get("pre_obs_step", -1)
            row = [-1] * 8
            if isinstance(pos, int) and 0 <= pos < n and pos != t:
                row[0] = seq_start + pos       # pre-mutation twin, full-array position (§8.1)
            forced_rows.append(row)
        n_total += n
    # S2 (§4.3 / G-COV): per-split coverage-demotion AFTER the echo purge — a content cell with
    # < _V3_GCOV_FLOOR surviving content steps in THIS split is demoted to "<cell>-lowcov" (out of
    # the pooled margin, into the report battery). At mint scale this demotes the held-out grep|hit
    # cells + any composed-pipe family that misses the floor. Frozen classes.json untouched.
    verbs, lowcov = _demote_lowcov(verbs, content)
    # §4.9 mint-confirmed θ1 echo/const demotion — head/tail/stat native are image-memorizable at
    # mint scale (axis-1 ≥ 0.59); exclude from the pooled margin (report-only). Frozen classes untouched.
    verbs, theta1 = _demote_theta1(verbs, content)
    forced = torch.tensor(forced_rows, dtype=torch.long) if forced_rows else torch.zeros(0, 8, dtype=torch.long)
    diag = {"purged": purged, "oou": oou, "lowcov": lowcov, "theta1_demoted": theta1,
            "n_forced": int((forced[:, 0] >= 0).sum()) if len(forced) else 0}
    return verbs, forced, diag


def _data_tensors(evalset, spec=None, raw_seqs=None):
    """Model-INDEPENDENT eval tensors (true/prev/verbs/cmd-embeddings) in flatten_predictions'
    step order — lets the objective-independent baselines be computed and cached once.
    spec (bench_versions.resolve): v2 roots mask ok_masked_verbs' failed steps out of the
    content pool by rewriting their verb to the excluded pseudo-verb "<verb>-miss".
    v3 roots (spec['cell_based'], raw_seqs given) rewrite each step's verb to its cell pseudo-verb
    and apply the per-step echo purge + build the counterfactual forced_foils tensor (§9.5/§4.6/§8.1)."""
    trues, prevs, cmds, oks = [], [], [], []
    for sq in evalset:
        ok = sq.get("ok")
        for t in range(sq["z_obs"].shape[0]):
            trues.append(sq["z_obs"][t])
            prevs.append(sq["z_obs"][t - 1] if t > 0 else torch.zeros(D))
            cmds.append(sq["cmds"][t])
            oks.append(True if ok is None else bool(ok[t]))
    forced, v3diag = None, None
    if spec and spec.get("cell_based") and raw_seqs is not None:
        verbs, forced, v3diag = _v3_cell_verbs(evalset, raw_seqs, spec)
    else:
        verbs = [M.verb_of(c) for c in cmds]
        if spec and spec["ok_masked_verbs"]:
            verbs = [f"{v}-miss" if (v in spec["ok_masked_verbs"] and not k) else v
                     for v, k in zip(verbs, oks)]
    cmd_embs = torch.stack([sq["z_cmd"][t] for sq in evalset for t in range(sq["z_obs"].shape[0])])
    # within-traj baseline predictions (constitution §5): nearest earlier cmd BY EMBEDDING in
    # the same trajectory -> that step's obs; zeros (predict_mean) when no earlier step.
    wt = []
    for sq in evalset:
        n = sq["z_obs"].shape[0]
        for t in range(n):
            if t == 0:
                wt.append(torch.zeros(D))
            else:
                d = ((sq["z_cmd"][:t] - sq["z_cmd"][t]) ** 2).sum(-1)
                wt.append(sq["z_obs"][int(d.argmin())])
    out = {"true": torch.stack(trues), "prev": torch.stack(prevs),
           "verbs": verbs, "_cmd_embs": cmd_embs, "_within_traj": torch.stack(wt)}
    if forced is not None:               # v3 only; absent for v1/v2 (forced_foils stays None)
        out["_forced_foils"] = forced
        out["_v3diag"] = v3diag
        # per-seq (image, length) manifest so the v3 arm loader can verify sst/wtm-val.pt align
        # PER SEQUENCE, not just by total step count (review F1: a stale precompute with a
        # coincidentally-matching total would silently corrupt every baseline arm otherwise).
        out["_seq_manifest"] = [(sq.get("image", "?"), int(sq["z_obs"].shape[0])) for sq in evalset]
    return out


def _load_v3_arm_tensors(val_root, split, evaldata, stats):
    """Load the precomputed SST/within_traj_mut arm embeddings (evolve.precompute_baselines) for a
    v3 root, slice to this split via the SAME image filter split_val applies, and standardize with
    the SAME obs stats (mo, so) the eval `true`/`prev` were standardized with. Returns
    (sst_std, determined_mask, wtm_std), all aligned to evaldata['true'] step order (§10.4b).
    ⊥ (undetermined) SST steps become ZEROS in standardized space (§10.3, sst arm = ⊥→zeros).
    FAIL-CLOSED if the tensors are absent or misaligned (a v3 root is unscorable without them)."""
    root = pathlib.Path(val_root)
    sst_p, wtm_p = root / "sst-val.pt", root / "wtm-val.pt"
    if not sst_p.exists() or not wtm_p.exists():
        raise ValueError(f"{val_root}: v3 scoring needs sst-val.pt + wtm-val.pt — run "
                         f"`python -m evolve.precompute_baselines --root {val_root}` (fail-closed, §10.4b)")
    sst_seqs = split_val(torch.load(sst_p, weights_only=False), split)
    wtm_seqs = split_val(torch.load(wtm_p, weights_only=False), split)
    z_sst = torch.cat([s["z"] for s in sst_seqs]) if sst_seqs else torch.zeros(0, D)
    det = torch.cat([s["determined"] for s in sst_seqs]) if sst_seqs else torch.zeros(0, dtype=torch.bool)
    z_wtm = torch.cat([s["z"] for s in wtm_seqs]) if wtm_seqs else torch.zeros(0, D)
    wdef = torch.cat([s["defined"] for s in wtm_seqs]) if wtm_seqs else torch.zeros(0, dtype=torch.bool)
    n = evaldata["true"].shape[0]
    if z_sst.shape[0] != n or z_wtm.shape[0] != n or det.shape[0] != n:
        raise ValueError(f"v3 arm-tensor misalignment for {val_root} [{split}]: sst {z_sst.shape[0]} "
                         f"wtm {z_wtm.shape[0]} det {det.shape[0]} != eval {n} — stale precompute "
                         f"(re-run precompute_baselines after re-encoding)")
    # F1: per-SEQ (image, length) identity — a matching TOTAL step count is not enough; a stale
    # sst-val.pt from a re-encode that reordered/swapped sequences would corrupt every arm silently.
    manifest = evaldata.get("_seq_manifest")
    if manifest is not None:
        arm_man = [(s.get("image", "?"), int(s["z"].shape[0])) for s in sst_seqs]
        wtm_man = [(s.get("image", "?"), int(s["z"].shape[0])) for s in wtm_seqs]
        if arm_man != manifest or wtm_man != manifest:
            raise ValueError(f"v3 arm-tensor PER-SEQ misalignment for {val_root} [{split}]: sst/wtm "
                             f"per-seq (image,len) != eval — stale precompute after a re-encode; "
                             f"re-run precompute_baselines")
    mo, so = stats
    sst_std = (z_sst - mo) / so
    sst_std[~det] = 0.0                                   # ⊥ -> zeros (chance) in std space (§10.3)
    wtm_std = (z_wtm - mo) / so
    wtm_std[~wdef] = 0.0                                  # undefined (t0, no earlier read) -> chance,
    return sst_std, det, wtm_std                          #   matching within_traj's t=0=zeros rule


def _base_for(split, seed, steps, fit, evaldata, device, data_root="data/dockerfs",
              val_root=None, stats_root=None, train_desc="full", stats=None):
    """max-baseline content-top1 + predict-mean calibration for a (data_root, split, seed, steps) —
    objective-independent, so computed once and cached. Returns (base, predict_mean).
    v1/v2: the 3 (or 4 with within_traj) ratified arms. v3 (spec['cell_based']): the SEVEN-arm max
    (§4.4) — adds sst / within_traj_mut / sst_composite from evolve.precompute_baselines, and the
    same-verb foils carry the counterfactual forced_foils (§8.1) so every arm and the WM face the
    IDENTICAL candidate set. `stats`=(mo, so) standardizes the precomputed arms (§10.4b signature
    change). val_root/stats_root/train_desc feed the v3 base_cache key (§13.2)."""
    spec = evaldata.get("_spec") or {"content": ("ls", "cat"), "within_traj_in_max": False}
    arm_names = ["retrieve_by_cmd", "no_history", "copy_prev"]
    if spec.get("within_traj_in_max"):
        arm_names.append("within_traj")
    if spec.get("cell_based"):
        arm_names += ["within_traj_mut", "sst", "sst_composite"]   # the 7-arm max (§4.4)
    key = _base_cache_key(data_root, split, seed, steps, spec, arm_names, val_root, stats_root, train_desc)
    cache = json.loads(BASE_CACHE.read_text()) if BASE_CACHE.exists() else {}
    if key in cache:
        return cache[key]["base"], cache[key]["predict_mean"]
    mlp = M.train_cmd_only(fit, device, steps=steps, seed=seed)
    with torch.no_grad():
        nohist = mlp(evaldata["_cmd_embs"].to(device)).cpu()
    ff = evaldata.get("_forced_foils")   # None for v1/v2 -> historical byte-identical retrieval
    ct = lambda p: M.content_retrieval(p, evaldata["true"], evaldata["verbs"],
                                       content=spec["content"], seed=seed,
                                       forced_foils=ff)["top1_sameverb"]
    rbc, noh = ct(M.retrieve_by_cmd_baseline(fit, evaldata)), ct(nohist)
    cpy, mean = ct(evaldata["prev"]), ct(torch.zeros_like(evaldata["true"]))
    arms = {"retrieve_by_cmd": rbc, "no_history": noh, "copy_prev": cpy}
    if spec.get("within_traj_in_max"):
        arms["within_traj"] = ct(evaldata["_within_traj"])   # constitution §5 (v2+)
    if spec.get("cell_based"):
        sst_std, det, wtm_std = _load_v3_arm_tensors(val_root or data_root, split, evaldata, stats)
        comp = wtm_std.clone()
        comp[det] = sst_std[det]                             # SST-where-determined else wtm
        arms["within_traj_mut"] = ct(wtm_std)
        arms["sst"] = ct(sst_std)
        arms["sst_composite"] = ct(comp)
    entry = {"base": max(arms.values()), "predict_mean": mean, **arms}
    cache[key] = entry
    BASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BASE_CACHE.write_text(json.dumps(cache, indent=1))
    return entry["base"], entry["predict_mean"]


def _train(genome, fit, device, loss_fn, seed, steps, target_mod, stream, head=None, head_p=None):
    """Train the world model with the genome's objective + arch + target + stream (+ optional head).
    Returns (net, ok); ok=False on NaN. The objective's loss compares the model's cmd-position
    prediction to target_mod.make_target(z_obs, z_prev) — e.g. the raw next obs (identity) or the
    residual z_obs - z_prev (delta). z_prev is the previous observation (strict-causal shift of tgt)."""
    torch.manual_seed(seed)
    build, aparams = G.load_arch(genome)
    net = build(**aparams)
    if getattr(target_mod, "LEARNED", False):
        # learned-target extension: the target impl provides an nn.Module (make_target/to_obs/reg)
        # whose params are registered on the net so the genome's optimizer trains them jointly.
        # The eval stays in the FIXED obs space (to_obs must reconstruct), which keeps a learned
        # target honest: collapsing the target space breaks reconstruction and is scored down.
        net.target_module = target_mod.make(D)
    head_state = None
    if head is not None:
        # head-axis extension: wrap may re-point net.forward and register readout/aux params on
        # net (trained jointly). aux_loss adds a train-only self-supervised term; passthrough is
        # a no-op returning None + 0.0. Must run BEFORE make_opt so aux params are optimized.
        head_state = head.wrap(net, D, **(head_p or {}))
    net = net.to(device)
    make_opt, bs = G.load_optim(genome)
    opt, sched = make_opt(net.parameters(), steps)
    # §8.2 strip seam: the batcher and stream.collate only ever see the stripped fit (target-only
    # keys removed). Identity pass-through for v1/v2 (no such keys) -> bit-identical.
    fit_stripped = _strip_target_only(fit)
    aux_live = fit_stripped is not fit   # v3 aux channels present -> plumbing active (but dormant)
    next_batch = G.load_batcher(genome)(fit_stripped, bs, seed)
    for step in range(1, steps + 1):
        idx = next_batch(step, steps)
        if len(idx) != bs or min(idx) < 0 or max(idx) >= len(fit_stripped):
            raise ValueError("batcher contract violation (len/bounds)")
        b = stream.collate([fit_stripped[i] for i in idx], device)
        if aux_live:
            # DORMANT v3.1 aux-target plumbing (§8.2): the harness-held ORIGINAL (unstripped) seq
            # dicts for this batch, indexed by the batcher's indices — the attach point for
            # multi-channel aux targets (exit_cls/z_delta). No sanctioned consumer in v3.0.
            _aux_originals = [fit[i] for i in idx]  # noqa: F841
        pred_full, _ = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = stream.extract_cmd_pred(pred_full, b)           # [B, maxn, D]
        tgt_full = b["tgt"]                                        # [B, maxn, D] = z_obs per step
        prev_full = torch.cat([torch.zeros_like(tgt_full[:, :1]), tgt_full[:, :-1]], dim=1)
        m = b["cmd_mask"]
        pred, tgt, prev = cmd_pred[m], tgt_full[m], prev_full[m]
        tmod = getattr(net, "target_module", None)
        if tmod is not None:
            loss = loss_fn(pred, tmod.make_target(tgt, prev)) + tmod.reg()
        else:
            loss = loss_fn(pred, target_mod.make_target(tgt, prev))
        if head is not None:
            loss = loss + head.aux_loss(head_state, b, net, device)  # 0.0 for passthrough
        if not torch.isfinite(loss):
            return net, False
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
    return net, True


def score_genome(genome, mode="proxy", data="data/dockerfs",
                 model="answerdotai/ModernBERT-base", proxy_steps=1000, split="inner",
                 save_dir=None, val_data=None, stats_root=None,
                 subsample_seqs=None, subsample_seed=0):
    """Return a fitness dict. mode='proxy' -> steps=proxy_steps, seeds=[0]; 'full' -> genome
    steps, seeds=[0,1,2]. split='inner' (fedora+mariadb, the optimization target) or 'final'
    (rockylinux+httpd, the untouched held-out-of-held-out test — champion validation only). Any
    guardrail failure -> fitness=-inf with a reason.

    v3 §11.5 ablate plumbing (all default-inert): val_data scores against a DIFFERENT val root than
    the train root; stats_root pins the standardization stats to another root (both arms standardize
    identically); subsample_seqs/subsample_seed do a train-side seeded per-image subsample feeding
    the §13.2 train-set descriptor. Defaults reproduce the historical single-root path exactly."""
    G.validate(genome)
    device = M.pick_device()
    train_full = _cached_encode(data, "train", model, device)
    val_root = val_data or data
    val_seqs = _cached_encode(val_root, "val", model, device)
    # standardization stats (§11.5): --stats-root pins the canonical full-root stats so an ablate/
    # subsampled arm standardizes IDENTICALLY to the full arm; default = the train root (as today).
    if stats_root and stats_root != data:
        mo, so, mc, sc = M.standardize_stats(_cached_encode(stats_root, "train", model, device))
    else:
        mo, so, mc, sc = M.standardize_stats(train_full)   # full-root stats (pre-subsample)
    # train-side seeded per-image subsample (§11.5) + the §13.2 train-set descriptor
    if subsample_seqs:
        train_seqs = _subsample_per_image(train_full, subsample_seqs, subsample_seed)
        train_desc = f"sub{subsample_seqs}:{subsample_seed}"
    else:
        train_seqs, train_desc = train_full, "full"
    M.apply_stats(train_seqs, mo, so, mc, sc)
    M.apply_stats(val_seqs, mo, so, mc, sc)
    inner = split_val(val_seqs, split)
    spec = BV.resolve(val_root)   # eval-side classes/content track the VAL root (== data by default)
    # v3: the cell rewrite + echo purge + forced_foils need the raw meta/output, which the encoded
    # cache does NOT carry (§13.1). Re-read val.jsonl and apply the SAME split image filter so the
    # raw seqs align 1:1 with `inner` (score-time recompute defines the cell, §8.2).
    raw_seqs = None
    if spec.get("cell_based"):
        raw_all = [json.loads(l) for l in open(pathlib.Path(val_root) / "val.jsonl")]
        raw_seqs = split_val(raw_all, split)

    loss_fn = G.load_objective(genome)
    target_mod = G.load_target(genome)
    stream = G.load_stream(genome)
    head, head_p = G.load_head(genome)
    if not head.leak_safe(head, head_p):
        return _fail("head_leak_fail (aux branch could leak future obs)", mode,
                     [0] if mode == "proxy" else [0, 1, 2], 0, split, [])
    seeds = [0] if mode == "proxy" else [0, 1, 2]
    steps = proxy_steps if mode == "proxy" else genome["chunks"]["optim"].get("steps", 4000)
    evaldata = _data_tensors(inner, spec, raw_seqs=raw_seqs)  # model-independent; base + wm score against it
    evaldata["_spec"] = spec

    per_seed = []
    for s in seeds:
        try:
            fit, _ = M.split_train_dev(train_seqs, seed=s)
            net, ok = _train(genome, fit, device, loss_fn, s, steps, target_mod, stream, head, head_p)
            if not ok:
                return _fail("train_diverged (NaN/inf loss)", mode, seeds, steps, split, per_seed)
            if not stream.leakage_ok(net, device):
                return _fail("leakage_fail (cmd_t prediction moved when obs_t corrupted)", mode, seeds, steps, split, per_seed)
            base, mean = _base_for(split, s, steps, fit, evaldata, device, data_root=data,
                                   val_root=val_root, stats_root=stats_root, train_desc=train_desc,
                                   stats=(mo, so))
            flat = stream.flatten_predictions(net, _strip_target_only(inner), device)
            tmod = getattr(net, "target_module", None)
            with torch.no_grad():  # learned to_obs runs on cpu tensors from flatten
                if tmod is not None:
                    pred_obs = tmod.cpu().to_obs(flat["pred"], flat["prev"])
                else:
                    pred_obs = target_mod.to_obs(flat["pred"], flat["prev"])  # reconstruct next-obs for retrieval
            wm = M.content_retrieval(pred_obs, evaldata["true"], evaldata["verbs"],
                                     content=spec["content"], seed=s,
                                     forced_foils=evaldata.get("_forced_foils"))["top1_sameverb"]
            per_seed.append({"seed": s, "wm": round(wm, 4), "base": round(base, 4),
                             "margin": round(wm - base, 4), "predict_mean": round(mean, 4)})
            if save_dir:
                # checkpoint hook (plan-eval / Phase-0): save AFTER scoring so training and
                # eval are byte-identical to a hook-less run; consumers must fidelity-gate
                # the saved seeds' margins against the archived record before use.
                p = pathlib.Path(save_dir); p.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": net.cpu().state_dict(), "genome": genome, "seed": s,
                            "steps": steps, "split": split, "data": data,
                            "margin": per_seed[-1]["margin"]},
                           p / f"{genome['id']}.s{s}.pt")
        except Exception as e:  # broken inventor code must not crash the loop
            return _fail(f"exception: {type(e).__name__}: {e}", mode, seeds, steps, split, per_seed)

    mean_cal = sum(p["predict_mean"] for p in per_seed) / len(per_seed)
    if mean_cal > CHANCE_SLACK:
        return _fail(f"calibration_fail (predict_mean top1={mean_cal:.3f} > {CHANCE_SLACK})",
                     mode, seeds, steps, split, per_seed)

    def mean(k):
        return round(sum(p[k] for p in per_seed) / len(per_seed), 4)

    result = {"fitness": mean("margin"), "guardrail": "pass", "mode": mode, "seeds": seeds,
              "steps": steps, "split": split, "wm_content_top1": mean("wm"),
              "base_content_top1": mean("base"), "eval_images": sorted({s["image"] for s in inner}),
              "per_seed": per_seed}
    # S2/§4.6 diagnostic: surface the v3 per-step purge + coverage-demotion for the report battery
    # (v3-only key; v1/v2 result dicts are byte-identical — no _v3diag present).
    if evaldata.get("_v3diag") is not None:
        result["v3_diag"] = evaldata["_v3diag"]
    return result


def _fail(reason, mode, seeds, steps, split, per_seed):
    return {"fitness": float("-inf"), "guardrail": reason, "mode": mode, "seeds": seeds,
            "steps": steps, "split": split, "per_seed": per_seed}
