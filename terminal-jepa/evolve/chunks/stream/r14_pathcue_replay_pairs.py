"""stream chunk (R14): PATH-CUED REPLAY PAIRS — before each real (cmd, obs) pair the collate
inserts a phantom pair [z_cmd_i, z_obs_j] where j < i is the most recent earlier step whose
command touched the SAME path (exact) or the PARENT/CHILD directory of the path that cmd_i
targets — hippocampal cued replay materialized in the token layout, keyed by LEXICAL path
structure the trunk's embedding-keyed memories cannot compute.

WHY THIS CAN MOVE THE cat MARGIN (stalled +.293 -> +.295 across two arch generations):
  * The champion arch's file memory is verb-QUOTIENT keyed: `ls /etc` writes under key(/etc),
    but `cat /etc/os-release` queries under key(/etc/os-release) — the parent-directory
    evidence (which files exist, distro-telltale names) is in memory yet UNADDRESSABLE,
    because parent/child is a lexical-compositional relation invisible in e5 key space.
    This collate performs that binding symbolically at data layout time: dirname matching on
    the raw command strings, then copies the bound observation NEXT TO the querying command.
  * The phantom pair is a genuine (cmd, obs) pair from the arch's point of view (stride-2
    typing [0,1,0,1] is preserved): the delta-rule file memory WRITES (key from z_cmd_i ->
    recalled obs) one pair before the real cmd_i READS with the identical z_cmd_i — a fresh,
    undecayed, exactly-keyed binding, i.e. collate-time cued recall feeding the same
    read-after-write alignment the memory already trains for on repeated commands. The
    arch's copy-prev mix channel (prev pair's obs) becomes copy-WHAT-I-SAW-AT-THIS-PATH at
    cued steps, and degrades exactly to baseline copy-prev via the echo fallback below.
  * tail/head/grep on a previously cat'd file, and first-time cat after a parent ls (the
    dominant v2 cat case — sequences open with identity-revealing ls/cat), all get their
    strongest in-trajectory evidence placed at attention distance 1-2 from the query.

Cross-domain source: hippocampal pattern completion / cued recall (CA3) + RETRO-style
retrieved-neighbor token injection (Borgeaud et al., arXiv:2112.04426) + pointer/copy
mechanisms — but the retrieval index here is the filesystem hierarchy itself.

LAYOUT (seen by the arch as a normal stride-2 interleave, L = 4*n):
  virtual step 2i   (phantom): tok[4i]   = cue cmd, tok[4i+1] = recalled obs
  virtual step 2i+1 (real)   : tok[4i+2] = z_cmd[i], tok[4i+3] = z_obs[i]
Phantom construction per real step i (deterministic, strictly causal, j < i):
  1. CUE  : most recent j with an exact absolute-path match -> (z_cmd[i], z_obs[j]);
            else most recent j with a parent/child path match -> (z_cmd[i], z_obs[j]).
  2. ECHO : no path relation -> (z_cmd[i-1], z_obs[i-1]) — a repeat of the previous pair, so
            the arch's pair-shifted copy-prev channel keeps its baseline semantics exactly.
  3. i==0 : fully padded phantom pair (key_pad True; the arch's write gate is pair-masked).
The VIRTUAL-STEP framing is deliberate: tgt/cmd_mask span 2n virtual steps with cmd_mask True
only at real steps, so tok.shape[1] == 2*tgt.shape[1] and live types alternate 0/1 — the
champion cued-recall head's interleave guard passes and its aux stays ACTIVE (it anchors and
queries only cmd_mask-valid steps, i.e. real ones). tgt at a phantom slot holds z_obs[i-1] so
the harness's shifted prev_full keeps exact baseline semantics for delta-style targets.

FIXED across streams: the per-step target is the untouched standardized z_obs; eval space,
foils, baselines, split unchanged — only what the predictor READS changes. No re-encode
needed: phantoms are copies of embeddings already in the root, bound by parsing `cmds`.

Causality: the phantom for step i holds only z_obs_{j<i} (or z_obs_{i-1}) and sits after real
pair i-1 and before real cmd_i — every token derived from obs_t appears strictly after cmd_t.
leakage_ok corrupts obs_3 AT THE SOURCE and re-collates, so the real obs_3 token AND every
downstream phantom copy of it are corrupted together; cmd predictions at steps <= 3 must be
bit-stable.
"""

import torch

from realenv import seq_worldmodel as M

NAME = "r14_pathcue_replay_pairs"
DESCRIPTION = ("Path-cued replay pairs: collate inserts a phantom (cue-cmd, recalled-obs) pair "
               "before each real pair, bound by exact/parent/child lexical path match on the "
               "command strings (echo of the previous pair as fallback); stride-2 pair typing "
               "preserved so the champion arch/head machinery reads it natively.")


# ------------------------------------------------------------------ path binding (pure text)

def _paths(cmd):
    """Absolute path tokens of a command string, trailing-slash-normalized."""
    out = []
    for t in (cmd or "").split():
        t = t.strip("'\";,()")
        if t.startswith("/"):
            p = t.rstrip("/")
            out.append(p if p else "/")
    return out


def _dirname(p):
    i = p.rfind("/")
    return p[:i] if i > 0 else "/"


def _recall_plan(cmds):
    """Per step: ("cue", j) exact-or-parent/child path recall, ("echo", i-1), or (None, None).
    Deterministic; strictly causal (j < i). Exact match preferred over parent/child; most
    recent within each kind."""
    step_paths = [_paths(c) for c in cmds]
    plans = []
    for i in range(len(cmds)):
        pi = set(step_paths[i])
        parents = {_dirname(p) for p in step_paths[i]}
        exact = rel = None
        for j in range(i - 1, -1, -1):
            pj = set(step_paths[j])
            if pi & pj:
                exact = j
                break
            if rel is None and pj and (parents & pj or pi & {_dirname(p) for p in step_paths[j]}):
                rel = j
        j = exact if exact is not None else rel
        if j is not None:
            plans.append(("cue", j))
        elif i > 0:
            plans.append(("echo", i - 1))
        else:
            plans.append((None, None))
    return plans


# ------------------------------------------------------------------ stream contract

def collate(batch, device):
    maxn = max(s["z_obs"].shape[0] for s in batch)
    V = 2 * maxn                                   # virtual steps: phantom, real, phantom, real ...
    L = 2 * V                                      # tokens; == 2 * tgt.shape[1] (head guard)
    B = len(batch)
    tok = torch.zeros(B, L, M.D)
    types = (torch.arange(L) % 2).long().unsqueeze(0).expand(B, L).contiguous()  # [0,1,0,1,...]
    key_pad = torch.ones(B, L, dtype=torch.bool)   # True = pad
    tgt = torch.zeros(B, V, M.D)                   # FIXED single-vector target (virtual-step frame)
    cmd_mask = torch.zeros(B, V, dtype=torch.bool)  # True only at REAL virtual steps
    for bi, s in enumerate(batch):
        n = s["z_obs"].shape[0]
        plans = _recall_plan(s["cmds"][:n])
        for i in range(n):
            base = 4 * i
            kind, j = plans[i]
            if kind == "cue":
                tok[bi, base] = s["z_cmd"][i]      # the cue: the very command about to run
                tok[bi, base + 1] = s["z_obs"][j]  # the recalled past observation (j < i)
                key_pad[bi, base] = False
                key_pad[bi, base + 1] = False
            elif kind == "echo":
                tok[bi, base] = s["z_cmd"][j]      # repeat of the previous pair -> copy-prev
                tok[bi, base + 1] = s["z_obs"][j]  # channel keeps baseline semantics
                key_pad[bi, base] = False
                key_pad[bi, base + 1] = False
            # (None, None): phantom pair stays fully padded (step 0)
            if i > 0:
                tgt[bi, 2 * i] = s["z_obs"][i - 1]  # so shifted prev_full at real steps == z_prev
            tok[bi, base + 2] = s["z_cmd"][i]
            tok[bi, base + 3] = s["z_obs"][i]
            key_pad[bi, base + 2] = False
            key_pad[bi, base + 3] = False
            tgt[bi, 2 * i + 1] = s["z_obs"][i]
            cmd_mask[bi, 2 * i + 1] = True
    return {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
            "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}


def extract_cmd_pred(pred_full, batch):
    # All cmd-typed positions (phantom + real alternating) -> [B, 2*maxn, D], aligned with the
    # virtual-step tgt/cmd_mask; the harness's cmd_mask selection keeps only real steps.
    return pred_full[:, 0::2]


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    """Mirrors seq_worldmodel.flatten_predictions (step order; prev/true from the single z_obs);
    predictions read at the REAL cmd token of each step (position 4t+2)."""
    net.eval()
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, h = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = pred[:, 2::4].cpu()
        cmd_h = h[:, 2::4].cpu()
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
    """Corrupt obs_3 AT THE SOURCE and re-collate — the real obs_3 token AND every phantom
    copy of it (the exact-match recall at step 5 below) are corrupted together; predictions
    at the real cmd tokens of steps 0..3 must be bit-stable."""
    net.eval()
    torch.manual_seed(0)
    cmds = ["ls /etc", "cat /etc/os-release", "ls /var", "cat /etc/os-release",
            "ls /etc", "cat /etc/os-release"]  # step5 exact-recalls step3; step1 parent-recalls step0
    z_obs = torch.randn(6, M.D)
    z_cmd = torch.randn(6, M.D)
    seq0 = [{"z_obs": z_obs, "z_cmd": z_cmd, "cmds": cmds, "image": "x"}]
    b0 = collate(seq0, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 2::4].clone().cpu()
    z_obs_bad = z_obs.clone()
    z_obs_bad[3] = torch.randn(M.D) * 100.0
    seq1 = [{"z_obs": z_obs_bad, "z_cmd": z_cmd, "cmds": cmds, "image": "x"}]
    b1 = collate(seq1, device)
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 2::4].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
