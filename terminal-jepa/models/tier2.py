"""Track B tier 2: learned dynamics over ORACLE keyed features — the predictor
architecture bake-off (status doc "Path to a working agent").

The representation is ground truth by construction (every path slot's symbolic
value), so battery differences between candidates are attributable to predictor
architecture alone. Both candidates share one trunk — a transformer over the
fixed key universe (258 file slots + 42 dir slots + 1 cwd slot + 1 action token)
— and differ ONLY in output parameterization:

- slotpred: re-predicts every slot's next value. Identity at init: logits =
  copy_margin * onehot(current value) + zero-init delta head (the project's
  identity-at-init lesson, finding 9, applied at the value level).
- editpred: copies the state structurally and predicts sparse edits — a
  per-slot change probability (bias-initialized near zero) plus new values,
  supervised only where the state truly changed.

Action-argument binding is TIED: a path argument embeds as the key embedding of
the slot it references (plus a role offset), so attention between the action
token and the referenced slot needs no table-matching to be learned.

Latents at battery time are hard symbolic states (argmax/threshold decode per
step), so `evals.dynamics` scores tier-2 candidates with the SAME symbolic edit
distance as the oracle/copy controls — numbers are directly comparable.

Pre-registered selection rule (recorded 2026-07-13, before training):
1. calibration: h1 rollout error on valid-no-op and invalid transitions must
   match the copy floor (0) within noise, with calibration AUC >= 0.9;
2. among calibrated candidates, lowest state-changing h1 rollout error;
3. ties broken by goal-ranking margin vs random.
Training budget is fixed and identical for both candidates; training loss is
never a selection criterion.

Usage:
  .venv/bin/python -m models.tier2 --data data/v1 --head slotpred --out runs/tier2/slotpred
  .venv/bin/python -m models.tier2 --data data/v1 --head editpred --out runs/tier2/editpred
  python3 -m evals.dynamics --data data/v1 --adapter tier2 --ckpt runs/tier2/slotpred/ckpt.pt --out runs/tier2/slotpred/battery.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import vocab  # noqa: E402
from env.state import FsState
from models.data import VERB_INDEX

N_FILES = len(vocab.FILE_PATHS)   # 258
N_DIRS = len(vocab.DIR_PATHS)     # 42
N_CWD = len(vocab.CWD_PATHS)      # 43
N_SLOTS = N_FILES + N_DIRS + 1    # 301: [files | dirs | cwd]
FILE_VALS = vocab.N_CONTENT + 1   # 0 = absent, k+1 = class k
DIR_VALS = 2
N_SPECIAL_ARGS = 11               # 0 empty, 1 root, 2 dot, 3..10 content c0..c7


def state_to_ids(state):
    f = state.features()
    ids = [c + 1 for c in f["file_class"]]
    ids += f["dir_exists"]
    ids.append(f["cwd_index"])
    return torch.tensor(ids, dtype=torch.long)


def ids_to_state(ids):
    ids = ids.tolist()
    files = {vocab.FILE_PATHS[i]: v - 1 for i, v in enumerate(ids[:N_FILES]) if v > 0}
    dirs = {vocab.DIR_PATHS[i] for i, v in enumerate(ids[N_FILES:N_FILES + N_DIRS]) if v}
    return FsState(dirs=dirs, files=files, cwd=vocab.CWD_PATHS[ids[-1]])


def arg_to_id(a):
    """Args map to either a special id (< N_SPECIAL_ARGS) or N_SPECIAL_ARGS + slot
    index, so path args can reuse the referenced slot's key embedding (tying)."""
    if a == "":
        return 0
    if a == "/":
        return 1
    if a == ".":
        return 2
    if not a.startswith("/"):
        return 3 + int(a[1:])  # content token "cK"
    p = vocab.str_to_path(a)
    if p in vocab.FILE_PATH_INDEX:
        return N_SPECIAL_ARGS + vocab.FILE_PATH_INDEX[p]
    return N_SPECIAL_ARGS + N_FILES + vocab.DIR_PATH_INDEX[p]


def action_to_ids(action):
    return torch.tensor(
        [VERB_INDEX[action[0]], arg_to_id(action[1]), arg_to_id(action[2])],
        dtype=torch.long,
    )


class Tier2Net(nn.Module):
    def __init__(self, head="slotpred", d=128, layers=2, nheads=4, copy_margin=10.0):
        super().__init__()
        assert head in ("slotpred", "editpred")
        self.head_kind = head
        self.copy_margin = copy_margin
        self.key_emb = nn.Embedding(N_SLOTS + 1, d)  # +1: the action token's key
        self.file_val = nn.Embedding(FILE_VALS, d)
        self.dir_val = nn.Embedding(DIR_VALS, d)
        self.cwd_val = nn.Embedding(N_CWD, d)
        self.verb_emb = nn.Embedding(len(VERB_INDEX), d)
        self.special_arg = nn.Embedding(N_SPECIAL_ARGS, d)
        self.role = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(
            d, nheads, 4 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.trunk = nn.TransformerEncoder(layer, layers)
        self.file_head = nn.Linear(d, FILE_VALS)
        self.dir_head = nn.Linear(d, DIR_VALS)
        self.cwd_head = nn.Linear(d, N_CWD)
        if head == "slotpred":
            for h in (self.file_head, self.dir_head, self.cwd_head):
                nn.init.zeros_(h.weight)  # delta head: identity at init
                nn.init.zeros_(h.bias)
        else:
            self.change_head = nn.Linear(d, 1)
            nn.init.constant_(self.change_head.bias, -3.0)  # near-copy at init

    def _arg_vec(self, arg_ids, role):
        is_slot = (arg_ids >= N_SPECIAL_ARGS).unsqueeze(-1)
        slot = self.key_emb(torch.clamp(arg_ids - N_SPECIAL_ARGS, min=0))
        spec = self.special_arg(torch.clamp(arg_ids, max=N_SPECIAL_ARGS - 1))
        return torch.where(is_slot, slot, spec) + self.role.weight[role]

    def forward(self, ids, act):
        """ids [B, 301], act [B, 3] -> dict of logits."""
        vals = torch.cat([
            self.file_val(ids[:, :N_FILES]),
            self.dir_val(ids[:, N_FILES:-1]),
            self.cwd_val(ids[:, -1:]),
        ], dim=1)
        toks = vals + self.key_emb.weight[:N_SLOTS]
        a = (self.key_emb.weight[N_SLOTS] + self.verb_emb(act[:, 0])
             + self._arg_vec(act[:, 1], 0) + self._arg_vec(act[:, 2], 1))
        h = self.trunk(torch.cat([toks, a.unsqueeze(1)], dim=1))[:, :N_SLOTS]
        out = {
            "file": self.file_head(h[:, :N_FILES]),
            "dir": self.dir_head(h[:, N_FILES:-1]),
            "cwd": self.cwd_head(h[:, -1]),
        }
        if self.head_kind == "slotpred":
            out["file"] = out["file"] + self.copy_margin * F.one_hot(ids[:, :N_FILES], FILE_VALS)
            out["dir"] = out["dir"] + self.copy_margin * F.one_hot(ids[:, N_FILES:-1], DIR_VALS)
            out["cwd"] = out["cwd"] + self.copy_margin * F.one_hot(ids[:, -1], N_CWD)
        else:
            out["change"] = self.change_head(h).squeeze(-1)
        return out

    @torch.no_grad()
    def step(self, ids, act):
        out = self(ids, act)
        nxt = torch.cat([
            out["file"].argmax(-1),
            out["dir"].argmax(-1),
            out["cwd"].argmax(-1, keepdim=True),
        ], dim=1)
        if self.head_kind == "editpred":
            nxt = torch.where(torch.sigmoid(out["change"]) > 0.5, nxt, ids)
        return nxt


def loss_fn(net, out, ids, nxt, pos_weight):
    tf, td, tc = nxt[:, :N_FILES], nxt[:, N_FILES:-1], nxt[:, -1]
    if net.head_kind == "slotpred":
        # One mean over ALL slot instances, never per-group means: per-group
        # averaging weights a changed slot's gradient by 1/group-size, a 258:6:1
        # cwd:dir:file asymmetry that left file edits entirely unlearned
        # (2026-07-13 bake-off round 1 — 0% fired on every file verb).
        ce = torch.cat([
            F.cross_entropy(out["file"].reshape(-1, FILE_VALS), tf.reshape(-1), reduction="none"),
            F.cross_entropy(out["dir"].reshape(-1, DIR_VALS), td.reshape(-1), reduction="none"),
            F.cross_entropy(out["cwd"], tc, reduction="none"),
        ])
        return ce.mean()
    changed = (nxt != ids).float()
    loss = F.binary_cross_entropy_with_logits(out["change"], changed, pos_weight=pos_weight)
    for logits, tgt, cur in ((out["file"], tf, ids[:, :N_FILES]),
                             (out["dir"], td, ids[:, N_FILES:-1]),
                             (out["cwd"].unsqueeze(1), tc.unsqueeze(1), ids[:, -1:])):
        m = tgt != cur
        if m.any():
            loss = loss + F.cross_entropy(logits[m], tgt[m])
    return loss


# -- data -------------------------------------------------------------------------


def featurize_split(jsonl_path, max_trajs):
    from evals.dynamics import load_transitions
    prev, nxt, acts, ttypes = [], [], [], []
    for tr in load_transitions(jsonl_path, max_trajs):
        for t, a in enumerate(tr["actions"]):
            prev.append(state_to_ids(tr["states"][t]))
            nxt.append(state_to_ids(tr["states"][t + 1]))
            acts.append(action_to_ids(a))
            ttypes.append(tr["ttypes"][t])
    return torch.stack(prev), torch.stack(nxt), torch.stack(acts), ttypes


@torch.no_grad()
def exact_match_eval(net, prev, nxt, acts, ttypes, device, chunk=512):
    hits = {}
    for i in range(0, prev.shape[0], chunk):
        pred = net.step(prev[i:i + chunk].to(device), acts[i:i + chunk].to(device)).cpu()
        ok = (pred == nxt[i:i + chunk]).all(dim=1)
        for j, t in enumerate(ttypes[i:i + chunk]):
            hits.setdefault(t, []).append(ok[j].item())
    em = {t: sum(v) / len(v) for t, v in hits.items()}
    em["all"] = sum(sum(v) for v in hits.values()) / sum(len(v) for v in hits.values())
    return em


# -- battery adapter ----------------------------------------------------------------


class Tier2Adapter:
    """Oracle representation + learned dynamics; latents are hard symbolic states so
    the battery's numbers are directly comparable to the oracle/copy controls."""

    def __init__(self, ckpt_path, device="cpu"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.net = Tier2Net(**ckpt["config"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(device).eval()
        self.device = device
        self.name = f"tier2-{ckpt['config']['head']}"

    def encode(self, state, ctx=None):
        return state

    def predict(self, state, action):
        ids = state_to_ids(state).unsqueeze(0).to(self.device)
        act = action_to_ids(action).unsqueeze(0).to(self.device)
        return ids_to_state(self.net.step(ids, act)[0].cpu())

    def distance(self, a, b):
        from evals.dynamics import OracleAdapter
        return OracleAdapter().distance(a, b)


# -- training ------------------------------------------------------------------------


def main(argv=None):
    from train.train import pick_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--head", required=True, choices=["slotpred", "editpred"])
    ap.add_argument("--max-train-trajs", type=int, default=800)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    root = pathlib.Path(args.data)
    prev, nxt, acts, ttypes = featurize_split(root / "train.jsonl", args.max_train_trajs)
    n_val = max(1, prev.shape[0] // 20)  # train-layout monitoring slice, NOT val.jsonl
    tr_sl, va_sl = slice(0, -n_val), slice(-n_val, None)
    print(f"transitions: train={prev.shape[0] - n_val} monitor={n_val}", flush=True)

    config = {"head": args.head, "d": 128, "layers": 2, "nheads": 4, "copy_margin": 10.0}
    net = Tier2Net(**config).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    # Identity-at-init guard (finding 9): both heads must copy before training.
    with torch.no_grad():
        chk = net.step(prev[:64].to(device), acts[:64].to(device)).cpu()
    assert (chk == prev[:64]).all(), "predictor is not the identity at init"
    print(f"head={args.head} params={n_params / 1e6:.2f}M identity-at-init: OK", flush=True)

    # Embeddings identified by module type, not name matching — name matching
    # silently decayed special_arg/role/file_val/dir_val/cwd_val (finding 19).
    emb_ids = {id(m.weight) for m in net.modules() if isinstance(m, nn.Embedding)}
    decay, no_decay = [], []
    for p in net.parameters():
        (no_decay if p.ndim <= 1 or id(p) in emb_ids else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)
    pos_weight = torch.tensor(args.pos_weight, device=device)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)
    n_train = prev.shape[0] - n_val
    log = []
    skipped = 0  # nonfinite-step guard (project convention for this machine's MPS
    # failure mode — status doc operational learnings; finding 19 flagged its absence)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n_train, (args.batch,), generator=g)
        ids, target = prev[idx].to(device), nxt[idx].to(device)
        act = acts[idx].to(device)
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net, net(ids, act), ids, target, pos_weight)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            skipped += 1
            continue
        opt.step()
        if step % 200 == 0 or step == args.steps:
            net.eval()
            em = exact_match_eval(net, prev[va_sl], nxt[va_sl], acts[va_sl], ttypes[-n_val:], device)
            rec = {"step": step, "loss": round(loss.item(), 5), "skipped": skipped,
                   "em": {k: round(v, 4) for k, v in em.items()}}
            log.append(rec)
            print(json.dumps(rec), flush=True)

    torch.save({"config": config, "state_dict": net.state_dict(),
                "train_args": vars(args)}, out / "ckpt.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out / 'ckpt.pt'}", flush=True)


if __name__ == "__main__":
    main()
