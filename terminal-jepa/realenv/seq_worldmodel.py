"""R4: a SEQUENCE world model over real Docker-shell exploration trajectories.

Each trajectory is an exploration of one Linux image: identify the system (`uname`,
`cat` of a config file), then navigate/inspect (`cd`/`ls`/`cat`). We embed every command
and every resulting observation with a FROZEN encoder (ModernBERT), then train a small
CAUSAL transformer over the interleaved token stream

    cmd_0  obs_0  cmd_1  obs_1  ...  cmd_t  obs_t

whose hidden state AT each command position must predict that command's resulting
observation embedding z_obs[t] (in latent space, standardized) — the world-model bet:
preview a command's consequence BEFORE running it, using the whole exploration history
(you cannot read obs_t; it doesn't exist yet). Only the transformer is learned; perception
is frozen.

The fair test is generalization to UNSEEN SYSTEMS (held-out Docker *images* — NOT held-out
tools; inferring a never-seen tool is a read-the-man-page capability and an eventual goal).
We report on two splits:
  - dev    : seen images, unseen sequences (in-distribution reference);
  - heldout: unseen images (fedora/rocky/mariadb/httpd) — the milestone.

Against honest no-model baselines on the SAME next-observation retrieval yardstick:
  - predict-mean   (train-mean obs; chance under retrieval by construction);
  - copy-prev-obs  (z_obs[t] := z_obs[t-1]);
  - retrieve-by-cmd(nearest train observation whose COMMAND embedding matches — lexical
                    memory, no world model).
Metric: given a prediction, rank the true next observation against foils (random foils and
hard same-verb foils) — top-1 accuracy + MRR. A world model that has learned system dynamics
should beat copy and lexical retrieval on UNSEEN images, not just memorize.

Usage:
  .venv/bin/python -m realenv.seq_worldmodel --data data/dockerfs --out runs/dockerfs/seq.json
"""

import argparse
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

D = 768
OBS_CAP = 1600  # chars of command output kept before encoding (median 37, p95 ~1k)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- rendering

def render_obs(step):
    """The OBSERVATION = what you see after running the command: resulting cwd + exit code
    + (truncated) output. cwd matters because `cd` has no stdout — its visible effect IS the
    new working directory."""
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"cwd={step.get('cwd', '/')} exit={step.get('exit', 0)}\n{out}"


def render_cmd(step):
    return step["cmd"]


def verb_of(cmd):
    p = cmd.split()
    return p[0] if p else ""


# ---------------------------------------------------------------- encoding

@torch.no_grad()
def encode_split(path, model, tok, device, bs=96):
    """Per-sequence arrays z_obs[n,768], z_cmd[n,768] (frozen mean-pooled), + cmds/image.
    All texts across all sequences are encoded once, then regrouped."""
    seqs = [json.loads(l) for l in open(path)]
    obs_texts, cmd_texts, spans = [], [], []
    for sq in seqs:
        start = len(obs_texts)
        for s in sq["steps"]:
            obs_texts.append(render_obs(s))
            cmd_texts.append(render_cmd(s))
        spans.append((start, len(obs_texts)))

    def enc(texts, tag):
        # length-sorted batching: group similar-length texts so padding is minimal (median
        # obs ~37 chars, p95 ~1k). Compute in sorted order, scatter back to original order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.zeros(len(texts), D)
        for i in range(0, len(order), bs):
            bidx = order[i:i + bs]
            e = tok([texts[j] for j in bidx], return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
            e = {k: v.to(device) for k, v in e.items()}
            h = model(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1)
            pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu()
            for k, j in enumerate(bidx):
                out[j] = pooled[k]
            if (i // bs) % 50 == 0:
                print(f"  enc {tag} {i}/{len(texts)}", flush=True)
        return out

    z_obs, z_cmd = enc(obs_texts, "obs"), enc(cmd_texts, "cmd")
    out = []
    for (a, b), sq in zip(spans, seqs):
        out.append({"z_obs": z_obs[a:b], "z_cmd": z_cmd[a:b],
                    "cmds": [s["cmd"] for s in sq["steps"]], "image": sq["image"],
                    # per-step success flag (exit 0 + non-empty output) — v2 class slicing
                    # (grep-miss exclusion); absent in v1 caches, consumers default all-True
                    "ok": [s.get("exit", 0) == 0 and bool((s.get("output") or "").strip())
                           for s in sq["steps"]]})
    return out


def cached_encode(data_root, split, model_name, device):
    cache = pathlib.Path(data_root) / f"emb-seq-{split}.pt"
    if cache.exists():
        print(f"  using cache {cache}", flush=True)
        return torch.load(cache, weights_only=False)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    seqs = encode_split(pathlib.Path(data_root) / f"{split}.jsonl", model, tok, device)
    torch.save(seqs, cache)
    print(f"  cached {len(seqs)} sequences -> {cache}", flush=True)
    return seqs


# ---------------------------------------------------------------- bag-of-tokens (gen twin)

def _iter_steps(data_root, split):
    for line in open(pathlib.Path(data_root) / f"{split}.jsonl"):
        for s in json.loads(line)["steps"]:
            yield s


def build_vocab(data_root, split, model_name, top_v=4000):
    """Multi-hot vocab over observation tokens, for the generative (surface-reconstruction)
    twin: the V most frequent observation tokens in the fit split."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    c = Counter()
    for step in _iter_steps(data_root, split):
        c.update(set(tok(render_obs(step), truncation=True, max_length=256)["input_ids"]))
    vmap = {tid: i for i, (tid, _) in enumerate(c.most_common(top_v))}
    return tok, vmap


# ---------------------------------------------------------------- data prep

def standardize_stats(seqs):
    allo = torch.cat([s["z_obs"] for s in seqs])
    allc = torch.cat([s["z_cmd"] for s in seqs])
    return (allo.mean(0, keepdim=True), allo.std(0, keepdim=True).clamp(min=1e-6),
            allc.mean(0, keepdim=True), allc.std(0, keepdim=True).clamp(min=1e-6))


def apply_stats(seqs, mo, so, mc, sc):
    for s in seqs:
        s["z_obs"] = (s["z_obs"] - mo) / so
        s["z_cmd"] = (s["z_cmd"] - mc) / sc
        if "z_obs_multi" in s:  # multi-vector stream segments live in the obs space
            s["z_obs_multi"] = (s["z_obs_multi"] - mo.unsqueeze(0)) / so.unsqueeze(0)


def bag_of(text, tok, vmap):
    v = torch.zeros(len(vmap))
    for tid in set(tok(text, truncation=True, max_length=256)["input_ids"]):
        if tid in vmap:
            v[vmap[tid]] = 1.0
    return v


def attach_bags(seqs, raw_path, tok, vmap):
    """Attach [n,V] observation token-bags to each seq dict (raw file order matches cached
    order; zip truncates to len(seqs) so a --limit smoke still works)."""
    raws = [json.loads(l) for l in open(raw_path)]
    assert len(raws) >= len(seqs), f"raw {len(raws)} < seqs {len(seqs)}"
    for s, r in zip(seqs, raws):
        assert s["image"] == r["image"], "cache/raw order mismatch"
        s["bag"] = torch.stack([bag_of(render_obs(st), tok, vmap) for st in r["steps"]])


# ---------------------------------------------------------------- model

class SeqWorldModel(nn.Module):
    """Causal transformer over interleaved [cmd_0,obs_0,cmd_1,obs_1,...] frozen-embedding
    tokens. `aux='jepa'` -> head at cmd positions predicts z_obs[t] (latent). `aux='recon'`
    -> predicts the obs token-bag (surface reconstruction; the generative twin). Both expose
    the pre-head hidden state at cmd positions for a common downstream probe."""

    def __init__(self, aux="jepa", vsize=0, d=192, layers=4, heads=4, dropout=0.1,
                 no_history=False):
        super().__init__()
        self.aux = aux
        self.d = d
        self.no_history = no_history  # self-only attention => matched-capacity, history-free control
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)   # 0=cmd, 1=obs
        self.pos_emb = nn.Embedding(64, d)   # max 2*32 steps
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, D if aux == "jepa" else vsize)

    def encode(self, tok_emb, types, key_pad):
        """tok_emb [B,L,D] frozen embeddings; types [B,L] in {0,1}; key_pad [B,L] True=pad.
        Returns hidden [B,L,d]. no_history=True masks all attention except self — same
        architecture/capacity as the full model but each token sees only itself, isolating the
        value of the exploration history from raw function-approximation capacity."""
        B, L, _ = tok_emb.shape
        pos = torch.arange(L, device=tok_emb.device)
        x = self.proj(tok_emb) + self.type_emb(types) + self.pos_emb(pos)[None]
        if self.no_history:
            mask = ~torch.eye(L, device=tok_emb.device, dtype=torch.bool)  # allow only the diagonal
        else:
            mask = torch.triu(torch.ones(L, L, device=tok_emb.device, dtype=torch.bool), 1)  # causal
        return self.tf(x, mask=mask, src_key_padding_mask=key_pad)

    def forward(self, tok_emb, types, key_pad):
        h = self.encode(tok_emb, types, key_pad)   # [B,L,d]
        return self.head(h), h                     # predictions at every position


def collate(batch, device):
    """batch: list of seq dicts. Interleave cmd/obs into token stream, pad, build masks and
    the target (standardized z_obs at each cmd position). Returns tensors on device."""
    maxn = max(s["z_obs"].shape[0] for s in batch)
    L = 2 * maxn
    B = len(batch)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)          # True = pad
    tgt = torch.zeros(B, maxn, D)
    bag = None
    if "bag" in batch[0]:
        bag = torch.zeros(B, maxn, batch[0]["bag"].shape[1])
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)     # valid cmd positions
    for bi, s in enumerate(batch):
        n = s["z_obs"].shape[0]
        for i in range(n):
            tok[bi, 2 * i] = s["z_cmd"][i]
            tok[bi, 2 * i + 1] = s["z_obs"][i]
            types[bi, 2 * i] = 0
            types[bi, 2 * i + 1] = 1
            key_pad[bi, 2 * i] = False
            key_pad[bi, 2 * i + 1] = False
            tgt[bi, i] = s["z_obs"][i]
            if bag is not None:
                bag[bi, i] = s["bag"][i]
            cmd_mask[bi, i] = True
    out = {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
           "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}
    if bag is not None:
        out["bag"] = bag.to(device)
    return out


def cmd_hidden(net, b):
    """Hidden state (and prediction) at cmd positions: [sum_valid, .]."""
    pred, h = net(b["tok"], b["types"], b["key_pad"])
    cmd_pred = pred[:, 0::2]          # [B, maxn, .] — cmd tokens are even positions
    cmd_h = h[:, 0::2]
    m = b["cmd_mask"]
    return cmd_pred[m], cmd_h[m], b["tgt"][m], (b["bag"][m] if "bag" in b else None)


def train_model(aux, fit, device, vsize=0, steps=4000, bs=64, lr=3e-4, seed=0, no_history=False,
                jepa_loss=None):
    """jepa_loss: optional callable(pred, tgt)->scalar overriding the default MSE for the 'jepa'
    aux (used by the evolve harness to train under an evolved objective; None = MSE baseline)."""
    torch.manual_seed(seed)
    net = SeqWorldModel(aux, vsize, no_history=no_history).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    bce = nn.functional.binary_cross_entropy_with_logits
    jloss = jepa_loss if jepa_loss is not None else (lambda p, t: ((p - t) ** 2).mean())
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(fit), (bs,), generator=g).tolist()
        b = collate([fit[i] for i in idx], device)
        pred, _, tgt, bag = cmd_hidden(net, b)
        loss = jloss(pred, tgt) if aux == "jepa" else bce(pred, bag)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"  [{aux}] step {step} loss {loss.item():.4f}", flush=True)
    return net


class CmdOnlyMLP(nn.Module):
    """History-FREE learned baseline: f(z_cmd) -> z_obs, no exploration context. The critical
    ablation — if the sequence world model can't beat this, the history/sequence buys nothing
    (it's just a per-command lookup, which is what a world model is supposed to transcend)."""

    def __init__(self, d=D, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, d))

    def forward(self, zc):
        return self.net(zc)


def train_cmd_only(fit, device, steps=4000, bs=256, lr=3e-4, seed=0):
    zc = torch.cat([s["z_cmd"] for s in fit]); zo = torch.cat([s["z_obs"] for s in fit])
    torch.manual_seed(seed)
    net = CmdOnlyMLP().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        idx = torch.randint(0, zc.shape[0], (bs,), generator=g)
        loss = ((net(zc[idx].to(device)) - zo[idx].to(device)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net


# ---------------------------------------------------------------- flat eval tensors

@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    """Run the model over every sequence; collect per-step predicted z_obs, hidden state,
    true z_obs, previous z_obs (copy baseline), command text, image. Order = step order."""
    net.eval()  # dropout off for inference (MPS fused attention also rejects nonzero dropout)
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, h = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = pred[:, 0::2].cpu(); cmd_h = h[:, 0::2].cpu()
        for bi, s in enumerate(chunk):
            n = s["z_obs"].shape[0]
            for t in range(n):
                preds.append(cmd_pred[bi, t]); hids.append(cmd_h[bi, t])
                trues.append(s["z_obs"][t])
                prevs.append(s["z_obs"][t - 1] if t > 0 else torch.zeros(D))
                cmds.append(s["cmds"][t]); imgs.append(s["image"])
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs,
            "verbs": [verb_of(c) for c in cmds]}


# ---------------------------------------------------------------- retrieval metric

def _foils_random(N, K, gen):
    return torch.randint(0, N, (N, K), generator=gen)


def _foils_sameverb(verbs, K, gen):
    """[N,K] foil indices, each row drawn (with replacement) from the SAME-verb steps."""
    by_verb = defaultdict(list)
    for j, v in enumerate(verbs):
        by_verb[v].append(j)
    pools = {v: torch.tensor(idxs) for v, idxs in by_verb.items()}
    foil = torch.empty(len(verbs), K, dtype=torch.long)
    for v, rows in by_verb.items():
        pool = pools[v]
        pick = pool[torch.randint(0, len(pool), (len(rows), K), generator=gen)]
        foil[torch.tensor(rows)] = pick
    return foil


@torch.no_grad()
def _rank_stats(pred, true, foil_idx, blk=1024):
    """Given per-step foil indices [N,K], score each candidate (true + K foils) by distance to
    pred and count foils strictly closer than the true. Returns (top1, mrr) over all N."""
    N = true.shape[0]
    top1 = 0.0; mrr = 0.0
    for s0 in range(0, N, blk):
        sl = slice(s0, min(s0 + blk, N))
        q = pred[sl].unsqueeze(1)                                    # [b,1,D]
        d_true = ((true[sl] - pred[sl]) ** 2).mean(-1, keepdim=True)  # [b,1]
        d_foil = ((true[foil_idx[sl]] - q) ** 2).mean(-1)            # [b,K]
        closer = (d_foil < d_true).sum(1)                            # strict: ties don't beat true
        top1 += (closer == 0).float().sum().item()
        mrr += (1.0 / (closer.float() + 1.0)).sum().item()
    return top1 / N, mrr / N


def retrieval(pred, true, verbs, n_foils=63, rounds=4, seed=0):
    """For each step, rank the TRUE next obs against foils by distance to `pred`: random foils
    AND hard same-verb foils. top-1 acc + MRR, averaged over rounds. `pred` is any prediction
    aligned with `true` (model output, z_prev for copy, zeros for predict-mean, ...). Ties are
    not counted as beating the true candidate, so predict-mean -> ~chance by construction."""
    N = true.shape[0]
    gen = torch.Generator().manual_seed(seed)
    t1s = mrrs = t1r = mrrr = 0.0
    for _ in range(rounds):
        f_h = _foils_sameverb(verbs, n_foils, gen); a, b = _rank_stats(pred, true, f_h)
        t1s += a; mrrs += b
        f_r = _foils_random(N, n_foils, gen); a, b = _rank_stats(pred, true, f_r)
        t1r += a; mrrr += b
    return {"top1_sameverb": t1s / rounds, "mrr_sameverb": mrrs / rounds,
            "top1_random": t1r / rounds, "mrr_random": mrrr / rounds}


VERBSET = ("uname", "ls", "cat", "cd")


def per_verb_breakdown(preds, true, verbs, seed, verbset=VERBSET):
    """top-1 (same-verb foils) restricted to each verb's steps. Exposes whether the world
    model's advantage is real (ls/cat: predict a listing/file's content on an unseen system)
    or trivial (cd: the observation is just `cwd=<target>`, echoable from the command)."""
    out = {}
    for v in verbset:
        idx = [i for i, vv in enumerate(verbs) if vv == v]
        if len(idx) < 20:
            continue
        ii = torch.tensor(idx); sub_true = true[ii]; sub_verbs = [v] * len(idx)
        row = {"n": len(idx)}
        for name, p in preds.items():
            row[name] = retrieval(p[ii], sub_true, sub_verbs, seed=seed)["top1_sameverb"]
        out[v] = row
    return out


def content_retrieval(pred, true, verbs, content=("ls", "cat"), seed=0):
    """Retrieval restricted to CONTENT verbs (ls/cat) — the observations that are NOT lexically
    echoable from the command (unlike cd's `cwd=<target>`). This is the honest headline: can the
    model predict a listing / file content on an unseen system? Foils are same-verb within the
    content subset."""
    idx = [i for i, v in enumerate(verbs) if v in content]
    ii = torch.tensor(idx)
    return retrieval(pred[ii], true[ii], [verbs[i] for i in idx], seed=seed)


def latent_mse(pred, true):
    return ((pred - true) ** 2).mean().item()


def cosine(pred, true):
    a = torch.nn.functional.normalize(pred, dim=-1)
    b = torch.nn.functional.normalize(true, dim=-1)
    return (a * b).sum(-1).mean().item()


# ---------------------------------------------------------------- retrieve-by-command baseline

def retrieve_by_cmd_baseline(fit_seqs, eval_flat):
    """No-model lexical memory: predict the eval step's obs as the obs of the TRAIN step whose
    COMMAND embedding is nearest (cosine). Returns a prediction tensor aligned with
    eval_flat['true']. This is the "just memorize commands" competitor a world model must beat."""
    train_cmd = torch.cat([s["z_cmd"] for s in fit_seqs])          # [M, D]
    train_obs = torch.cat([s["z_obs"] for s in fit_seqs])          # [M, D]
    q = eval_flat["_cmd_embs"]                                     # [N, D]
    tn = torch.nn.functional.normalize(train_cmd, dim=-1)
    preds = []
    for i in range(0, q.shape[0], 256):
        qn = torch.nn.functional.normalize(q[i:i + 256], dim=-1)
        preds.append(train_obs[(qn @ tn.T).argmax(1)])
    return torch.cat(preds)


# ---------------------------------------------------------------- splits

def split_train_dev(fit_seqs, frac=0.1, seed=0):
    rng = random.Random(f"dev:{seed}")
    idx = list(range(len(fit_seqs)))
    rng.shuffle(idx)
    k = max(1, int(len(idx) * frac))
    dev = set(idx[:k])
    return ([s for i, s in enumerate(fit_seqs) if i not in dev],
            [s for i, s in enumerate(fit_seqs) if i in dev])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dockerfs")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--top-v", type=int, default=4000)
    ap.add_argument("--gen-twin", action="store_true", help="also run the generative twin comparison")
    ap.add_argument("--ablation", default="", choices=["", "history"],
                    help="history: full vs same-architecture self-only (history-masked) transformer")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap #sequences per split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = pick_device()
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"device {device} | seeds {seeds}", flush=True)

    train_seqs = cached_encode(args.data, "train", args.model, device)
    val_seqs = cached_encode(args.data, "val", args.model, device)
    if args.limit:
        train_seqs = train_seqs[:args.limit]; val_seqs = val_seqs[:args.limit]

    mo, so, mc, sc = standardize_stats(train_seqs)
    apply_stats(train_seqs, mo, so, mc, sc); apply_stats(val_seqs, mo, so, mc, sc)

    if args.ablation == "history":
        print("=== history ablation: full vs self-only (matched-capacity) transformer ===", flush=True)
        report = run_history_ablation(train_seqs, val_seqs, device, args.steps, seeds)
        print("=== HISTORY ABLATION ===\n" + json.dumps(report, indent=1), flush=True)
        if args.out:
            p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=1)); print(f"wrote {args.out}", flush=True)
        return report

    tok = vmap = None
    if args.gen_twin:
        tok, vmap = build_vocab(args.data, "train", args.model, args.top_v)
        attach_bags(train_seqs, pathlib.Path(args.data) / "train.jsonl", tok, vmap)
        attach_bags(val_seqs, pathlib.Path(args.data) / "val.jsonl", tok, vmap)

    per_seed = []
    for s in seeds:
        print(f"=== seed {s} ===", flush=True)
        fit, dev = split_train_dev(train_seqs, seed=s)
        net = train_model("jepa", fit, device, steps=args.steps, seed=s)
        mlp = train_cmd_only(fit, device, steps=args.steps, seed=s)  # history-free ablation

        res = {}
        for name, seqs in (("dev", dev), ("heldout", val_seqs)):
            flat = flatten_predictions(net, seqs, device)
            flat["_cmd_embs"] = torch.stack([sq["z_cmd"][t]
                                             for sq in seqs for t in range(sq["z_obs"].shape[0])])
            with torch.no_grad():
                nohist_pred = mlp(flat["_cmd_embs"].to(device)).cpu()
            mean_pred = torch.zeros_like(flat["true"])
            preds = {"wm": flat["pred"], "wm_no_history": nohist_pred, "copy_prev": flat["prev"],
                     "predict_mean": mean_pred, "retrieve_by_cmd": retrieve_by_cmd_baseline(fit, flat)}
            res[name] = {"n": flat["true"].shape[0]}
            for k, p in preds.items():
                res[name][k] = retrieval(p, flat["true"], flat["verbs"], seed=s)
            res[name]["by_verb"] = per_verb_breakdown(
                {"wm": flat["pred"], "wm_no_history": nohist_pred,
                 "retrieve_by_cmd": preds["retrieve_by_cmd"]},
                flat["true"], flat["verbs"], seed=s)
            res[name].update({
                "wm_latent_mse": latent_mse(flat["pred"], flat["true"]),
                "nohist_latent_mse": latent_mse(nohist_pred, flat["true"]),
                "copy_latent_mse": latent_mse(flat["prev"], flat["true"]),
                "mean_latent_mse": latent_mse(mean_pred, flat["true"]),
                "wm_cosine": cosine(flat["pred"], flat["true"]),
            })
            print(f"  {name}: wm top1(sv)={res[name]['wm']['top1_sameverb']:.3f} "
                  f"nohist={res[name]['wm_no_history']['top1_sameverb']:.3f} "
                  f"copy={res[name]['copy_prev']['top1_sameverb']:.3f} "
                  f"retr={res[name]['retrieve_by_cmd']['top1_sameverb']:.3f} "
                  f"mean={res[name]['predict_mean']['top1_sameverb']:.3f} "
                  f"| wm_mse={res[name]['wm_latent_mse']:.3f} copy_mse={res[name]['copy_latent_mse']:.3f}",
                  flush=True)

        if args.gen_twin:
            res["gen_twin"] = run_gen_twin(fit, val_seqs, device, vmap, args.steps, s,
                                           res["heldout"]["wm"])
        per_seed.append(res)

    report = aggregate(per_seed, args, seeds)
    print("=== AGGREGATE ===\n" + json.dumps(report["heldout"], indent=1), flush=True)
    if args.out:
        p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}", flush=True)
    return report


@torch.no_grad()
def gen_flatten(net, seqs, device, bs=64):
    """For the recon twin: predicted obs token-bag LOGITS + true bags at each cmd position."""
    net.eval()
    logits, bags, verbs = [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, _ = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = pred[:, 0::2].cpu()
        for bi, s in enumerate(chunk):
            for t in range(s["z_obs"].shape[0]):
                logits.append(cmd_pred[bi, t]); bags.append(s["bag"][t]); verbs.append(verb_of(s["cmds"][t]))
    return torch.stack(logits), torch.stack(bags), verbs


def bag_retrieval(logits, bags, verbs, n_foils=63, rounds=4, seed=0, blk=64):
    """Recon twin's NATIVE-space retrieval, mirroring `retrieval` but scoring candidate bags by
    BCE(pred_logits, bag) instead of latent distance. Same metric (top-1 + MRR) so jepa (latent)
    and recon (tokens) compete apples-to-apples on 'pick the true next observation'."""
    import torch.nn.functional as Fn
    N = bags.shape[0]
    gen = torch.Generator().manual_seed(seed)

    def rank(foil_idx):
        top1 = mrr = 0.0
        for s0 in range(0, N, blk):
            sl = slice(s0, min(s0 + blk, N)); lg = logits[sl]                      # [b,V]
            d_true = Fn.binary_cross_entropy_with_logits(lg, bags[sl], reduction="none").mean(-1, keepdim=True)
            fb = bags[foil_idx[sl]]                                                # [b,K,V]
            d_foil = Fn.binary_cross_entropy_with_logits(
                lg.unsqueeze(1).expand_as(fb), fb, reduction="none").mean(-1)      # [b,K]
            closer = (d_foil < d_true).sum(1)
            top1 += (closer == 0).float().sum().item(); mrr += (1.0 / (closer.float() + 1.0)).sum().item()
        return top1 / N, mrr / N

    t1s = mrrs = t1r = mrrr = 0.0
    for _ in range(rounds):
        a, b = rank(_foils_sameverb(verbs, n_foils, gen)); t1s += a; mrrs += b
        a, b = rank(_foils_random(N, n_foils, gen)); t1r += a; mrrr += b
    return {"top1_sameverb": t1s / rounds, "mrr_sameverb": mrrs / rounds,
            "top1_random": t1r / rounds, "mrr_random": mrrr / rounds}


def run_gen_twin(fit, val_seqs, device, vmap, steps, seed, jepa_ret):
    """JEPA (latent-prediction) vs a compute-matched GENERATIVE twin (obs token-reconstruction),
    each judged in ITS OWN native space by the SAME metric — next-observation retrieval on
    held-out images. No probe (a probe's target would equal the jepa objective and bias the
    test). The jepa arm reuses the main world model's held-out retrieval (identical net + eval);
    here we only train the recon twin and score it in token-bag space. Asks the JEPA question
    directly: does predicting the next observation's abstract EMBEDDING generalize to unseen
    systems better than predicting its surface TOKENS?"""
    rnet = train_model("recon", fit, device, vsize=len(vmap), steps=steps, seed=seed)
    rlog, rbag, rverb = gen_flatten(rnet, val_seqs, device)
    recon = bag_retrieval(rlog, rbag, rverb, seed=seed)
    print(f"    gen-twin [recon] top1(sv)={recon['top1_sameverb']:.3f} "
          f"vs [jepa] {jepa_ret['top1_sameverb']:.3f}", flush=True)
    return {"jepa": jepa_ret, "recon": recon,
            "jepa_minus_recon_top1_sameverb": round(jepa_ret["top1_sameverb"] - recon["top1_sameverb"], 4)}


def run_history_ablation(train_seqs, val_seqs, device, steps, seeds, jepa_loss=None):
    """The reviewer-required control for 'history helps': compare the FULL causal transformer
    against the SAME transformer with self-only attention (no_history) — identical architecture,
    depth, and parameter count, differing ONLY in whether each command position may attend to the
    exploration history. If full > masked on held-out content verbs, the sequence/history is doing
    real work (not just function-approximation capacity). Paired per seed. jepa_loss overrides the
    training objective (used to confirm history still drives the gain under an evolved objective)."""
    rows = []
    for s in seeds:
        fit, _ = split_train_dev(train_seqs, seed=s)
        r = {}
        for name, no_hist in (("full", False), ("masked", True)):
            net = train_model("jepa", fit, device, steps=steps, seed=s, no_history=no_hist,
                              jepa_loss=jepa_loss)
            flat = flatten_predictions(net, val_seqs, device)
            cv = content_retrieval(flat["pred"], flat["true"], flat["verbs"], seed=s)
            pv = per_verb_breakdown({"m": flat["pred"]}, flat["true"], flat["verbs"], seed=s)
            r[name] = {"content_top1_sameverb": cv["top1_sameverb"],
                       "content_mrr_sameverb": cv["mrr_sameverb"],
                       "ls_top1": pv.get("ls", {}).get("m"), "cat_top1": pv.get("cat", {}).get("m"),
                       "cd_top1": pv.get("cd", {}).get("m"),
                       "latent_mse": latent_mse(flat["pred"], flat["true"])}
        r["history_gain_content_top1"] = round(
            r["full"]["content_top1_sameverb"] - r["masked"]["content_top1_sameverb"], 4)
        rows.append(r)
        print(f"  seed {s}: full content_top1={r['full']['content_top1_sameverb']:.3f} "
              f"masked={r['masked']['content_top1_sameverb']:.3f} "
              f"gain={r['history_gain_content_top1']:.3f}", flush=True)

    def ms(f):
        v = [f(x) for x in rows]; v = [z for z in v if isinstance(z, (int, float)) and z == z]
        m = sum(v) / len(v)
        return {"mean": round(m, 4), "std": round((sum((z - m) ** 2 for z in v) / len(v)) ** 0.5, 4)}
    return {"seeds": seeds, "steps": steps, "n_val": val_seqs and sum(sq["z_obs"].shape[0] for sq in val_seqs),
            "full": {k: ms(lambda x, k=k: x["full"][k]) for k in ("content_top1_sameverb", "ls_top1", "cat_top1", "cd_top1", "latent_mse")},
            "masked": {k: ms(lambda x, k=k: x["masked"][k]) for k in ("content_top1_sameverb", "ls_top1", "cat_top1", "cd_top1", "latent_mse")},
            "history_gain_content_top1": ms(lambda x: x["history_gain_content_top1"])}


def aggregate(per_seed, args, seeds):
    def mean_std(vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        if not vals:
            return {"mean": float("nan"), "std": float("nan")}
        m = sum(vals) / len(vals)
        return {"mean": round(m, 4), "std": round((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5, 4)}

    report = {"data": args.data, "model": args.model, "seeds": seeds, "steps": args.steps}
    for split in ("dev", "heldout"):
        sec = {"n": per_seed[0][split]["n"]}
        for pred in ("wm", "wm_no_history", "copy_prev", "predict_mean", "retrieve_by_cmd"):
            for m in ("top1_sameverb", "mrr_sameverb", "top1_random", "mrr_random"):
                sec[f"{pred}.{m}"] = mean_std([p[split][pred][m] for p in per_seed])
        for m in ("wm_latent_mse", "nohist_latent_mse", "copy_latent_mse", "mean_latent_mse", "wm_cosine"):
            sec[m] = mean_std([p[split][m] for p in per_seed])
        # per-verb (mean over seeds); verbs present in every seed's breakdown
        verbs_present = set.intersection(*[set(p[split]["by_verb"].keys()) for p in per_seed]) \
            if all(p[split].get("by_verb") for p in per_seed) else set()
        bv = {}
        for v in sorted(verbs_present):
            bv[v] = {"n": per_seed[0][split]["by_verb"][v]["n"]}
            for pred in ("wm", "wm_no_history", "retrieve_by_cmd"):
                bv[v][f"{pred}.top1_sameverb"] = mean_std(
                    [p[split]["by_verb"][v][pred] for p in per_seed])
        sec["by_verb"] = bv
        report[split] = sec
    if args.gen_twin:
        g = {}
        for aux in ("jepa", "recon"):
            for m in ("top1_sameverb", "mrr_sameverb", "top1_random", "mrr_random"):
                g[f"{aux}.{m}"] = mean_std([p["gen_twin"][aux][m] for p in per_seed])
        g["jepa_minus_recon_top1_sameverb"] = mean_std([p["gen_twin"]["jepa_minus_recon_top1_sameverb"] for p in per_seed])
        report["gen_twin_heldout"] = g
    return report


if __name__ == "__main__":
    main()
