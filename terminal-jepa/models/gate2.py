"""Track A gate 2: the tier-2-winning predictor shape over FROZEN ModernBERT
path-keyed features (status doc "Path to a working agent"; findings 14-16).

Design decisions carried in, each decided by prior measurement:
- Representation: frozen ModernBERT-base per-line vectors keyed by path
  (finding 14 — gate 1 passed at ceiling). Slot universe: files 0..257,
  dirs 258..299, cwd 300.
- Targets render CLEAN — banner=None, noise=None (finding 15a: raw-target
  nuisance power is 79% of signal). Inputs train in BOTH regimes
  (contaminated and clean, sampled per draw) so the battery can be run twice
  — clean inputs and contaminated inputs — and the delta is an end-to-end
  nuisance-robustness measurement (the A3-fallback test).
- Predictor shape: the tier-2 bake-off winner (finding 16) — full
  re-prediction over the fixed key universe with identity at init, here as a
  residual anchor (present slots: their input vector; absent slots: zero)
  plus a zero-initialized delta head. Absent slots target the FIXED zero
  vector, so existence is readable from predicted norms and the target space
  contains no learnable sink.
- Loss: one mean over all slot instances (finding 16's normalization lesson).
- Training is 1-step teacher-forced; the battery's horizon-3 rollouts measure
  compounding, and a rollout loss is the queued fix if h=3 drifts.

Usage:
  .venv/bin/python -m models.gate2 --mode precompute --data data/v1 --out runs/gate2
  .venv/bin/python -m models.gate2 --mode train --cache runs/gate2 --out runs/gate2/slotpred
  python3 -m evals.dynamics --data data/v1 --adapter gate2 --ckpt runs/gate2/slotpred/ckpt.pt --out runs/gate2/slotpred/battery-clean.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from env import render, vocab  # noqa: E402
from models.data import VERB_INDEX
from models.tier2 import N_SPECIAL_ARGS, arg_to_id, action_to_ids  # noqa: F401

N_FILES = len(vocab.FILE_PATHS)
N_DIRS = len(vocab.DIR_PATHS)
N_SLOTS = N_FILES + N_DIRS + 1
CWD_SLOT = N_SLOTS - 1
FEAT_D = 768


def rec_to_slots(rec):
    """encode_batch record -> (idx LongTensor[K], vecs HalfTensor[K, 768]) over the
    unified slot space. cwd is always present."""
    idx = list(rec["file_idx"]) + [N_FILES + i for i in rec["dir_idx"]] + [CWD_SLOT]
    vecs = torch.cat([rec["file_vecs"], rec["dir_vecs"], rec["cwd"].unsqueeze(0)])
    return torch.tensor(idx, dtype=torch.long), vecs


def assemble_dense(slot_list, device=None):
    """list of (idx, vecs) -> feats [B, 301, 768] float32 (absent = 0)."""
    B = len(slot_list)
    feats = torch.zeros(B, N_SLOTS, FEAT_D)
    for b, (idx, vecs) in enumerate(slot_list):
        feats[b, idx] = vecs.float()
    return feats.to(device) if device else feats


class Gate2Net(nn.Module):
    def __init__(self, d=128, layers=2, nheads=4, presence_norm=1.0):
        super().__init__()
        self.presence_norm = presence_norm  # slot counts as present if ||vec|| exceeds
        self.proj = nn.Linear(FEAT_D, d)
        self.absent = nn.Parameter(torch.zeros(d))
        self.key_emb = nn.Embedding(N_SLOTS + 1, d)
        self.verb_emb = nn.Embedding(len(VERB_INDEX), d)
        self.special_arg = nn.Embedding(N_SPECIAL_ARGS, d)
        self.role = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(
            d, nheads, 4 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.trunk = nn.TransformerEncoder(layer, layers)
        self.delta = nn.Linear(d, FEAT_D)
        nn.init.zeros_(self.delta.weight)  # identity at init: output = anchor
        nn.init.zeros_(self.delta.bias)

    def _arg_vec(self, arg_ids, role):
        is_slot = (arg_ids >= N_SPECIAL_ARGS).unsqueeze(-1)
        slot = self.key_emb(torch.clamp(arg_ids - N_SPECIAL_ARGS, min=0))
        spec = self.special_arg(torch.clamp(arg_ids, max=N_SPECIAL_ARGS - 1))
        return torch.where(is_slot, slot, spec) + self.role.weight[role]

    def forward(self, feats, act):
        """feats [B, 301, 768] (absent = 0), act [B, 3] -> predicted next feats,
        clean target space, [B, 301, 768]."""
        present = (feats.norm(dim=-1, keepdim=True) > self.presence_norm)
        v = torch.where(present, self.proj(feats), self.absent.expand_as(self.proj(feats)))
        toks = v + self.key_emb.weight[:N_SLOTS]
        a = (self.key_emb.weight[N_SLOTS] + self.verb_emb(act[:, 0])
             + self._arg_vec(act[:, 1], 0) + self._arg_vec(act[:, 2], 1))
        h = self.trunk(torch.cat([toks, a.unsqueeze(1)], dim=1))[:, :N_SLOTS]
        anchor = feats * present  # absent slots anchor at the fixed zero target
        return anchor + self.delta(h)


class Gate2PretrainedNet(nn.Module):
    """Round 4 (A1 gate-2, pretrained-initialized predictor; direction 2026-07-13).

    CORRECTION (2026-07-14, per user): this is NOT a faithful VL-JEPA replication,
    and the earlier "recipe-faithful / VL-JEPA's actual recipe ingredient" wording
    was wrong. Here the predictor is initialized from the last N layers of the SAME
    ModernBERT ENCODER that produces the features — a bidirectional MLM *encoder*
    reused as a predictor. VL-JEPA's actual predictor is a SEPARATE, decoder-only
    *generative* LLM (Llama-3.2-1B layers, ~490M) over the frozen features — a
    different model TYPE (generative decoder vs MLM encoder), IDENTITY (a distinct
    model vs the encoder's own layers), and SCALE (~1/30th). So this arm bounds a
    "recycle the encoder's own layers into a small predictor" variant; VL-JEPA's
    recipe (a generative-LLM predictor over frozen features) is untested here. A
    faithful arm (`Gate2GenLLMNet`) is scoped separately.

    init="random" builds the identical truncated architecture from config with
    random weights: the attribution control finding 19 requires (the same control
    whose absence undermined finding 14's pretraining attribution). Operates
    natively at 768-d. Keeps the project's validated identity-at-init: residual
    anchor + zero-init delta. Slot-identity/verb/arg embeddings init at std 0.02
    so inputs stay near the line-vector distribution the layers were trained on."""

    def __init__(self, model_name="answerdotai/ModernBERT-base", n_layers=3,
                 presence_norm=1.0, init="pretrained"):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if init == "pretrained":
            lm = AutoModel.from_pretrained(model_name)
        else:
            lm = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        lm.layers = lm.layers[-n_layers:]
        d = lm.config.hidden_size
        # inputs_embeds bypasses the token table; stub it so 38.6M dead params
        # don't sit in the optimizer (embeddings.norm/drop still apply).
        lm.embeddings.tok_embeddings = nn.Embedding(1, d)
        self.trunk_lm = lm
        self.presence_norm = presence_norm
        self.absent = nn.Parameter(torch.zeros(d))
        self.key_emb = nn.Embedding(N_SLOTS + 1, d)
        self.verb_emb = nn.Embedding(len(VERB_INDEX), d)
        self.special_arg = nn.Embedding(N_SPECIAL_ARGS, d)
        self.role = nn.Embedding(2, d)
        for e in (self.key_emb, self.verb_emb, self.special_arg, self.role):
            nn.init.normal_(e.weight, std=0.02)
        self.delta = nn.Linear(d, d)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def _arg_vec(self, arg_ids, role):
        is_slot = (arg_ids >= N_SPECIAL_ARGS).unsqueeze(-1)
        slot = self.key_emb(torch.clamp(arg_ids - N_SPECIAL_ARGS, min=0))
        spec = self.special_arg(torch.clamp(arg_ids, max=N_SPECIAL_ARGS - 1))
        return torch.where(is_slot, slot, spec) + self.role.weight[role]

    def forward(self, feats, act):
        present = (feats.norm(dim=-1, keepdim=True) > self.presence_norm)
        v = torch.where(present, feats, self.absent.expand_as(feats))
        toks = v + self.key_emb.weight[:N_SLOTS]
        a = (self.key_emb.weight[N_SLOTS] + self.verb_emb(act[:, 0])
             + self._arg_vec(act[:, 1], 0) + self._arg_vec(act[:, 2], 1))
        x = torch.cat([toks, a.unsqueeze(1)], dim=1)
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        h = self.trunk_lm(inputs_embeds=x, attention_mask=mask).last_hidden_state[:, :N_SLOTS]
        anchor = feats * present
        return anchor + self.delta(h)


def build_net(config):
    if config.get("trunk", "scratch") == "scratch":
        return Gate2Net(**{k: config[k] for k in ("d", "layers", "nheads", "presence_norm")})
    return Gate2PretrainedNet(config["model_name"], config["pretrained_layers"],
                              config["presence_norm"], init=config.get("init", "pretrained"))


# -- precompute ---------------------------------------------------------------------


def precompute(args, device):
    from transformers import AutoModel, AutoTokenizer

    from probes.frozen_probe import encode_batch, load_trajs

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, max_trajs in (("train", args.max_train_trajs), ("val", args.max_val_trajs)):
        trajs = load_trajs(pathlib.Path(args.data) / f"{split}.jsonl", "both", max_trajs)
        cache = {"model": args.model, "trajs": []}
        for ti, tr in enumerate(trajs):
            # rendered twice per state: contaminated (deployed regime) + clean (targets)
            texts_dirty = tr["texts"]
            texts_clean = [render.render_full(st, None, None) for st in tr["states"]]
            recs_dirty, recs_clean = [], []
            for i in range(0, len(texts_dirty), args.batch_size):
                recs_dirty.extend(encode_batch(model, tok, texts_dirty[i:i + args.batch_size], device))
                recs_clean.extend(encode_batch(model, tok, texts_clean[i:i + args.batch_size], device))
            entry = {
                "dirty": [rec_to_slots(r) for r in recs_dirty],
                "clean": [rec_to_slots(r) for r in recs_clean],
                "acts": None,
                "ttypes": None,
            }
            cache["trajs"].append(entry)
            if ti % 50 == 0:
                print(f"  {split}: {ti}/{len(trajs)} trajs", flush=True)
        # actions/ttypes come from the raw jsonl (load_trajs keeps states/texts only)
        from evals.dynamics import load_transitions
        for entry, tr in zip(cache["trajs"], load_transitions(
                pathlib.Path(args.data) / f"{split}.jsonl", max_trajs)):
            entry["acts"] = torch.stack([action_to_ids(a) for a in tr["actions"]])
            entry["ttypes"] = tr["ttypes"]
        torch.save(cache, out / f"cache-{split}.pt")
        print(f"saved {out / f'cache-{split}.pt'} ({len(cache['trajs'])} trajs)", flush=True)


# -- training -----------------------------------------------------------------------


def load_cache_transitions(cache_path):
    cache = torch.load(cache_path, weights_only=True)
    trans = []  # (traj_idx, t) pairs; features assembled at batch time
    for ti, tr in enumerate(cache["trajs"]):
        for t in range(tr["acts"].shape[0]):
            trans.append((ti, t))
    return cache, trans


def batch_from(cache, trans, picks, regime_rng, device):
    ins, acts, tgts, cleans, ttypes = [], [], [], [], []
    for p in picks:
        ti, t = trans[p]
        tr = cache["trajs"][ti]
        regime = "clean" if regime_rng.random() < 0.5 else "dirty"
        ins.append(tr[regime][t])
        tgts.append(tr["clean"][t + 1])
        cleans.append(tr["clean"][t])
        acts.append(tr["acts"][t])
        ttypes.append(tr["ttypes"][t])
    return (assemble_dense(ins, device), torch.stack(acts).long().to(device),
            assemble_dense(tgts, device), assemble_dense(cleans, device), ttypes)


@torch.no_grad()
def copy_ratio_eval(net, cache, trans, device, n=512, seed=1):
    """Per-type mean L2(pred, target) / L2(anchor, target): < 1 beats copy."""
    import random as _random
    rng = _random.Random(f"gate2-eval:{seed}")
    picks = [rng.randrange(len(trans)) for _ in range(n)]
    feats, acts, tgts, _, ttypes = batch_from(cache, trans, picks,
                                              _random.Random("eval-regime"), device)
    pred = net(feats, acts)
    present = (feats.norm(dim=-1, keepdim=True) > net.presence_norm)
    anchor = feats * present
    e_pred = (pred - tgts).norm(dim=-1).mean(dim=-1)     # [B]
    e_copy = (anchor - tgts).norm(dim=-1).mean(dim=-1)
    out = {}
    for tt in set(ttypes):
        m = torch.tensor([t == tt for t in ttypes])
        out[tt] = {"pred": e_pred[m].mean().item(), "copy": e_copy[m].mean().item()}
        out[tt]["ratio"] = out[tt]["pred"] / max(out[tt]["copy"], 1e-9)
    return out


def train(args, device):
    import random as _random

    torch.manual_seed(args.seed)
    cache_dir = pathlib.Path(args.cache)
    cache, trans = load_cache_transitions(cache_dir / "cache-train.pt")
    n_val = max(1, len(trans) // 20)
    trans_tr, trans_va = trans[:-n_val], trans[-n_val:]
    print(f"transitions: train={len(trans_tr)} monitor={n_val}", flush=True)

    config = {"d": 128, "layers": 2, "nheads": 4, "presence_norm": 1.0,
              "trunk": args.trunk, "pretrained_layers": args.pretrained_layers,
              "model_name": args.model, "init": args.trunk_init}
    net = build_net(config).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    # Identity-at-init guard (finding 9): prediction must equal the anchor.
    feats, acts, _, _, _ = batch_from(cache, trans_tr, list(range(16)),
                                      _random.Random("chk"), device)
    with torch.no_grad():
        present = (feats.norm(dim=-1, keepdim=True) > net.presence_norm)
        assert torch.allclose(net(feats, acts), feats * present, atol=1e-5), \
            "gate2 predictor is not the identity at init"
    print(f"params={n_params / 1e6:.2f}M identity-at-init: OK", flush=True)

    # Embeddings identified by module type (finding 19: name matching silently
    # decayed special_arg/role tables).
    emb_ids = {id(m.weight) for m in net.modules() if isinstance(m, nn.Embedding)}
    decay, no_decay = [], []
    for p in net.parameters():
        (no_decay if p.ndim <= 1 or id(p) in emb_ids else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = _random.Random(f"gate2:{args.seed}")
    regime_rng = _random.Random(f"gate2-regime:{args.seed}")
    log = []
    skipped = 0  # nonfinite-step guard (finding 19)
    for step in range(1, args.steps + 1):
        picks = [g.randrange(len(trans_tr)) for _ in range(args.batch)]
        feats, acts, tgts, cleans, _ = batch_from(cache, trans_tr, picks, regime_rng, device)
        net.train()
        opt.zero_grad(set_to_none=True)
        err = (net(feats, acts) - tgts).norm(dim=-1)  # [B, 301]
        # Changed-slot weighting (round 2): a uniform mean pays ~300 slots of
        # diffuse haze over the 1 slot that semantically changed — measured round 1
        # at VoE 0.48 (chance) and calibration AUC 0.65. Finding 16's per-edit
        # gradient-balance lesson, continuous form. Changed = clean_t vs clean_t+1.
        changed = ((cleans - tgts).norm(dim=-1) > 1.0).float()
        w = 1.0 + args.edit_weight * changed
        loss = (w * err).sum() / w.sum()
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
            ratios = copy_ratio_eval(net, cache, trans_va, device)
            rec = {"step": step, "loss": round(loss.item(), 5), "skipped": skipped,
                   "copy_ratio": {k: round(v["ratio"], 4) for k, v in ratios.items()}}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save({"config": config, "model": cache["model"],
                        "state_dict": net.state_dict(), "train_args": vars(args),
                        "step": step}, out / "ckpt.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out / 'ckpt.pt'}", flush=True)


# -- codebook grounding (round 3) -----------------------------------------------------


def _gather_labeled(cache_path, data_jsonl, regimes=("clean",)):
    from evals.dynamics import load_transitions

    cache = torch.load(cache_path, weights_only=True)
    trajs_gt = load_transitions(data_jsonl, len(cache["trajs"]))
    fv, fl, cv, cl = [], [], [], []
    for tr_c, tr_g in zip(cache["trajs"], trajs_gt):
        for regime in regimes:
            for (idx, vecs), st in zip(tr_c[regime], tr_g["states"]):
                f = st.features()
                for i, v in zip(idx.tolist(), vecs):
                    if i < N_FILES:
                        fv.append(v)
                        fl.append(f["file_class"][i])
                    elif i == CWD_SLOT:
                        cv.append(v)
                        cl.append(f["cwd_index"])
    return (torch.stack(fv).float(), torch.tensor(fl),
            torch.stack(cv).float(), torch.tensor(cl))


def build_codebook(args, device):
    """Linear-decoder symbol grounding (finding 14's actual result: content and cwd
    are 100% LINEARLY decodable from line vectors — nearest-centroid decoding was
    measured at only 52%/81%, since Euclidean distance lets path/context variance
    swamp class variance). Fits protocol-v2-style linear heads on ground-truth-
    labeled training features over BOTH render regimes (privileged at build time,
    like datagen; inference-time decode is nonprivileged). Validated on the held-out
    val cache before save."""
    import torch.nn as _nn

    from probes.probe import PROBE_FIT, fit_head, make_head

    cache_dir = pathlib.Path(args.cache)
    root = pathlib.Path(args.data)
    fv, fl, cv, cl = _gather_labeled(cache_dir / "cache-train.pt",
                                     root / "train.jsonl", ("clean", "dirty"))
    mu_f, sd_f = fv.mean(0, keepdim=True), fv.std(0, keepdim=True).clamp(min=1e-6)
    mu_c, sd_c = cv.mean(0, keepdim=True), cv.std(0, keepdim=True).clamp(min=1e-6)
    torch.manual_seed(args.seed)
    ce = _nn.functional.cross_entropy
    h_cls = fit_head(make_head("linear", FEAT_D, vocab.N_CONTENT),
                     (fv - mu_f) / sd_f, fl, ce, device)
    h_cwd = fit_head(make_head("linear", FEAT_D, len(vocab.CWD_PATHS)),
                     (cv - mu_c) / sd_c, cl, ce, device)
    cb = {"cls_head": {k: v.cpu() for k, v in h_cls.state_dict().items()},
          "cwd_head": {k: v.cpu() for k, v in h_cwd.state_dict().items()},
          "mu_f": mu_f, "sd_f": sd_f, "mu_c": mu_c, "sd_c": sd_c,
          "probe_fit": dict(PROBE_FIT)}
    vv, vl, wv, wl = _gather_labeled(cache_dir / "cache-val.pt",
                                     root / "val.jsonl", ("clean", "dirty"))
    with torch.no_grad():
        acc_cls = (h_cls(((vv - mu_f) / sd_f).to(device)).argmax(-1).cpu() == vl).float().mean().item()
        acc_cwd = (h_cwd(((wv - mu_c) / sd_c).to(device)).argmax(-1).cpu() == wl).float().mean().item()
    cb["val_decode_acc"] = {"file_class": acc_cls, "cwd": acc_cwd}
    print(f"linear decoder val acc (both regimes pooled): "
          f"{json.dumps(cb['val_decode_acc'])}", flush=True)
    torch.save(cb, cache_dir / "codebook.pt")
    print(f"saved {cache_dir / 'codebook.pt'}", flush=True)


class Gate2CodebookAdapter:
    """Round 3: frozen encoder + linear decoder as zero-shot state reader, plus the
    ALREADY-TRAINED tier-2 symbolic dynamics model (finding 16's winner, reused
    unchanged). Latents are decoded symbolic states, so battery numbers are directly
    comparable to the tier-2 and oracle rows. Existence decodes structurally (line
    presence); content and cwd decode through the fitted linear heads."""

    def __init__(self, tier2_ckpt, codebook_path, input_regime="clean", device=None):
        import torch.nn as _nn

        from transformers import AutoModel, AutoTokenizer

        from models.tier2 import Tier2Adapter
        from train.train import pick_device

        self.t2 = Tier2Adapter(tier2_ckpt)
        self.cb = torch.load(codebook_path, weights_only=True)
        self.h_cls = _nn.Linear(FEAT_D, vocab.N_CONTENT)
        self.h_cls.load_state_dict(self.cb["cls_head"])
        self.h_cwd = _nn.Linear(FEAT_D, len(vocab.CWD_PATHS))
        self.h_cwd.load_state_dict(self.cb["cwd_head"])
        self.h_cls.eval()
        self.h_cwd.eval()
        self.device = pick_device(device or "auto")
        model_name = "answerdotai/ModernBERT-base"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.input_regime = input_regime
        self.name = f"gate2-codebook-{input_regime}"

    @torch.no_grad()
    def encode(self, state, ctx=None):
        from env.state import FsState
        from probes.frozen_probe import encode_batch

        if self.input_regime == "both" and ctx is not None:
            text = render.render_full(state, ctx["banner_id"], ctx["noise_seed"],
                                      step=ctx["step"])
        else:
            text = render.render_full(state, None, None)
        rec = encode_batch(self.lm, self.tok, [text], self.device)[0]
        files = {}
        if rec["file_idx"]:
            fv = (rec["file_vecs"].float() - self.cb["mu_f"]) / self.cb["sd_f"]
            ks = self.h_cls(fv).argmax(-1).tolist()
            files = {vocab.FILE_PATHS[i]: k for i, k in zip(rec["file_idx"], ks)}
        dirs = {vocab.DIR_PATHS[i] for i in rec["dir_idx"]}
        cv = (rec["cwd"].float().unsqueeze(0) - self.cb["mu_c"]) / self.cb["sd_c"]
        cwd_i = int(self.h_cwd(cv).argmax(-1))
        return FsState(dirs=dirs, files=files, cwd=vocab.CWD_PATHS[cwd_i])

    def predict(self, state, action):
        return self.t2.predict(state, action)

    def distance(self, a, b):
        return self.t2.distance(a, b)


# -- battery adapter ------------------------------------------------------------------


class Gate2Adapter:
    """Latents are dense [301, 768] slot-feature maps. encode() renders and encodes
    live (exemplars and battery states are novel); input_regime picks whether battery
    inputs carry nuisance ('both') or not ('clean') — run the battery once with each
    and the delta is the end-to-end nuisance-robustness measurement."""

    def __init__(self, ckpt_path, input_regime="clean", device=None):
        from transformers import AutoModel, AutoTokenizer

        from train.train import pick_device

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.net = build_net(ckpt["config"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.device = pick_device(device or "auto")
        self.net.to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(ckpt["model"])
        self.lm = AutoModel.from_pretrained(ckpt["model"]).to(self.device).eval()
        self.input_regime = input_regime
        self.name = f"gate2-{input_regime}"

    def encode(self, state, ctx=None):
        from probes.frozen_probe import encode_batch

        if self.input_regime == "both" and ctx is not None:
            text = render.render_full(state, ctx["banner_id"], ctx["noise_seed"],
                                      step=ctx["step"])
        else:
            text = render.render_full(state, None, None)
        rec = encode_batch(self.lm, self.tok, [text], self.device)[0]
        return assemble_dense([rec_to_slots(rec)])[0]

    def predict(self, latent, action):
        with torch.no_grad():
            return self.net(latent.unsqueeze(0).to(self.device),
                            action_to_ids(action).unsqueeze(0).to(self.device))[0].cpu()

    def distance(self, a, b):
        return (a - b).norm(dim=-1).sum().item()


def main(argv=None):
    from train.train import pick_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["precompute", "train", "codebook"])
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--cache", default="runs/gate2")
    ap.add_argument("--max-train-trajs", type=int, default=400)
    ap.add_argument("--max-val-trajs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)  # precompute LM batching
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--edit-weight", type=float, default=100.0,
                    help="extra loss weight on semantically-changed slots (round 2); "
                         "0 = uniform")
    ap.add_argument("--trunk", default="scratch", choices=["scratch", "pretrained"])
    ap.add_argument("--trunk-init", default="pretrained",
                    choices=["pretrained", "random"],
                    help="pretrained trunk weights, or the same architecture "
                         "random-init (the finding-19 attribution control)")
    ap.add_argument("--pretrained-layers", type=int, default=3,
                    help="trunk=pretrained: how many final encoder layers to reuse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="required for --mode train")
    args = ap.parse_args(argv)
    if args.mode in ("train", "precompute") and not args.out:
        ap.error(f"--mode {args.mode} requires --out")
    device = pick_device(args.device)
    if args.mode == "precompute":
        precompute(args, device)
    elif args.mode == "codebook":
        build_codebook(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
