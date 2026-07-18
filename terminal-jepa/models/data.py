"""Tokenizer and dataset for Phase 1 (terminal-jepa.md §4).

Tokenization is compositional (path components, single digits), matching the plan's
requirement that both observation and action encoders handle unseen combinations of
known symbols. The vocabulary is built deterministically from env.vocab, closed over
everything the full-obs renderer and action grammar can emit; UNK exists for safety.

Observations are rendered from stored states at load time, so the distractor regime
(clean / banner / dynamic / both) is chosen here, not at datagen time.
"""

import json
import re

import torch

from env import render, vocab
from env.state import FsState

_TOKEN_RE = re.compile(r"[A-Za-z_.\-]+|\d|\S")

PAD, UNK, CLS = 0, 1, 2


def tokenize(text):
    return _TOKEN_RE.findall(text)


def build_token_vocab():
    corpus = list(vocab.BANNERS)
    corpus += ["cwd: / tree:", "[ts 0123456789] [pid 0123456789]"]
    corpus += [vocab.path_to_str(p) + "/" for p in vocab.DIR_PATHS]
    corpus += [f"{vocab.path_to_str(p)} [c0]" for p in vocab.FILE_PATHS]
    corpus += vocab.CONTENT_TOKENS + [f"c{k}" for k in range(vocab.N_CONTENT)]
    corpus += ["cd ls cat mkdir touch rm cp mv write . user@sandbox:$"]
    toks = sorted({t for s in corpus for t in tokenize(s)})
    itos = ["<pad>", "<unk>", "<cls>"] + toks
    return {t: i for i, t in enumerate(itos)}


TOKEN_VOCAB = build_token_vocab()

# Action label spaces for the IDM ablation head (classification over enumerations).
ARG_VOCAB = (
    ["", ".", "/"]
    + [vocab.path_to_str(p) for p in vocab.DIR_PATHS]
    + [vocab.path_to_str(p) for p in vocab.FILE_PATHS]
    + [f"c{k}" for k in range(vocab.N_CONTENT)]
)
ARG_INDEX = {a: i for i, a in enumerate(ARG_VOCAB)}
VERB_INDEX = {v: i for i, v in enumerate(
    ["cd", "ls", "cat", "mkdir", "touch", "rm", "cp", "mv", "write"]
)}


def encode_text(text, max_len):
    """Returns (ids, truncated). Truncation drops trailing tree entries while keeping
    the banner/noise header, so it must never pass silently — callers count it."""
    ids = [CLS] + [TOKEN_VOCAB.get(t, UNK) for t in tokenize(text)]
    truncated = len(ids) > max_len
    ids = ids[:max_len]
    return ids + [PAD] * (max_len - len(ids)), truncated


def action_to_text(action):
    return render.action_to_cmd(tuple(action))


REGIMES = {
    "clean": (False, False),
    "banner": (True, False),
    "dynamic": (False, True),
    "both": (True, True),
}


class TrajectoryData:
    """Loads a datagen split, reconstructs state sequences, renders + tokenizes
    observations for one regime, and serves random training windows.

    Tokenized data is held as int16 tensors (~64MB for the full v0 split) rather than
    Python int lists (~1GB): macOS jetsam killed training runs under memory pressure.
    Training (keep_states=False) also drops the FsState objects after rendering."""

    def __init__(self, jsonl_path, regime="both", max_trajs=None,
                 obs_len=640, act_len=24, keep_states=True):
        self.obs_len, self.act_len = obs_len, act_len
        use_banner, use_noise = REGIMES[regime]
        self.trajs = []
        self.truncated_obs = 0
        self.truncated_act = 0
        with open(jsonl_path) as fh:
            for line in fh:
                if max_trajs is not None and len(self.trajs) >= max_trajs:
                    break
                t = json.loads(line)
                states = [FsState.from_json(t["layout"])]
                for s in t["steps"]:
                    states.append(FsState.from_json(s["state_after"]))
                banner = t["banner_id"] if use_banner else None
                noise = t["noise_seed"] if use_noise else None
                obs_ids = []
                for i, st in enumerate(states):
                    ids, trunc = encode_text(
                        render.render_full(st, banner, noise, step=i), obs_len
                    )
                    self.truncated_obs += trunc
                    obs_ids.append(ids)
                acts = [tuple(s["action"]) for s in t["steps"]]
                act_ids = []
                for a in acts:
                    ids, trunc = encode_text(action_to_text(a), act_len)
                    self.truncated_act += trunc
                    act_ids.append(ids)
                act_labels = [
                    (
                        VERB_INDEX[a[0]],
                        ARG_INDEX.get(a[1], 0),
                        ARG_INDEX.get(a[2], 0),
                    )
                    for a in acts
                ]
                self.trajs.append({
                    "states": states if keep_states else None,
                    "obs": torch.tensor(obs_ids, dtype=torch.int16),
                    "acts": torch.tensor(act_ids, dtype=torch.int16),
                    "act_labels": torch.tensor(act_labels, dtype=torch.long),
                    "banner_id": t["banner_id"],
                    "layout_id": t["layout_id"],
                })
        if self.truncated_obs:
            print(f"WARNING: {self.truncated_obs} observations truncated at "
                  f"{obs_len} tokens — state entries silently dropped", flush=True)
        if self.truncated_act:
            print(f"WARNING: {self.truncated_act} actions truncated at "
                  f"{act_len} tokens", flush=True)

    def sample_windows(self, batch_size, horizon, rng):
        """Returns dict of tensors: obs [B, horizon+1, L], acts [B, horizon, La],
        act_labels [B, horizon, 3]."""
        obs, acts, labels = [], [], []
        for _ in range(batch_size):
            tr = self.trajs[rng.randrange(len(self.trajs))]
            t = rng.randrange(tr["acts"].shape[0] - horizon + 1)
            obs.append(tr["obs"][t : t + horizon + 1])
            acts.append(tr["acts"][t : t + horizon])
            labels.append(tr["act_labels"][t : t + horizon])
        obs_t = torch.stack(obs).long()
        acts_t = torch.stack(acts).long()
        # Dynamic padding: observations average ~250 of 640 tokens; trimming to the
        # batch max cuts attention cost ~2.5x.
        obs_max = int((obs_t != PAD).sum(-1).max())
        acts_max = int((acts_t != PAD).sum(-1).max())
        return {
            "obs": obs_t[..., :obs_max],
            "acts": acts_t[..., :acts_max],
            "act_labels": torch.stack(labels),
        }

    def probe_examples(self):
        """(obs_ids_tensor, features, banner_id) per state, for the probing harness.
        Requires keep_states=True."""
        out = []
        for tr in self.trajs:
            for i, st in enumerate(tr["states"]):
                out.append((tr["obs"][i], st.features(), tr["banner_id"]))
        return out
