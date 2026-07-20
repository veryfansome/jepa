"""M2 attack: independently verify 'pred-vs-true sqL2 ~1859 > random-pair ~1478' with the
certified champion checkpoints + cached inner-val data, and quantify whether magnitude
artifacts actually break CROSS-PREDICTION comparability (the stated motivation for cosine)."""
import json, pathlib, sys
sys.path.insert(0, "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa")
import torch
from realenv import seq_worldmodel as M
from evolve import genome as G

CK = pathlib.Path("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/plan/pod/ckpts")
GEN = json.load(open("/private/tmp/claude-501/-Users-fanzhu-PyCharmProjects-jepa/d198d7af-d892-4db5-b49e-72d82a05b137/scratchpad/r9/genomes/r9-arch-chunked-codex.json"))
DATA = "/Users/fanzhu/PyCharmProjects/jepa/terminal-jepa/data/dockerfs-e5"

device = M.pick_device()
train = M.cached_encode(DATA, "train", "x", device)
val = M.cached_encode(DATA, "val", "x", device)
mo, so, mc, sc = M.standardize_stats(train)
M.apply_stats(val, mo, so, mc, sc)
inner = [s for s in val if s["image"] in ("fedora:latest", "mariadb:latest")]
print(f"inner-val seqs: {len(inner)} (of {len(val)})")

build, ap_ = G.load_arch(GEN)
g = torch.Generator().manual_seed(123)
for s in (0, 1, 2):
    ck = torch.load(CK / f"r9-arch-chunked-codex.s{s}.pt", weights_only=False)
    net = build(**ap_); net.load_state_dict(ck["state_dict"]); net = net.to(device).eval()
    flat = M.flatten_predictions(net, inner, device)
    p, t = flat["pred"], flat["true"]
    N = p.shape[0]
    perm = torch.randperm(N, generator=g)
    matched = ((p - t) ** 2).sum(-1)
    p_rand_t = ((p - t[perm]) ** 2).sum(-1)
    perm2 = torch.randperm(N, generator=g)
    t_rand_t = ((t - t[perm2]) ** 2).sum(-1)
    cos = torch.nn.functional.cosine_similarity(p, t, dim=-1)
    print(f"seed {s}: N={N}")
    print(f"  matched pred-true sqL2      mean {matched.mean():.0f}  median {matched.median():.0f}")
    print(f"  pred vs RANDOM true sqL2    mean {p_rand_t.mean():.0f}")
    print(f"  true vs true (random pair)  mean {t_rand_t.mean():.0f}")
    print(f"  ||pred||^2 mean {(p**2).sum(-1).mean():.0f}   ||true||^2 mean {(t**2).sum(-1).mean():.0f}")
    print(f"  matched cosine mean {cos.mean():.3f}")
    # per-dim error concentration
    mse_d = ((p - t) ** 2).mean(0)
    top = torch.topk(mse_d, 10).values
    print(f"  per-dim MSE: mean {mse_d.mean():.2f}  top10 {top.round().tolist()}  frac in top16 dims {torch.topk(mse_d,16).values.sum()/mse_d.sum():.3f}")
    if s == 0:
        # cross-prediction comparability: for same-sequence steps, decompose sqL2 to a common
        # target: d(p_i,gz) = ||p_i||^2 - 2 p_i.gz + ||gz||^2. Spread of ||p_i||^2 across
        # candidate predictions vs spread of the cross term decides whether norm artifacts
        # dominate. Approximate candidate sets by all preds within one sequence vs that seq's
        # last true obs as 'goal'.
        import collections
        by_seq = collections.defaultdict(list)
        k = 0
        for sq in inner:
            n = sq["z_obs"].shape[0]
            by_seq[id(sq)] = list(range(k, k + n)); k += n
        norm_spread, cross_spread = [], []
        for ids in by_seq.values():
            if len(ids) < 4: continue
            P = p[torch.tensor(ids)]
            gz = t[ids[-1]]
            nrm = (P ** 2).sum(-1)
            crs = 2 * (P @ gz)
            norm_spread.append(nrm.std().item()); cross_spread.append(crs.std().item())
        ns = torch.tensor(norm_spread); cs = torch.tensor(cross_spread)
        print(f"  within-seq spread: std(||p||^2) mean {ns.mean():.0f} vs std(2 p.g) mean {cs.mean():.0f}  ratio {ns.mean()/cs.mean():.2f}")
