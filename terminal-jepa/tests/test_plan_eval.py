"""Guards for the Phase-0 planning probe (realenv/plan_eval.py): candidate-set hygiene,
strict plan@1, random-model calibration at the 1/K floor, lexical-planner sanity, and the
no-goal-in-input contract (the true obs_t must be zeroed out of the WM planner's stream)."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from realenv import seq_worldmodel as M
from realenv import plan_eval as P

D = M.D


def synth_seqs(n_seqs=8, n_steps=8, seed=0, image="img:a"):
    g = torch.Generator().manual_seed(seed)
    seqs = []
    for i in range(n_seqs):
        seqs.append({
            "image": image,
            "cmds": [f"cat /etc/f{i}_{t}" for t in range(n_steps)],
            "z_cmd": torch.randn(n_steps, D, generator=g),
            "z_obs": torch.randn(n_steps, D, generator=g),
        })
    return seqs


def test_pools_and_candidates():
    seqs = synth_seqs()
    pools = P.build_pools(seqs)
    assert ("img:a", "cat") in pools and len(pools[("img:a", "cat")]) == 64
    goal = (0, 3, "cat")
    cands = P.draw_candidates(seqs, pools, goal, k=8, seed=7, dedup_cos=0.99)
    assert cands is not None and len(cands) == 8
    assert cands[0]["true"] and cands[0]["cmd"] == seqs[0]["cmds"][3]
    assert all(not c["true"] for c in cands[1:])
    assert len({c["cmd"] for c in cands}) == 8  # distinct commands
    assert all(c["goal_cos"] < 0.99 for c in cands[1:])


def test_eligibility_absolute_args_only():
    assert P.eligible("cat /etc/passwd", "cat")
    assert P.eligible("ls -la /usr/lib", "ls")
    assert not P.eligible("ls", "ls")            # bare ls: context-dependent
    assert not P.eligible("cat etc/passwd", "cat")
    assert not P.eligible("cd ..", "cd")
    assert P.eligible("cd /usr", "cd")


def test_plan_at_1_strict():
    assert P.plan_at_1(torch.tensor([1.0, 2.0, 3.0]))
    assert not P.plan_at_1(torch.tensor([2.0, 1.0, 3.0]))
    assert not P.plan_at_1(torch.tensor([1.0, 1.0, 3.0]))  # tie -> not a hit


def test_random_model_calibrates_to_floor():
    torch.manual_seed(0)
    seqs = synth_seqs(n_seqs=10, n_steps=8)
    pools = P.build_pools(seqs)
    goals = P.sample_goals(seqs, pools, ("cat",), n=60, seed=1)
    net = M.SeqWorldModel("jepa", 0).eval()

    class Identity:
        @staticmethod
        def to_obs(pred, prev):
            return pred

    k, hits, n = 8, 0, 0
    for gi, goal in enumerate(goals):
        cands = P.draw_candidates(seqs, pools, goal, k=k, seed=gi, dedup_cos=0.99)
        if cands is None:
            continue
        n += 1
        hits += P.plan_at_1(P.wm_distances(net, Identity, seqs, goal, cands, "cpu"))
    rate = hits / n
    assert n >= 40 and abs(rate - 1 / k) < 0.12, f"random-model plan@1 {rate} far from 1/{k}"


def test_lexical_planner_wins_when_goal_echoes_cmd():
    # construct a case where z_cmd(true) == z_goal exactly: cosine ranking must pick it
    seqs = synth_seqs(n_seqs=6, n_steps=6)
    si, t = 0, 3
    seqs[si]["z_obs"][t] = seqs[si]["z_cmd"][t].clone()
    pools = P.build_pools(seqs)
    cands = P.draw_candidates(seqs, pools, (si, t, "cat"), k=6, seed=3, dedup_cos=0.99)
    z_goal = seqs[si]["z_obs"][t]
    gn = z_goal / z_goal.norm()
    dl = torch.tensor([-float((c["z_cmd"] / c["z_cmd"].norm()) @ gn) for c in cands])
    assert P.plan_at_1(dl)


def test_true_obs_never_in_wm_input():
    seqs = synth_seqs(n_seqs=6, n_steps=6)
    pools = P.build_pools(seqs)
    goal = (0, 3, "cat")
    cands = P.draw_candidates(seqs, pools, goal, k=4, seed=5, dedup_cos=0.99)

    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, tok, types, key_pad):
            seen["obs_row"] = tok[:, 2 * 3 + 1].clone()
            return torch.zeros(tok.shape[0], tok.shape[1], D), torch.zeros(tok.shape[0], tok.shape[1], 8)

    class Identity:
        @staticmethod
        def to_obs(pred, prev):
            return pred

    P.wm_distances(Spy(), Identity, seqs, goal, cands, "cpu")
    assert seen["obs_row"].abs().max().item() == 0.0, "true obs_t leaked into the planner input"
