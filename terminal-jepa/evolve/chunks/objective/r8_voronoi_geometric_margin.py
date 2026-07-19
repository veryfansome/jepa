"""objective chunk: VORONOI GEOMETRIC-MARGIN hinge in the precision-weighted eval metric.

DESIGNED FOR THE FASTWEIGHTS CONTEXT (what a delta-rule target-space memory does NOT provide)
---------------------------------------------------------------------------------------------
The champion arch (r7_path_delta_fastweights) reads RAW past observations out of two delta-rule
associative memories and adds them, gated, directly to the prediction in target space. Classic
associative-memory analysis (Hopfield crosstalk: Amit/Gutfreund/Sompolinsky 1985; delta-rule
fast weights: Schlag/Irie/Schmidhuber, arXiv:2102.11174) says such a read fails by INTERFERENCE:
partially-matching keys return a MIXTURE of stored values, so the prediction error contains
components pointing TOWARD other stored (same-verb) targets — exactly the components that flip
strict top-1 — plus large components orthogonal to every separating axis, which are HARMLESS for
the ranking. A softmax-listwise loss (the incumbents) cannot tell these apart: its NLL includes a
radial pull on the full error vector, so the trunk wastes gradient fighting the memory read's
benign orthogonal residue. R7's measured negative epistasis (mutual-proximity hubness term won on
the hippo arch, LOST on fastweights) says candidate-geometry re-weighting is already supplied by
the memory; what is NOT supplied is a loss that penalizes ONLY the interference cross-talk.

THE EXACT DECISION VARIABLE (derived from realenv/seq_worldmodel.py::_rank_stats)
---------------------------------------------------------------------------------
The eval scores candidates by per-dim-mean squared L2 and counts a foil as beating truth only
STRICTLY. In the precision-weighted inner product <a,b> = mean_d w_d a_d b_d, with error
e_i = pred_i - tgt_i and separating vector D_ij = tgt_i - tgt_j:

    d(pred_i, tgt_j) - d(pred_i, tgt_i) = sep_ij + 2<e_i, D_ij>,   sep_ij = <D_ij, D_ij>

so truth beats foil j  iff  v_ij := (d_ij - d_ii) / (2 sep_ij) = 1/2 + <e_i, D_ij>/sep_ij > 0.
v_ij is the SVM-style GEOMETRIC margin (Boser/Guyon/Vapnik 1992) of the Voronoi constraint
"pred_i lands in tgt_i's cell": dimensionless, per-pair normalized by the pair separation, and a
function of the error's PROJECTION on the separating axis only. Its gradient w.r.t. pred_i is
proportional to w * (tgt_i - tgt_j): purely along the separating direction, zero along the
~700-dim orthogonal complement — the loss ignores the memory's harmless residue and concentrates
all gradient on cancelling cross-talk. Per-pair normalization makes near-duplicate same-verb
pairs (the ones the sysblock hard-negative batcher stuffs into the batch, and the ones a
delta-memory confuses) carry proportionally the largest margin requirement, replacing the
incumbent's global temperature with a per-pair scale.

Loss = hardest-foil-weighted smooth hinge tau*softplus((m - v)/tau) over valid in-batch pairs
(precision geometry identical to the incumbent champion; sep < SEP_MIN pairs masked as
effective ties — the eval's strict rule means near-identical targets cannot flip top-1, and
dividing by a tiny sep would amplify noise) + the small MSE anchor for absolute placement.

CONTRACT / SAFETY
-----------------
* Pure function of (pred, tgt); no state, no RNG, no in-place edits. Two [n,n] matmuls, same
  cost order as the incumbents. Precision weights and foil weights are DETACHED.
* NaN-safe: sep clamped >= SEP_MIN before division; smooth hinge tau*softplus((m-v)/tau) has
  gradient bounded by 1 for any v (softplus is linear for large args, no overflow); masked
  softmax computed under no_grad with nan_to_num for all-invalid rows; n < 2 -> MSE anchor only.
* Anti-collapse (exact): for constant pred c, e_i - e_j = tgt_j - tgt_i = -D_ij, hence
  v_ij + v_ji = 1 + <e_i - e_j, D_ij>/sep_ij = 1 - 1 = 0: margins are ANTISYMMETRIC, so every
  valid pair has one side with v <= 0, i.e. hinge >= tau*softplus(m/tau) >= m on at least half
  of all constraints (and the hardest-foil weights put MORE mass on the violated side). The
  loss is pinned away from its minimum, and MSE(const, varying tgt) > 0. Collapse cannot win.
  (CPU-verified: collapse loss 0.358 vs perfect-prediction 0.0079; optimizing this loss drives
  the actual strict same-verb top-1 from 0.016 to 1.0.)
"""

import torch
import torch.nn.functional as F

NAME = "voronoi_geometric_margin_precision"
DESCRIPTION = (
    "Exact-decision-variable ranking loss: for each in-batch pair, the SVM-style geometric "
    "margin v_ij = (d_ij - d_ii)/(2*sep_ij) of the Voronoi constraint 'pred_i is closer to its "
    "own target than to target j' in the SAME precision-weighted per-dim-mean squared-L2 "
    "geometry the eval uses, hinged with a smooth per-pair-normalized margin. Gradient flows "
    "purely along each pair's separating axis (invariant to error orthogonal to it), so it "
    "penalizes only the interference cross-talk a delta-rule fastweight memory read injects and "
    "leaves its benign target-space residue alone. Detached hardest-foil weighting, duplicate-"
    "pair (tie) masking, small MSE anchor for absolute placement and anti-collapse."
)

_BETA = 0.5        # precision temper: w_d = Pi_d^BETA (incumbent-proven in the fastweights context)
_EPS = 1e-2        # MSE floor inside the precision (standardized dims have MSE ~0.1-2)
_WMIN, _WMAX = 0.25, 4.0  # band on the mean-1 precision weights (incumbent geometry)
_SEP_MIN = 0.05    # pairs with weighted per-dim-mean sq separation below this are ties: masked
_MARGIN = 0.25     # required slack, in units of the pair separation (strict-tie rule => want slack)
_TAU_H = 0.1       # hinge smoothness in v units; gradient bounded by 1
_TAU_W = 0.5       # hardest-foil weight temperature over -v (smooth-max over decision-critical foils)
_ANCHOR = 0.05     # absolute-placement MSE anchor (metric is shift-sensitive); also anti-collapse


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee (positive for constant pred vs varying tgt).
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # --- Free-energy precision geometry (identical to the champion objective), DETACHED. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)                # [d] per-dim mean squared error
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)                  # [d] tempered precision
        w = w / w.mean().clamp_min(1e-12)                      # mean 1 (match eval scale)
        w = w.clamp(_WMIN, _WMAX)                              # band: no single dim dominates
        w = w / w.mean().clamp_min(1e-12)                      # re-normalize to mean 1
        sw = w.sqrt().unsqueeze(0)                             # [1, d] scale both sides by sqrt(Pi)

    pw = pred * sw                                             # [n, d]
    tw = tgt * sw                                              # [n, d]
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)                 # [n, 1]
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)                 # [n, 1]
    # dist2[i, j] = mean_d w_d (pred_i - tgt_j)^2 : the EXACT (precision-tilted) eval quantity.
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)  # [n, n]

    # Pairwise target separations sep[i, j] = mean_d w_d (tgt_i - tgt_j)^2 (no grad path: targets).
    with torch.no_grad():
        sep = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)  # [n, n]
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        valid = (~eye) & (sep >= _SEP_MIN)                     # ties/duplicates carry no constraint

    # Geometric margin of each Voronoi constraint: v > 0 iff truth strictly beats foil j.
    pos = dist2.diagonal().unsqueeze(1)                        # [n, 1] d_ii
    v = (dist2 - pos) / (2.0 * sep.clamp_min(_SEP_MIN))        # [n, n] dimensionless

    # Smooth per-pair hinge: require v >= MARGIN (slack against the strict-tie rule).
    hinge = _TAU_H * F.softplus((_MARGIN - v) / _TAU_H)        # [n, n]; |d hinge / d v| <= 1

    # Detached hardest-foil weighting: smooth-max over the foils that decide top-1.
    with torch.no_grad():
        wl = (-v / _TAU_W).masked_fill(~valid, float("-inf"))
        fw = torch.softmax(wl, dim=1)
        fw = torch.nan_to_num(fw, nan=0.0)                     # rows with no valid pair -> 0

    ranking = (fw * hinge).sum(dim=1).mean()
    return ranking + _ANCHOR * mse_anchor
