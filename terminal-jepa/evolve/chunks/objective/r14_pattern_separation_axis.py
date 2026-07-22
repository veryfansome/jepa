"""objective chunk: DENTATE-GYRUS PATTERN-SEPARATION AXIS contrastive — champion
free-energy precision-weighted focal listwise L2 with ring-reweighted negatives, whose
anti-retrieval push is a DIRECTIONAL, decision-variable-faithful pattern-separation term
that penalizes a prediction for landing ON the trajectory's recalled earlier observation.

THE GREP / WITHIN-TRAJECTORY LENS (measured, v2 inner). On grep the binding baseline is
WITHIN-TRAJECTORY retrieval (.513, harness.py §"within-traj"): for a command step it recalls
the observation of the nearest EARLIER command (by command embedding) in the SAME trajectory.
On grep this is strong because a grep step's own earlier cat/head of the file embeds NEAR the
grep answer (matching lines are a subset of that earlier content). The WM only earns margin
where the TRUE observation DIFFERS from that recalled earlier one but the WM still predicts the
true one. The residual failure is a PATTERN-SEPARATION failure: grep-output and the earlier
cat-output of the same file are two SIMILAR patterns the model must keep apart.

CROSS-DOMAIN MECHANISM (hippocampal memory, dentate gyrus). The hippocampal DG performs
PATTERN SEPARATION: it orthogonalizes similar inputs so a cue that would recall a stored
pattern instead maps to a distinct representation, the complement of CA3 PATTERN COMPLETION /
recall (Marr 1971; McNaughton & Morris 1987; Leutgeb et al., Science 2007 — DG remaps sharply
to small input changes while CA3 pattern-completes; Bakker et al., Science 2008 — DG/CA3
mismatch detection). The failure the margin exposes on grep is CA3-style completion (reproduce
the recalled earlier observation) where DG-style separation is required (resolve grep-vs-cat).
We translate the DG computation directly: for each example find its RECALL-CONFUSABLE target
set (close-but-distinct targets — exactly what within-trajectory recall would return), form the
DISCRIMINATIVE AXIS that separates the truth from that recalled pattern, and require the
prediction to resolve the ambiguity ALONG THAT AXIS, past the perpendicular-bisector decision
boundary toward truth.

WHY DIRECTIONAL, NOT ISOTROPIC. The eval is squared-L2 top-1: the WM out-ranks a confusable
target t_j iff ||pred - t_i||^2 < ||pred - t_j||^2, i.e. iff pred lies on the truth side of the
PERPENDICULAR BISECTOR of t_i and t_j — a purely DIRECTIONAL condition on the projection of
pred onto u = (t_i - t_j)/||t_i - t_j||. The champion's repulsion hinge acts on full isotropic
squared distances (spends effort on dimensions irrelevant to the t_i-vs-recall decision). This
term instead measures the eval's ACTUAL decision variable — the normalized position of the
prediction along the single axis on which truth and the recalled observation differ:
  c_i    = sum_j q_ij t_j                (RECALL CENTROID: the confusable set = what recall returns)
  delta  = t_i - c_i,   dn = ||delta||,   u = delta / dn      (the discriminative axis, DETACHED)
  s_i    = (pred . u - c_i . u) / dn                          (0 at the recalled obs, 1 at truth)
  L_sep  = gate_i * softplus(((1 - RHO) - s_i) / TAU)         (be past the boundary toward truth)
s_i is SCALE-FREE (dn cancels), so the margin (1-RHO) means "at least a fraction 1-RHO of the
way from the recalled observation to the truth" — one interpretable knob, decision-boundary at
s=0.5. A prediction that COPIES the recalled earlier observation (pred=c_i) gives s_i=0 and the
maximal penalty softplus((1-RHO)/TAU); a prediction at truth gives s_i=1 and ~0 penalty. This
is precisely a penalty on predictions that merely reproduce an earlier observation of the same
trajectory when the true observation differs.

The confusable set is found WITHOUT metadata (the loss(pred,tgt) contract has none) via the
champion's band-pass CONFUSABILITY RING on target-target distances (Robinson et al.
arXiv:2010.04592 hard negatives; Chuang et al. arXiv:2007.00224 debiased — near-identical
targets, the same config file catted twice, are down-weighted to ~0 as FALSE negatives; only
close-but-DISTINCT targets, the recall-confusable ones, get weight). The sysblock batcher packs
a trajectory's cmd positions into the flattened batch, so a grep step's own earlier observations
sit in the target rows and populate that ring — exactly the within-trajectory candidate set.

The retrieval backbone is the champion free-energy precision-weighted focal listwise L2
contrastive with the SAME ring importance-weighting of in-batch negatives (which lifts the other
verbs); this term REPLACES the champion's isotropic repulsion hinge with the directional
separation term above, sharpening the anti-recall push onto the grep/within-trajectory residual.

Contract / safety:
  * Pure function of (pred, tgt). Ring, q, gate, precision, c_i, u, dn all DETACHED; the only
    grad path in L_sep is pred . u (linear in pred). No state, no in-place edits of inputs.
  * NaN-safe: eps floors in precision, lambda, ring mass, dn; dist2/tt clamp_min(0); n<2 -> MSE.
  * Anti-collapse: a constant pred cannot minimize the listwise NLL (log a is target-only and
    the diagonal cannot be favored) NOR L_sep (u_i, c_i.u, dn are target-only and point in many
    directions, so no single pred vector puts every s_i past the boundary -> strictly positive)
    NOR the MSE anchor (constant vs varying tgt > 0). Collapse cannot minimize the loss.
"""

import torch
import torch.nn.functional as F

NAME = "pattern_separation_axis"
DESCRIPTION = (
    "Champion free-energy precision-weighted focal listwise L2 contrastive with ring-reweighted "
    "in-batch negatives, whose anti-retrieval push is a DIRECTIONAL dentate-gyrus pattern-"
    "separation term: for each example it builds the discriminative axis u between the truth and "
    "its recall-confusable centroid (the close-but-distinct targets a within-trajectory recall "
    "would return) and penalizes, via a scale-free softplus margin on the prediction's normalized "
    "position along u, any prediction that lands on the recalled earlier observation instead of "
    "resolving past the squared-L2 perpendicular-bisector decision boundary toward truth."
)

# ---- champion constants (unchanged) ----
_TEMP = 0.25       # softmax temperature on mean-1-normalized per-dim-mean sqL2
_GAMMA = 1.0       # focal focus on not-yet-#1 rows
_ANCHOR = 0.05     # small MSE anchor (anti-collapse + absolute placement)
_BETA = 0.5        # precision temper Pi_d^BETA
_EPS = 1e-2        # MSE floor inside the precision
_WMIN, _WMAX = 0.25, 4.0

# ---- ring (recall-confusable set) constants ----
_KAPPA = 4.0       # peak extra negative mass on a maximally confusable pair
_DELTA = 0.05      # per-dim sq-dist below which two targets are the SAME answer (false neg -> ring~0)
_LAM_FRAC = 0.5    # confusability kernel scale = _LAM_FRAC * mean off-diag target-target dist

# ---- pattern-separation (directional anti-recall) constants ----
_LAMBDA_SEP = 0.1  # weight of the directional separation term
_RHO = 0.35        # require s_i >= 1-RHO = 0.65: at least 65% of the way from recall to truth
                   # (decision boundary is s=0.5; the extra 0.15 is the training margin)
_TAU_SEP = 0.25    # softplus sharpness on the scale-free position s_i
_GEPS = 1e-3       # ring-mass floor inside the row gate
_DNEPS = 1e-4      # floor on the discriminative-axis norm (guards near-duplicate centroids)


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee.
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # --- Free-energy precision: Pi_d = 1 / Var(error_d), batch-estimated, DETACHED. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)                # [d]
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)                  # tempered precision
        w = w / w.mean().clamp_min(1e-12)                      # mean 1 (match eval scale)
        w = w.clamp(_WMIN, _WMAX)                              # band
        w = w / w.mean().clamp_min(1e-12)                      # re-normalize
        sw = w.sqrt().unsqueeze(0)                             # [1, d]

    # Precision-weighted per-dim-mean squared L2 (champion geometry).
    pw = pred * sw                                             # [n, d]
    tw = tgt * sw                                              # [n, d]
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)                 # [n, 1]
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)                 # [n, 1]
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())            # [n, n]
    dist2 = dist2.clamp_min(0.0) / float(d)

    # --- Confusability ring on TARGET-TARGET distances (all DETACHED). ---
    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)  # [n, n]
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        confus = torch.exp(-tt / lam)                          # 1 at identical -> 0 far
        dupmask = 1.0 - torch.exp(-tt / _DELTA)                # ~0 identical -> ~1 distinct
        ring = (confus * dupmask).masked_fill(eye, 0.0)        # [n, n] band-pass, zero diag

        # Importance weights for the listwise negatives (off-diag mean 1 per row).
        a_raw = 1.0 + _KAPPA * ring
        row_mean = a_raw.masked_fill(eye, 0.0).sum(dim=1, keepdim=True) / (n - 1)
        a = (a_raw / row_mean.clamp_min(1e-6)).masked_fill(eye, 1.0)
        log_a = a.clamp_min(1e-6).log()                        # [n, n]

        # Recall-confusable distribution + smooth row gate for the separation term.
        mass = ring.sum(dim=1)                                 # [n]
        q = ring / mass.clamp_min(1e-6).unsqueeze(1)           # [n, n], rows ~1 (or ~0)
        gate = mass / (mass + _GEPS)                           # [n] in [0,1): ~0 if no confusables

        # RECALL CENTROID and DISCRIMINATIVE AXIS (target-only, detached).
        c = q @ tgt                                            # [n, d] the recalled-observation centroid
        delta = tgt - c                                        # [n, d] axis: recalled obs -> truth
        dn = delta.norm(dim=1).clamp_min(_DNEPS)               # [n]
        u = delta / dn.unsqueeze(1)                            # [n, d] unit discriminative axis
        c_proj = (c * u).sum(dim=1)                            # [n] projection of the recalled obs

    # --- Champion listwise term with importance-weighted negatives. ---
    logits = -dist2 / _TEMP + log_a
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    # --- Directional pattern-separation: resolve truth-vs-recall along the discriminative axis. ---
    # s_i in [~0 at recalled obs, ~1 at truth]; the ONLY grad path is (pred . u), linear in pred.
    p_proj = (pred * u).sum(dim=1)                             # [n], carries grad
    s = (p_proj - c_proj) / dn                                 # [n] scale-free position along axis
    sep = (gate * F.softplus(((1.0 - _RHO) - s) / _TAU_SEP)).mean()

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_SEP * sep
