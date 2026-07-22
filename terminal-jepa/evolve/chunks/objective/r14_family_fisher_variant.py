"""objective chunk: LOCAL FISHER-DISCRIMINANT within-family variant contrastive.

THE CAT STALL, RE-DIAGNOSED (R14 lens). cat's margin has not moved through two champion
generations while every other content verb rose. cat has the highest retrieve-by-cmd
baseline (.531): the corpus already returns the correct file for every path whose content
is IDENTICAL across systems. So the entire remaining cat margin lives in SYSTEM-VARIANT
file bodies — same path (/etc/os-release, distro configs), DIFFERENT content per system.
For those, retrieve-by-cmd returns a WRONG-BUT-CLOSE variant (a sibling from the same
family), and the world model can only win by predicting the SPECIFIC variant this system
has. Under the eval's uniform per-dim squared-L2 metric, the true target and its family
siblings share a large "file template" component and differ only along a SMALL, system-
specific RESIDUAL direction; the top-1 decision inside a variant family rides ENTIRELY on
that residual.

WHY THE CHAMPION STRUCTURALLY CANNOT MOVE IT. The champion (free-energy precision +
anti-retrieval ring) weights every embedding dimension by a GLOBAL precision Pi_d =
1/Var(error_d): it DOWN-weights dimensions the model predicts poorly, treating high-error
dims as noise to abstract away. But a variant family's distinguishing residual dims ARE
high-error — system-specific content is exactly the hardest thing to predict — so the
global precision suppresses precisely the dimensions that carry cat's margin. Denoising and
discrimination pull opposite ways here: the champion's ring re-weights which NEGATIVE PAIRS
matter (the confusable siblings) but still measures them in the champion's globally-denoised
geometry, where the family residual is dim. Re-weighting pairs is not enough; the GEOMETRY
itself has to turn toward the residual.

THE NEW MECHANISM — a LOCAL (family-conditional) Fisher discriminant. For each anchor i,
form its confusable VARIANT FAMILY as a band-passed soft neighborhood over TARGET-TARGET
distances (close-but-distinct: near-identical targets excluded as the identical-across-
systems FALSE negatives retrieve-by-cmd already gets right; far targets excluded as easy).
Then compute the family's PER-DIM within-family target variance
    v_i,d = sum_j w_i,j (t_j,d - mu_i,d)^2 ,   mu_i,d = sum_j w_i,j t_j,d
and build a per-anchor DISCRIMINABILITY weight g_i,d = (v_i,d + eps)^rho, mean-1 over dims.
This is Fisher's linear discriminant reduced to its diagonal / the LDA between-alternatives
scatter (Fisher 1936; Duda-Hart-Stork): because the targets are globally standardized
(per-dim var 1), v_i,d < 1 on the template dims the family SHARES and ~1 (or the ratio to
global) on the dims it VARIES along, so g UP-weights the exact residual directions that
separate this variant from its siblings — regardless of whether they are globally hard to
predict. We then run a listwise contrastive in this LOCAL geometry:
    dist_i,j = mean_d g_i,d (pred_i,d - t_j,d)^2 ,   logits = -dist / tau ,  label = i
so the ranking gradient is spent on matching the true target along its family-distinguishing
residual — the eval's own within-family decision variable — instead of re-earning the
template separations retrieve-by-cmd banks for free. Rows with no confusable family (g falls
back to uniform) reduce to a plain per-dim-mean-L2 contrastive, so non-cat verbs are
unaffected. A detached focal weight concentrates gradient on rows not yet ranked #1.

CROSS-DOMAIN ROOTS. Fisher's between/within-class scatter (LDA, 1936) supplies the
discriminant direction; the LOCAL (per-anchor, neighborhood-conditional) estimation of it is
local discriminant analysis / neighbourhood components analysis (Goldberger et al. 2004) and
mirrors the neuroscience of PATTERN SEPARATION in hippocampal CA3/dentate gyrus: overlapping
memories that share most features are orthogonalized along their distinguishing dimensions so
they can be told apart at recall (Yassa & Stark 2011). The champion pools two overlapping
config files; this loss separates them by amplifying the axis on which they disagree.

WHY THE MARGIN CAN RISE. margin grows only where the WM beats BOTH baselines; for cat that
is exactly the variant families where retrieve-by-cmd returns a sibling. A geometry that
concentrates the ranking gradient on the within-family residual optimizes that residual
directly, whereas the global-precision champion averages it away. A light champion-style
GLOBAL listwise term + MSE anchor are retained at reduced weight so template placement and
the other verbs do not regress.

Contract / safety:
  * Pure function of (pred, tgt); family weights, mu, v, g, focal, global precision all
    DETACHED; no state, no in-place edits of inputs. Ops: a handful of [n,n]/[n,d] matmuls
    (same order as the champion). n < 2 -> MSE anchor only.
  * NaN-safe: eps floors in precision, family kernel scale, ring mass, within-family variance,
    per-row g normalization; dist/tt clamp_min(0); g banded to [WMIN,WMAX] then re-normalized;
    empty-family rows -> uniform g (well-defined). log_softmax is numerically stable.
  * Anti-collapse: for a CONSTANT prediction pred_i = c, the local logit row
    logit_i,j ∝ 2 g_i·(c ⊙ t_j) - g_i·(t_j ⊙ t_j) varies with j only through t_j and gives
    no reason for the diagonal j=i to be the maximum -> the row-softmax cannot pin mass on the
    true index, so the NLL stays away from its minimum; the GLOBAL listwise term (identical
    logit rows across i, champion argument) likewise cannot; and MSE(const, varying tgt) is
    strictly positive. Collapse cannot minimize the loss.
"""

import torch
import torch.nn.functional as F

NAME = "family_fisher_variant_contrastive"
DESCRIPTION = (
    "Local Fisher-discriminant contrastive for cat: per anchor, form its confusable VARIANT "
    "FAMILY as a band-passed soft neighborhood over target-target distances (near-identical "
    "excluded as false negatives, far excluded as easy), compute the per-dim WITHIN-FAMILY "
    "target variance, and UP-weight the dimensions the family varies along — the system-specific "
    "residual that separates a file's variant from its siblings and that the champion's GLOBAL "
    "free-energy precision structurally suppresses (those dims are globally high-error). A "
    "listwise L2 contrastive is then run in this per-anchor local geometry (LDA / hippocampal "
    "pattern separation), with a light champion global-precision listwise term + MSE anchor "
    "retained at reduced weight for template placement and anti-collapse."
)

# ---- global (champion) geometry constants ----
_TEMP = 0.25        # softmax temperature on mean-1-normalized per-dim-mean sqL2
_GAMMA = 1.0        # focal focus on not-yet-#1 rows
_ANCHOR = 0.05      # small MSE anchor (absolute placement + anti-collapse)
_BETA = 0.5         # global precision temper Pi_d^BETA (used ONLY for the tt/family geometry
                    # and the retained global listwise term)
_EPS = 1e-2         # MSE floor inside the precision
_WMIN, _WMAX = 0.25, 4.0
_GLOBAL = 0.5       # weight on the retained champion-style GLOBAL listwise term (keeps other
                    # verbs + template placement while the local term targets cat)

# ---- local Fisher-discriminant / variant-family constants ----
_LAM_FRAC = 0.5     # family kernel scale = _LAM_FRAC * mean off-diag target-target distance
_DELTA = 0.05       # per-dim sq-dist below which two targets are the SAME answer (identical-
                    # across-systems file -> excluded from the family: a FALSE negative)
_RHO = 0.5          # discriminability exponent g = (within-family var)^RHO (0 -> uniform
                    # champion-like local geometry; 1 -> full within-family variance tilt)
_TEMP_L = 0.25      # temperature for the LOCAL discriminant contrastive
_GEPS = 1e-3        # floor inside the per-dim within-family variance / g normalization


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee.
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    labels = torch.arange(n, device=pred.device)

    # --- Global free-energy precision Pi_d = 1/Var(error_d), batch-estimated, DETACHED. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)                 # [d]
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)                   # tempered precision
        w = w / w.mean().clamp_min(1e-12)                       # mean 1 (match eval scale)
        w = w.clamp(_WMIN, _WMAX)                               # band
        w = w / w.mean().clamp_min(1e-12)                       # re-normalize
        sw = w.sqrt().unsqueeze(0)                              # [1, d]

    # Precision-weighted per-dim-mean squared L2 (champion geometry) for the GLOBAL term
    # and for defining the confusable family on target-target distances.
    pw = pred * sw                                             # [n, d]
    tw = tgt * sw                                              # [n, d]
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)                 # [n, 1]
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)                 # [n, 1]
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())           # [n, n]
    dist2 = dist2.clamp_min(0.0) / float(d)

    # --- Champion-style GLOBAL listwise term (retained at reduced weight). ---
    logits_g = -dist2 / _TEMP
    logp_g = F.log_softmax(logits_g, dim=1)
    nll_g = -logp_g.gather(1, labels[:, None]).squeeze(1)      # [n]
    with torch.no_grad():
        p_true_g = (-nll_g).exp().clamp(0.0, 1.0)
        focal_g = (1.0 - p_true_g).pow(_GAMMA)
    listwise_global = (focal_g * nll_g).mean()

    # --- Confusable VARIANT FAMILY: band-passed soft neighborhood over target-target dist. ---
    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)   # [n, n]
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)   # scalar mean off-diag distance
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        confus = torch.exp(-tt / lam)                          # 1 at identical -> 0 far
        dupmask = 1.0 - torch.exp(-tt / _DELTA)                # ~0 identical -> ~1 distinct
        ring = (confus * dupmask).masked_fill(eye, 0.0)        # [n, n] band-pass, zero diag

        # Soft family membership (rows sum to ~1, or ~0 if no confusable siblings in batch).
        fam_mass = ring.sum(dim=1, keepdim=True)               # [n, 1]
        w_fam = ring / fam_mass.clamp_min(_GEPS)               # [n, n]

        # Per-dim within-family target statistics (unweighted-standardized targets t = tgt).
        t2 = tgt * tgt                                         # [n, d]
        mu = w_fam @ tgt                                       # [n, d] family centroid
        e_t2 = w_fam @ t2                                      # [n, d] family E[t^2]
        v = (e_t2 - mu * mu).clamp_min(0.0)                    # [n, d] within-family variance

        # Discriminability weight: up-weight dims the family VARIES along, mean-1 per row.
        g = (v + _GEPS).pow(_RHO)                              # [n, d]
        g = g / g.mean(dim=1, keepdim=True).clamp_min(1e-12)   # mean 1 over dims
        g = g.clamp(_WMIN, _WMAX)                              # band: no single dim dominates
        g = g / g.mean(dim=1, keepdim=True).clamp_min(1e-12)   # re-normalize mean 1
        # Rows with no family (fam_mass ~ 0) -> v ~ 0 -> g uniform (all ones): local term
        # reduces to a plain per-dim-mean-L2 contrastive there. Well-defined, no NaN.

    # --- LOCAL Fisher-discriminant contrastive: per-anchor g-weighted squared L2. ---
    #   dist_L[i,j] = mean_d g[i,d] (pred[i,d] - tgt[j,d])^2
    #              = ( sum_d g[i,d] pred[i,d]^2  - 2 sum_d g[i,d] pred[i,d] tgt[j,d]
    #                                            +   sum_d g[i,d] tgt[j,d]^2 ) / d
    gp = g * pred                                             # [n, d]  (grad via pred; g detached)
    term1 = (gp * pred).sum(dim=1, keepdim=True)             # [n, 1]  sum_d g pred^2
    term2 = gp @ tgt.t()                                     # [n, n]  sum_d g pred t_j
    term3 = g @ t2.t()                                       # [n, n]  sum_d g t_j^2
    dist_L = (term1 - 2.0 * term2 + term3).clamp_min(0.0) / float(d)   # [n, n]

    logits_l = -dist_L / _TEMP_L
    logp_l = F.log_softmax(logits_l, dim=1)
    nll_l = -logp_l.gather(1, labels[:, None]).squeeze(1)     # [n]
    with torch.no_grad():
        p_true_l = (-nll_l).exp().clamp(0.0, 1.0)
        focal_l = (1.0 - p_true_l).pow(_GAMMA)
    listwise_local = (focal_l * nll_l).mean()

    return listwise_local + _GLOBAL * listwise_global + _ANCHOR * mse_anchor
