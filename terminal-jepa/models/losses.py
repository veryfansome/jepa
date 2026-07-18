"""Anti-collapse regularizers and auxiliary losses (terminal-jepa.md §4)."""

import torch


def sigreg(z, n_directions=1024, n_t=17, t_max=4.0, generator=None):
    """Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv 2511.08544): project the
    batch onto random unit directions and score each 1-D projection against N(0,1) with
    the Epps-Pulley characteristic-function statistic. By Cramér-Wold, matching all 1-D
    projections pushes the joint embedding distribution toward the isotropic Gaussian —
    the provably collapse-free target. Directions are resampled every call."""
    n, d = z.shape
    u = torch.randn(d, n_directions, device=z.device, generator=generator)
    u = u / u.norm(dim=0, keepdim=True)
    p = z @ u  # [N, M]

    t = torch.linspace(-t_max, t_max, n_t, device=z.device)
    w = torch.exp(-0.5 * t**2)
    w = w / w.sum()
    tp = t.view(-1, 1, 1) * p.unsqueeze(0)  # [T, N, M]
    c_hat = torch.cos(tp).mean(dim=1)  # [T, M]
    s_hat = torch.sin(tp).mean(dim=1)
    target = torch.exp(-0.5 * t**2).view(-1, 1)
    ep = (w.view(-1, 1) * ((c_hat - target) ** 2 + s_hat**2)).sum(dim=0)
    return ep.mean()


def _centered_cov(z):
    zc = z - z.mean(dim=0)
    return (zc.T @ zc) / (z.shape[0] - 1)


def vicreg_var_cov(z, z_b=None, var_target=1.0, eps=1e-4, var_weight=25.0):
    """Variance hinge + off-diagonal covariance penalty at VICReg's normalization AND
    its 25:1 variance:covariance weighting (arXiv 2105.04906 uses mu=25, nu=1) —
    restoring the normalization without the weighting made contraction the
    regularizer's own optimum (adversarial review, 2026-07-10: at d=1024, N=128 the
    plain covariance term has a rank floor of d/(N-1)-1 ~ 7 for ANY unit-variance code,
    scaling as scale^4, so z_std ~ 0.36 minimized it).

    With z_b given, the covariance penalty is CROSS-FITTED: sum_{i!=j} C^A_ij * C^B_ij
    over two disjoint-trajectory halves. Its expectation is the population penalty, so
    the rank floor and the temporal-persistence tax (pooled consecutive timesteps are
    ~0.99-correlated in this domain, inflating the plain estimator ~4x) both vanish,
    while duplication — a population property — is still detected."""
    z_all = z if z_b is None else torch.cat([z, z_b])
    std = torch.sqrt(z_all.var(dim=0) + eps)
    var_loss = torch.relu(var_target - std).mean()
    d = z.shape[1]
    if z_b is None:
        cov = _centered_cov(z)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    else:
        ca, cb = _centered_cov(z), _centered_cov(z_b)
        prod = ca * cb
        off_diag = prod.sum() - prod.diagonal().sum()
    return var_weight * var_loss + off_diag / d


def sigreg_per_index(z_slots, n_directions=256, n_t=17, t_max=4.0):
    """SIGReg run separately for each slot index: z_slots [N, K, d] -> K independent
    Epps-Pulley tests against N(0,1), sharing one direction sketch. Pooling slots as
    exchangeable samples lets static per-slot mean offsets masquerade as sample variance
    (fidelity audit: 87% of the pooled test's variance target was met by identity
    offsets); testing per index makes any nonzero slot mean itself a penalty. K x
    n_directions univariate tests (16*256 = 4096) exceeds the papers' 1024 sketches."""
    n, k, d = z_slots.shape
    u = torch.randn(d, n_directions, device=z_slots.device)
    u = u / u.norm(dim=0, keepdim=True)
    p = torch.einsum("nkd,dm->knm", z_slots, u)  # [K, N, M]

    t = torch.linspace(-t_max, t_max, n_t, device=z_slots.device)
    w = torch.exp(-0.5 * t**2)
    w = w / w.sum()
    tp = t.view(-1, 1, 1, 1) * p.unsqueeze(0)  # [T, K, N, M]
    c_hat = torch.cos(tp).mean(dim=2)  # [T, K, M]
    s_hat = torch.sin(tp).mean(dim=2)
    target = torch.exp(-0.5 * t**2).view(-1, 1, 1)
    ep = (w.view(-1, 1, 1) * ((c_hat - target) ** 2 + s_hat**2)).sum(dim=0)
    return ep.mean()


def temporal_similarity(z_t, z_t1):
    """PLDM-style smoothness: consecutive latents should be similar."""
    return (1 - torch.nn.functional.cosine_similarity(z_t, z_t1, dim=-1)).mean()
