"""OPTIM chunk (v2-loss-surface, role-aware): the proven r8 Muon-on-addressing optimizer,
GENERALIZED with a third parameter role for the v2 surface — the ZERO-INITIALIZED
fade-in modulation gates (the FiLM output projections that carry the cat/system-variant
margin) — which get a developmentally PHASED schedule and no weight decay, while the
trunk keeps the exact incumbent AdamW+schedule and the delta-rule addressing matrices
keep the exact incumbent Muon.

WHY A DEDICATED ROLE FOR THE FADE-IN GATES (v2-specific, not generic)
---------------------------------------------------------------------
r8_muon was tuned on the v1 surface. The v2 champion arch (r13_syscond_film_content)
adds two ZERO-INITIALIZED output projections — the view-FiLM (film_out) and the
system-identity FiLM (sysfilm_out). By construction they emit exactly zero at init, so
the module IS the champion at step 0 and the affine correction "fades in only where it
reduces loss." That correction is precisely the mechanism aimed at the STALLED cat
margin (displace the corpus-modal /etc/passwd prediction toward the current system's
variant). Under the incumbent, these gates are pooled into one AdamW group and driven
by ONE schedule + full weight decay — a mismatch on two counts:

1. TIMING. A multiplicative gain field that modulates the trunk's content prediction is
   only useful once the trunk PRODUCES a usable corpus-modal prediction. When film_out
   is trained from step 0, its earliest gradients modulate a still-random trunk output —
   the affine channel spends early capacity fitting noise, then has to be re-learned
   once the trunk stabilizes. This mirrors cortical development and the two-stream
   gain-modulation picture in neuroscience: feedforward (driving) pathways mature
   BEFORE the neuromodulatory / top-down GAIN fields that multiplicatively scale them
   (attention/ACh/DA gain control; Salinas & Thier 2000, "Gain modulation: a major
   computational principle"; Ferguson & Cardin 2020). The correction should come ONLINE
   after the base map exists — a phase shift, not a cold co-start.

2. DECAY DIRECTION. Weight decay pulls a zero-init gate BACK toward zero — the identity
   configuration — directly opposing the loss signal that is trying to grow it. Decay
   here is pure friction against activation (the same reason the incumbent zeroes wd on
   the scale-invariant Muon keys). Homeostatic synaptic scaling regulates a gain to a
   useful operating range by the POSTSYNAPTIC effect, never by shrinking the gain toward
   silence (Turrigiano 2008). So: wd = 0 on the fade-in gates.

MECHANISM — three roles, resolved structurally at make() time (no names, no metadata):
  * ADDRESSING (Muon, unchanged): 2D (key_d, in) matrices with in not in {key_d, D} that
    come as >=2 identical-shape siblings — the delta-rule read/write projections. Exact
    incumbent Muon (Newton-Schulz orthogonalized momentum, RMS-matched, wd 0, incumbent
    schedule). Verbatim from r8_muon.
  * FADE-IN GATES (new role): 2D matrices that are EXACTLY all-zero at init — the
    universal signature of a zero-initialized fade-in projection (FiLM out, LayerScale,
    zero-init residual/adapter). Detected by count_nonzero==0 on the freshly built net
    (make() runs before any step). AdamW, wd=0, and a PHASED lambda: frozen for the
    first delay_frac of steps, then a short warmup to a BOOSTED peak (mod_boost x, to
    pay back the lost steps and the exact-zero cold start), then the SAME cosine-to-floor
    tail as the trunk so all roles anneal together.
  * TRUNK (incumbent): everything else. Exact incumbent AdamW(5e-4, wd 5e-4, b2 .95) +
    warmup(4%)-hold(30%)-cosine-to-floor(5%).

STRICT GENERALIZATION / SAFETY
  * On an arch with NEITHER addressing siblings NOR a zero-init 2D gate (baseline
    transformer, etc.) make() returns the EXACT incumbent (single AdamW + one LambdaLR),
    bit-identical to go_warmup_holdcos_floor.
  * On an arch with addressing but no zero-init gate, it returns the EXACT r8_muon
    two-optimizer configuration (bit-identical to the current champion optim).
  * The new role only activates where a zero-init 2D gate exists — the v2 FiLM archs.
  * count_nonzero(p)==0 cannot false-positive on randn-init projections (addressing,
    codebooks), Kaiming-uniform Linear weights, or ones-init LayerNorm weights; it can
    only catch deliberately zero-initialized 2D fade-in weights. 1-D zero biases and
    LayerNorm zero-biases are excluded (ndim==2 gate), so they stay in the trunk group
    with standard handling.
  * NaN-safe (Muon skips non-finite grads; NS normalizes by a clamped norm; AdamW
    unchanged). No RNG -> deterministic given the harness seed. Only consumes gradients;
    never touches eval/causality/loss; a constant prediction still cannot minimize the
    contrastive objective (anti-collapse unaffected). The delayed onset never DISABLES a
    gate at eval — it only shifts WHEN it learns; the gate is still trained (with boost)
    for the majority of steps.

Contract: make(params, steps, **kw) -> (optimizer, scheduler). The harness calls
opt.zero_grad(set_to_none=True), opt.step(), then scheduler.step() once per iteration.
Pure/self-contained; torch only; no file/network/global state.

Refs: Muon (Jordan 2024; Liu 2025, arXiv:2502.16982); FiLM (Perez 2017, arXiv:1709.07871);
Salinas & Thier 2000 (gain modulation); Ferguson & Cardin 2020 (mechanisms of gain
control); Turrigiano 2008 (homeostatic synaptic scaling); Schlag/Irie/Schmidhuber 2021
(fast-weight key geometry, arXiv:2102.11174).
"""
import math
import torch

NAME = "r14_neuromod_gainfield_phased"
DESCRIPTION = (
    "Role-aware v2 optimizer: incumbent AdamW+schedule on the trunk, incumbent Muon "
    "(orthogonalized momentum, RMS-matched, wd 0) on the delta-rule addressing "
    "siblings, and a NEW role for zero-init fade-in FiLM gates (count_nonzero==0 at "
    "init) — AdamW, wd 0, and a phased schedule (frozen for delay_frac, then a boosted "
    "warmup, shared cosine-to-floor tail) so the multiplicative gain field comes online "
    "AFTER the trunk map, not co-trained from a random start. Exact r8_muon where no "
    "zero-init gate exists; exact incumbent where no addressing sibling exists."
)

D = 768  # frozen encoder dim; addressing matrices never touch it on either side


def _ns_orth(g, steps=5, eps=1e-7):
    """Approximate polar factor of g via the quintic Newton-Schulz iteration (Muon's
    coefficients). Semi-orthogonal output, singular values ~1. Float32; NaN-safe."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        s = x @ x.mT
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class _MuonKeys(torch.optim.Optimizer):
    """Muon for the addressing matrices only (verbatim from r8_muon). group['lr'] is
    driven by LambdaLR on the shared incumbent schedule; the update is RMS-matched to
    AdamW via 0.2*sqrt(max(n,m)); wd=0 (keys are scale-invariant)."""

    def __init__(self, params, lr, momentum=0.95, ns_steps=5, rms_match=0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps, rms_match=rms_match))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]
            lr = group["lr"]
            ns = group["ns_steps"]
            rms = group["rms_match"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue  # NaN-safe
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu)
                o = _ns_orth(u, steps=ns)
                scale = rms * math.sqrt(max(p.shape[0], p.shape[1]))
                p.add_(o, alpha=-lr * scale)
        return loss


class _MultiOpt:
    """Composite exposing the surface the harness uses (zero_grad / step / param_groups)
    over an arbitrary list of sub-optimizers. Only used when >1 role is present."""

    def __init__(self, opts):
        self.opts = list(opts)

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {str(i): o.state_dict() for i, o in enumerate(self.opts)}


class _MultiSched:
    """Steps a list of LambdaLRs together (each carries its own role's shape)."""

    def __init__(self, scheds):
        self.scheds = list(scheds)

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return [lr for s in self.scheds for lr in s.get_last_lr()]


def _incumbent_lambda(steps, warmup_frac, hold_frac, floor_ratio):
    """go_warmup_holdcos_floor multiplier: warmup(4%, min 20) -> hold(30%) -> cosine to
    floor_ratio (never 0)."""
    warm = max(20, int(warmup_frac * steps))
    hold = int(hold_frac * steps)
    decay_start = warm + hold
    decay_len = max(1, steps - decay_start)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return floor_ratio + (1.0 - floor_ratio) * cos

    return lr_lambda


def _phased_gate_lambda(steps, warmup_frac, hold_frac, floor_ratio, delay_frac,
                        mwarm_frac, boost):
    """Fade-in-gate multiplier: 0 for the first delay_frac (trunk establishes the base
    map), a short warmup to `boost`x peak (repay lost steps + the exact-zero cold start),
    hold, then the SAME cosine-to-floor tail as the trunk so all roles anneal together."""
    warm = max(20, int(warmup_frac * steps))
    hold = int(hold_frac * steps)
    decay_start = warm + hold
    decay_len = max(1, steps - decay_start)
    delay = max(0, int(delay_frac * steps))
    mwarm = max(10, int(mwarm_frac * steps))
    onset_end = min(delay + mwarm, decay_start)   # keep the ramp before the cosine tail
    ramp_len = max(1, onset_end - delay)

    def lr_lambda(step):
        if step < delay:
            return 0.0
        if step < onset_end:
            return boost * (step - delay + 1) / ramp_len
        if step < decay_start:
            return boost
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return boost * (floor_ratio + (1.0 - floor_ratio) * cos)

    return lr_lambda


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, delay_frac=0.15, mwarm_frac=0.06, mod_boost=2.0):
    params = [p for p in params]

    # -- role 1: delta-rule ADDRESSING siblings (Muon), exactly as r8_muon --
    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    # -- role 2: zero-init 2D FADE-IN GATES (count_nonzero==0 at init) --
    gates = [p for p in params
             if p.ndim == 2 and id(p) not in key_ids
             and torch.count_nonzero(p).item() == 0]
    gate_ids = {id(p) for p in gates}

    # -- role 3: TRUNK (everything else) --
    rest = [p for p in params if id(p) not in key_ids and id(p) not in gate_ids]

    trunk_lambda = _incumbent_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    # Fast path: no special roles -> EXACT incumbent (bit-identical).
    if not keys and not gates:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, trunk_lambda)

    opts, scheds = [], []

    if rest:  # trunk group (always nonempty on a real net; guarded for robustness)
        adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        opts.append(adamw)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(adamw, trunk_lambda))

    if keys:  # exact incumbent Muon role
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps,
                         rms_match=rms_match)
        opts.append(muon)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, trunk_lambda))

    if gates:  # new phased fade-in-gate role: wd=0, delayed boosted schedule
        gate_opt = torch.optim.AdamW(gates, lr=lr, weight_decay=0.0, betas=(0.9, beta2))
        gate_lambda = _phased_gate_lambda(steps, warmup_frac, hold_frac, floor_ratio,
                                          delay_frac, mwarm_frac, mod_boost)
        opts.append(gate_opt)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(gate_opt, gate_lambda))

    return _MultiOpt(opts), _MultiSched(scheds)
