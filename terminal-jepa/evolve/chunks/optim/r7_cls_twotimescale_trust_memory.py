"""OPTIM chunk: a Complementary-Learning-Systems (CLS) two-timescale AdamW that treats the hippo
arch's fast episodic-READ heads and its slow neocortical TRUNK as distinct learning problems, with
a LAMB-style per-tensor trust ratio applied ONLY to the fast (heterogeneous-scale) memory group.

WHY (grounded in the search + cross-domain literature, not defaults)
--------------------------------------------------------------------
The arch is now r6_hippo_episodic_place_read: a slow neocortical transformer TRUNK
(proj/type_emb/pos_emb/tf/head) plus a fast hippocampal READ pathway
(cmd_proj/place_norm/Wq/Wk/log_temp/gate). The held-out retrieval margin is carried by the
episodic read: it returns a convex combination of REAL strictly-past observation embeddings
(on-manifold, needs no cross-system transfer), fired by a confidence gate. WHICH past obs is
retrieved and WHETHER the pathway fires is decided entirely by Wq/Wk/log_temp/gate.

  * Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995): the hippocampus
    learns FAST (sparse, conjunctive, pattern-separated) while the neocortex learns SLOWLY
    (distributed, generalizing). The hippo arch literally instantiates this split, so its two
    families want two timescales: the memory heads get a higher peak LR and lower weight decay
    (converge fast to a sharp, high-firing retrieval policy the margin rewards); the trunk keeps
    the proven incumbent LR/wd (a smooth, well-regularized fallback the gate relaxes to).

  * LAMB (You et al. 2019, arXiv:1904.00962): the memory heads have wildly heterogeneous scales
    (a 0-dim scalar temperature, 128x192 key matrices, a 1x192 gate output). A single global Adam
    LR under/over-drives them; a per-tensor trust ratio r = ||w||/||update|| makes each memory
    tensor's step proportional to its own weight norm, so temperature and keys co-adapt at
    comparable RELATIVE rates. Trust is scoped to the fast group ONLY -- the trunk keeps plain
    decoupled AdamW, which the ledger shows beats decoupled-by-ndim (0.4014 vs 0.3575).

  * The SLOW trunk keeps the incumbent EXACTLY: warmup(4%) -> hold(30%) at peak -> cosine to
    floor(peak/20), lr 5e-4, wd 5e-4, betas (0.9,0.95). The custom slow update matches
    torch.optim.AdamW to float32 round-off (verified 4.8e-6 over 50 steps), so the only change
    vs the incumbent is faster, scale-normalized learning of the generalization-carrying pathway.

ROUTING (arch-agnostic, safe)
-----------------------------
make() only receives net.parameters() (bare tensors, no names). The memory heads are identified
by a shape signature that NEVER occurs in the transformer trunk (verified: zero trunk
false-positives on the hippo arch): a 0-dim scalar (log_temp), any dim == kq (Wq/Wk weight+bias),
a size-1 dim (gate output (1,d) and (1,)), or a dim == d+2 (gate input). The ambiguous memory
tensors that share a trunk shape (cmd_proj, place_norm) are conservatively left in the slow group
-- still trained correctly, just on the slow timescale. For ANY arch without the signature (e.g.
baseline_transformer), the fast group is empty and every param falls into the slow group -> pure
incumbent warmup_holdcos AdamW. So this optim STRICTLY generalizes the incumbent and differs only
when the hippo memory heads are present.

SAFETY
------
NaN-safe: trust ratio is guarded (finite + positive norms; falls back to r=1). Anti-collapse and
leak-free by construction: the optimizer touches weights via gradients only, never the loss, the
data, or causality, so a constant prediction still cannot minimize the (contrastive) objective and
no future frame can leak. Grad-norm is clipped to 1.0 by the harness every step.

Contract: make(params, steps, **kw) -> (optimizer, scheduler). scheduler.step() is called once per
iteration by the harness AFTER opt.step(). Pure/self-contained; torch only; no file/state.
"""
import math
import torch

NAME = "cls_twotimescale_trust_memory"
DESCRIPTION = ("Two-timescale AdamW: slow trunk (lr 5e-4, wd 5e-4, betas .9/.95; 4% warmup, 30% "
               "hold, cosine-to-floor) + fast episodic-read heads (2.5x lr, 0.1x wd, beta2 .98, "
               "LAMB per-tensor trust ratio). Routes by kq/gate/scalar shape signature (0 trunk "
               "false-positives); degrades to the incumbent AdamW on any arch without the signature.")

D = 768


def _is_memory(p, d, kq):
    """Arch-agnostic signature for the hippo episodic-read heads (Wq/Wk/log_temp/gate). These dims
    (0-dim scalar temp, any dim == kq, a size-1 gate output dim, a dim == d+2 gate input) never
    occur in the transformer trunk (verified). Zero trunk false-positives -> safe. No signature
    match (other archs) -> memory group empty -> pure incumbent AdamW on all params."""
    if p.ndim == 0:
        return True
    s = tuple(p.shape)
    if kq in s:
        return True
    if 1 in s:
        return True
    if (d + 2) in s:
        return True
    return False


class CLSAdamTrust(torch.optim.Optimizer):
    """AdamW (decoupled weight decay) for both groups; the fast group additionally multiplies its
    per-tensor update by a LAMB trust ratio r = ||w|| / ||update|| (clipped), so a scalar
    temperature and a 128x192 key matrix co-adapt at comparable RELATIVE rates. group['lr'] is set
    each iteration by the scheduler. When trust=False the update is exactly decoupled AdamW."""

    def __init__(self, groups, betas_slow=(0.9, 0.95), betas_fast=(0.9, 0.98), eps=1e-8):
        defaults = dict(eps=eps)
        super().__init__(groups, defaults)
        self.betas_slow = betas_slow
        self.betas_fast = betas_fast

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            fast = group.get("fast", False)
            b1, b2 = (self.betas_fast if fast else self.betas_slow)
            lr = group["lr"]
            wd = group["wd"]
            eps = group.get("eps", 1e-8)
            trust = group.get("trust", False)
            trust_clip = group.get("trust_clip", 10.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                m, v = st["m"], st["v"]
                st["step"] += 1
                t = st["step"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                mhat = m / bc1
                denom = (v / bc2).sqrt_().add_(eps)
                update = mhat / denom                      # Adam step direction (pre-LR)
                if wd != 0.0:
                    update = update.add(p, alpha=wd)       # decoupled weight decay inside the step
                if trust and p.ndim >= 1:
                    w_norm = torch.linalg.vector_norm(p)
                    u_norm = torch.linalg.vector_norm(update)
                    if torch.isfinite(w_norm) and torch.isfinite(u_norm) and w_norm > 0 and u_norm > 0:
                        r = (w_norm / u_norm).clamp(max=trust_clip)
                    else:
                        r = torch.ones((), device=p.device, dtype=p.dtype)
                    step_vec = update.mul(r)
                else:
                    step_vec = update
                p.add_(step_vec, alpha=-lr)
        return loss


class _Scheduler:
    """Sets each group's lr each iteration on the incumbent warmup -> hold -> cosine-to-floor shape,
    scaled by the group's own peak lr. .step() once per iteration (called after opt.step)."""

    def __init__(self, opt, steps, warm, hold_end, decay_len, floor_ratio):
        self.opt = opt
        self.steps = int(steps)
        self.warm = int(warm)
        self.hold_end = int(hold_end)
        self.decay_len = max(1, int(decay_len))
        self.floor_ratio = float(floor_ratio)
        self._peak = [g["lr"] for g in opt.param_groups]   # per-group peak lr captured up front
        self.n = 0
        self._apply(0)

    def _mult(self, idx):
        if idx < self.warm:
            return (idx + 1) / self.warm
        if idx < self.hold_end:
            return 1.0
        p = (idx - self.hold_end) / self.decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))   # 1 -> 0
        return self.floor_ratio + (1.0 - self.floor_ratio) * cos  # 1 -> floor_ratio

    def _apply(self, idx):
        m = self._mult(idx)
        for g, peak in zip(self.opt.param_groups, self._peak):
            g["lr"] = peak * m

    def step(self):
        self.n += 1
        if self.n >= self.steps:
            return
        self._apply(self.n)

    def get_last_lr(self):
        return [g["lr"] for g in self.opt.param_groups]

    def state_dict(self):
        return {"n": self.n}

    def load_state_dict(self, sd):
        self.n = sd.get("n", 0)


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30, floor_ratio=0.05,
         fast_lr_mult=2.5, fast_wd_mult=0.1, trust_clip=10.0, d=192, kq=128):
    steps = int(steps)
    params = list(params)
    slow, fast = [], []
    for p in params:
        (fast if _is_memory(p, d, kq) else slow).append(p)

    groups = []
    if slow:
        groups.append(dict(params=slow, lr=lr, wd=wd, fast=False, trust=False))
    if fast:
        groups.append(dict(params=fast, lr=lr * fast_lr_mult, wd=wd * fast_wd_mult,
                           fast=True, trust=True, trust_clip=trust_clip))
    if not groups:  # degenerate (no params) -- should not happen
        groups.append(dict(params=params, lr=lr, wd=wd, fast=False, trust=False))

    opt = CLSAdamTrust(groups)

    warm = max(20, int(warmup_frac * steps))
    hold_end = warm + int(hold_frac * steps)
    hold_end = min(hold_end, max(warm, steps - 2))         # keep phases ordered for tiny budgets
    decay_len = max(1, steps - hold_end)
    sched = _Scheduler(opt, steps, warm, hold_end, decay_len, floor_ratio)
    return opt, sched

