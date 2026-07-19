"""R8 optimizer: fastweight-aware homeostatic consolidation AdamW.

The current champion arch writes each completed (command, observation) pair into online
delta-rule memories and later reads them in target space. The learned bottleneck is therefore
not a stored memory tensor; it is the addressing/control system: repeated thin key maps
(content_read/content_write/path_read/path_write, usually key_d x d), small read/write gates,
and scalar decay/position controls.

Mechanism translated from homeostatic synaptic scaling and Oja-style normalization:
  row_norm_i <- row_norm_i * (target / row_norm_i) ** eta
  U <- normalize_rows(W); U <- normalize_rows(U - alpha * ((U U^T - I) U))
The first term slowly keeps each key channel near an initial set point while preserving learned
directions; the second softly prevents key-channel collapse. This complements the delta memory by
keeping its address basis high-capacity rather than adding another retrieval loss/path.

Concrete grounding:
  - Synaptic scaling stabilizes plastic networks while preserving learned relative structure:
    https://arxiv.org/abs/1304.2266 and https://arxiv.org/abs/1709.05633
  - Oja/weight-normalization style constraints decouple direction learning from norm drift:
    https://www.scholarpedia.org/article/Oja_learning_rule and https://arxiv.org/abs/1602.07868

Fallback: if no repeated thin key matrices are present, all parameters use the incumbent-like
AdamW(lr 5e-4, wd 5e-4, beta2 .95) warmup -> hold -> cosine-to-floor schedule.
"""
import math
import torch

NAME = "r8_fastweight_homeostatic_consolidation"
DESCRIPTION = (
    "Fastweight-aware AdamW. Repeated thin key matrices get faster low-decay LR, Oja/synaptic-"
    "scaling row homeostasis, and tail EMA consolidation; small gates/scalars get slow no-decay "
    "control LR; all other params keep the incumbent warmup-hold-cosine-to-floor AdamW."
)


def _shape_counts(params):
    counts = {}
    for p in params:
        if p.ndim == 2:
            s = tuple(p.shape)
            counts[s] = counts.get(s, 0) + 1
    return counts


def _is_assoc_key(p, counts):
    if p.ndim != 2:
        return False
    rows, cols = tuple(p.shape)
    if counts.get((rows, cols), 0) < 2:
        return False
    return 16 <= rows <= 144 and 128 <= cols <= 256 and rows < cols


def _is_small_control(p, has_assoc):
    if not has_assoc:
        return False
    if p.ndim == 0:
        return True
    if p.ndim == 1 and p.numel() <= 4:
        return True
    if p.ndim == 2:
        rows, cols = tuple(p.shape)
        return rows <= 2 and 64 <= cols <= 512
    return False


class HomeostaticAdamW(torch.optim.Optimizer):
    def __init__(self, groups, eps=1e-8):
        defaults = dict(
            lr=5e-4,
            betas=(0.9, 0.95),
            eps=eps,
            weight_decay=0.0,
            homeo=False,
            homeo_strength=0.0,
            ortho_strength=0.0,
            homeo_delay=0.10,
            homeo_ramp=0.35,
            progress=0.0,
        )
        super().__init__(groups, defaults)

    @torch.no_grad()
    def _homeostat(self, p, state, group):
        progress = float(group.get("progress", 0.0))
        delay = float(group.get("homeo_delay", 0.10))
        ramp = max(1e-6, float(group.get("homeo_ramp", 0.35)))
        scale = max(0.0, min(1.0, (progress - delay) / ramp))
        eta = float(group.get("homeo_strength", 0.0)) * scale
        alpha = float(group.get("ortho_strength", 0.0)) * scale
        if eta <= 0.0 and alpha <= 0.0:
            return

        w = p.detach().float()
        if not bool(torch.isfinite(w).all().item()):
            return

        eps = float(group.get("eps", 1e-8))
        norm = w.norm(dim=1, keepdim=True).clamp_min(eps)

        target = state.get("row_target")
        if target is None:
            target = norm.mean().detach().expand_as(norm).clone()
            state["row_target"] = target
        target = target.to(device=w.device, dtype=w.dtype)

        rows = w / norm
        if alpha > 0.0 and rows.shape[0] <= rows.shape[1]:
            eye = state.get("eye")
            if eye is None or eye.shape[0] != rows.shape[0] or eye.device != rows.device:
                eye = torch.eye(rows.shape[0], device=rows.device, dtype=rows.dtype)
                state["eye"] = eye
            gram = rows @ rows.t()
            rows = rows - alpha * ((gram - eye) @ rows)
            rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(eps)

        if eta > 0.0:
            gain = (target / norm).clamp(0.5, 2.0).pow(eta)
            new_w = rows * norm * gain
        else:
            new_w = rows * norm

        if bool(torch.isfinite(new_w).all().item()):
            p.copy_(new_w.to(dtype=p.dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("HomeostaticAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group.get("homeo", False) and p.ndim == 2:
                        row_norm = p.detach().float().norm(dim=1, keepdim=True).clamp_min(eps)
                        state["row_target"] = row_norm.mean().detach().expand_as(row_norm).clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bc2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-(lr / bc1))

                if group.get("homeo", False) and p.ndim == 2:
                    self._homeostat(p, state, group)

        return loss


class _ScheduleAndConsolidate:
    def __init__(self, opt, steps, avg_start_frac=0.72, avg_every_frac=0.01, ema_alpha=0.04):
        self.opt = opt
        self.steps = max(1, int(steps))
        self.n = 0
        self.peaks = [float(g["lr"]) for g in opt.param_groups]
        self.avg_start = min(max(0, int(avg_start_frac * self.steps)), self.steps - 1)
        self.avg_every = max(1, int(avg_every_frac * self.steps))
        self.ema_alpha = float(ema_alpha)
        self.avg_params = [
            p for g in opt.param_groups if g.get("assoc_avg", False)
            for p in g["params"] if p.requires_grad
        ]
        self.avg = None
        self.n_avg = 0
        self._apply(0)

    def _mult(self, group, idx):
        warm_frac = float(group.get("warm_frac", 0.04))
        hold_frac = float(group.get("hold_frac", 0.30))
        floor = float(group.get("floor_ratio", 0.05))
        power = float(group.get("cool_power", 1.0))
        warm_min = int(group.get("warm_min", 20))

        warm = min(max(1, max(warm_min, int(warm_frac * self.steps))), self.steps)
        decay_start = warm + int(hold_frac * self.steps)
        decay_start = min(decay_start, max(warm, self.steps - 1))
        decay_len = max(1, self.steps - decay_start)

        if idx < warm:
            return (idx + 1) / warm
        if idx < decay_start:
            return 1.0

        p = (idx - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        if power != 1.0:
            cos = cos ** power
        return floor + (1.0 - floor) * cos

    def _apply(self, idx):
        progress = idx / max(1, self.steps - 1)
        for group, peak in zip(self.opt.param_groups, self.peaks):
            group["lr"] = peak * self._mult(group, idx)
            group["progress"] = progress

    @torch.no_grad()
    def _accumulate(self):
        if not self.avg_params:
            return
        if self.avg is None:
            self.avg = [p.detach().clone().float() for p in self.avg_params]
            self.n_avg = 1
            return
        self.n_avg += 1
        a = self.ema_alpha
        for avg, p in zip(self.avg, self.avg_params):
            avg.lerp_(p.detach().float(), a)

    @torch.no_grad()
    def _finalize(self):
        if self.avg is None:
            return
        for avg, p in zip(self.avg, self.avg_params):
            if bool(torch.isfinite(avg).all().item()):
                p.copy_(avg.to(device=p.device, dtype=p.dtype))

    def step(self):
        completed = self.n
        last = completed >= self.steps - 1

        if completed >= self.avg_start:
            if ((completed - self.avg_start) % self.avg_every == 0) or last:
                self._accumulate()

        if last:
            self._finalize()
            self.n += 1
            return

        self.n += 1
        self._apply(self.n)

    def get_last_lr(self):
        return [group["lr"] for group in self.opt.param_groups]

    def state_dict(self):
        return {"n": self.n, "n_avg": self.n_avg}

    def load_state_dict(self, state_dict):
        self.n = int(state_dict.get("n", 0))
        self.n_avg = int(state_dict.get("n_avg", 0))
        self._apply(min(self.n, self.steps - 1))


def make(
    params,
    steps,
    lr=5e-4,
    wd=5e-4,
    assoc_lr_mult=1.6,
    assoc_wd_mult=0.10,
    control_lr_mult=0.85,
    homeo_strength=0.012,
    ortho_strength=0.0015,
):
    plist = [p for p in params if p.requires_grad]
    if not plist:
        raise ValueError("optimizer got an empty parameter list")

    counts = _shape_counts(plist)
    assoc = [p for p in plist if _is_assoc_key(p, counts)]
    has_assoc = len(assoc) > 0
    assoc_ids = {id(p) for p in assoc}
    control = [p for p in plist if id(p) not in assoc_ids and _is_small_control(p, has_assoc)]
    control_ids = {id(p) for p in control}
    trunk = [p for p in plist if id(p) not in assoc_ids and id(p) not in control_ids]

    groups = []
    if trunk:
        groups.append(dict(
            params=trunk,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=wd,
            kind="trunk",
            warm_frac=0.04,
            hold_frac=0.30,
            floor_ratio=0.05,
            cool_power=1.0,
        ))

    if assoc:
        groups.append(dict(
            params=assoc,
            lr=lr * assoc_lr_mult,
            betas=(0.9, 0.92),
            weight_decay=wd * assoc_wd_mult,
            kind="assoc_key",
            warm_frac=0.06,
            hold_frac=0.14,
            floor_ratio=0.012,
            cool_power=1.7,
            homeo=True,
            homeo_strength=homeo_strength,
            ortho_strength=ortho_strength,
            homeo_delay=0.08,
            homeo_ramp=0.40,
            assoc_avg=True,
        ))

    if control:
        groups.append(dict(
            params=control,
            lr=lr * control_lr_mult,
            betas=(0.9, 0.98),
            weight_decay=0.0,
            kind="control",
            warm_frac=0.10,
            hold_frac=0.45,
            floor_ratio=0.08,
            cool_power=0.8,
        ))

    opt = HomeostaticAdamW(groups)
    sched = _ScheduleAndConsolidate(opt, steps)
    return opt, sched
