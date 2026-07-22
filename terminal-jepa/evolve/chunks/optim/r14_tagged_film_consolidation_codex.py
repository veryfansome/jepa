"""OPTIM chunk: role-aware tagged FiLM consolidation.

The R14 surface is no longer the old uniform R4 loss: it has antiretrieval ring
negatives, zero-init system FiLM modulators, and a train-only cued-recall probe.
This optimizer treats those roles differently from the slow transformer trunk.

Cross-domain mechanism: synaptic tagging and capture.  For modulation and
head/readout tensors, a detached EMA of role-local gradient RMS is the "tag":
when that role receives unusually large loss pressure, its Adam step is
temporarily amplified; late in training those tagged fast-plastic tensors are
consolidated by EMA weight capture and the average is copied back on the final
scheduler step.  The delta-memory addressing matrices keep the incumbent Muon
orthogonalized-momentum update, but syscond FiLM/value tensors are kept out of
that group when a bias-paired shape reveals they are modulators rather than
biasless key projections.

Contract: make(params, steps, **kw) -> (optimizer, scheduler_or_None).  The
harness calls opt.zero_grad(set_to_none=True), opt.step(), scheduler.step().
Self-contained, deterministic, torch+stdlib only; no data/eval access.
"""
import math
import torch

NAME = "r14_tagged_film_consolidation_codex"
DESCRIPTION = (
    "Role-aware optimizer for the v2 loss surface: incumbent AdamW trunk, Muon only on "
    "biasless repeated key-addressing matrices, fast no-decay synaptic-tag Adam for "
    "FiLM/system modulation tensors, faster low-decay Adam for eval+aux heads, slow "
    "no-decay gates/control, and late EMA consolidation of mod/head weights."
)

D = 768


def _finite_all(x):
    return bool(torch.isfinite(x).all().item())


def _smooth01(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _ns_orth(g, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.detach().float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / x.norm().clamp_min(eps)
    for _ in range(int(steps)):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.mT
    return x.to(dtype=g.dtype)


def _shape_counts(params):
    counts = {}
    for p in params:
        if p.ndim == 2:
            s = tuple(int(v) for v in p.shape)
            counts[s] = counts.get(s, 0) + 1
    return counts


def _followed_by_bias(params, i, rows):
    return (
        i + 1 < len(params)
        and params[i + 1].ndim == 1
        and int(params[i + 1].numel()) == int(rows)
    )


def _partition(params, key_d):
    counts = _shape_counts(params)
    mod_ids, head_ids, key_ids, control_ids = set(), set(), set(), set()
    mod_hidden = set()

    for p in params:
        if p.ndim == 2 and int(p.shape[0]) == 2 * D:
            mod_ids.add(id(p))
            mod_hidden.add(int(p.shape[1]))
        elif p.ndim == 1 and int(p.numel()) == 2 * D:
            mod_ids.add(id(p))

    for i, p in enumerate(params):
        if id(p) in mod_ids or p.ndim != 2:
            continue
        rows, cols = int(p.shape[0]), int(p.shape[1])
        if rows in mod_hidden and rows < cols < D:
            mod_ids.add(id(p))
            if _followed_by_bias(params, i, rows):
                mod_ids.add(id(params[i + 1]))
        if (
            rows == key_d
            and cols not in (key_d, D)
            and counts.get((rows, cols), 0) >= 2
            and _followed_by_bias(params, i, rows)
        ):
            mod_ids.add(id(p))
            mod_ids.add(id(params[i + 1]))

    for i, p in enumerate(params):
        if id(p) in mod_ids:
            continue
        if p.ndim == 2 and int(p.shape[0]) == D and int(p.shape[1]) != D:
            head_ids.add(id(p))
            if _followed_by_bias(params, i, D):
                head_ids.add(id(params[i + 1]))
        elif p.ndim == 1 and int(p.numel()) == D:
            head_ids.add(id(p))

    for i, p in enumerate(params):
        if id(p) in mod_ids or id(p) in head_ids or p.ndim != 2:
            continue
        rows, cols = int(p.shape[0]), int(p.shape[1])
        if (
            rows == key_d
            and cols not in (key_d, D)
            and counts.get((rows, cols), 0) >= 2
            and not _followed_by_bias(params, i, rows)
        ):
            key_ids.add(id(p))

    for p in params:
        pid = id(p)
        if pid in mod_ids or pid in head_ids or pid in key_ids:
            continue
        if p.ndim == 0:
            control_ids.add(pid)
        elif p.ndim == 1 and int(p.numel()) <= 16:
            control_ids.add(pid)
        elif p.ndim == 2:
            rows, cols = int(p.shape[0]), int(p.shape[1])
            if rows <= 4 or (rows <= 16 and cols == key_d):
                control_ids.add(pid)

    roles = {k: [] for k in ("trunk", "norm", "key", "mod", "head", "control")}
    for p in params:
        pid = id(p)
        if pid in mod_ids:
            roles["mod"].append(p)
        elif pid in head_ids:
            roles["head"].append(p)
        elif pid in key_ids:
            roles["key"].append(p)
        elif pid in control_ids:
            roles["control"].append(p)
        elif p.ndim < 2:
            roles["norm"].append(p)
        else:
            roles["trunk"].append(p)
    return roles


class _TaggedRoleOptimizer(torch.optim.Optimizer):
    def __init__(self, groups):
        defaults = dict(
            lr=5e-4,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            mode="adam",
            momentum=0.95,
            ns_steps=5,
            rms_match=0.2,
            capture=False,
            tag_power=0.0,
            tag_decay=0.96,
            gain_clip=(1.0, 1.0),
            tag_threshold=1.0,
        )
        super().__init__(groups, defaults)
        self._ref_tag = 0.0

    def _group_grad_rms(self, group):
        total, n = 0.0, 0
        for p in group["params"]:
            g = p.grad
            if g is None or g.is_sparse or not _finite_all(g):
                continue
            v = float(g.detach().float().pow(2).mean().sqrt().item())
            if math.isfinite(v):
                total += v
                n += 1
        return total / max(1, n)

    def _update_tags(self):
        for group in self.param_groups:
            rms = self._group_grad_rms(group)
            old = group.get("_tag", None)
            decay = float(group.get("tag_decay", 0.96))
            group["_tag"] = rms if old is None else decay * float(old) + (1.0 - decay) * rms

        refs = [
            float(g.get("_tag", 0.0))
            for g in self.param_groups
            if g.get("role") in ("trunk", "norm", "key") and float(g.get("_tag", 0.0)) > 0.0
        ]
        if not refs:
            refs = [float(g.get("_tag", 0.0)) for g in self.param_groups if float(g.get("_tag", 0.0)) > 0.0]
        ref = sum(refs) / len(refs) if refs else 1.0
        self._ref_tag = ref if self._ref_tag <= 0.0 else 0.95 * self._ref_tag + 0.05 * ref

        for group in self.param_groups:
            group["_gain"] = 1.0
            group["_capture_gate"] = 0.0
            if not group.get("capture", False):
                continue
            ratio = float(group.get("_tag", 0.0)) / (self._ref_tag + 1e-12)
            lo, hi = group.get("gain_clip", (1.0, 1.0))
            power = float(group.get("tag_power", 0.0))
            raw = ratio ** power if ratio > 0.0 else float(lo)
            gain = max(float(lo), min(float(hi), raw))
            phase = float(group.get("capture_phase", 1.0))
            group["_gain"] = 1.0 + phase * (gain - 1.0)
            thresh = max(1e-6, float(group.get("tag_threshold", 1.0)))
            group["_capture_gate"] = phase * (ratio / (ratio + thresh)) if ratio > 0.0 else 0.0

    @torch.no_grad()
    def _adam_step_group(self, group):
        beta1, beta2 = group["betas"]
        lr = float(group["lr"]) * float(group.get("_gain", 1.0))
        decay_lr = float(group["lr"])
        wd = float(group["weight_decay"])
        eps = float(group["eps"])
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            if g.is_sparse:
                raise RuntimeError("TaggedRoleOptimizer does not support sparse gradients")
            if not _finite_all(g):
                continue
            st = self.state[p]
            if len(st) == 0:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg, exp_avg_sq = st["exp_avg"], st["exp_avg_sq"]
            st["step"] += 1
            t = st["step"]

            if wd != 0.0:
                p.mul_(1.0 - decay_lr * wd)

            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            denom = exp_avg_sq.sqrt().div(math.sqrt(bc2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-(lr / bc1))

    @torch.no_grad()
    def _muon_step_group(self, group):
        lr = float(group["lr"]) * float(group.get("_gain", 1.0))
        mu = float(group["momentum"])
        ns_steps = int(group["ns_steps"])
        rms_match = float(group["rms_match"])
        for p in group["params"]:
            g = p.grad
            if g is None or p.ndim != 2 or not _finite_all(g):
                continue
            st = self.state[p]
            if "buf" not in st:
                st["buf"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            buf = st["buf"]
            buf.mul_(mu).add_(g)
            u = g.add(buf, alpha=mu)
            o = _ns_orth(u, steps=ns_steps)
            scale = rms_match * math.sqrt(max(int(p.shape[0]), int(p.shape[1])))
            p.add_(o, alpha=-lr * scale)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._update_tags()
        for group in self.param_groups:
            if group.get("mode") == "muon":
                self._muon_step_group(group)
            else:
                self._adam_step_group(group)
        return loss


class _TaggedSchedule:
    def __init__(self, opt, steps, avg_start_frac=0.68, avg_every_frac=0.01, ema_alpha=0.08):
        self.opt = opt
        self.steps = max(1, int(steps))
        self.n = 0
        self.peaks = [float(g["lr"]) for g in opt.param_groups]
        self.avg_start = min(max(0, int(avg_start_frac * self.steps)), self.steps - 1)
        self.avg_every = max(1, int(avg_every_frac * self.steps))
        self.ema_alpha = float(ema_alpha)
        self.avg_pairs = [
            (g, p)
            for g in opt.param_groups
            if g.get("role") in ("mod", "head")
            for p in g["params"]
            if p.requires_grad
        ]
        self.avg = None
        self.n_avg = 0
        self._apply(0)

    def _cos_floor(self, idx, warm_frac, hold_frac, floor, power=1.0, warm_min=20):
        warm = min(max(1, max(int(warm_min), int(warm_frac * self.steps))), self.steps)
        hold_end = min(warm + int(hold_frac * self.steps), max(warm, self.steps - 1))
        if idx < warm:
            return (idx + 1) / warm
        if idx < hold_end:
            return 1.0
        p = (idx - hold_end) / max(1, self.steps - hold_end)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return float(floor) + (1.0 - float(floor)) * (cos ** float(power))

    def _role_mult(self, role, idx):
        if role == "mod":
            return self._cos_floor(idx, 0.025, 0.45, 0.015, power=1.35, warm_min=10)
        if role == "head":
            return self._cos_floor(idx, 0.025, 0.25, 0.020, power=1.15, warm_min=10)
        if role == "control":
            return self._cos_floor(idx, 0.08, 0.20, 0.08, power=1.0, warm_min=20)
        return self._cos_floor(idx, 0.04, 0.30, 0.05, power=1.0, warm_min=20)

    def _capture_phase(self, role, idx):
        p = idx / max(1, self.steps - 1)
        if role == "mod":
            return 1.0 - _smooth01((p - 0.58) / 0.22)
        if role == "head":
            return 1.0 - _smooth01((p - 0.50) / 0.22)
        return 0.0

    def _apply(self, idx):
        progress = idx / max(1, self.steps - 1)
        for group, peak in zip(self.opt.param_groups, self.peaks):
            role = group.get("role", "trunk")
            group["lr"] = peak * self._role_mult(role, idx)
            group["progress"] = progress
            group["capture_phase"] = self._capture_phase(role, idx)

    @torch.no_grad()
    def _accumulate(self):
        if not self.avg_pairs:
            return
        if self.avg is None:
            self.avg = [p.detach().clone().float() for _, p in self.avg_pairs]
            self.n_avg = 1
            return
        self.n_avg += 1
        for avg, (group, p) in zip(self.avg, self.avg_pairs):
            gate = max(0.0, min(1.0, float(group.get("_capture_gate", 0.0))))
            alpha = self.ema_alpha * (0.25 + 0.75 * gate)
            avg.lerp_(p.detach().float(), alpha)

    @torch.no_grad()
    def _finalize(self):
        if self.avg is None:
            return
        for avg, (_, p) in zip(self.avg, self.avg_pairs):
            if _finite_all(avg):
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
        return [float(g["lr"]) for g in self.opt.param_groups]

    def state_dict(self):
        return {"n": self.n, "n_avg": self.n_avg}

    def load_state_dict(self, sd):
        self.n = int(sd.get("n", 0))
        self.n_avg = int(sd.get("n_avg", 0))
        self._apply(min(self.n, self.steps - 1))


def make(
    params,
    steps,
    lr=5e-4,
    wd=5e-4,
    key_d=64,
    trunk_lr_mult=1.0,
    key_lr_mult=1.0,
    mod_lr_mult=2.4,
    head_lr_mult=1.4,
    control_lr_mult=0.45,
    norm_lr_mult=0.80,
    momentum=0.95,
    ns_steps=5,
    rms_match=0.2,
    avg_start_frac=0.68,
    avg_every_frac=0.01,
    ema_alpha=0.08,
):
    plist = [p for p in params if p.requires_grad]
    if not plist:
        raise ValueError("optimizer got an empty parameter list")

    roles = _partition(plist, int(key_d))
    groups = []

    def add(role, ps, lr_mult, weight_decay, betas, mode="adam", capture=False,
            tag_power=0.0, tag_decay=0.96, gain_clip=(1.0, 1.0), tag_threshold=1.0):
        if not ps:
            return
        groups.append(dict(
            params=ps,
            role=role,
            lr=float(lr) * float(lr_mult),
            weight_decay=float(weight_decay),
            betas=betas,
            eps=1e-8,
            mode=mode,
            momentum=float(momentum),
            ns_steps=int(ns_steps),
            rms_match=float(rms_match),
            capture=bool(capture),
            tag_power=float(tag_power),
            tag_decay=float(tag_decay),
            gain_clip=gain_clip,
            tag_threshold=float(tag_threshold),
        ))

    add("trunk", roles["trunk"], trunk_lr_mult, wd, (0.9, 0.95))
    add("norm", roles["norm"], norm_lr_mult, 0.0, (0.9, 0.95))
    add("key", roles["key"], key_lr_mult, 0.0, (0.9, 0.95), mode="muon")
    add("mod", roles["mod"], mod_lr_mult, 0.0, (0.85, 0.90),
        capture=True, tag_power=0.35, tag_decay=0.93, gain_clip=(0.60, 1.85), tag_threshold=0.70)
    add("head", roles["head"], head_lr_mult, 1e-4, (0.9, 0.96),
        capture=True, tag_power=0.25, tag_decay=0.95, gain_clip=(0.70, 1.45), tag_threshold=0.85)
    add("control", roles["control"], control_lr_mult, 0.0, (0.9, 0.98))

    opt = _TaggedRoleOptimizer(groups)
    sched = _TaggedSchedule(
        opt,
        steps,
        avg_start_frac=avg_start_frac,
        avg_every_frac=avg_every_frac,
        ema_alpha=ema_alpha,
    )
    return opt, sched
