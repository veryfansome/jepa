"""OPTIM chunk: AdamW on the proven warmup->hold->anneal trajectory, but the tail is a MODERATE
CONSTANT-LR window over which parameter iterates are UNIFORMLY AVERAGED (SWA / LAWA), and the
running average is written back into the live parameters on the final iteration.

Why (grounded in the recorded search, not defaults):
- Fitness is held-out next-obs retrieval on UNSEEN systems; the ledger shows a real generalization
  gap (e5 champion proxy 0.54 -> full-final 0.50, and the general proxy->full shrinkage). SWA
  (Izmailov et al., arXiv:1803.05407) and LAWA (Sanyal et al., arXiv:2306.03241) show that
  averaging iterates sampled along a trajectory run at a MODERATELY HIGH, non-decaying LR lands in
  a wider/flatter basin that generalizes better than any single low-LR iterate. That is exactly the
  lever an unseen-system retrieval metric rewards, and it is orthogonal to the objective/arch/encoder
  levers already stacked into the global best.
- The incumbent warmup_holdcos_floor wins by holding LR high and refusing to decay to 0 (cosine to
  floor/20). This pushes that logic further: decaying the tail to ~0 collapses the iterates to a
  single point, so averaging buys nothing. Instead we anneal to a moderate SWA LR (0.25*peak) and
  HOLD it constant over the last ~40% of steps, so consecutive iterates spread across the basin rim;
  the uniform average then sits at the flat center.
- LAWA's finding that spaced checkpoints amplify the gain -> we sample the average every ~1% of
  steps (not every step), giving ~40 spaced snapshots at both proxy (1000) and full (4000) budgets.
- Hyperparameters recombine the CHAMPION optim (peak lr 5e-4, wd 5e-4, betas (0.9,0.95), plain
  non-decoupled AdamW): the decoupled-by-ndim variant scored LOWER in the ledger (0.3575 vs 0.4014),
  so we keep plain wd. Grad-norm is clipped to 1.0 by the harness every step, so a 5e-4 peak with a
  short warmup is safe.

In-contract mechanism (no EMA-for-eval extension): the harness calls scheduler.step() once per
iteration AFTER opt.step(), so at each call the params hold the just-completed iterate. The custom
scheduler accumulates a uniform running average of those iterates over the SWA window and, on the
final call, copies the average into the live parameters. The harness evaluates `net` immediately
after training -> evaluation happens at the averaged (flat) weights, with zero change to the loop.

Safety: averaging does not touch the loss, so it cannot cause representation collapse (the objective
stays contrastive; a constant prediction still cannot minimize it) and cannot leak future frames
(the optimizer never sees data/causality). The finalize step guards against non-finite averages and
leaves the last live iterate untouched if anything is not finite.

Contract: make(params, steps, **kw) -> (optimizer, scheduler). scheduler.step() is called once per
iteration by the harness. Pure/self-contained; torch only; no file/state.
"""
import math
import torch

NAME = "r6_swa_tailavg_hold"
DESCRIPTION = ("AdamW(lr 5e-4, wd 5e-4, betas (0.9,0.95)); 4% warmup, 20% hold at peak, cosine "
               "anneal to 0.25*peak by 60% of steps, then a constant-LR SWA tail where iterates "
               "are uniformly averaged (sampled every ~1% of steps) and written into the params "
               "on the final step for flat-minimum, generalization-oriented eval.")


class _SWATailScheduler:
    """LR schedule + tail weight-averaging in one .step()-per-iteration object.

    Timeline in iteration index space (idx = 0 .. steps-1, the index of each opt.step()):
      [0, warm)            linear warmup 0 -> peak
      [warm, hold_end)     HOLD at peak
      [hold_end, swa_start) cosine anneal peak -> swa_lr (= swa_ratio * peak)
      [swa_start, steps)   CONSTANT swa_lr  <- SWA window: iterates averaged here
    On the final .step() the uniform average over the sampled tail iterates is copied into params.
    """

    def __init__(self, opt, steps, warm, hold_end, swa_start, swa_ratio, avg_every):
        self.opt = opt
        self.steps = int(steps)
        self.warm = int(warm)
        self.hold_end = int(hold_end)
        self.swa_start = int(swa_start)
        self.swa_ratio = float(swa_ratio)
        self.avg_every = max(1, int(avg_every))
        self.anneal_len = max(1, self.swa_start - self.hold_end)
        # base (peak) lr per group, captured before any override
        self._base = [g["lr"] for g in opt.param_groups]
        # flat, ordered list of trainable params (canonical order from the optimizer)
        self._params = [p for g in opt.param_groups for p in g["params"] if p.requires_grad]
        self._avg = None          # list of running-average tensors (lazy-alloc on first capture)
        self._n_avg = 0           # number of snapshots accumulated
        self.n = 0                # number of .step() calls made
        self._set_lr(0)           # set LR for the first opt.step() (idx 0)

    def _mult(self, idx):
        if idx < self.warm:
            return (idx + 1) / self.warm
        if idx < self.hold_end:
            return 1.0
        if idx < self.swa_start:
            p = (idx - self.hold_end) / self.anneal_len       # 0 -> 1
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))  # 1 -> 0
            return self.swa_ratio + (1.0 - self.swa_ratio) * cos  # peak -> swa_lr
        return self.swa_ratio                                   # constant SWA tail

    def _set_lr(self, idx):
        m = self._mult(idx)
        for g, b in zip(self.opt.param_groups, self._base):
            g["lr"] = b * m

    @torch.no_grad()
    def _accumulate(self):
        if self._avg is None:
            self._avg = [p.detach().clone().float() for p in self._params]
            self._n_avg = 1
            return
        self._n_avg += 1
        inv = 1.0 / self._n_avg
        for a, p in zip(self._avg, self._params):
            a.add_(p.detach().float() - a, alpha=inv)  # incremental uniform mean

    @torch.no_grad()
    def _finalize(self):
        if self._avg is None or self._n_avg == 0:
            return
        for a, p in zip(self._avg, self._params):
            if torch.isfinite(a).all():
                p.data.copy_(a.to(dtype=p.dtype, device=p.device))
            # else: keep the live iterate for this tensor (NaN-safe)

    def step(self):
        self.n += 1
        completed = self.n - 1                 # index of the opt.step() just finished
        last = (self.n >= self.steps)
        if completed >= self.swa_start:
            if ((completed - self.swa_start) % self.avg_every == 0) or last:
                self._accumulate()
        if last:
            self._finalize()
            return
        self._set_lr(self.n)                    # LR for the next opt.step()

    # harness never needs these, but keep a torch-scheduler-like surface
    def get_last_lr(self):
        return [g["lr"] for g in self.opt.param_groups]

    def state_dict(self):
        return {"n": self.n, "n_avg": self._n_avg}

    def load_state_dict(self, sd):
        self.n = sd.get("n", 0)
        self._n_avg = sd.get("n_avg", 0)


def make(params, steps, lr=5e-4, wd=5e-4, betas=(0.9, 0.95), eps=1e-8,
         warmup_frac=0.04, hold_frac=0.20, swa_start_frac=0.60, swa_ratio=0.25,
         avg_every_frac=0.01):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)

    steps = int(steps)
    warm = max(20, int(warmup_frac * steps))
    hold_end = warm + int(hold_frac * steps)
    swa_start = int(swa_start_frac * steps)
    # keep the phases well-ordered even for tiny/odd budgets, and leave >=1 tail step
    hold_end = min(hold_end, max(warm, steps - 2))
    swa_start = min(max(swa_start, hold_end + 1), steps - 1)
    avg_every = max(1, int(avg_every_frac * steps))

    sched = _SWATailScheduler(opt, steps, warm, hold_end, swa_start, swa_ratio, avg_every)
    return opt, sched

