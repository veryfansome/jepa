"""Sandbox: ties state + actions + renderers into a stepped episode."""

from . import actions, render


class Sandbox:
    def __init__(self, layout_state, banner_id=None, noise_seed=None):
        layout_state.check_invariants()
        self.state = layout_state.copy()
        self.banner_id = banner_id
        self.noise_seed = noise_seed
        self.step_count = 0
        self.last_action = ("", "", "")
        self.last_stdout = ""

    def step(self, action):
        cwd_before = self.state.cwd
        res = actions.apply(self.state, action)
        self.state = res.state
        self.step_count += 1
        self.last_action = action
        self.last_stdout = res.stdout
        res.cwd_before = cwd_before
        return res

    def obs_full(self, with_distractors=True):
        b = self.banner_id if with_distractors else None
        n = self.noise_seed if with_distractors else None
        return render.render_full(self.state, b, n, self.step_count)

    def obs_partial(self, cwd_before, with_distractors=True):
        b = self.banner_id if with_distractors else None
        n = self.noise_seed if with_distractors else None
        return render.render_partial(
            cwd_before, self.last_action, self.last_stdout, b, n, self.step_count
        )
