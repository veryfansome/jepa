"""Trajectory policies (terminal-jepa.md §3). Policies emit valid-intent actions only;
the invalid-action quota is enforced at the trajectory level by datagen.generate so the
realized invalid rate tracks the quota regardless of policy mix."""

from env import actions, vocab
from env.state import CwdIs, FileAbsent, FileExists, FileExistsWithClass


def plan_for(pred, state):
    """Constructive plan for one predicate from the true state. Used only by datagen's
    scripted policy (data collection may be privileged; planners may not be)."""
    steps = []
    if isinstance(pred, (FileExistsWithClass, FileExists)):
        parent = pred.path[:-1]
        for i in range(1, len(parent) + 1):
            if parent[:i] not in state.dirs:
                steps.append(("mkdir", vocab.path_to_str(parent[:i]), ""))
        cls = pred.cls if isinstance(pred, FileExistsWithClass) else 0
        steps.append(("write", vocab.path_to_str(pred.path), f"c{cls}"))
    elif isinstance(pred, FileAbsent):
        if pred.path in state.files:
            steps.append(("rm", vocab.path_to_str(pred.path), ""))
    elif isinstance(pred, CwdIs):
        for i in range(1, len(pred.path) + 1):
            if pred.path[:i] not in state.dirs:
                steps.append(("mkdir", vocab.path_to_str(pred.path[:i]), ""))
        steps.append(("cd", vocab.path_to_str(pred.path), ""))
    else:
        raise ValueError(f"unsupported predicate: {pred!r}")
    return steps


class GoalReacher:
    """Executes constructive plans with epsilon-noise (random *valid* actions); samples a
    fresh goal from the allowed predicate pool whenever the current one is satisfied (or
    its plan is exhausted), so trajectories stay busy for their full length."""

    def __init__(self, predicate_pool, rng, epsilon):
        self.pool = predicate_pool
        self.rng = rng
        self.epsilon = epsilon
        self.queue = []
        self.goal = None

    def _new_goal(self, state):
        for _ in range(50):
            pred = self.rng.choice(self.pool)
            if not pred.check(state):
                plan = plan_for(pred, state)
                if plan:
                    self.goal, self.queue = pred, plan
                    return True
        return False

    def next_action(self, state):
        if self.rng.random() < self.epsilon:
            return actions.sample_valid(state, self.rng)
        if not self.queue or (self.goal and self.goal.check(state)):
            if not self._new_goal(state):
                return actions.sample_valid(state, self.rng)
        # Replan when noise knocked the plan off course (a queued step may have become
        # invalid, e.g. its parent dir was rm'd by an epsilon action).
        action = self.queue[0]
        if actions.apply(state, action).ttype == actions.INVALID:
            self.queue = plan_for(self.goal, state)
            if not self.queue:
                return actions.sample_valid(state, self.rng)
            action = self.queue[0]
        self.queue = self.queue[1:]
        return action
