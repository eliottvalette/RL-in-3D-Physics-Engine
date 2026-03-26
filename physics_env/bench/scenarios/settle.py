"""Deterministic no-actuation settle scenario."""

import numpy as np


def _reset_env_state(env):
    env.quadruped.reset()
    env.circles_passed.clear()
    env.prev_potential = None
    env.consecutive_steps_below_critical_height = 0
    env.consecutive_steps_above_critical_height = 0
    env.prev_radius = None


class SettleScenario:
    name = "settle"

    def reset(self, env):
        _reset_env_state(env)

    def actions(self, env, step_idx):
        del env, step_idx
        return np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
