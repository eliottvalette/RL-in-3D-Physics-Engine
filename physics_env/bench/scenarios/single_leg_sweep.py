"""Single-leg sweep benchmark."""

import numpy as np

from ..scenario_utils import BenchScenario, align_lowest_vertex_to_ground, reset_env_state, set_base_state, zero_actions


class SingleLegSweepScenario(BenchScenario):
    name = "single_leg_sweep"
    category = "moving_geometry"
    description = "One front leg sweeps repeatedly while the body starts grounded."

    def reset(self, env):
        reset_env_state(env)
        set_base_state(env, position=[0.0, 5.5, 0.0], rotation=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])
        align_lowest_vertex_to_ground(env, clearance=0.0)

    def actions(self, env, step_idx):
        del env
        phase = step_idx % 120
        shoulders, elbows = zero_actions()
        if phase < 40:
            shoulders[0] = 1.0
            elbows[0] = -1.0
        elif phase < 80:
            shoulders[0] = -1.0
            elbows[0] = 1.0
        return shoulders, elbows
