"""Flat drop benchmark."""

from ..scenario_utils import BenchScenario, reset_env_state, set_base_state, zero_actions


class DropFlatScenario(BenchScenario):
    name = "drop_flat"
    category = "free_dynamics"
    description = "Neutral quadruped dropped from rest onto flat ground."

    def reset(self, env):
        reset_env_state(env)
        set_base_state(env, position=[0.0, 8.0, 0.0], rotation=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])

    def actions(self, env, step_idx):
        del env, step_idx
        return zero_actions()
