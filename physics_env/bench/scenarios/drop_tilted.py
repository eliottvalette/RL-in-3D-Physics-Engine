"""Tilted drop benchmark."""

from ..scenario_utils import BenchScenario, reset_env_state, set_base_state, zero_actions


class DropTiltedScenario(BenchScenario):
    name = "drop_tilted"
    category = "free_dynamics"
    description = "Tilted quadruped dropped from rest to expose contact asymmetry."

    def reset(self, env):
        reset_env_state(env)
        set_base_state(env, position=[0.0, 8.5, 0.0], rotation=[0.25, 0.10, 0.35], velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])

    def actions(self, env, step_idx):
        del env, step_idx
        return zero_actions()
