"""Deterministic no-actuation settle scenario."""

from ..scenario_utils import BenchScenario, align_lowest_vertex_to_ground, reset_env_state, set_base_state, zero_actions


class SettleScenario(BenchScenario):
    name = "settle"
    category = "free_dynamics"
    description = "Neutral quadruped settles onto flat ground from near-contact."

    def reset(self, env):
        reset_env_state(env)
        set_base_state(env, position=[0.0, 5.5, 0.0], rotation=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])
        align_lowest_vertex_to_ground(env, clearance=0.0)

    def actions(self, env, step_idx):
        del env, step_idx
        return zero_actions()
