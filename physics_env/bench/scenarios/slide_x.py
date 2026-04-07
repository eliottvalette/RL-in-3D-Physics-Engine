"""Horizontal sliding benchmark."""

from ..scenario_utils import BenchScenario, align_lowest_vertex_to_ground, reset_env_state, set_base_state, zero_actions


class SlideXScenario(BenchScenario):
    name = "slide_x"
    category = "contact_friction"
    description = "Quadruped aligned on the ground with initial lateral velocity."

    def reset(self, env):
        reset_env_state(env)
        set_base_state(env, position=[0.0, 5.5, 0.0], rotation=[0.0, 0.0, 0.0], velocity=[2.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])
        align_lowest_vertex_to_ground(env, clearance=0.0)

    def actions(self, env, step_idx):
        del env, step_idx
        return zero_actions()
