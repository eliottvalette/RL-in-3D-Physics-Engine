"""Alternating gait smoke benchmark."""

from ..scenario_utils import BenchScenario, align_lowest_vertex_to_ground, reset_env_state, set_base_state, zero_actions


class GaitSmokeScenario(BenchScenario):
    name = "gait_smoke"
    category = "moving_geometry"
    description = "Alternating diagonal leg commands to reveal traction asymmetry."

    def reset(self, env):
        reset_env_state(env)
        set_base_state(env, position=[0.0, 5.5, 0.0], rotation=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])
        align_lowest_vertex_to_ground(env, clearance=0.0)

    def actions(self, env, step_idx):
        del env
        phase = (step_idx // 20) % 4
        shoulders, elbows = zero_actions()
        if phase in (0, 2):
            shoulders[0] = 1.0
            shoulders[3] = -1.0
            elbows[0] = -1.0
            elbows[3] = 1.0
        else:
            shoulders[1] = 1.0
            shoulders[2] = -1.0
            elbows[1] = -1.0
            elbows[2] = 1.0
        return shoulders, elbows
