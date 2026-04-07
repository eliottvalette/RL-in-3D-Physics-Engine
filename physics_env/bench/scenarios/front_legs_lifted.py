"""Rear-support / front-lift benchmark."""

from ..scenario_utils import BenchScenario, align_lowest_vertex_to_ground, reset_env_state, set_base_state, set_joint_pose, zero_actions


class FrontLegsLiftedScenario(BenchScenario):
    name = "front_legs_lifted"
    category = "support_polygon"
    description = "Front legs folded upward to stress rear support and tipping."

    def reset(self, env):
        reset_env_state(env)
        set_base_state(env, position=[0.0, 5.5, 0.0], rotation=[0.0, 0.0, 0.15], velocity=[0.0, 0.0, 0.0], angular_velocity=[0.0, 0.0, 0.0])
        set_joint_pose(env, shoulders=[1.0, 1.0, 0.0, 0.0], elbows=[-0.5, -0.5, 0.0, 0.0])
        align_lowest_vertex_to_ground(env, clearance=0.0)

    def actions(self, env, step_idx):
        del env, step_idx
        return zero_actions()
