"""Bench scenario for the interactive X-toggle gait animation."""

from ..scenario_utils import BenchScenario, align_lowest_vertex_to_ground, reset_env_state, set_base_state


class DemoGaitAnimationScenario(BenchScenario):
    name = "demo_gait_animation"
    category = "moving_geometry"
    description = "Runs the same shoulder/elbow action gait used by the X-toggle animation."

    def reset(self, env):
        reset_env_state(env)
        env.demo_gait_step = 0
        set_base_state(
            env,
            position=[0.0, 5.5, 0.0],
            rotation=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
        )
        align_lowest_vertex_to_ground(env, clearance=0.0)

    def actions(self, env, step_idx):
        del step_idx
        shoulders, elbows = env._demo_gait_actions()
        env.demo_gait_step += 1
        return shoulders, elbows
