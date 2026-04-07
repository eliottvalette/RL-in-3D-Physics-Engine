"""Free-space angular integration benchmark."""

from ..scenario_utils import BenchScenario, reset_env_state, set_base_state, zero_actions


class AirSpinScenario(BenchScenario):
    name = "air_spin"
    category = "free_dynamics"
    description = "Quadruped spun in free space to isolate orientation integration."

    def reset(self, env):
        reset_env_state(env)
        set_base_state(env, position=[0.0, 12.0, 0.0], rotation=[0.2, -0.1, 0.15], velocity=[0.0, 0.0, 0.0], angular_velocity=[1.5, 0.8, 1.2])

    def actions(self, env, step_idx):
        del env, step_idx
        return zero_actions()

    def should_stop(self, env, step_idx, done):
        del env, done
        return step_idx >= 119
