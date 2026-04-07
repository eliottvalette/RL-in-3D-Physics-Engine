import unittest
import numpy as np

from physics_env.core.config import MAX_CONSECUTIVE_JOINT_LIMIT_STEPS, STATE_SIZE
from physics_env.envs.quadruped_env import QuadrupedEnv
from physics_env.quadruped.quadruped import Quadruped
from physics_env.quadruped.quadruped_points import create_quadruped_vertices, get_quadruped_vertices


class EnvStateFeaturesTest(unittest.TestCase):
    def test_joint_limit_progress_and_prev_action_are_appended_to_state(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=True)

        env.consecutive_shoulder_limit_steps[:] = [0, 10, MAX_CONSECUTIVE_JOINT_LIMIT_STEPS, MAX_CONSECUTIVE_JOINT_LIMIT_STEPS + 30]
        env.consecutive_elbow_limit_steps[:] = [5, 25, 40, MAX_CONSECUTIVE_JOINT_LIMIT_STEPS]
        env.prev_action = np.array([-1.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0], dtype=np.float32)

        state = env.get_state()

        self.assertEqual(len(state), STATE_SIZE)

        expected_progress = [
            0.0,
            10 / MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
            1.0,
            1.0,
            5 / MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
            25 / MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
            40 / MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
            1.0,
        ]
        actual_progress = state[-16:-8]
        actual_prev_action = state[-8:]

        for actual, expected in zip(actual_progress, expected_progress):
            self.assertAlmostEqual(actual, expected, places=6)

        for actual, expected in zip(actual_prev_action, env.prev_action.tolist()):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_leg_height_features_use_upper_and_lower_vertices(self):
        vertices = np.zeros((72, 3), dtype=np.float64)
        vertices_dict = create_quadruped_vertices()
        quadruped = Quadruped(
            position=np.array([0.0, 5.5, 0.0], dtype=np.float64),
            vertices=get_quadruped_vertices(),
            vertices_dict=vertices_dict,
        )

        for idx in range(72):
            vertices[idx, 1] = 100.0
        vertices[8:16, 1] = 10.0
        vertices[40:48, 1] = -3.0
        quadruped.rotated_vertices = vertices

        state = quadruped.get_state()
        first_leg_min_y, first_leg_max_y = state[31:33]

        self.assertAlmostEqual(first_leg_min_y, -3.0, places=6)
        self.assertAlmostEqual(first_leg_max_y, 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
