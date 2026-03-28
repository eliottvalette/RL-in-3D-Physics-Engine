import unittest
import numpy as np

from physics_env.core.config import MAX_CONSECUTIVE_JOINT_LIMIT_STEPS, STATE_SIZE
from physics_env.envs.quadruped_env import QuadrupedEnv


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


if __name__ == "__main__":
    unittest.main()
