import unittest

import numpy as np

from physics_env.core.config import (
    CRITICAL_TILT_ANGLE,
    JOINT_LIMIT_THRESHOLD,
    MAX_CONSECUTIVE_CRITICAL_TILT_STEPS,
    MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
    TERMINAL_PENALTY_CRITICAL_TILT,
    TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT,
    TERMINAL_PENALTY_TOO_HIGH,
)
from physics_env.envs.quadruped_env import QuadrupedEnv


class TerminalRewardsTest(unittest.TestCase):
    def test_too_high_applies_terminal_penalty(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.quadruped.position[1] = 6.2
        env.consecutive_steps_above_critical_height = 21

        _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "too_high")
        self.assertAlmostEqual(env.last_reward_components["terminal_event_reward"], TERMINAL_PENALTY_TOO_HIGH, places=6)
        self.assertLessEqual(reward, TERMINAL_PENALTY_TOO_HIGH + 10.0)

    def test_critical_tilt_applies_terminal_penalty(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.quadruped.orientation = env.quadruped._euler_to_quaternion(np.array([0.0, 0.0, CRITICAL_TILT_ANGLE + 0.2]))
        env.quadruped.sync_euler_from_orientation()
        env.consecutive_steps_critical_tilt = MAX_CONSECUTIVE_CRITICAL_TILT_STEPS

        _, _, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "critical_tilt")
        self.assertAlmostEqual(env.last_reward_components["terminal_event_reward"], TERMINAL_PENALTY_CRITICAL_TILT, places=6)

    def test_joint_limit_timeout_applies_terminal_penalty(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.quadruped.shoulder_angles[0] = JOINT_LIMIT_THRESHOLD + 0.05
        env.consecutive_shoulder_limit_steps[0] = MAX_CONSECUTIVE_JOINT_LIMIT_STEPS

        _, _, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "joint_limit_timeout")
        self.assertAlmostEqual(env.last_reward_components["terminal_event_reward"], TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT, places=6)


if __name__ == "__main__":
    unittest.main()
