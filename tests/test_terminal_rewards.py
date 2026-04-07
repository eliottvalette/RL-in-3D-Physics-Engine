import unittest
from unittest.mock import patch

import numpy as np

from physics_env.core.config import (
    JOINT_LIMIT_THRESHOLD,
    MAX_BODY_HEIGHT,
    MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
    MIN_BODY_HEIGHT,
    PROGRESS_REWARD_COEF,
    CRITICAL_TILT_ANGLE,
    TERMINAL_PENALTY_CRITICAL_TILT,
    TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT,
    TERMINAL_PENALTY_TOO_HIGH,
    TERMINAL_PENALTY_TOO_LOW,
    TILT_NO_REWARD_ANGLE,
)
from physics_env.envs.quadruped_env import QuadrupedEnv


def _freeze_physics():
    return patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None)


class TerminalRewardsTest(unittest.TestCase):
    def test_too_low_is_terminal_and_does_not_pay_locomotion(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[1] = MIN_BODY_HEIGHT - 0.1
        env.quadruped.position[2] = -0.5

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "too_low")
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_TOO_LOW, places=6)
        self.assertEqual(env.last_reward_components["locomotion_reward"], 0.0)

    def test_too_high_is_terminal_and_does_not_pay_locomotion(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[1] = MAX_BODY_HEIGHT + 0.1
        env.quadruped.position[2] = -0.5

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "too_high")
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_TOO_HIGH, places=6)
        self.assertEqual(env.last_reward_components["locomotion_reward"], 0.0)

    def test_excess_tilt_is_terminal_and_does_not_pay_locomotion(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = -0.5
        env.quadruped.orientation = env.quadruped._euler_to_quaternion(
            np.array([0.0, 0.0, CRITICAL_TILT_ANGLE + 0.05])
        )
        env.quadruped.sync_euler_from_orientation()

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "critical_tilt")
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_CRITICAL_TILT, places=6)
        self.assertEqual(env.last_reward_components["locomotion_reward"], 0.0)

    def test_tilt_between_pose_and_critical_threshold_is_not_terminal(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = -0.02
        env.quadruped.orientation = env.quadruped._euler_to_quaternion(
            np.array([0.0, 0.0, TILT_NO_REWARD_ANGLE + 0.05])
        )
        env.quadruped.sync_euler_from_orientation()

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_done_reason, "running")
        self.assertAlmostEqual(reward, PROGRESS_REWARD_COEF * 0.02, places=6)

    def test_terminal_failure_keeps_only_terminal_penalty(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.cumulative_locomotion_reward = 12.0
        env.prev_potential = 0.0
        env.quadruped.position[2] = -0.5
        env.quadruped.orientation = env.quadruped._euler_to_quaternion(
            np.array([0.0, 0.0, CRITICAL_TILT_ANGLE + 0.05])
        )
        env.quadruped.sync_euler_from_orientation()

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertNotIn("failure_clawback", env.last_reward_components)
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_CRITICAL_TILT, places=6)

    def test_joint_limit_timeout_is_terminal_and_does_not_pay_locomotion(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = -0.5
        env.quadruped.shoulder_angles[0] = JOINT_LIMIT_THRESHOLD + 0.05
        env.consecutive_shoulder_limit_steps[0] = MAX_CONSECUTIVE_JOINT_LIMIT_STEPS

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "joint_limit_timeout")
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT, places=6)
        self.assertEqual(env.last_reward_components["locomotion_reward"], 0.0)

    def test_forward_progress_is_the_only_positive_locomotion_reward(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = -0.02
        env.quadruped.velocity[2] = -0.5

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_done_reason, "running")
        self.assertGreater(env.last_reward_components["distance_reward"], 0.0)
        self.assertAlmostEqual(
            env.last_reward_components["distance_reward"],
            PROGRESS_REWARD_COEF * 0.02,
            places=6,
        )
        self.assertGreater(env.last_reward_components["forward_speed_signal"], 0.0)
        self.assertEqual(env.last_reward_components["z_speed_reward"], 0.0)
        self.assertAlmostEqual(reward, env.last_reward_components["distance_reward"], places=6)

    def test_forward_speed_without_position_progress_does_not_pay(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.0
        env.quadruped.velocity[2] = -1.0

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_reward_components["distance_reward"], 0.0)
        self.assertEqual(env.last_reward_components["z_speed_reward"], 0.0)
        self.assertGreater(env.last_reward_components["forward_speed_signal"], 0.0)
        self.assertEqual(reward, 0.0)

    def test_backward_motion_gets_negative_progress_reward(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.8
        env.quadruped.velocity[2] = 1.0

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertLess(env.last_reward_components["distance_reward"], 0.0)
        self.assertEqual(env.last_reward_components["sparse_reward"], 0.0)
        self.assertEqual(env.last_reward_components["z_speed_reward"], 0.0)
        self.assertLess(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
