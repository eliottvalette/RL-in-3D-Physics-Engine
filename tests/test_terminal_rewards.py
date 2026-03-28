import unittest
from unittest.mock import patch

import numpy as np

from physics_env.core.config import (
    CRITICAL_TILT_ANGLE,
    HEIGHT_REWARD_DECAY_MARGIN,
    JOINT_LIMIT_THRESHOLD,
    MAX_BODY_HEIGHT,
    MAX_CONSECUTIVE_CRITICAL_TILT_STEPS,
    MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
    TERMINAL_PENALTY_CRITICAL_TILT,
    TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT,
    TERMINAL_PENALTY_TOO_HIGH,
    TILT_NO_REWARD_ANGLE,
)
from physics_env.envs.quadruped_env import QuadrupedEnv


class TerminalRewardsTest(unittest.TestCase):
    def test_too_high_applies_terminal_penalty_without_ending_episode(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.quadruped.position[1] = 6.2
        env.consecutive_steps_above_critical_height = 21

        with patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None):
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_done_reason, "too_high")
        self.assertAlmostEqual(env.last_reward_components["terminal_event_reward"], TERMINAL_PENALTY_TOO_HIGH, places=6)
        self.assertLessEqual(reward, TERMINAL_PENALTY_TOO_HIGH + 10.0)

    def test_critical_tilt_applies_terminal_penalty_without_ending_episode(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.quadruped.orientation = env.quadruped._euler_to_quaternion(np.array([0.0, 0.0, CRITICAL_TILT_ANGLE + 0.2]))
        env.quadruped.sync_euler_from_orientation()
        env.consecutive_steps_critical_tilt = MAX_CONSECUTIVE_CRITICAL_TILT_STEPS

        with patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None):
            _, _, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_done_reason, "critical_tilt")
        self.assertAlmostEqual(env.last_reward_components["terminal_event_reward"], TERMINAL_PENALTY_CRITICAL_TILT, places=6)

    def test_joint_limit_timeout_applies_terminal_penalty_without_ending_episode(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.quadruped.shoulder_angles[0] = JOINT_LIMIT_THRESHOLD + 0.05
        env.consecutive_shoulder_limit_steps[0] = MAX_CONSECUTIVE_JOINT_LIMIT_STEPS

        with patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None):
            _, _, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_done_reason, "joint_limit_timeout")
        self.assertAlmostEqual(env.last_reward_components["terminal_event_reward"], TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT, places=6)

    def test_positive_rewards_are_scaled_down_smoothly_when_tilt_grows(self):
        upright_env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        high_tilt_env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)

        for env in (upright_env, high_tilt_env):
            env.prev_potential = 0.0
            env.quadruped.position[2] = -0.02
            env.quadruped.velocity[2] = -0.5

        high_tilt_env.quadruped.orientation = high_tilt_env.quadruped._euler_to_quaternion(
            np.array([0.0, 0.0, TILT_NO_REWARD_ANGLE + 0.05])
        )
        high_tilt_env.quadruped.sync_euler_from_orientation()

        with patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None):
            _, upright_reward, upright_done, _ = upright_env.step([0, 0, 0, 0], [0, 0, 0, 0])
            _, high_tilt_reward, high_tilt_done, _ = high_tilt_env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(upright_done)
        self.assertFalse(high_tilt_done)

        upright_components = upright_env.last_reward_components
        high_tilt_components = high_tilt_env.last_reward_components

        self.assertGreater(upright_components["distance_reward"], high_tilt_components["distance_reward"])
        self.assertGreater(high_tilt_components["distance_reward"], 0.0)
        self.assertGreater(upright_components["z_speed_reward"], high_tilt_components["z_speed_reward"])
        self.assertGreater(high_tilt_components["z_speed_reward"], 0.0)
        self.assertGreater(high_tilt_components["tilt_reward_scale"], 0.0)
        self.assertLess(high_tilt_components["tilt_reward_scale"], 1.0)
        self.assertGreater(upright_components["locomotion_reward_scale"], high_tilt_components["locomotion_reward_scale"])
        self.assertLess(high_tilt_reward, upright_reward)

    def test_positive_rewards_are_scaled_down_smoothly_when_height_drifts(self):
        upright_env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        high_env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)

        for env in (upright_env, high_env):
            env.prev_potential = 0.0
            env.quadruped.position[2] = -0.02
            env.quadruped.velocity[2] = -0.5

        high_env.quadruped.position[1] = MAX_BODY_HEIGHT + HEIGHT_REWARD_DECAY_MARGIN * 0.4

        with patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None):
            _, upright_reward, upright_done, _ = upright_env.step([0, 0, 0, 0], [0, 0, 0, 0])
            _, high_reward, high_done, _ = high_env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(upright_done)
        self.assertFalse(high_done)

        upright_components = upright_env.last_reward_components
        high_components = high_env.last_reward_components

        self.assertEqual(upright_components["height_reward_scale"], 1.0)
        self.assertGreater(high_components["height_reward_scale"], 0.0)
        self.assertLess(high_components["height_reward_scale"], 1.0)
        self.assertGreater(upright_components["distance_reward"], high_components["distance_reward"])
        self.assertGreater(high_components["distance_reward"], 0.0)
        self.assertGreater(upright_components["z_speed_reward"], high_components["z_speed_reward"])
        self.assertGreater(high_components["z_speed_reward"], 0.0)
        self.assertGreater(upright_components["locomotion_reward_scale"], high_components["locomotion_reward_scale"])
        self.assertLess(high_reward, upright_reward)

    def test_backward_motion_does_not_earn_forward_progress_or_checkpoints(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.8
        env.quadruped.velocity[2] = 1.0

        with patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None):
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_reward_components["distance_reward"], 0.0)
        self.assertEqual(env.last_reward_components["sparse_reward"], 0.0)
        self.assertEqual(env.last_reward_components["z_speed_reward"], 0.0)
        self.assertLessEqual(reward, 0.0)

    def test_action_smoothness_penalizes_large_action_jumps(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_action[:] = -1.0

        with patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None):
            _, reward, done, _ = env.step([1, 1, 1, 1], [1, 1, 1, 1])

        self.assertFalse(done)
        self.assertLess(env.last_reward_components["action_smoothness_penalty"], 0.0)
        self.assertLess(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
