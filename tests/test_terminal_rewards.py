import unittest
from unittest.mock import patch

import numpy as np

from physics_env.core.config import (
    ACTION_CHANGE_PENALTY_COEF,
    ANGULAR_VELOCITY_PENALTY_COEF,
    HEIGHT_SOFT_REWARD_MARGIN,
    CONTACT_QUALITY_SCALE_FLOOR,
    CRITICAL_TILT_ANGLE,
    DIAGONAL_GAIT_REWARD_MAX,
    DT,
    FOOT_UNUSED_GRACE_STEPS,
    FOOT_UNUSED_MAX_PENALTY,
    FOOT_UNUSED_WINDOW_STEPS,
    FOOT_SLIP_MAX_PENALTY,
    FORWARD_SPEED_REWARD_MAX,
    FORWARD_SPEED_TARGET_M_S,
    TERMINAL_PENALTY_AIRBORNE,
    FOOT_SLIP_PENALTY_COEF,
    FOOT_SLIP_SPEED_THRESHOLD,
    MAX_AIRBORNE_STEPS,
    MAX_BODY_HEIGHT,
    MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
    MIN_BODY_HEIGHT,
    NON_DIAGONAL_SUPPORT_PENALTY_MAX,
    PROGRESS_REWARD_COEF,
    SHOULDER_ANGLE_MAX,
    SWING_CLEARANCE_REWARD_MAX,
    SWING_CLEARANCE_TARGET_UNITS,
    TERMINAL_PENALTY_CRITICAL_TILT,
    TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT,
    TERMINAL_PENALTY_TOO_HIGH,
    TERMINAL_PENALTY_TOO_LOW,
    TILT_SOFT_REWARD_MARGIN,
    UNIT_SCALE_M,
)
from physics_env.envs.quadruped_env import QuadrupedEnv


def _freeze_physics():
    return patch("physics_env.envs.quadruped_env.update_quadruped", lambda quadruped: None)


def _set_ground_contact(env):
    env.quadruped.active_contact_indices = np.array([0], dtype=np.int64)


def _set_two_foot_contacts(env):
    env.quadruped.active_contact_indices = np.array([40, 48], dtype=np.int64)


def _set_diagonal_foot_contacts(env):
    env.quadruped.active_contact_indices = np.array([40, 64], dtype=np.int64)


def _forward_speed_reward_scale(progress_delta_units: float) -> float:
    speed_m_s = max(0.0, progress_delta_units) / DT * UNIT_SCALE_M
    return min(1.0, speed_m_s / FORWARD_SPEED_TARGET_M_S)


def _target_progress_scale(progress_delta_units: float) -> float:
    speed_scale = _forward_speed_reward_scale(progress_delta_units)
    return CONTACT_QUALITY_SCALE_FLOOR + (1.0 - CONTACT_QUALITY_SCALE_FLOOR) * speed_scale


class TerminalRewardsTest(unittest.TestCase):
    def test_too_low_is_terminal_and_does_not_pay_locomotion(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[1] = MIN_BODY_HEIGHT - 0.1
        env.quadruped.position[2] = 0.5

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
        env.quadruped.position[2] = 0.5

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "too_high")
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_TOO_HIGH, places=6)
        self.assertEqual(env.last_reward_components["locomotion_reward"], 0.0)

    def test_excess_tilt_is_terminal_and_does_not_pay_locomotion(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.5
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
        _set_ground_contact(env)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.02
        env.quadruped.orientation = env.quadruped._euler_to_quaternion(
            np.array([0.0, 0.0, np.deg2rad(10.0) + 0.05])
        )
        env.quadruped.sync_euler_from_orientation()

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_done_reason, "running")
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, PROGRESS_REWARD_COEF * 0.02)

    def test_near_critical_tilt_softly_reduces_progress_reward(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_two_foot_contacts(env)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.02
        target_tilt = CRITICAL_TILT_ANGLE - 0.5 * TILT_SOFT_REWARD_MARGIN
        env.quadruped.orientation = env.quadruped._euler_to_quaternion(
            np.array([0.0, 0.0, target_tilt])
        )
        env.quadruped.sync_euler_from_orientation()

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertAlmostEqual(env.last_reward_components["tilt_reward_scale"], 0.5, places=6)
        expected_target_scale = _target_progress_scale(0.02)
        self.assertAlmostEqual(
            env.last_reward_components["target_progress_scale"],
            expected_target_scale,
            places=6,
        )
        self.assertAlmostEqual(
            env.last_reward_components["locomotion_reward"],
            PROGRESS_REWARD_COEF * 0.01 * expected_target_scale,
            places=6,
        )
        expected_speed_scale = _forward_speed_reward_scale(0.02)
        self.assertAlmostEqual(
            env.last_reward_components["forward_speed_reward"],
            0.5 * expected_speed_scale * FORWARD_SPEED_REWARD_MAX,
            places=6,
        )
        self.assertAlmostEqual(
            env.last_reward_components["non_diagonal_support_penalty"],
            -0.5 * expected_speed_scale * NON_DIAGONAL_SUPPORT_PENALTY_MAX,
            places=6,
        )
        self.assertAlmostEqual(
            reward,
            PROGRESS_REWARD_COEF * 0.01 * expected_target_scale
            + 0.5 * expected_speed_scale * FORWARD_SPEED_REWARD_MAX
            - 0.5 * expected_speed_scale * NON_DIAGONAL_SUPPORT_PENALTY_MAX,
            places=6,
        )

    def test_terminal_failure_keeps_only_terminal_penalty(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.cumulative_locomotion_reward = 12.0
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.5
        env.quadruped.orientation = env.quadruped._euler_to_quaternion(
            np.array([0.0, 0.0, CRITICAL_TILT_ANGLE + 0.05])
        )
        env.quadruped.sync_euler_from_orientation()

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_CRITICAL_TILT, places=6)

    def test_joint_limit_timeout_is_terminal_and_does_not_pay_locomotion(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.5
        env.quadruped.shoulder_angles[0] = SHOULDER_ANGLE_MAX
        env.consecutive_shoulder_limit_steps[0] = MAX_CONSECUTIVE_JOINT_LIMIT_STEPS

        with _freeze_physics():
            _, reward, done, _ = env.step([1, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "joint_limit_timeout")
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT, places=6)
        self.assertEqual(env.last_reward_components["locomotion_reward"], 0.0)

    def test_forward_progress_and_speed_reward_pay_clean_motion(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_two_foot_contacts(env)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.02
        env.quadruped.velocity[2] = 0.5

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
        self.assertAlmostEqual(env.last_reward_components["forward_speed"], 0.5, places=6)
        expected_speed_scale = _forward_speed_reward_scale(0.02)
        self.assertAlmostEqual(
            env.last_reward_components["forward_speed_reward"],
            expected_speed_scale * FORWARD_SPEED_REWARD_MAX,
            places=6,
        )
        self.assertAlmostEqual(
            env.last_reward_components["non_diagonal_support_penalty"],
            -expected_speed_scale * NON_DIAGONAL_SUPPORT_PENALTY_MAX,
            places=6,
        )
        self.assertAlmostEqual(
            reward,
            env.last_reward_components["locomotion_reward"]
            + env.last_reward_components["forward_speed_reward"]
            + env.last_reward_components["non_diagonal_support_penalty"],
            places=6,
        )

    def test_diagonal_support_adds_gait_quality_reward(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_diagonal_foot_contacts(env)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.02
        foot_centers_world = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, SWING_CLEARANCE_TARGET_UNITS, 0.0],
                [0.0, SWING_CLEARANCE_TARGET_UNITS, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        with patch.object(env.quadruped, "get_foot_centers_world", return_value=foot_centers_world):
            with _freeze_physics():
                _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertAlmostEqual(env.last_reward_components["diagonal_contact_score"], 1.0, places=6)
        self.assertAlmostEqual(env.last_reward_components["swing_clearance_score"], 1.0, places=6)
        expected_speed_scale = _forward_speed_reward_scale(0.02)
        self.assertAlmostEqual(
            env.last_reward_components["forward_speed_reward"],
            expected_speed_scale * FORWARD_SPEED_REWARD_MAX,
            places=6,
        )
        self.assertAlmostEqual(
            env.last_reward_components["diagonal_gait_reward"],
            expected_speed_scale * DIAGONAL_GAIT_REWARD_MAX,
            places=6,
        )
        self.assertAlmostEqual(
            env.last_reward_components["swing_clearance_reward"],
            expected_speed_scale * SWING_CLEARANCE_REWARD_MAX,
            places=6,
        )
        self.assertEqual(env.last_reward_components["non_diagonal_support_penalty"], 0.0)
        self.assertAlmostEqual(
            reward,
            PROGRESS_REWARD_COEF * 0.02 * _target_progress_scale(0.02)
            + expected_speed_scale * FORWARD_SPEED_REWARD_MAX
            + expected_speed_scale * DIAGONAL_GAIT_REWARD_MAX
            + expected_speed_scale * SWING_CLEARANCE_REWARD_MAX,
            places=6,
        )

    def test_near_low_height_limit_softly_reduces_progress_reward(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_two_foot_contacts(env)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.02
        env.quadruped.position[1] = MIN_BODY_HEIGHT + 0.5 * HEIGHT_SOFT_REWARD_MARGIN

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertAlmostEqual(env.last_reward_components["height_reward_scale"], 0.5, places=6)
        expected_target_scale = _target_progress_scale(0.02)
        self.assertAlmostEqual(
            env.last_reward_components["locomotion_reward"],
            PROGRESS_REWARD_COEF * 0.01 * expected_target_scale,
            places=6,
        )
        expected_speed_scale = _forward_speed_reward_scale(0.02)
        self.assertAlmostEqual(
            reward,
            PROGRESS_REWARD_COEF * 0.01 * expected_target_scale
            + 0.5 * expected_speed_scale * FORWARD_SPEED_REWARD_MAX
            - 0.5 * expected_speed_scale * NON_DIAGONAL_SUPPORT_PENALTY_MAX,
            places=6,
        )

    def test_forward_progress_without_recent_contact_does_not_pay(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.02
        env.quadruped.velocity[2] = 0.5

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_done_reason, "running")
        self.assertAlmostEqual(env.last_reward_components["raw_distance_reward"], 0.02, places=6)
        self.assertEqual(env.last_reward_components["distance_reward"], 0.0)
        self.assertEqual(env.last_reward_components["contact_reward_scale"], 0.0)
        self.assertEqual(reward, 0.0)

    def test_airborne_timeout_is_terminal_after_first_ground_contact(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.has_had_ground_contact = True
        env.steps_since_contact = MAX_AIRBORNE_STEPS
        env.prev_potential = 0.0

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertTrue(done)
        self.assertEqual(env.last_done_reason, "airborne")
        self.assertAlmostEqual(reward, TERMINAL_PENALTY_AIRBORNE, places=6)
        self.assertEqual(env.last_reward_components["locomotion_reward"], 0.0)

    def test_angular_velocity_penalty_discourages_body_oscillation(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_two_foot_contacts(env)
        env.prev_potential = 0.0
        env.quadruped.angular_velocity = np.array([0.0, 2.0, 0.0], dtype=np.float64)

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_done_reason, "running")
        self.assertAlmostEqual(env.last_reward_components["angular_velocity_norm"], 2.0, places=6)
        self.assertAlmostEqual(
            env.last_reward_components["angular_velocity_penalty"],
            -ANGULAR_VELOCITY_PENALTY_COEF * 2.0,
            places=6,
        )
        self.assertAlmostEqual(reward, -ANGULAR_VELOCITY_PENALTY_COEF * 2.0, places=6)

    def test_single_leg_support_reduces_locomotion_quality_without_blocking_progress(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_ground_contact(env)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.02

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertAlmostEqual(env.last_reward_components["contact_quality_scale"], CONTACT_QUALITY_SCALE_FLOOR, places=6)
        self.assertAlmostEqual(
            env.last_reward_components["locomotion_reward"],
            PROGRESS_REWARD_COEF * 0.02 * CONTACT_QUALITY_SCALE_FLOOR * _target_progress_scale(0.02),
            places=6,
        )
        self.assertAlmostEqual(
            env.last_reward_components["forward_speed_reward"],
            CONTACT_QUALITY_SCALE_FLOOR * _forward_speed_reward_scale(0.02) * FORWARD_SPEED_REWARD_MAX,
            places=6,
        )
        self.assertEqual(env.last_reward_components["non_diagonal_support_penalty"], 0.0)
        self.assertAlmostEqual(
            reward,
            env.last_reward_components["locomotion_reward"]
            + env.last_reward_components["forward_speed_reward"],
            places=6,
        )

    def test_action_change_penalty_discourages_command_dithering(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_two_foot_contacts(env)
        env.prev_action = np.ones(8, dtype=np.float32)
        env.prev_potential = 0.0

        with _freeze_physics():
            _, reward, done, _ = env.step([-1, -1, -1, -1], [-1, -1, -1, -1])

        self.assertFalse(done)
        self.assertAlmostEqual(env.last_reward_components["action_delta_mean_abs"], 2.0, places=6)
        self.assertAlmostEqual(env.last_reward_components["action_sign_flip_rate"], 1.0, places=6)
        self.assertAlmostEqual(
            env.last_reward_components["action_change_penalty"],
            -ACTION_CHANGE_PENALTY_COEF * 2.0,
            places=6,
        )
        self.assertAlmostEqual(reward, env.last_reward_components["action_change_penalty"], places=6)

    def test_action_change_penalty_does_not_tax_useful_start_stop(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_two_foot_contacts(env)
        env.prev_action = np.zeros(8, dtype=np.float32)
        env.prev_potential = 0.0

        with _freeze_physics():
            _, reward, done, _ = env.step([1, 1, 1, 1], [1, 1, 1, 1])

        self.assertFalse(done)
        self.assertAlmostEqual(env.last_reward_components["action_delta_mean_abs"], 1.0, places=6)
        self.assertAlmostEqual(env.last_reward_components["action_sign_flip_rate"], 0.0, places=6)
        self.assertEqual(env.last_reward_components["action_change_penalty"], 0.0)
        self.assertEqual(reward, 0.0)

    def test_foot_unused_penalty_starts_after_grace(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_two_foot_contacts(env)
        env.prev_potential = 0.0
        env.consecutive_unused_foot_steps[:] = FOOT_UNUSED_GRACE_STEPS

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        expected_normalized_excess_mean = 0.5 / FOOT_UNUSED_WINDOW_STEPS
        self.assertFalse(done)
        self.assertAlmostEqual(
            env.last_reward_components["foot_unused_penalty"],
            -FOOT_UNUSED_MAX_PENALTY * expected_normalized_excess_mean,
            places=6,
        )
        self.assertAlmostEqual(reward, env.last_reward_components["foot_unused_penalty"], places=6)

    def test_foot_slip_penalty_uses_contact_foot_planar_speed(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.quadruped.active_contact_indices = np.array([40, 56], dtype=np.int64)
        env.prev_potential = 0.0

        with patch.object(
            env.quadruped,
            "get_foot_center_velocities_local",
            return_value=np.array(
                [
                    [FOOT_SLIP_SPEED_THRESHOLD + 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [FOOT_SLIP_SPEED_THRESHOLD + 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        ):
            with _freeze_physics():
                _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertAlmostEqual(
            env.last_reward_components["foot_slip_penalty"],
            -min(FOOT_SLIP_MAX_PENALTY, FOOT_SLIP_PENALTY_COEF * 2.0),
            places=6,
        )
        self.assertAlmostEqual(reward, env.last_reward_components["foot_slip_penalty"], places=6)

    def test_forward_speed_without_position_progress_does_not_pay(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        env.prev_potential = 0.0
        env.quadruped.position[2] = 0.0
        env.quadruped.velocity[2] = 1.0

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertEqual(env.last_reward_components["distance_reward"], 0.0)
        self.assertAlmostEqual(env.last_reward_components["forward_speed"], 1.0, places=6)
        self.assertEqual(reward, 0.0)

    def test_backward_motion_gets_negative_progress_reward(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=False)
        _set_ground_contact(env)
        env.prev_potential = 0.0
        env.quadruped.position[2] = -0.8
        env.quadruped.velocity[2] = -1.0

        with _freeze_physics():
            _, reward, done, _ = env.step([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertFalse(done)
        self.assertLess(env.last_reward_components["distance_reward"], 0.0)
        self.assertAlmostEqual(env.last_reward_components["forward_speed"], -1.0, places=6)
        self.assertLess(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
