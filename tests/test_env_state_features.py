import unittest
import numpy as np

from physics_env.core.config import (
    MAX_AIRBORNE_STEPS,
    MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
    RESET_JOINT_ANGLE_JITTER,
    RESET_VERTICAL_AXIS_ROTATION_JITTER,
    STATE_SIZE,
)
from physics_env.envs.quadruped_env import QuadrupedEnv
from physics_env.quadruped.quadruped import QUADRUPED_TOTAL_MASS, Quadruped
from physics_env.quadruped.quadruped_points import create_quadruped_vertices, get_quadruped_vertices


class EnvStateFeaturesTest(unittest.TestCase):
    def test_state_exposes_joint_progress_actions_and_contact_history(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=True)

        env.consecutive_shoulder_limit_steps[:] = [0, 10, MAX_CONSECUTIVE_JOINT_LIMIT_STEPS, MAX_CONSECUTIVE_JOINT_LIMIT_STEPS + 30]
        env.consecutive_elbow_limit_steps[:] = [5, 25, 40, MAX_CONSECUTIVE_JOINT_LIMIT_STEPS]
        env.prev_action = np.array([-1.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0], dtype=np.float32)
        env.steps_since_contact = 3
        env.has_had_ground_contact = True
        env.steps_since_foot_contact[:] = [0, 2, 10, 1]
        env.has_had_foot_ground_contact[:] = [True, True, False, True]
        env.quadruped.active_contact_indices = np.array([40, 65], dtype=np.int64)

        components = env.get_state_components()
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

        for actual, expected in zip(components["joint_limit_progress"], expected_progress):
            self.assertAlmostEqual(actual, expected, places=6)

        for actual, expected in zip(components["prev_action"], env.prev_action.tolist()):
            self.assertAlmostEqual(actual, expected, places=6)

        np.testing.assert_allclose(components["foot_contact"], np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(
            components["steps_since_foot_contact_norm"],
            np.array([0.0, 2 / 5, 1.0, 1 / 5], dtype=np.float32),
        )
        self.assertAlmostEqual(float(components["steps_since_contact_norm"][0]), 3 / MAX_AIRBORNE_STEPS, places=6)
        self.assertAlmostEqual(float(components["grounded_recently"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(components["has_had_ground_contact"][0]), 1.0, places=6)

    def test_foot_position_features_use_lower_leg_bottom_face_centers(self):
        vertices = np.zeros((72, 3), dtype=np.float64)
        vertices_dict = create_quadruped_vertices()
        quadruped = Quadruped(
            position=np.array([0.0, 5.5, 0.0], dtype=np.float64),
            vertices=get_quadruped_vertices(),
            vertices_dict=vertices_dict,
        )

        vertices[40:44] = np.array(
            [
                [0.0, -4.0, 1.0],
                [0.0, -4.0, 3.0],
                [2.0, -2.0, 1.0],
                [2.0, -2.0, 3.0],
            ],
            dtype=np.float64,
        )
        quadruped.local_transformed_vertices = vertices.copy()
        quadruped.rotated_vertices = vertices.copy()
        quadruped.last_local_articulation_velocities = np.zeros_like(vertices)
        quadruped._needs_update = False

        components = quadruped.get_state_components()
        first_foot_position = components["foot_positions_body"][:3]
        first_foot_velocity = components["foot_velocities_body"][:3]
        first_foot_height = float(components["foot_heights_world"][0])

        np.testing.assert_allclose(first_foot_position, np.array([1.0, -3.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(first_foot_velocity, np.zeros(3, dtype=np.float32))
        self.assertAlmostEqual(first_foot_height, -3.0, places=6)

    def test_body_frame_features_are_semantic(self):
        vertices_dict = create_quadruped_vertices()
        quadruped = Quadruped(
            position=np.array([0.0, 5.5, 0.0], dtype=np.float64),
            vertices=get_quadruped_vertices(),
            vertices_dict=vertices_dict,
            velocity=np.array([1.0, 2.0, -3.0], dtype=np.float64),
        )

        quadruped.angular_velocity = np.array([0.4, -0.2, 0.1], dtype=np.float64)
        components = quadruped.get_state_components()
        expected_com_velocity = quadruped.get_center_of_mass_velocity()
        expected_task_velocity = np.array(
            [expected_com_velocity[0], expected_com_velocity[1], -expected_com_velocity[2]],
            dtype=np.float32,
        )

        self.assertAlmostEqual(float(components["body_height"][0]), 5.5, places=6)
        self.assertAlmostEqual(float(components["body_height_error"][0]), 0.5, places=6)
        np.testing.assert_allclose(components["gravity_body"], np.array([0.0, -1.0, 0.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(components["task_forward_body"], np.array([0.0, 0.0, -1.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(components["linear_velocity_task"], expected_task_velocity, atol=1e-6)
        np.testing.assert_allclose(
            components["joint_limit_fraction"],
            np.zeros(8, dtype=np.float32),
            atol=1e-6,
        )

    def test_part_masses_are_derived_from_uniform_density(self):
        vertices_dict = create_quadruped_vertices()
        quadruped = Quadruped(
            position=np.array([0.0, 5.5, 0.0], dtype=np.float64),
            vertices=get_quadruped_vertices(),
            vertices_dict=vertices_dict,
        )

        part_volumes = np.array([float(np.prod(dimensions)) for dimensions in quadruped.part_dimensions])
        part_densities = quadruped.part_masses / part_volumes

        self.assertAlmostEqual(quadruped.mass, QUADRUPED_TOTAL_MASS, places=6)
        np.testing.assert_allclose(part_densities, np.full_like(part_densities, part_densities[0]))
        self.assertGreater(quadruped.part_masses[0], quadruped.part_masses[1])

    def test_pose_jitter_reset_randomizes_joints_without_velocity_kick(self):
        env = QuadrupedEnv(rendering=False, headless=True, bench_mode=True)
        env.reset_episode(pose_jitter=True)

        np.testing.assert_allclose(env.quadruped.position, env.quadruped.initial_position)
        np.testing.assert_allclose(env.quadruped.velocity, env.quadruped.initial_velocity)
        np.testing.assert_allclose(env.quadruped.angular_velocity, env.quadruped.initial_angular_velocity)
        np.testing.assert_allclose(env.quadruped.shoulder_velocities, np.zeros(4))
        np.testing.assert_allclose(env.quadruped.elbow_velocities, np.zeros(4))

        self.assertLessEqual(float(np.abs(env.quadruped.shoulder_angles).max()), RESET_JOINT_ANGLE_JITTER)
        self.assertLessEqual(float(np.abs(env.quadruped.elbow_angles).max()), RESET_JOINT_ANGLE_JITTER)
        self.assertAlmostEqual(float(env.quadruped.rotation[0]), 0.0, places=6)
        self.assertLessEqual(abs(float(env.quadruped.rotation[1])), RESET_VERTICAL_AXIS_ROTATION_JITTER)
        self.assertAlmostEqual(float(env.quadruped.rotation[2]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
