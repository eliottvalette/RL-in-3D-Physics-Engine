import unittest

import numpy as np

from physics_env.core.config import ELBOW_DELTA, MOTOR_STOP_EPS, SHOULDER_DELTA
from physics_env.quadruped.quadruped import Quadruped
from physics_env.quadruped.quadruped_points import create_quadruped_vertices, get_quadruped_vertices


def make_quadruped():
    vertices_dict = create_quadruped_vertices()
    return Quadruped(
        position=np.array([0.0, 5.5, 0.0], dtype=np.float64),
        vertices=get_quadruped_vertices(),
        vertices_dict=vertices_dict,
    )


class JointMotorDynamicsTest(unittest.TestCase):
    def test_idle_command_brakes_shoulder_velocity(self):
        quadruped = make_quadruped()

        for _ in range(8):
            quadruped.adjust_shoulder_angle(0, SHOULDER_DELTA)

        driven_velocity = float(quadruped.shoulder_velocities[0])
        self.assertGreater(driven_velocity, 0.0)

        for _ in range(12):
            quadruped.adjust_shoulder_angle(0, 0.0)

        braked_velocity = float(quadruped.shoulder_velocities[0])
        self.assertLess(abs(braked_velocity), abs(driven_velocity))
        self.assertGreaterEqual(braked_velocity, 0.0)

        for _ in range(40):
            quadruped.adjust_shoulder_angle(0, 0.0)

        self.assertLessEqual(abs(float(quadruped.shoulder_velocities[0])), MOTOR_STOP_EPS)

    def test_reverse_command_crosses_zero_progressively(self):
        quadruped = make_quadruped()

        for _ in range(8):
            quadruped.adjust_elbow_angle(0, ELBOW_DELTA)

        initial_forward_velocity = float(quadruped.elbow_velocities[0])
        self.assertGreater(initial_forward_velocity, 0.0)

        quadruped.adjust_elbow_angle(0, -ELBOW_DELTA)
        first_reverse_velocity = float(quadruped.elbow_velocities[0])
        self.assertLess(first_reverse_velocity, initial_forward_velocity)
        self.assertGreater(first_reverse_velocity, -ELBOW_DELTA)

        for _ in range(10):
            quadruped.adjust_elbow_angle(0, -ELBOW_DELTA)

        self.assertLess(float(quadruped.elbow_velocities[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
