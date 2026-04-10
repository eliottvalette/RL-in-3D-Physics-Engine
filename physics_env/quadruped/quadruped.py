# quadruped.py
import numpy as np
import pygame
from ..core.config import *
from ..rendering.camera import Camera3D
from .quadruped_calibration import INITIAL_ELBOW_ANGLES, INITIAL_SHOULDER_ANGLES
import copy
import math

# --- Mass model ---------------------------------------------------
# The total mass is preserved from the previous hand-tuned distribution:
# 3.0 kg body + 4 * (0.35 kg upper leg + 0.15 kg lower leg) = 5.0 kg.
# Per-part masses are derived from volume with a uniform density.
QUADRUPED_TOTAL_MASS = 5.0

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800


def _quat_normalize(quaternion):
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _clip_angle(angle, angle_min, angle_max):
    return max(float(angle_min), min(float(angle_max), float(angle)))


def _joint_limit_fraction(angles, angle_min, angle_max):
    angles = np.asarray(angles, dtype=np.float64)
    fractions = np.zeros_like(angles, dtype=np.float64)
    positive_mask = angles >= 0.0

    if angle_max > 0.0:
        fractions[positive_mask] = angles[positive_mask] / float(angle_max)
    if angle_min < 0.0:
        fractions[~positive_mask] = angles[~positive_mask] / float(angle_min)

    return np.clip(fractions, 0.0, 1.0)


def _action_edge_color(action_value):
    action_value = float(action_value)
    if action_value > 0.0:
        return (255, 0, 0)
    if action_value < 0.0:
        return (0, 0, 255)
    return (0, 0, 0)


class Quadruped:
    def __init__(self, position, vertices, vertices_dict, rotation = np.array([0.0, 0.0, 0.0]), velocity = np.array([0.0, 0.0, 0.0]), color = (255, 255, 255)):
        self.initial_position = position.copy()
        self.initial_velocity = velocity.copy()
        self.initial_rotation = rotation.copy()
        self.initial_angular_velocity = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_rotation = rotation.copy()
        self.initial_vertices = vertices.copy()
        self.initial_vertices_dict = vertices_dict.copy()
        self.color = color

        self.position = position # position en x, y, z du centre du quadruped
        self.velocity = velocity
        self.rotation = rotation # rotation en radians
        self.orientation = self._euler_to_quaternion(rotation)
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vertices = vertices.copy()
        self.vertices_dict = vertices_dict.copy()
        
        # Get shoulder and elbow positions from vertices_dict if available
        self.shoulder_positions = vertices_dict.get('shoulder_positions', []) if vertices_dict else []
        self.elbow_positions = vertices_dict.get('elbow_positions', []) if vertices_dict else []

        # Articulations (épaules et coudes) - angles en radians
        # Épaules: 0=Front Right, 1=Front Left, 2=Back Right, 3=Back Left
        self.shoulder_angles = INITIAL_SHOULDER_ANGLES.copy()
        self.shoulder_velocities = np.array([0.0, 0.0, 0.0, 0.0])
        # Coudes: 0=Front Right, 1=Front Left, 2=Back Right, 3=Back Left
        self.elbow_angles = INITIAL_ELBOW_ANGLES.copy()
        self.elbow_velocities = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Angles initiaux
        self.initial_shoulder_angles = self.shoulder_angles.copy()
        self.initial_elbow_angles = self.elbow_angles.copy()
        self.initial_shoulder_velocities = self.shoulder_velocities.copy()
        self.initial_elbow_velocities = self.elbow_velocities.copy()
        self.prev_vertices = None
        self.prev_local_transformed_vertices = None
        self.last_local_articulation_velocities = np.zeros((len(self.vertices), 3), dtype=np.float64)
        self.active_contact_indices = np.empty(0, dtype=np.int64)
        self._needs_update = True
        self.rotated_vertices = None
        self.local_transformed_vertices = None
        self.part_local_rotations = []
        self.part_dimensions = self._compute_part_dimensions()
        self.part_masses, self.mass_density = self._compute_uniform_density_masses(self.part_dimensions)
        self.local_center_of_mass = np.zeros(3, dtype=np.float64)
        
        # --- masse & inertie réalistes --------------------------
        self.mass = float(self.part_masses.sum())
        self.I_body = np.eye(3, dtype=np.float64)
        self.rotated_vertices = self.get_vertices()
        self.prev_local_transformed_vertices = self.local_transformed_vertices.copy()
        self.motor_delay =  4

        # Danger zones
        self.too_high = False
        self.too_low = False
        self.steps_since_too_high = 0
        self.steps_since_too_low = 0
    
    def reset_random(self):
        self.position = self.initial_position.copy()
        self.vertices = self.initial_vertices.copy()
        self.velocity = self.initial_velocity.copy() + np.random.rand(3)
        self.rotation = self.initial_rotation.copy() + np.random.rand(3)
        self.orientation = self._euler_to_quaternion(self.rotation)
        self.angular_velocity = self.initial_angular_velocity.copy() + np.random.rand(3)
        self.shoulder_angles = self.initial_shoulder_angles.copy()
        self.elbow_angles = self.initial_elbow_angles.copy()
        self.shoulder_velocities = self.initial_shoulder_velocities.copy()
        self.elbow_velocities = self.initial_elbow_velocities.copy()
        self.prev_vertices = None
        self.prev_local_transformed_vertices = None
        self.last_local_articulation_velocities = np.zeros((len(self.vertices), 3), dtype=np.float64)
        self.active_contact_indices = np.empty(0, dtype=np.int64)
        self._needs_update = True
        self.rotated_vertices = self.get_vertices()
        self.prev_local_transformed_vertices = self.local_transformed_vertices.copy()
    
    def reset(self):
        self.position = self.initial_position.copy()
        self.vertices = self.initial_vertices.copy()
        self.velocity = self.initial_velocity.copy()
        self.rotation = self.initial_rotation.copy()
        self.orientation = self._euler_to_quaternion(self.rotation)
        self.angular_velocity = self.initial_angular_velocity.copy()
        self.shoulder_angles = self.initial_shoulder_angles.copy()
        self.elbow_angles = self.initial_elbow_angles.copy()
        self.shoulder_velocities = self.initial_shoulder_velocities.copy()
        self.elbow_velocities = self.initial_elbow_velocities.copy()
        self.prev_vertices = None
        self.prev_local_transformed_vertices = None
        self.last_local_articulation_velocities = np.zeros((len(self.vertices), 3), dtype=np.float64)
        self.active_contact_indices = np.empty(0, dtype=np.int64)
        self._needs_update = True
        self.rotated_vertices = self.get_vertices()
        self.prev_local_transformed_vertices = self.local_transformed_vertices.copy()

    def reset_pose_jitter(self):
        self.reset()
        self.rotation[1] += np.random.uniform(
            -RESET_VERTICAL_AXIS_ROTATION_JITTER,
            RESET_VERTICAL_AXIS_ROTATION_JITTER,
        )
        self.orientation = self._euler_to_quaternion(self.rotation)
        self.shoulder_angles = np.clip(
            self.initial_shoulder_angles
            + np.random.uniform(
                -RESET_JOINT_ANGLE_JITTER,
                RESET_JOINT_ANGLE_JITTER,
                size=4,
            ).astype(np.float64),
            SHOULDER_ANGLE_MIN,
            SHOULDER_ANGLE_MAX,
        )
        self.elbow_angles = np.clip(
            self.initial_elbow_angles
            + np.random.uniform(
                -RESET_JOINT_ANGLE_JITTER,
                RESET_JOINT_ANGLE_JITTER,
                size=4,
            ).astype(np.float64),
            ELBOW_ANGLE_MIN,
            ELBOW_ANGLE_MAX,
        )
        self.shoulder_velocities = self.initial_shoulder_velocities.copy()
        self.elbow_velocities = self.initial_elbow_velocities.copy()
        self.last_local_articulation_velocities = np.zeros((len(self.vertices), 3), dtype=np.float64)
        self._needs_update = True
        self.rotated_vertices = self.get_vertices()
        self.prev_local_transformed_vertices = self.local_transformed_vertices.copy()

    def get_rotation_matrix(self):
        w, x, y, z = _quat_normalize(self.orientation)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    def get_world_center_of_mass_offset(self):
        return self.get_rotation_matrix() @ self.local_center_of_mass

    def get_world_center_of_mass(self):
        return self.position + self.get_world_center_of_mass_offset()

    def get_center_of_mass_velocity(self):
        return self.velocity + np.cross(self.angular_velocity, self.get_world_center_of_mass_offset())

    def get_world_inverse_inertia(self):
        rotation_matrix = self.get_rotation_matrix()
        body_inverse_inertia = np.linalg.inv(self.I_body)
        return rotation_matrix @ body_inverse_inertia @ rotation_matrix.T

    def get_world_inertia(self):
        rotation_matrix = self.get_rotation_matrix()
        return rotation_matrix @ self.I_body @ rotation_matrix.T

    def sync_orientation_from_euler(self):
        self.orientation = self._euler_to_quaternion(self.rotation)

    def sync_euler_from_orientation(self):
        self.rotation = self._quaternion_to_euler(self.orientation)

    def integrate_orientation(self, dt):
        angular_speed = float(np.linalg.norm(self.angular_velocity))
        if angular_speed <= 1e-12:
            return
        axis = self.angular_velocity / angular_speed
        half_angle = 0.5 * angular_speed * dt
        sin_half = math.sin(half_angle)
        delta_quaternion = np.array([
            math.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
        ], dtype=np.float64)
        self.orientation = _quat_normalize(_quat_multiply(delta_quaternion, self.orientation))
        self.sync_euler_from_orientation()

    def get_local_articulation_velocities(self, dt):
        if self.local_transformed_vertices is None:
            self._needs_update = True
            self.get_vertices()

        current_vertices = self.local_transformed_vertices
        if (
            self.prev_local_transformed_vertices is None
            or self.prev_local_transformed_vertices.shape != current_vertices.shape
            or dt <= 0.0
        ):
            return np.zeros_like(current_vertices)

        return (current_vertices - self.prev_local_transformed_vertices) / dt

    def snapshot_local_geometry(self):
        if self.local_transformed_vertices is None:
            self._needs_update = True
            self.get_vertices()
        self.prev_local_transformed_vertices = self.local_transformed_vertices.copy()

    def _euler_to_quaternion(self, euler):
        roll, pitch, yaw = euler
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return _quat_normalize(np.array([
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ], dtype=np.float64))

    def _quaternion_to_euler(self, quaternion):
        w, x, y, z = _quat_normalize(quaternion)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def _compute_part_dimensions(self):
        dimensions = []
        for part_idx in range(len(self.initial_vertices) // 8):
            part_vertices = np.array(self.initial_vertices[part_idx * 8:(part_idx + 1) * 8], dtype=np.float64)
            mins = part_vertices.min(axis=0)
            maxs = part_vertices.max(axis=0)
            dimensions.append(maxs - mins)
        return dimensions

    @staticmethod
    def _compute_uniform_density_masses(part_dimensions):
        part_volumes = np.array([float(np.prod(dimensions)) for dimensions in part_dimensions], dtype=np.float64)
        total_volume = float(part_volumes.sum())
        if total_volume <= 1e-12:
            raise ValueError("Quadruped geometry has zero volume; cannot derive mass from density.")

        mass_density = QUADRUPED_TOTAL_MASS / total_volume
        return part_volumes * mass_density, mass_density

    def _build_local_geometry(self):
        cos_sh = np.cos(self.shoulder_angles)
        sin_sh = np.sin(self.shoulder_angles)
        cos_el = np.cos(self.elbow_angles)
        sin_el = np.sin(self.elbow_angles)

        shoulder_rotations = []
        elbow_rotations = []

        for leg_idx in range(4):
            shoulder_rotations.append(np.array([
                [1, 0, 0],
                [0, cos_sh[leg_idx], -sin_sh[leg_idx]],
                [0, sin_sh[leg_idx], cos_sh[leg_idx]]
            ], dtype=np.float64))
            elbow_rotations.append(np.array([
                [1, 0, 0],
                [0, cos_el[leg_idx], -sin_el[leg_idx]],
                [0, sin_el[leg_idx], cos_el[leg_idx]]
            ], dtype=np.float64))

        result = np.empty((len(self.vertices), 3), dtype=np.float64)
        part_rotations = [np.eye(3, dtype=np.float64) for _ in range(len(self.vertices) // 8)]

        for i, vertex in enumerate(self.vertices):
            part_index = i // 8
            v = vertex.copy()

            if part_index == 0:
                part_rotations[part_index] = np.eye(3, dtype=np.float64)
            elif 1 <= part_index <= 4:
                leg_index = part_index - 1
                shoulder_center = self.shoulder_positions[leg_index]
                relative_pos = v - shoulder_center
                part_rotations[part_index] = shoulder_rotations[leg_index]
                v = shoulder_center + shoulder_rotations[leg_index] @ relative_pos
            elif 5 <= part_index <= 8:
                leg_index = part_index - 5
                shoulder_center = self.shoulder_positions[leg_index]
                relative_pos = v - shoulder_center
                v = shoulder_center + shoulder_rotations[leg_index] @ relative_pos

                elbow_center_original = self.elbow_positions[leg_index]
                elbow_relative_pos = elbow_center_original - shoulder_center
                elbow_center_transformed = shoulder_center + shoulder_rotations[leg_index] @ elbow_relative_pos
                relative_pos = v - elbow_center_transformed
                v = elbow_center_transformed + elbow_rotations[leg_index] @ relative_pos
                part_rotations[part_index] = elbow_rotations[leg_index] @ shoulder_rotations[leg_index]

            result[i] = v

        return result, part_rotations

    def _update_mass_properties(self, local_vertices, part_rotations):
        masses = self.part_masses
        part_centers = []
        for part_idx in range(len(local_vertices) // 8):
            part_vertices = local_vertices[part_idx * 8:(part_idx + 1) * 8]
            part_centers.append(part_vertices.mean(axis=0))

        weighted_sum = np.sum([mass * center for mass, center in zip(masses, part_centers)], axis=0)
        local_center_of_mass = weighted_sum / self.mass

        inertia_body = np.zeros((3, 3), dtype=np.float64)
        for part_idx, (mass, center, rotation_matrix, dims) in enumerate(zip(masses, part_centers, part_rotations, self.part_dimensions)):
            dx, dy, dz = dims
            part_inertia_diag = np.array([
                mass * (dy ** 2 + dz ** 2) / 12.0,
                mass * (dx ** 2 + dz ** 2) / 12.0,
                mass * (dx ** 2 + dy ** 2) / 12.0,
            ], dtype=np.float64)
            part_inertia = rotation_matrix @ np.diag(part_inertia_diag) @ rotation_matrix.T
            offset = center - local_center_of_mass
            offset_sq = float(offset @ offset)
            inertia_body += part_inertia + mass * (offset_sq * np.eye(3) - np.outer(offset, offset))

        self.local_center_of_mass = local_center_of_mass
        self.I_body = inertia_body
    
    def get_vertices(self):
        """Retourne les vertices. Recalcule uniquement si nécessaire."""
        if not self._needs_update and self.rotated_vertices is not None:
            return self.rotated_vertices

        R_global = self.get_rotation_matrix()
        local_vertices, part_rotations = self._build_local_geometry()
        self.local_transformed_vertices = local_vertices
        self.part_local_rotations = part_rotations
        self._update_mass_properties(local_vertices, part_rotations)

        result = np.empty((len(self.vertices), 3), dtype=np.float64)
        for i, vertex in enumerate(local_vertices):
            result[i] = R_global @ vertex + self.position

        self.rotated_vertices = result
        self._needs_update = False
        
        return result
    
    def get_vertices_dict(self):
        return self.vertices_dict

    def get_body_frame_vector(self, world_vector):
        return self.get_rotation_matrix().T @ np.asarray(world_vector, dtype=np.float64)

    def get_foot_centers_local(self):
        if self.local_transformed_vertices is None:
            self._needs_update = True
            self.get_vertices()

        foot_centers = np.empty((4, 3), dtype=np.float64)
        for leg_idx in range(4):
            lower_start = (5 + leg_idx) * 8
            lower_leg_vertices = self.local_transformed_vertices[lower_start : lower_start + 8]
            foot_centers[leg_idx] = lower_leg_vertices[:4].mean(axis=0)
        return foot_centers

    def get_foot_centers_world(self):
        self.get_vertices()

        foot_centers = np.empty((4, 3), dtype=np.float64)
        for leg_idx in range(4):
            lower_start = (5 + leg_idx) * 8
            lower_leg_vertices = self.rotated_vertices[lower_start : lower_start + 8]
            foot_centers[leg_idx] = lower_leg_vertices[:4].mean(axis=0)
        return foot_centers

    def get_foot_center_velocities_local(self):
        foot_velocities = np.empty((4, 3), dtype=np.float64)
        for leg_idx in range(4):
            lower_start = (5 + leg_idx) * 8
            lower_leg_velocities = self.last_local_articulation_velocities[lower_start : lower_start + 8]
            foot_velocities[leg_idx] = lower_leg_velocities[:4].mean(axis=0)
        return foot_velocities

    def get_state_components(self):
        self.get_vertices()

        gravity_norm = float(np.linalg.norm(GRAVITY))
        gravity_direction_world = GRAVITY / gravity_norm if gravity_norm > 1e-12 else np.array([0.0, -1.0, 0.0], dtype=np.float64)
        task_forward_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        rotation_matrix = self.get_rotation_matrix()
        world_to_body = rotation_matrix.T
        com_offset_world = rotation_matrix @ self.local_center_of_mass
        center_of_mass_velocity_world = self.velocity + np.cross(self.angular_velocity, com_offset_world)
        task_frame_velocity = np.array(
            [center_of_mass_velocity_world[0], center_of_mass_velocity_world[1], -center_of_mass_velocity_world[2]],
            dtype=np.float64,
        )
        joint_limit_fraction = np.concatenate(
            [
                _joint_limit_fraction(self.shoulder_angles, SHOULDER_ANGLE_MIN, SHOULDER_ANGLE_MAX),
                _joint_limit_fraction(self.elbow_angles, ELBOW_ANGLE_MIN, ELBOW_ANGLE_MAX),
            ],
            dtype=np.float64,
        )
        foot_centers_local = self.get_foot_centers_local()
        foot_centers_world = np.empty((4, 3), dtype=np.float64)
        for leg_idx in range(4):
            foot_centers_world[leg_idx] = rotation_matrix @ foot_centers_local[leg_idx] + self.position
        foot_velocities_local = self.get_foot_center_velocities_local()
        foot_heights_world = foot_centers_world[:, 1]

        return {
            "body_height": np.array([self.position[1]], dtype=np.float32),
            "body_height_error": np.array([self.position[1] - 0.5 * (MIN_BODY_HEIGHT + MAX_BODY_HEIGHT)], dtype=np.float32),
            "linear_velocity_body": (world_to_body @ center_of_mass_velocity_world).astype(np.float32),
            "linear_velocity_task": task_frame_velocity.astype(np.float32),
            "angular_velocity_body": (world_to_body @ self.angular_velocity).astype(np.float32),
            "gravity_body": (world_to_body @ gravity_direction_world).astype(np.float32),
            "task_forward_body": (world_to_body @ task_forward_world).astype(np.float32),
            "shoulder_angles": self.shoulder_angles.astype(np.float32),
            "elbow_angles": self.elbow_angles.astype(np.float32),
            "shoulder_velocities": self.shoulder_velocities.astype(np.float32),
            "elbow_velocities": self.elbow_velocities.astype(np.float32),
            "joint_limit_fraction": np.clip(joint_limit_fraction, 0.0, 1.0).astype(np.float32),
            "foot_positions_body": foot_centers_local.astype(np.float32).reshape(-1),
            "foot_velocities_body": foot_velocities_local.astype(np.float32).reshape(-1),
            "foot_heights_world": foot_heights_world.astype(np.float32),
        }

    def set_shoulder_angle(self, leg_index, angle):
        capped_angle = _clip_angle(angle, SHOULDER_ANGLE_MIN, SHOULDER_ANGLE_MAX)
        self.shoulder_angles[leg_index] = capped_angle
        self._needs_update = True
    
    def set_elbow_angle(self, leg_index, angle):
        capped_angle = _clip_angle(angle, ELBOW_ANGLE_MIN, ELBOW_ANGLE_MAX)
        self.elbow_angles[leg_index] = capped_angle
        self._needs_update = True

    def _update_joint_motor(self, angles, velocities, leg_index, target_speed, angle_min, angle_max):
        target_speed = float(target_speed)
        ease_factor = 1.0 - float(np.clip(MOTOR_DIFFICULTY, 0.0, 1.0))
        if abs(target_speed) > 1e-9:
            response_gain = MOTOR_RESPONSE_GAIN + ease_factor * MOTOR_RESPONSE_DIFFICULTY_BOOST
        else:
            response_gain = MOTOR_IDLE_BRAKE_GAIN + ease_factor * MOTOR_BRAKE_DIFFICULTY_BOOST
        response_gain = float(np.clip(response_gain, 0.0, 1.0))
        velocity_damping = float(np.clip(
            MOTOR_VELOCITY_DAMPING + ease_factor * MOTOR_DAMPING_DIFFICULTY_BOOST,
            0.0,
            0.95,
        ))
        current_velocity = float(velocities[leg_index])

        current_velocity += (target_speed - current_velocity) * response_gain
        current_velocity *= (1.0 - velocity_damping)

        if abs(current_velocity) < MOTOR_STOP_EPS:
            current_velocity = 0.0

        new_angle = angles[leg_index] + current_velocity
        capped_angle = _clip_angle(new_angle, angle_min, angle_max)
        angles[leg_index] = capped_angle

        if (
            capped_angle >= angle_max and current_velocity > 0.0
        ) or (
            capped_angle <= angle_min and current_velocity < 0.0
        ):
            current_velocity = 0.0

        velocities[leg_index] = current_velocity
        self._needs_update = True
    
    def adjust_shoulder_angle(self, leg_index, delta_angle):
        self._update_joint_motor(
            angles=self.shoulder_angles,
            velocities=self.shoulder_velocities,
            leg_index=leg_index,
            target_speed=delta_angle,
            angle_min=SHOULDER_ANGLE_MIN,
            angle_max=SHOULDER_ANGLE_MAX,
        )
    
    def adjust_elbow_angle(self, leg_index, delta_angle):
        self._update_joint_motor(
            angles=self.elbow_angles,
            velocities=self.elbow_velocities,
            leg_index=leg_index,
            target_speed=delta_angle,
            angle_min=ELBOW_ANGLE_MIN,
            angle_max=ELBOW_ANGLE_MAX,
        )
    
    def get_state(self):
        components = self.get_state_components()
        state = np.concatenate(list(components.values()), dtype=np.float32)
        return state.tolist()

    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le quadruped 3D avec projection et profondeur (arêtes seulement)"""        
        # Utiliser les vertices déjà calculés
        vertices = self.rotated_vertices
        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:  # projected peut etre None si le point est derrière la caméra
                projected_vertices.append(projected)
        
        if len(projected_vertices) < 8:
            return  # Le quadruped est partiellement hors champ de vision, on ne dessine pas
        
        # Définir les couleurs pour chaque partie
        colors = [
            (255, 255, 255),  # Body (white)
            (0, 0, 255),      # Upper leg 0 - Front right (blue)
            (255, 0, 0),      # Upper leg 1 - Front left (red)
            (0, 255, 0),      # Upper leg 2 - Back right (green)
            (255, 255, 255),  # Upper leg 3 - Back left (white)
            (0, 0, 255),      # Lower leg 0 - Front right (blue)
            (255, 0, 0),      # Lower leg 1 - Front left (red)
            (0, 255, 0),      # Lower leg 2 - Back right (green)
            (255, 255, 255)   # Lower leg 3 - Back left (white)
        ]
        
        # Dessiner chaque partie du quadruped (body + 8 legs)
        # Chaque partie a 8 vertices, donc on dessine par groupes de 8
        parts_per_leg = 8
        total_parts = len(projected_vertices) // parts_per_leg
        
        for part_idx in range(total_parts):
            start_idx = part_idx * parts_per_leg
            end_idx = start_idx + parts_per_leg
            
            if end_idx <= len(projected_vertices):
                part_vertices = projected_vertices[start_idx:end_idx]
                part_color = colors[part_idx] if part_idx < len(colors) else self.color
                
                # Définir les arêtes pour chaque partie (cube)
                edges = [
                    (0, 1), (1, 3), (3, 2), (2, 0),  # Face avant
                    (4, 5), (5, 7), (7, 6), (6, 4),  # Face arrière
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
                ]
                
                # Dessiner les arêtes de cette partie
                for edge in edges:
                    if edge[0] < len(part_vertices) and edge[1] < len(part_vertices):
                        start = part_vertices[edge[0]][:2]
                        end = part_vertices[edge[1]][:2]
                        
                        # Vérifier que les coordonnées sont valides
                        if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                            0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                            pygame.draw.line(screen, part_color, start, end, 2)
    
    def draw_premium(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le quadruped 3D avec faces pleines et dégradé de gris basé sur la profondeur"""
        # Utiliser les vertices déjà calculés
        vertices = self.rotated_vertices
        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:  # projected peut etre None si le point est derrière la caméra
                projected_vertices.append(projected)
        
        if len(projected_vertices) < 8:
            return  # Le quadruped est partiellement hors champ de vision, on ne dessine pas
        
        # Définir les faces pour chaque cube (6 faces par cube)
        # Chaque face est définie par 4 indices de vertices
        cube_faces = [
            [0, 1, 3, 2],  # Face avant (bottom)
            [4, 5, 7, 6],  # Face arrière (top)
            [0, 1, 5, 4],  # Face gauche
            [2, 3, 7, 6],  # Face droite
            [0, 2, 6, 4],  # Face inférieure
            [1, 3, 7, 5]   # Face supérieure
        ]
        cube_edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        
        # Dessiner chaque partie du quadruped (body + 8 legs)
        # Chaque partie a 8 vertices, donc on dessine par groupes de 8
        parts_per_leg = 8
        total_parts = len(projected_vertices) // parts_per_leg
        current_actions = getattr(self, "render_action_vector", np.zeros(8, dtype=np.float32))
        
        # Trier les faces par profondeur pour le rendu correct (painter's algorithm)
        all_faces = []
        
        for part_idx in range(total_parts):
            start_idx = part_idx * parts_per_leg
            end_idx = start_idx + parts_per_leg
            
            if end_idx <= len(projected_vertices):
                part_vertices = projected_vertices[start_idx:end_idx]
                
                # Calculer le centre de profondeur de cette partie pour le tri
                part_depth = sum(v[2] for v in part_vertices) / len(part_vertices)
                
                # Créer les faces pour cette partie
                for face_indices in cube_faces:
                    face_vertices = [part_vertices[i] for i in face_indices if i < len(part_vertices)]
                    
                    if len(face_vertices) == 4:
                        # Calculer la profondeur moyenne de la face
                        face_depth = sum(v[2] for v in face_vertices) / len(face_vertices)
                        
                        # Calculer la normale de la face pour déterminer si elle est visible
                        # Simplification : on dessine toutes les faces pour l'instant
                        
                        all_faces.append({
                            'vertices': face_vertices,
                            'depth': face_depth,
                            'part_idx': part_idx
                        })
        
        # Trier les faces par profondeur (les plus éloignées en premier)
        all_faces.sort(key=lambda face: face['depth'], reverse=True)
        
        # Dessiner les faces triées
        for face in all_faces:
            face_vertices = face['vertices']
            depth = face['depth']
            part_idx = face['part_idx']
            
            # Calculer la couleur basée sur la profondeur (dégradé de gris)
            # Plus la profondeur est grande, plus la couleur est sombre
            base_intensity = 200  # Gris clair de base
            depth_factor = max(0, min(1, depth / 50))  # Normaliser la profondeur
            intensity = int(base_intensity * (1 - depth_factor * 0.7))  # Réduire jusqu'à 30% de l'intensité
            
            color = (intensity, intensity, intensity)
            
            # Convertir les vertices en points 2D pour pygame
            points_2d = [(int(v[0]), int(v[1])) for v in face_vertices]
            
            # Vérifier que tous les points sont dans les limites de l'écran
            valid_points = True
            for x, y in points_2d:
                if not (0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT):
                    valid_points = False
                    break
            
            if valid_points and len(points_2d) >= 3:
                # Dessiner la face pleine
                pygame.draw.polygon(screen, color, points_2d)
                
                # Optionnel : dessiner les contours des faces pour plus de définition
                pygame.draw.polygon(screen, (50, 50, 50), points_2d, 1)

        for part_idx in range(1, total_parts):
            start_idx = part_idx * parts_per_leg
            end_idx = start_idx + parts_per_leg
            if end_idx > len(projected_vertices):
                continue

            part_vertices = projected_vertices[start_idx:end_idx]
            action_index = part_idx - 1
            if action_index >= len(current_actions):
                action_value = 0.0
            else:
                action_value = current_actions[action_index]
            edge_color = _action_edge_color(action_value)

            for edge_start, edge_end in cube_edges:
                start = part_vertices[edge_start][:2]
                end = part_vertices[edge_end][:2]
                if (
                    0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT
                    and 0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT
                ):
                    pygame.draw.line(screen, edge_color, start, end, 2)
