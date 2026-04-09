# quadruped_env.py
import time
import pygame
import numpy as np
from pygame.locals import *

from ..core.config import *
from ..rendering.camera import Camera3D
from ..quadruped.quadruped import Quadruped
from ..quadruped.quadruped_points import get_quadruped_vertices, create_quadruped_vertices
from ..rendering.ground import Ground
from ..quadruped.update_quad import update_quadruped

class QuadrupedEnv:
    """
    Main game class for simulating and controlling a quadruped robot in a 3D environment using Pygame.
    Handles camera movement, joint controls, rendering, and UI.
    """
    # Key mappings for joint controls
    SHOULDER_KEYS = [
        (K_r, 0, 1), (K_f, 0, -1),  # Front Right
        (K_t, 2, 1), (K_g, 2, -1),  # Front Left
        (K_y, 1, 1), (K_h, 1, -1),  # Back Right
        (K_u, 3, 1), (K_j, 3, -1),  # Back Left
    ]
    ELBOW_KEYS = [
        (K_1, 0, 1), (K_5, 0, -1),  # Front Right
        (K_2, 2, 1), (K_6, 2, -1),  # Front Left
        (K_3, 1, 1), (K_7, 1, -1),  # Back Right
        (K_4, 3, 1), (K_8, 3, -1),  # Back Left
    ]
    INSTRUCTIONS = [
        "Contrôles:",
        "ZQSD - Déplacer caméra",
        "AE - Monter/Descendre caméra",
        "Flèches - Rotation caméra",
        "Espace - Reset quadruped",
        "B - Reset articulations",
        "R/F/T/G/Y/H/U/J - Épaules (FR,FL,BR,BL)",
        "1-8 - Coudes (FR,FL,BR,BL)",
        "P - Afficher les sommets",
        "Échap - Quitter"
    ]
    def __init__(self, rendering=True, headless=False, bench_mode=False):
        """Initialize the game, Pygame, and world objects."""
        pygame.init()
        self.headless = headless
        self.bench_mode = bench_mode
        if headless:
            self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Moteur Physique 3D - Pygame")
        self.clock = pygame.time.Clock()
        self.rendering = rendering

        # World objects
        self.camera = Camera3D()
        self.ground = Ground(size=20)
        self.quadruped_vertices_dict = create_quadruped_vertices()
        self.quadruped = Quadruped(
            position=np.array([0.0, 4.5, 0.0]),
            vertices=get_quadruped_vertices(),
            vertices_dict=self.quadruped_vertices_dict
        )
        self.camera_speed = 0.1
        self.rotation_speed = 0.02
        self.font = pygame.font.Font(None, 24)

        self.prev_potential = None
        self.cumulative_locomotion_reward = 0.0
        self.consecutive_steps_below_critical_height = 0
        self.consecutive_steps_above_critical_height = 0
        self.consecutive_steps_critical_tilt = 0
        self.consecutive_shoulder_limit_steps = np.zeros(4, dtype=np.int32)
        self.consecutive_elbow_limit_steps = np.zeros(4, dtype=np.int32)
        self.steps_since_contact = 0
        self.has_had_ground_contact = False
        self.steps_since_foot_contact = np.zeros(4, dtype=np.int32)
        self.has_had_foot_ground_contact = np.zeros(4, dtype=bool)
        self.prev_action = np.zeros(8, dtype=np.float32)
        self.last_reward_components = {}
        self.last_done_reason = "running"

    def _reset_safety_counters(self):
        self.consecutive_steps_below_critical_height = 0
        self.consecutive_steps_above_critical_height = 0
        self.consecutive_steps_critical_tilt = 0
        self.consecutive_shoulder_limit_steps.fill(0)
        self.consecutive_elbow_limit_steps.fill(0)

    def _compute_body_tilt_angles(self):
        body_up_world = self.quadruped.get_rotation_matrix()[:, 1]
        forward_tilt = abs(np.arctan2(body_up_world[0], body_up_world[1]))
        side_tilt = abs(np.arctan2(body_up_world[2], body_up_world[1]))
        return forward_tilt, side_tilt

    @staticmethod
    def _soft_reward_scale(distance_to_failure, soft_margin):
        if soft_margin <= 0.0:
            return 1.0 if distance_to_failure > 0.0 else 0.0
        return float(np.clip(distance_to_failure / soft_margin, 0.0, 1.0))

    @staticmethod
    def _joint_limit_push_mask(values, actions, angle_min, angle_max):
        values = np.asarray(values, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        at_lower_limit = values <= float(angle_min) + JOINT_LIMIT_ANGLE_EPS
        at_upper_limit = values >= float(angle_max) - JOINT_LIMIT_ANGLE_EPS
        pushing_lower = actions < 0.0
        pushing_upper = actions > 0.0
        return np.logical_or(
            np.logical_and(at_lower_limit, pushing_lower),
            np.logical_and(at_upper_limit, pushing_upper),
        )

    @staticmethod
    def _compute_joint_limit_penalty(max_push_steps):
        if max_push_steps <= JOINT_LIMIT_STUCK_GRACE_STEPS:
            return 0.0

        available_steps = max(1, MAX_CONSECUTIVE_JOINT_LIMIT_STEPS - JOINT_LIMIT_STUCK_GRACE_STEPS)
        normalized_progress = min(
            float(max_push_steps - JOINT_LIMIT_STUCK_GRACE_STEPS) / float(available_steps),
            1.0,
        )
        exp_numerator = np.expm1(JOINT_LIMIT_STUCK_EXP_RATE * normalized_progress)
        exp_denominator = np.expm1(JOINT_LIMIT_STUCK_EXP_RATE)
        if exp_denominator <= 0.0:
            return -JOINT_LIMIT_PENALTY_COEF * normalized_progress
        return -JOINT_LIMIT_PENALTY_COEF * float(exp_numerator / exp_denominator)

    def reset_episode_state(self):
        self.prev_potential = None
        self.cumulative_locomotion_reward = 0.0
        self.prev_action.fill(0.0)
        self._reset_safety_counters()
        self.steps_since_contact = 0
        self.has_had_ground_contact = False
        self.steps_since_foot_contact.fill(0)
        self.has_had_foot_ground_contact.fill(False)
        self.last_reward_components = {}
        self.last_done_reason = "running"
        self.quadruped.too_high = False
        self.quadruped.steps_since_too_high = 0
        self.quadruped.too_low = False
        self.quadruped.steps_since_too_low = 0

    @staticmethod
    def _normalize_contact_steps(steps_since_contact, has_contact_history, max_steps):
        if not has_contact_history:
            return 1.0
        return float(np.clip(steps_since_contact / max_steps, 0.0, 1.0))

    def _get_lower_leg_contact_flags(self):
        active_contact_indices = np.asarray(getattr(self.quadruped, "active_contact_indices", []), dtype=np.int64)
        lower_leg_contact_flags = np.zeros(4, dtype=np.float32)
        for vertex_idx in active_contact_indices.tolist():
            part_index = int(vertex_idx // 8)
            if 5 <= part_index <= 8:
                lower_leg_contact_flags[part_index - 5] = 1.0
        return lower_leg_contact_flags

    def get_state_components(self):
        quadruped_components = self.quadruped.get_state_components()
        foot_positions_body = quadruped_components["foot_positions_body"].reshape(4, 3)
        foot_contact = self._get_lower_leg_contact_flags()
        grounded_recently = 1.0 if (self.has_had_ground_contact and self.steps_since_contact <= CONTACT_GRACE_STEPS) else 0.0
        grounded_leg_count_norm = float(foot_contact.mean())

        if np.any(foot_contact > 0.5):
            support_centroid_body_xz = foot_positions_body[foot_contact > 0.5][:, [0, 2]].mean(axis=0).astype(np.float32)
            com_minus_support_centroid_body_xz = (
                self.quadruped.local_center_of_mass[[0, 2]] - support_centroid_body_xz.astype(np.float64)
            ).astype(np.float32)
        else:
            support_centroid_body_xz = np.zeros(2, dtype=np.float32)
            com_minus_support_centroid_body_xz = np.zeros(2, dtype=np.float32)

        foot_contact_steps = np.array(
            [
                self._normalize_contact_steps(int(steps), bool(has_contact), CONTACT_GRACE_STEPS)
                for steps, has_contact in zip(self.steps_since_foot_contact, self.has_had_foot_ground_contact)
            ],
            dtype=np.float32,
        )
        contact_count = int(len(getattr(self.quadruped, "active_contact_indices", [])))

        components = {
            **quadruped_components,
            "joint_limit_progress": self._get_joint_limit_progress(),
            "prev_action": self.prev_action.astype(np.float32),
            "foot_contact": foot_contact,
            "steps_since_foot_contact_norm": foot_contact_steps,
            "contact_count_norm": np.array([min(contact_count, MAX_CONTACT_POINTS) / MAX_CONTACT_POINTS], dtype=np.float32),
            "steps_since_contact_norm": np.array(
                [self._normalize_contact_steps(self.steps_since_contact, self.has_had_ground_contact, MAX_AIRBORNE_STEPS)],
                dtype=np.float32,
            ),
            "grounded_recently": np.array([grounded_recently], dtype=np.float32),
            "has_had_ground_contact": np.array([1.0 if self.has_had_ground_contact else 0.0], dtype=np.float32),
            "grounded_leg_count_norm": np.array([grounded_leg_count_norm], dtype=np.float32),
            "support_centroid_body_xz": support_centroid_body_xz,
            "com_minus_support_centroid_body_xz": com_minus_support_centroid_body_xz,
        }
        return components

    def reset_episode(self, randomize=False, pose_jitter=False):
        if randomize and pose_jitter:
            raise ValueError("reset_episode cannot use randomize=True and pose_jitter=True at the same time")
        if randomize:
            self.quadruped.reset_random()
        elif pose_jitter:
            self.quadruped.reset_pose_jitter()
        else:
            self.quadruped.reset()
        self.reset_episode_state()
        return self.get_state()

    def _get_joint_limit_progress(self):
        shoulder_progress = np.clip(
            self.consecutive_shoulder_limit_steps.astype(np.float32) / MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
            0.0,
            1.0,
        )
        elbow_progress = np.clip(
            self.consecutive_elbow_limit_steps.astype(np.float32) / MAX_CONSECUTIVE_JOINT_LIMIT_STEPS,
            0.0,
            1.0,
        )
        return np.concatenate([shoulder_progress, elbow_progress], dtype=np.float32)
        
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()
            keys = pygame.key.get_pressed()
            shoulder_actions, elbow_actions = self.handle_joint_controls(keys)
            camera_actions = self.handle_camera_controls(keys)
            reset_actions = [0, 0]
            if keys[K_SPACE]:
                reset_actions[0] = 1
            if keys[K_b]:
                reset_actions[1] = 1

            if keys[K_p]:
                state = self.get_state()
                print(f"state: {state}, len: {len(state)}")
                time.sleep(0.1)

            reward = 0.0
            done = False
            step_time = 0.0
            for substep_idx in range(PHYSICS_STEPS_PER_RENDER):
                step_camera_actions = camera_actions if substep_idx == 0 else [0] * 10
                step_reset_actions = reset_actions if substep_idx == 0 else [0, 0]
                _, reward, done, substep_time = self.step(
                    shoulder_actions,
                    elbow_actions,
                    step_camera_actions,
                    step_reset_actions,
                )
                step_time += substep_time
            step_time /= PHYSICS_STEPS_PER_RENDER

            if self.rendering:
                self.render(reward, done, step_time)
            self.clock.tick(RENDER_FPS)
        pygame.quit()

    def handle_events(self):
        """Handle Pygame events. Returns False if the game should exit."""
        if self.headless:
            return True
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
        return True

    def handle_camera_controls(self, keys):
        """Handle camera movement and rotation based on key input."""
        camera_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Movement
        if keys[K_z]:
            camera_actions[0] = 1
        if keys[K_s]:
            camera_actions[1] = 1
        if keys[K_q]:
            camera_actions[2] = 1
        if keys[K_d]:
            camera_actions[3] = 1
        if keys[K_e]:
            camera_actions[4] = 1
        if keys[K_a]:
            camera_actions[5] = 1
        # Rotation
        if keys[K_LEFT]:
            camera_actions[6] = 1
        if keys[K_RIGHT]:
            camera_actions[7] = 1
        if keys[K_DOWN]:
            camera_actions[8] = 1
        if keys[K_UP]:
            camera_actions[9] = 1

        return camera_actions

    def handle_joint_controls(self, keys):
        """Handle joint (shoulder and elbow) controls based on key input."""
        shoulder_actions = [0, 0, 0, 0]
        elbow_actions = [0, 0, 0, 0]
        for key, idx, sign in self.SHOULDER_KEYS:
            if keys[key]:
                shoulder_actions[idx] = sign
        for key, idx, sign in self.ELBOW_KEYS:
            if keys[key]:
                elbow_actions[idx] = sign
        return shoulder_actions, elbow_actions
    
    def step(self, shoulder_actions, elbow_actions, camera_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], reset_actions = [0, 0]):
        """Step the quadruped in the environment.
        
        actions_array is a vector composed only of 0 and 1.
        --- Selected by the agent ---
        The first 4 elements are the actions for the shoulders. (FR, FL, BR, BL)
        The following 4 elements are the actions for the elbows. (FR, FL, BR, BL)
        --- Only in manual mode ---
        The following 10 are for the camera. (z, s, q, d, e, a, left, right, up, down)
        The following 2 are for the reset (no-rd and rd). (space and b)
        The following 1 for the vertices display. (p)
        
        The action is the following:
        - 0: no action
        - 1: action
        """

        start_step_time = time.time()

        # Update all Shoulder and Elbow angles
        for idx, action in enumerate(shoulder_actions):
            self.quadruped.adjust_shoulder_angle(idx, SHOULDER_DELTA * action)
        for idx, action in enumerate(elbow_actions):
            self.quadruped.adjust_elbow_angle(idx, ELBOW_DELTA * action)

        # Update camera
        for idx, action in enumerate(camera_actions):
            if action:
                self.camera.update_camera(idx, self.camera_speed, self.rotation_speed)
        
        # Reset quadruped
        if reset_actions[0]:
            self.reset_episode(randomize=True)
        if reset_actions[1]:
            self.reset_episode(pose_jitter=True)

        # Update quadruped
        update_quadruped(self.quadruped)

        if self.bench_mode:
            self._reset_safety_counters()
            self.quadruped.too_high = False
            self.quadruped.steps_since_too_high = 0
            self.quadruped.too_low = False
            self.quadruped.steps_since_too_low = 0
            self.last_reward_components = {
                "distance_reward": 0.0,
                "locomotion_reward": 0.0,
                "cumulative_locomotion_reward": 0.0,
                "forward_progress": 0.0,
                "clipped_forward_progress": 0.0,
                "progress_delta": 0.0,
                "forward_speed": 0.0,
                "terminal_event_reward": 0.0,
                "raw_distance_reward": 0.0,
                "angular_velocity_penalty": 0.0,
                "angular_velocity_norm": 0.0,
                "contact_count": float(len(getattr(self.quadruped, "active_contact_indices", []))),
                "steps_since_contact": float(self.steps_since_contact),
                "grounded_recently": 1.0,
                "contact_reward_scale": 1.0,
                "tilt_reward_scale": 1.0,
                "height_reward_scale": 1.0,
                "raw_pose_scale": 1.0,
                "locomotion_reward_scale": 1.0,
                "joint_limit_penalty": 0.0,
                "joint_limit_push_steps_max": 0.0,
                "joint_limit_push_active": 0.0,
                "forward_tilt_deg": 0.0,
                "side_tilt_deg": 0.0,
            }
            self.last_done_reason = "bench_mode"
            self.prev_action = np.concatenate(
                [
                    np.asarray(shoulder_actions, dtype=np.float32),
                    np.asarray(elbow_actions, dtype=np.float32),
                ],
                dtype=np.float32,
            )
            next_state = self.get_state()
            end_step_time = time.time()
            step_time = end_step_time - start_step_time
            return next_state, 0.0, False, step_time
        current_action = np.concatenate(
            [
                np.asarray(shoulder_actions, dtype=np.float32),
                np.asarray(elbow_actions, dtype=np.float32),
            ],
            dtype=np.float32,
        )
        # ---- REWARD ---
        # ----------  a) progression avant signee --------------
        forward_progress = -float(self.quadruped.position[2])
        progress_delta = 0.0
        if self.prev_potential is None:
            raw_distance_reward = 0.0
        else:
            progress_delta = forward_progress - self.prev_potential
            raw_distance_reward = PROGRESS_REWARD_COEF * float(progress_delta)
        self.prev_potential = forward_progress

        # ----------  b) metriques de mouvement et posture -------------------------
        forward_speed = -float(self.quadruped.velocity[2])

        forward_tilt, side_tilt = self._compute_body_tilt_angles()
        max_tilt = max(forward_tilt, side_tilt)
        angular_velocity_norm = float(np.linalg.norm(self.quadruped.angular_velocity))
        angular_velocity_penalty = -ANGULAR_VELOCITY_PENALTY_COEF * angular_velocity_norm

        contact_count = int(len(getattr(self.quadruped, "active_contact_indices", [])))
        if contact_count > 0:
            self.has_had_ground_contact = True
            self.steps_since_contact = 0
        else:
            self.steps_since_contact += 1
        foot_contact_flags = self._get_lower_leg_contact_flags()
        foot_contact_mask = foot_contact_flags > 0.5
        self.has_had_foot_ground_contact = np.logical_or(self.has_had_foot_ground_contact, foot_contact_mask)
        self.steps_since_foot_contact = np.where(
            foot_contact_mask,
            0,
            self.steps_since_foot_contact + 1,
        )
        grounded_recently = self.has_had_ground_contact and self.steps_since_contact <= CONTACT_GRACE_STEPS
        airborne_timeout = self.has_had_ground_contact and self.steps_since_contact > MAX_AIRBORNE_STEPS
        distance_reward = raw_distance_reward if grounded_recently else 0.0

        # ----------  c) contraintes terminales -------------------------------------
        body_height = float(self.quadruped.position[1])
        below_critical_height = body_height < MIN_BODY_HEIGHT
        above_critical_height = body_height > MAX_BODY_HEIGHT

        if below_critical_height:
            self.consecutive_steps_below_critical_height += 1
            self.consecutive_steps_above_critical_height = 0
        elif above_critical_height:
            self.consecutive_steps_above_critical_height += 1
            self.consecutive_steps_below_critical_height = 0
        else:
            self.consecutive_steps_below_critical_height = 0
            self.consecutive_steps_above_critical_height = 0

        critically_tilted = max_tilt > CRITICAL_TILT_ANGLE
        if critically_tilted:
            self.consecutive_steps_critical_tilt += 1
        else:
            self.consecutive_steps_critical_tilt = 0

        shoulder_pushing_limit = self._joint_limit_push_mask(
            self.quadruped.shoulder_angles,
            shoulder_actions,
            SHOULDER_ANGLE_MIN,
            SHOULDER_ANGLE_MAX,
        )
        elbow_pushing_limit = self._joint_limit_push_mask(
            self.quadruped.elbow_angles,
            elbow_actions,
            ELBOW_ANGLE_MIN,
            ELBOW_ANGLE_MAX,
        )
        self.consecutive_shoulder_limit_steps = np.where(
            shoulder_pushing_limit,
            self.consecutive_shoulder_limit_steps + 1,
            0,
        )
        self.consecutive_elbow_limit_steps = np.where(
            elbow_pushing_limit,
            self.consecutive_elbow_limit_steps + 1,
            0,
        )
        max_joint_limit_push_steps = int(
            max(
                int(self.consecutive_shoulder_limit_steps.max()),
                int(self.consecutive_elbow_limit_steps.max()),
            )
        )
        joint_limit_penalty = self._compute_joint_limit_penalty(max_joint_limit_push_steps)

        tilt_reward_scale = self._soft_reward_scale(
            CRITICAL_TILT_ANGLE - max_tilt,
            TILT_SOFT_REWARD_MARGIN,
        )
        height_reward_scale = self._soft_reward_scale(
            body_height - MIN_BODY_HEIGHT,
            HEIGHT_SOFT_REWARD_MARGIN,
        )
        raw_pose_scale = min(tilt_reward_scale, height_reward_scale)
        locomotion_reward_scale = raw_pose_scale

        joint_limit_timeout = (
            int(self.consecutive_shoulder_limit_steps.max()) > MAX_CONSECUTIVE_JOINT_LIMIT_STEPS
            or int(self.consecutive_elbow_limit_steps.max()) > MAX_CONSECUTIVE_JOINT_LIMIT_STEPS
        )

        done_reason = "running"
        terminal_event_reward = 0.0

        if below_critical_height:
            done_reason = "too_low"
            terminal_event_reward = TERMINAL_PENALTY_TOO_LOW
        elif above_critical_height:
            done_reason = "too_high"
            terminal_event_reward = TERMINAL_PENALTY_TOO_HIGH
        elif max_tilt > CRITICAL_TILT_ANGLE:
            done_reason = "critical_tilt"
            terminal_event_reward = TERMINAL_PENALTY_CRITICAL_TILT
        elif joint_limit_timeout:
            done_reason = "joint_limit_timeout"
            terminal_event_reward = TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT
        elif airborne_timeout:
            done_reason = "airborne"
            terminal_event_reward = TERMINAL_PENALTY_AIRBORNE
        done = done_reason != "running"

        # Inform the quadruped that it is in danger
        self.quadruped.too_high = above_critical_height
        self.quadruped.steps_since_too_high = self.consecutive_steps_above_critical_height
        self.quadruped.too_low = below_critical_height
        self.quadruped.steps_since_too_low = self.consecutive_steps_below_critical_height

        # ----------  d)  Somme finale -------------------------
        # Objectif: le progres ne compte que dans un regime de locomotion au sol.
        # Une petite penalite de vitesse angulaire limite les oscillations du body.
        locomotion_reward = (distance_reward * locomotion_reward_scale) if not done else 0.0
        if done:
            reward = terminal_event_reward
        else:
            reward = locomotion_reward + angular_velocity_penalty + joint_limit_penalty
            self.cumulative_locomotion_reward += locomotion_reward
        self.last_reward_components = {
            "distance_reward": float(distance_reward),
            "raw_distance_reward": float(raw_distance_reward),
            "locomotion_reward": float(locomotion_reward),
            "cumulative_locomotion_reward": float(self.cumulative_locomotion_reward),
            "forward_progress": float(forward_progress),
            "clipped_forward_progress": float(max(0.0, forward_progress)),
            "progress_delta": float(progress_delta),
            "forward_speed": float(forward_speed),
            "terminal_event_reward": float(terminal_event_reward),
            "angular_velocity_penalty": float(angular_velocity_penalty),
            "angular_velocity_norm": float(angular_velocity_norm),
            "contact_count": float(contact_count),
            "steps_since_contact": float(self.steps_since_contact),
            "grounded_recently": 1.0 if grounded_recently else 0.0,
            "contact_reward_scale": 1.0 if grounded_recently else 0.0,
            "tilt_reward_scale": float(tilt_reward_scale),
            "height_reward_scale": float(height_reward_scale),
            "raw_pose_scale": float(raw_pose_scale),
            "locomotion_reward_scale": float(locomotion_reward_scale * (1.0 if grounded_recently else 0.0)),
            "joint_limit_penalty": float(joint_limit_penalty),
            "joint_limit_push_steps_max": float(max_joint_limit_push_steps),
            "joint_limit_push_active": 1.0 if max_joint_limit_push_steps > 0 else 0.0,
            "forward_tilt_deg": float(np.degrees(forward_tilt)),
            "side_tilt_deg": float(np.degrees(side_tilt)),
        }
        self.last_done_reason = done_reason
        self.prev_action = current_action
        next_state = self.get_state()
        # -----------------------------------------------------
        end_step_time = time.time()
        step_time = end_step_time - start_step_time

        return next_state, reward, done, step_time

    def render(self, reward, done = False, step_time = 0.0, state_value = None):
        """Render the 3D world and UI."""
        self.screen.fill(BLACK)
        self.ground.draw_premium(self.screen, self.camera)
        self.ground.draw_axes(self.screen, self.camera)
        self.quadruped.draw_premium(self.screen, self.camera)
        self.render_ui(reward, done, step_time, state_value)
        if not self.headless:
            pygame.display.flip()
    
    def get_state(self):
        """Get the current state of the quadruped."""
        state = np.concatenate(list(self.get_state_components().values()), dtype=np.float32)
        return state.tolist()

    def render_ui(self, reward, done = False, step_time = 0.0, state_value = None):
        """Render the UI overlays (info and instructions)."""
        # Info texts
        pos_text = f"Position: ({self.quadruped.position[0]:.2f}, {self.quadruped.position[1]:.2f}, {self.quadruped.position[2]:.2f})"
        vel_text = f"Vitesse: ({self.quadruped.velocity[0]:.2f}, {self.quadruped.velocity[1]:.2f}, {self.quadruped.velocity[2]:.2f})"
        cam_text = f"Caméra: ({self.camera.position[0]:.1f}, {self.camera.position[1]:.1f}, {self.camera.position[2]:.1f})"
        rot_text = f"Rotation: ({self.quadruped.rotation[0]:.2f}, {self.quadruped.rotation[1]:.2f}, {self.quadruped.rotation[2]:.2f})"
        shoulder_text = (
            f"Épaules: FR({self.quadruped.shoulder_angles[0]:.2f}) "
            f"FL({self.quadruped.shoulder_angles[1]:.2f}) "
            f"BR({self.quadruped.shoulder_angles[2]:.2f}) "
            f"BL({self.quadruped.shoulder_angles[3]:.2f})"
        )
        elbow_text = (
            f"Coudes: FR({self.quadruped.elbow_angles[0]:.2f}) "
            f"FL({self.quadruped.elbow_angles[1]:.2f}) "
            f"BR({self.quadruped.elbow_angles[2]:.2f}) "
            f"BL({self.quadruped.elbow_angles[3]:.2f})"
        )
        reward_text = f"Récompense: {reward:.2f}"
        done_text = f"Terminé ? {done}"
        step_time_text = f"Step time: {(step_time*1000):.6f}ms"
        surfaces = [
            self.font.render(pos_text, True, WHITE),
            self.font.render(vel_text, True, WHITE),
            self.font.render(cam_text, True, WHITE),
            self.font.render(rot_text, True, WHITE),
            self.font.render(shoulder_text, True, WHITE),
            self.font.render(elbow_text, True, WHITE),
            self.font.render(reward_text, True, WHITE),
            self.font.render(done_text, True, WHITE),
            self.font.render(step_time_text, True, WHITE),
        ]
        for i, surf in enumerate(surfaces):
            self.screen.blit(surf, (10, 10 + i * 25))
        # Instructions
        for i, instruction in enumerate(self.INSTRUCTIONS):
            inst_surface = self.font.render(instruction, True, GRAY)
            self.screen.blit(inst_surface, (10, WINDOW_HEIGHT - 140 + i * 20))

        # State value
        if state_value is not None:
            state_value_text = f"State value: {state_value:.2f}"
            state_value_surface = self.font.render(state_value_text, True, WHITE)
            self.screen.blit(state_value_surface, (WINDOW_WIDTH - 200, 10))

if __name__ == "__main__":
    import cProfile

    if PROFILING:
        profiler = cProfile.Profile()
        profiler.enable()

    game = QuadrupedEnv(rendering=True)
    game.run()

    if PROFILING:
        profiler.disable()
        profiler.dump_stats("profiling/physics_engine_only.prof")
