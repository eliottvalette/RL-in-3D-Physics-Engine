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
    CIRCLE_RADII = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

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
            position=np.array([0.0, 5.5, 0.0]),
            vertices=get_quadruped_vertices(),
            vertices_dict=self.quadruped_vertices_dict
        )
        self.camera_speed = 0.1
        self.rotation_speed = 0.02
        self.font = pygame.font.Font(None, 24)

        self.circle_radii   = self.CIRCLE_RADII
        self.circles_passed = set()         # stocke les rayons déjà comptés
        self.prev_radius   = None     # distance horizontale au pas t‑1
        self.prev_potential = None
        self.rot_penalty_coef = 0.5
        self.gait_reward_coef = 0.5
        self.consecutive_steps_below_critical_height = 0
        self.consecutive_steps_above_critical_height = 0
        self.consecutive_steps_critical_tilt = 0
        self.consecutive_shoulder_limit_steps = np.zeros(4, dtype=np.int32)
        self.consecutive_elbow_limit_steps = np.zeros(4, dtype=np.int32)
        self.last_reward_components = {}
        self.last_done_reason = "not_started"

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

    def _compute_tilt_reward_scale(self, max_tilt):
        if max_tilt <= STABILITY_TILT_THRESHOLD:
            return 1.0
        if max_tilt >= CRITICAL_TILT_ANGLE:
            return 0.0

        tilt_window = CRITICAL_TILT_ANGLE - STABILITY_TILT_THRESHOLD
        if tilt_window <= 0.0:
            return 0.0

        return float(np.clip((CRITICAL_TILT_ANGLE - max_tilt) / tilt_window, 0.0, 1.0))

    def _compute_height_reward_scale(self, body_height):
        if MIN_BODY_HEIGHT <= body_height <= MAX_BODY_HEIGHT:
            return 1.0

        if body_height < MIN_BODY_HEIGHT:
            overshoot = MIN_BODY_HEIGHT - body_height
        else:
            overshoot = body_height - MAX_BODY_HEIGHT

        if HEIGHT_REWARD_DECAY_MARGIN <= 0.0:
            return 0.0

        return float(np.clip(1.0 - overshoot / HEIGHT_REWARD_DECAY_MARGIN, 0.0, 1.0))

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
            self.quadruped.reset_random()
            self.circles_passed.clear()
            self.prev_potential = None
            self.prev_radius    = None
            self._reset_safety_counters()
        if reset_actions[1]:
            self.quadruped.reset()
            self.circles_passed.clear()
            self.prev_potential = None
            self.prev_radius    = None
            self._reset_safety_counters()

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
                "z_speed_reward": 0.0,
                "sparse_reward": 0.0,
                "tilt_penalty": 0.0,
                "gait_reward": 0.0,
                "joint_limit_penalty": 0.0,
                "pitch_rate_penalty": 0.0,
                "height_penalty": 0.0,
                "stability_bonus": 0.0,
                "terminal_event_reward": 0.0,
                "forward_tilt_deg": 0.0,
                "side_tilt_deg": 0.0,
            }
            self.last_done_reason = "bench_mode"
            next_state = self.get_state()
            end_step_time = time.time()
            step_time = end_step_time - start_step_time
            return next_state, 0.0, False, step_time

        next_state = self.get_state()
        # ---- REWARD ---
        # ----------  a) progrès vers l'avant (potentiel signe sur -Z) --------------
        # L'avant du quadruped est defini par une progression vers les z negatifs.
        # Utiliser abs(z) rewardait aussi les derivees vers l'arriere.
        forward_progress = max(0.0, -float(self.quadruped.position[2]))
        if self.prev_potential is None:
            distance_reward = 0.0
        else:
            distance_reward = PROGRESS_REWARD_COEF * (forward_progress - self.prev_potential)
        self.prev_potential = forward_progress
        self.prev_radius = forward_progress

        # ----------  b) vitesse avant (faible shaping) ----------------------------
        forward_speed = -float(self.quadruped.velocity[2])
        z_speed_reward = FORWARD_SPEED_REWARD_COEF * float(np.clip(forward_speed, -1.0, 1.0))

        # ----------  c)  Pénalité d'inclinaison du corps (pénalité brute) --------------
        forward_tilt, side_tilt = self._compute_body_tilt_angles()
        max_tilt = max(forward_tilt, side_tilt)
        tilt_excess = max(0.0, forward_tilt - TILT_DEADZONE) + max(0.0, side_tilt - TILT_DEADZONE)
        tilt_penalty = -TILT_PENALTY_COEF * tilt_excess

        # ----------  d) checkpoints clairsemés le long de l'axe d'avance ----------
        sparse_reward = 0.0
        for r in self.circle_radii:
            if forward_progress >= r and r not in self.circles_passed:
                sparse_reward += 5.0
                self.circles_passed.add(r)

        # ----------  Termination checker --------------
        body_height = float(self.quadruped.position[1])
        below_critical_height = body_height < MIN_BODY_HEIGHT
        above_critical_height = body_height > MAX_BODY_HEIGHT
        height_penalty = HEIGHT_PENALTY if (below_critical_height or above_critical_height) else 0.0

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

        shoulder_near_limit = np.abs(self.quadruped.shoulder_angles) > JOINT_LIMIT_THRESHOLD
        elbow_near_limit = np.abs(self.quadruped.elbow_angles) > JOINT_LIMIT_THRESHOLD
        self.consecutive_shoulder_limit_steps = np.where(
            shoulder_near_limit,
            self.consecutive_shoulder_limit_steps + 1,
            0,
        )
        self.consecutive_elbow_limit_steps = np.where(
            elbow_near_limit,
            self.consecutive_elbow_limit_steps + 1,
            0,
        )
        joint_limit_progress = self._get_joint_limit_progress()
        joint_limit_penalty = -JOINT_LIMIT_PROGRESS_PENALTY_COEF * float(np.mean(np.square(joint_limit_progress)))
        gait_reward = joint_limit_penalty

        pitch_rate = abs(float(self.quadruped.angular_velocity[0]))
        pitch_rate_penalty = -PITCH_RATE_PENALTY_COEF * min(pitch_rate, MAX_ANGULAR_VELOCITY)

        tilt_reward_scale = self._compute_tilt_reward_scale(max_tilt)
        height_reward_scale = self._compute_height_reward_scale(body_height)
        locomotion_reward_scale = tilt_reward_scale * height_reward_scale

        # On attenue progressivement les rewards locomotion quand le corps se couche
        # ou s'eloigne de sa fenetre de hauteur utile, au lieu de coupures brutales.
        distance_reward *= locomotion_reward_scale
        z_speed_reward *= locomotion_reward_scale
        sparse_reward *= locomotion_reward_scale

        stable_motion = forward_speed > 0.0
        stability_bonus = STABILITY_BONUS * locomotion_reward_scale if stable_motion else 0.0

        joint_limit_timeout = (
            int(self.consecutive_shoulder_limit_steps.max()) > MAX_CONSECUTIVE_JOINT_LIMIT_STEPS
            or int(self.consecutive_elbow_limit_steps.max()) > MAX_CONSECUTIVE_JOINT_LIMIT_STEPS
        )

        done_reason = "running"
        terminal_event_reward = 0.0

        # Les etats graves restent penalises, mais ne terminent plus l'episode.
        # L'objectif est d'eviter que l'agent apprenne a provoquer une terminaison
        # anticipee plutot qu'a se corriger et continuer a marcher.
        if self.consecutive_steps_above_critical_height > 20:
            done_reason = "too_high"
            terminal_event_reward = TERMINAL_PENALTY_TOO_HIGH
        elif self.consecutive_steps_critical_tilt > MAX_CONSECUTIVE_CRITICAL_TILT_STEPS:
            done_reason = "critical_tilt"
            terminal_event_reward = TERMINAL_PENALTY_CRITICAL_TILT
        elif joint_limit_timeout:
            done_reason = "joint_limit_timeout"
            terminal_event_reward = TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT
        done = False

        # Inform the quadruped that it is in danger
        self.quadruped.too_high = above_critical_height
        self.quadruped.steps_since_too_high = self.consecutive_steps_above_critical_height
        self.quadruped.too_low = below_critical_height
        self.quadruped.steps_since_too_low = self.consecutive_steps_below_critical_height

        # ----------  d)  Somme finale -------------------------
        reward = (distance_reward
                  + z_speed_reward
                  + sparse_reward
                  + tilt_penalty
                  + joint_limit_penalty
                  + pitch_rate_penalty
                  + height_penalty
                  + stability_bonus
                  + terminal_event_reward)
        self.last_reward_components = {
            "distance_reward": float(distance_reward),
            "z_speed_reward": float(z_speed_reward),
            "sparse_reward": float(sparse_reward),
            "tilt_penalty": float(tilt_penalty),
            "gait_reward": float(gait_reward),
            "joint_limit_penalty": float(joint_limit_penalty),
            "pitch_rate_penalty": float(pitch_rate_penalty),
            "height_penalty": float(height_penalty),
            "stability_bonus": float(stability_bonus),
            "terminal_event_reward": float(terminal_event_reward),
            "tilt_reward_scale": float(tilt_reward_scale),
            "height_reward_scale": float(height_reward_scale),
            "locomotion_reward_scale": float(locomotion_reward_scale),
            "forward_tilt_deg": float(np.degrees(forward_tilt)),
            "side_tilt_deg": float(np.degrees(side_tilt)),
        }
        self.last_done_reason = done_reason
        # -----------------------------------------------------
        end_step_time = time.time()
        step_time = end_step_time - start_step_time

        return next_state, reward, done, step_time

    def render(self, reward, done = False, step_time = 0.0, state_value = None):
        """Render the 3D world and UI."""
        self.screen.fill(BLACK)
        self.ground.draw_premium(self.screen, self.camera)
        self.ground.draw_axes(self.screen, self.camera)
        self.draw_checkpoint_circles()
        self.quadruped.draw_premium(self.screen, self.camera)
        self.render_ui(reward, done, step_time, state_value)
        if not self.headless:
            pygame.display.flip()
    
    def get_state(self):
        """Get the current state of the quadruped."""
        base_state = np.asarray(self.quadruped.get_state(), dtype=np.float32)
        joint_limit_progress = self._get_joint_limit_progress()
        state = np.concatenate([base_state, joint_limit_progress], dtype=np.float32)
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
        score_text = f"Score cercles: {len(self.circles_passed)}"
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
            self.font.render(score_text, True, WHITE),
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

    def draw_checkpoint_circles(self):
        """Dessine les cercles de récompense au sol."""
        segments = 36
        for r in self.circle_radii:
            pts = []
            for theta in np.linspace(0, 2*np.pi, segments, endpoint=False):
                world_pt = np.array([r*np.cos(theta), 0.0, r*np.sin(theta)])
                proj = self.camera.project_3d_to_2d(world_pt)
                if proj:                 # point visible
                    pts.append(proj[:2])
            if len(pts) > 1:
                color = (0, 255, 0) if r in self.circles_passed else (100, 100, 100)
                pygame.draw.lines(self.screen, color, True, pts, 1)

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
