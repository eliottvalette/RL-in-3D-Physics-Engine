"""Bench metric helpers."""

from dataclasses import asdict, dataclass
import math

from physics_env.core.config import CONTACT_THRESHOLD_BASE, DT, GRAVITY


def _normalize_angle(angle):
    return ((angle + math.pi) % (2 * math.pi)) - math.pi


def _convex_hull(points):
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _polygon_area(points):
    if len(points) < 3:
        return 0.0

    area = 0.0
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) * 0.5


@dataclass
class BenchMetrics:
    scenario: str
    seed: int
    category: str = "generic"
    description: str = ""
    steps_executed: int = 0
    done: bool = False
    env_done_seen: bool = False
    first_env_done_time_s: float | None = None
    total_reward: float = 0.0
    total_step_time_s: float = 0.0
    initial_body_height: float = 0.0
    max_ground_penetration: float = 0.0
    min_body_height: float = math.inf
    max_body_height: float = -math.inf
    final_body_height: float = 0.0
    initial_position_x: float = 0.0
    initial_position_z: float = 0.0
    final_position_x: float = 0.0
    final_position_z: float = 0.0
    max_abs_x_drift: float = 0.0
    final_horizontal_distance: float = 0.0
    max_abs_tilt: float = 0.0
    max_pitch_abs: float = 0.0
    max_yaw_abs: float = 0.0
    max_roll_abs: float = 0.0
    tip_time_s: float | None = None
    settle_time_s: float | None = None
    settle_distance: float | None = None
    final_speed: float = 0.0
    max_speed: float = 0.0
    max_angular_speed: float = 0.0
    initial_total_energy: float = 0.0
    final_total_energy: float = 0.0
    min_total_energy: float = math.inf
    max_total_energy: float = 0.0
    energy_drift_pct: float = 0.0
    final_contact_count: int = 0
    max_contact_count: int = 0
    avg_contact_count: float = 0.0
    final_support_polygon_area: float = 0.0
    max_support_polygon_area: float = 0.0
    grounded_frames: int = 0
    first_contact_time_s: float | None = None
    grounded_height_mean: float = 0.0
    grounded_height_jitter_rms: float = 0.0
    grounded_height_peak_to_peak: float = 0.0
    grounded_vertical_speed_rms: float = 0.0
    grounded_angular_speed_rms: float = 0.0
    grounded_vertical_velocity_zero_crossings: int = 0
    avg_contact_churn: float = 0.0
    max_contact_churn: int = 0
    avg_support_polygon_area_delta: float = 0.0
    max_support_polygon_area_delta: float = 0.0

    _settle_counter: int = 0
    _grounded_height_sum: float = 0.0
    _grounded_height_sq_sum: float = 0.0
    _grounded_vertical_speed_sq_sum: float = 0.0
    _grounded_angular_speed_sq_sum: float = 0.0
    _grounded_min_height: float = math.inf
    _grounded_max_height: float = -math.inf
    _contact_churn_samples: int = 0
    _contact_churn_sum: float = 0.0
    _support_area_delta_samples: int = 0
    _support_area_delta_sum: float = 0.0
    _prev_contact_indices: tuple[int, ...] | None = None
    _prev_support_polygon_area: float | None = None
    _prev_vertical_velocity: float | None = None

    def update(self, env, reward: float, step_time: float, done: bool, env_done: bool | None = None):
        quadruped = env.quadruped
        position = quadruped.position
        velocity = quadruped.velocity
        rotation = quadruped.rotation
        omega = quadruped.angular_velocity

        translational_energy = 0.5 * quadruped.mass * float((velocity ** 2).sum())
        rotational_energy = 0.5 * float((quadruped.I_body * (omega ** 2)).sum())
        potential_energy = quadruped.mass * abs(float(GRAVITY[1])) * float(position[1])
        total_energy = translational_energy + rotational_energy + potential_energy
        min_vertex_y = float(quadruped.rotated_vertices[:, 1].min())
        penetration = max(0.0, -min_vertex_y)
        pitch_abs = abs(_normalize_angle(float(rotation[0])))
        yaw_abs = abs(_normalize_angle(float(rotation[1])))
        roll_abs = abs(_normalize_angle(float(rotation[2])))
        tilt = pitch_abs + yaw_abs + roll_abs
        speed = float(math.sqrt(float((velocity ** 2).sum())))
        angular_speed = float(math.sqrt(float((omega ** 2).sum())))
        contact_mask = quadruped.rotated_vertices[:, 1] <= CONTACT_THRESHOLD_BASE
        contact_indices = tuple(int(idx) for idx in range(contact_mask.shape[0]) if bool(contact_mask[idx]))
        contact_points = quadruped.rotated_vertices[contact_mask]
        contact_count = int(contact_points.shape[0])
        support_polygon_area = _polygon_area(_convex_hull([(float(point[0]), float(point[2])) for point in contact_points]))

        if self.steps_executed == 0:
            self.initial_body_height = float(position[1])
            self.initial_position_x = float(position[0])
            self.initial_position_z = float(position[2])
            self.initial_total_energy = total_energy

        self.steps_executed += 1
        self.done = done
        if env_done is None:
            env_done = done
        if env_done:
            self.env_done_seen = True
            if self.first_env_done_time_s is None:
                self.first_env_done_time_s = self.steps_executed * DT
        self.total_reward += reward
        self.total_step_time_s += step_time
        self.max_ground_penetration = max(self.max_ground_penetration, penetration)
        self.min_body_height = min(self.min_body_height, float(position[1]))
        self.max_body_height = max(self.max_body_height, float(position[1]))
        self.final_body_height = float(position[1])
        self.final_position_x = float(position[0])
        self.final_position_z = float(position[2])
        self.max_abs_x_drift = max(self.max_abs_x_drift, abs(float(position[0])))
        self.final_horizontal_distance = math.dist((self.initial_position_x, self.initial_position_z), (self.final_position_x, self.final_position_z))
        self.max_abs_tilt = max(self.max_abs_tilt, tilt)
        self.max_pitch_abs = max(self.max_pitch_abs, pitch_abs)
        self.max_yaw_abs = max(self.max_yaw_abs, yaw_abs)
        self.max_roll_abs = max(self.max_roll_abs, roll_abs)
        self.final_speed = speed
        self.max_speed = max(self.max_speed, speed)
        self.max_angular_speed = max(self.max_angular_speed, angular_speed)
        self.final_total_energy = total_energy
        self.min_total_energy = min(self.min_total_energy, total_energy)
        self.max_total_energy = max(self.max_total_energy, total_energy)
        self.final_contact_count = contact_count
        self.max_contact_count = max(self.max_contact_count, contact_count)
        self.avg_contact_count += (contact_count - self.avg_contact_count) / self.steps_executed
        self.final_support_polygon_area = support_polygon_area
        self.max_support_polygon_area = max(self.max_support_polygon_area, support_polygon_area)

        if contact_count > 0:
            self.grounded_frames += 1
            if self.first_contact_time_s is None:
                self.first_contact_time_s = self.steps_executed * DT

            height = float(position[1])
            vertical_speed = float(abs(velocity[1]))
            self._grounded_height_sum += height
            self._grounded_height_sq_sum += height * height
            self._grounded_vertical_speed_sq_sum += vertical_speed * vertical_speed
            self._grounded_angular_speed_sq_sum += angular_speed * angular_speed
            self._grounded_min_height = min(self._grounded_min_height, height)
            self._grounded_max_height = max(self._grounded_max_height, height)

            grounded_frames_f = float(self.grounded_frames)
            self.grounded_height_mean = self._grounded_height_sum / grounded_frames_f
            height_variance = max(0.0, self._grounded_height_sq_sum / grounded_frames_f - self.grounded_height_mean ** 2)
            self.grounded_height_jitter_rms = math.sqrt(height_variance)
            self.grounded_height_peak_to_peak = self._grounded_max_height - self._grounded_min_height
            self.grounded_vertical_speed_rms = math.sqrt(self._grounded_vertical_speed_sq_sum / grounded_frames_f)
            self.grounded_angular_speed_rms = math.sqrt(self._grounded_angular_speed_sq_sum / grounded_frames_f)

            if self._prev_contact_indices is not None:
                churn = len(set(contact_indices).symmetric_difference(self._prev_contact_indices))
                self._contact_churn_samples += 1
                self._contact_churn_sum += churn
                self.max_contact_churn = max(self.max_contact_churn, churn)
                self.avg_contact_churn = self._contact_churn_sum / self._contact_churn_samples

            if self._prev_support_polygon_area is not None:
                support_delta = abs(support_polygon_area - self._prev_support_polygon_area)
                self._support_area_delta_samples += 1
                self._support_area_delta_sum += support_delta
                self.max_support_polygon_area_delta = max(self.max_support_polygon_area_delta, support_delta)
                self.avg_support_polygon_area_delta = self._support_area_delta_sum / self._support_area_delta_samples

            current_vertical_velocity = float(velocity[1])
            if (
                self._prev_vertical_velocity is not None
                and abs(current_vertical_velocity) > 1e-4
                and abs(self._prev_vertical_velocity) > 1e-4
                and current_vertical_velocity * self._prev_vertical_velocity < 0.0
            ):
                self.grounded_vertical_velocity_zero_crossings += 1

            self._prev_contact_indices = contact_indices
            self._prev_support_polygon_area = support_polygon_area
            self._prev_vertical_velocity = current_vertical_velocity
        else:
            self._prev_contact_indices = None
            self._prev_support_polygon_area = None
            self._prev_vertical_velocity = None

        if self.initial_total_energy > 1e-9:
            self.energy_drift_pct = 100.0 * (self.final_total_energy - self.initial_total_energy) / self.initial_total_energy

        if self.tip_time_s is None and max(pitch_abs, roll_abs) >= 0.5:
            self.tip_time_s = self.steps_executed * DT

        if speed < 0.05 and angular_speed < 0.05 and penetration < CONTACT_THRESHOLD_BASE:
            self._settle_counter += 1
        else:
            self._settle_counter = 0

        if self.settle_time_s is None and self._settle_counter >= 20:
            self.settle_time_s = self.steps_executed * DT
            self.settle_distance = self.final_horizontal_distance

    def to_dict(self):
        payload = asdict(self)
        if payload["min_body_height"] is math.inf:
            payload["min_body_height"] = None
        if payload["max_body_height"] is -math.inf:
            payload["max_body_height"] = None
        if payload["min_total_energy"] is math.inf:
            payload["min_total_energy"] = None
        payload.pop("_settle_counter", None)
        payload.pop("_grounded_height_sum", None)
        payload.pop("_grounded_height_sq_sum", None)
        payload.pop("_grounded_vertical_speed_sq_sum", None)
        payload.pop("_grounded_angular_speed_sq_sum", None)
        payload.pop("_grounded_min_height", None)
        payload.pop("_grounded_max_height", None)
        payload.pop("_contact_churn_samples", None)
        payload.pop("_contact_churn_sum", None)
        payload.pop("_support_area_delta_samples", None)
        payload.pop("_support_area_delta_sum", None)
        payload.pop("_prev_contact_indices", None)
        payload.pop("_prev_support_polygon_area", None)
        payload.pop("_prev_vertical_velocity", None)
        return payload
