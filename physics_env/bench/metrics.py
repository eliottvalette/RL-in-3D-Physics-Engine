"""Bench metric helpers."""

from dataclasses import asdict, dataclass
import math


@dataclass
class BenchMetrics:
    scenario: str
    seed: int
    steps_executed: int = 0
    done: bool = False
    total_reward: float = 0.0
    total_step_time_s: float = 0.0
    max_ground_penetration: float = 0.0
    min_body_height: float = math.inf
    max_body_height: float = -math.inf
    final_body_height: float = 0.0
    final_position_x: float = 0.0
    final_position_z: float = 0.0
    max_abs_x_drift: float = 0.0
    max_abs_tilt: float = 0.0
    final_speed: float = 0.0
    max_speed: float = 0.0
    final_total_energy: float = 0.0
    max_total_energy: float = 0.0

    def update(self, env, reward: float, step_time: float, done: bool):
        quadruped = env.quadruped
        position = quadruped.position
        velocity = quadruped.velocity
        rotation = quadruped.rotation
        omega = quadruped.angular_velocity

        translational_energy = 0.5 * quadruped.mass * float((velocity ** 2).sum())
        rotational_energy = 0.5 * float((quadruped.I_body * (omega ** 2)).sum())
        total_energy = translational_energy + rotational_energy
        min_vertex_y = float(quadruped.rotated_vertices[:, 1].min())
        penetration = max(0.0, -min_vertex_y)
        tilt = float(sum(abs(angle) for angle in rotation))
        speed = float(math.sqrt(float((velocity ** 2).sum())))

        self.steps_executed += 1
        self.done = done
        self.total_reward += reward
        self.total_step_time_s += step_time
        self.max_ground_penetration = max(self.max_ground_penetration, penetration)
        self.min_body_height = min(self.min_body_height, float(position[1]))
        self.max_body_height = max(self.max_body_height, float(position[1]))
        self.final_body_height = float(position[1])
        self.final_position_x = float(position[0])
        self.final_position_z = float(position[2])
        self.max_abs_x_drift = max(self.max_abs_x_drift, abs(float(position[0])))
        self.max_abs_tilt = max(self.max_abs_tilt, tilt)
        self.final_speed = speed
        self.max_speed = max(self.max_speed, speed)
        self.final_total_energy = total_energy
        self.max_total_energy = max(self.max_total_energy, total_energy)

    def to_dict(self):
        payload = asdict(self)
        if payload["min_body_height"] is math.inf:
            payload["min_body_height"] = None
        if payload["max_body_height"] is -math.inf:
            payload["max_body_height"] = None
        return payload
