"""Formatting helpers for bench output."""


def format_metrics(metrics: dict) -> str:
    lines = [
        f"scenario: {metrics['scenario']}",
        f"seed: {metrics['seed']}",
        f"steps_executed: {metrics['steps_executed']}",
        f"done: {metrics['done']}",
        f"total_reward: {metrics['total_reward']:.6f}",
        f"avg_step_time_ms: {(1000 * metrics['total_step_time_s'] / max(metrics['steps_executed'], 1)):.6f}",
        f"max_ground_penetration: {metrics['max_ground_penetration']:.6f}",
        f"min_body_height: {metrics['min_body_height']}",
        f"max_body_height: {metrics['max_body_height']}",
        f"final_body_height: {metrics['final_body_height']:.6f}",
        f"final_position_x: {metrics['final_position_x']:.6f}",
        f"final_position_z: {metrics['final_position_z']:.6f}",
        f"max_abs_x_drift: {metrics['max_abs_x_drift']:.6f}",
        f"max_abs_tilt: {metrics['max_abs_tilt']:.6f}",
        f"final_speed: {metrics['final_speed']:.6f}",
        f"max_speed: {metrics['max_speed']:.6f}",
        f"final_total_energy: {metrics['final_total_energy']:.6f}",
        f"max_total_energy: {metrics['max_total_energy']:.6f}",
    ]
    return "\n".join(lines)
