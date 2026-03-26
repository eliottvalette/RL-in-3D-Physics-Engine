"""Formatting helpers for bench output."""


def format_metrics(metrics: dict) -> str:
    ordered_keys = [
        "scenario",
        "category",
        "description",
        "seed",
        "steps_executed",
        "done",
        "env_done_seen",
        "first_env_done_time_s",
        "avg_step_time_ms",
        "max_ground_penetration",
        "min_body_height",
        "max_body_height",
        "final_body_height",
        "final_position_x",
        "final_position_z",
        "final_horizontal_distance",
        "max_abs_x_drift",
        "max_abs_tilt",
        "max_pitch_abs",
        "max_yaw_abs",
        "max_roll_abs",
        "tip_time_s",
        "settle_time_s",
        "settle_distance",
        "final_speed",
        "max_speed",
        "max_angular_speed",
        "initial_total_energy",
        "final_total_energy",
        "min_total_energy",
        "max_total_energy",
        "energy_drift_pct",
        "final_contact_count",
        "max_contact_count",
        "avg_contact_count",
        "final_support_polygon_area",
        "max_support_polygon_area",
        "final_support_margin",
        "min_support_margin",
        "outside_support_frames",
        "grounded_frames",
        "first_contact_time_s",
        "grounded_height_mean",
        "grounded_height_jitter_rms",
        "grounded_height_peak_to_peak",
        "grounded_vertical_speed_rms",
        "grounded_angular_speed_rms",
        "grounded_vertical_velocity_zero_crossings",
        "avg_contact_churn",
        "max_contact_churn",
        "avg_support_polygon_area_delta",
        "max_support_polygon_area_delta",
    ]

    values = dict(metrics)
    values["avg_step_time_ms"] = 1000 * metrics["total_step_time_s"] / max(metrics["steps_executed"], 1)
    hidden_keys = {"total_step_time_s", "total_reward"}

    lines = []
    for key in ordered_keys:
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")

    for key in sorted(values):
        if key in ordered_keys or key in hidden_keys:
            continue
        value = values[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
