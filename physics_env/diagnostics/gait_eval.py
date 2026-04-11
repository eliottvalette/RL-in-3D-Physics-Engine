"""Gait diagnostics for policy rollouts run from test.py."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

from physics_env.core.config import (
    DEBUG_GAIT_EVAL_EPISODES,
    DEBUG_GAIT_EVAL_JSON_PATH,
    DEBUG_GAIT_EVAL_MAX_STEPS,
    DEBUG_GAIT_EVAL_SAVE_JSON,
    DT,
    PHYSICS_HZ,
    RENDER_FPS,
    UNIT_SCALE_M,
)
from physics_env.envs.quadruped_env import QuadrupedEnv


LEG_NAMES = ("front_right", "front_left", "back_right", "back_left")
DIAGONAL_A = (0, 3)
DIAGONAL_B = (1, 2)
REWARD_COMPONENT_KEYS = (
    "locomotion_reward",
    "angular_velocity_penalty",
    "joint_limit_penalty",
    "foot_slip_penalty",
    "action_change_penalty",
    "support_degeneracy_penalty",
    "foot_unused_penalty",
    "terminal_event_reward",
)


@dataclass
class GaitEvalResult:
    episodes: list[dict[str, Any]]
    global_summary: dict[str, Any]
    timeseries: list[dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def _max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def _count_sign_changes(values: np.ndarray) -> int:
    nonzero = values[np.abs(values) > 1e-9]
    if nonzero.size < 2:
        return 0
    signs = np.sign(nonzero)
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _action_switch_rate(actions: np.ndarray, duration_s: float) -> float:
    if actions.shape[0] < 2 or duration_s <= 0.0:
        return 0.0
    switches = np.count_nonzero(actions[1:] != actions[:-1])
    return float(switches / (actions.shape[1] * duration_s))


def _contact_switches_per_second(contacts: np.ndarray, duration_s: float) -> dict[str, float]:
    if contacts.shape[0] < 2 or duration_s <= 0.0:
        return {leg_name: 0.0 for leg_name in LEG_NAMES}
    switches = np.count_nonzero(contacts[1:] != contacts[:-1], axis=0)
    return {
        leg_name: float(switches[leg_idx] / duration_s)
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }


def _run_durations_s(mask: np.ndarray, target_value: bool) -> list[float]:
    durations: list[float] = []
    current_length = 0
    for value in mask.astype(bool):
        if value == target_value:
            current_length += 1
        elif current_length > 0:
            durations.append(current_length * DT)
            current_length = 0
    if current_length > 0:
        durations.append(current_length * DT)
    return durations


def _support_pattern(contact_row: np.ndarray) -> str:
    active = [LEG_NAMES[idx] for idx, value in enumerate(contact_row) if value > 0.5]
    return "+".join(active) if active else "airborne"


def _sample_step(
    env: QuadrupedEnv,
    episode_idx: int,
    step_idx: int,
    action: np.ndarray,
    reward: float,
    done: bool,
    step_time: float,
    state_value: float | None,
) -> dict[str, Any]:
    components = env.get_state_components()
    foot_contact = np.asarray(components["foot_contact"], dtype=np.float32)
    foot_heights = np.asarray(components["foot_heights_world"], dtype=np.float32)
    foot_positions_body = np.asarray(components["foot_positions_body"], dtype=np.float32).reshape(4, 3)
    foot_velocities_body = np.asarray(components["foot_velocities_body"], dtype=np.float32).reshape(4, 3)
    reward_components = env.last_reward_components
    com_minus_support = np.asarray(components["com_minus_support_centroid_body_xz"], dtype=np.float32)
    forward_tilt_deg = _safe_float(reward_components.get("forward_tilt_deg"))
    side_tilt_deg = _safe_float(reward_components.get("side_tilt_deg"))
    applied_reward_components = {
        key: _safe_float(reward_components.get(key))
        for key in REWARD_COMPONENT_KEYS
    }
    if done:
        for key in REWARD_COMPONENT_KEYS:
            if key != "terminal_event_reward":
                applied_reward_components[key] = 0.0

    return {
        "episode": int(episode_idx),
        "step": int(step_idx),
        "time_s": float(step_idx * DT),
        "reward": float(reward),
        "done": bool(done),
        "done_reason": str(env.last_done_reason),
        "step_wall_time_s": float(step_time),
        "state_value": None if state_value is None else float(state_value),
        "forward_progress_units": -float(env.quadruped.position[2]),
        "forward_progress_m": -float(env.quadruped.position[2]) * UNIT_SCALE_M,
        "forward_speed_units_s": -float(env.quadruped.velocity[2]),
        "forward_speed_m_s": -float(env.quadruped.velocity[2]) * UNIT_SCALE_M,
        "body_height_units": float(env.quadruped.position[1]),
        "body_height_m": float(env.quadruped.position[1]) * UNIT_SCALE_M,
        "body_vertical_speed_units_s": float(env.quadruped.velocity[1]),
        "body_vertical_speed_m_s": float(env.quadruped.velocity[1]) * UNIT_SCALE_M,
        "forward_tilt_deg": forward_tilt_deg,
        "side_tilt_deg": side_tilt_deg,
        "max_abs_tilt_deg": max(abs(forward_tilt_deg), abs(side_tilt_deg)),
        "angular_velocity_norm": _safe_float(reward_components.get("angular_velocity_norm")),
        "grounded_leg_count": int(np.count_nonzero(foot_contact > 0.5)),
        "com_minus_support_centroid_body_xz_units": com_minus_support.astype(float).tolist(),
        "com_minus_support_centroid_norm_units": float(np.linalg.norm(com_minus_support)),
        "com_minus_support_centroid_norm_m": float(np.linalg.norm(com_minus_support)) * UNIT_SCALE_M,
        "foot_contact": foot_contact.astype(int).tolist(),
        "support_pattern": _support_pattern(foot_contact),
        "foot_heights_world_units": foot_heights.astype(float).tolist(),
        "foot_heights_world_m": (foot_heights * UNIT_SCALE_M).astype(float).tolist(),
        "foot_positions_body_units": foot_positions_body.astype(float).reshape(-1).tolist(),
        "foot_velocities_body_units_s": foot_velocities_body.astype(float).reshape(-1).tolist(),
        "shoulder_angles": env.quadruped.shoulder_angles.astype(float).tolist(),
        "elbow_angles": env.quadruped.elbow_angles.astype(float).tolist(),
        "shoulder_velocities": env.quadruped.shoulder_velocities.astype(float).tolist(),
        "elbow_velocities": env.quadruped.elbow_velocities.astype(float).tolist(),
        "action": action.astype(float).tolist(),
        "locomotion_reward": applied_reward_components["locomotion_reward"],
        "distance_reward": _safe_float(reward_components.get("distance_reward")),
        "raw_distance_reward": _safe_float(reward_components.get("raw_distance_reward")),
        "angular_velocity_penalty": applied_reward_components["angular_velocity_penalty"],
        "joint_limit_penalty": applied_reward_components["joint_limit_penalty"],
        "locomotion_reward_scale": _safe_float(reward_components.get("locomotion_reward_scale")),
        "joint_limit_push_steps_max": _safe_float(reward_components.get("joint_limit_push_steps_max")),
        "terminal_event_reward": applied_reward_components["terminal_event_reward"],
        "foot_slip_penalty": applied_reward_components["foot_slip_penalty"],
        "foot_slip_speed_mean": _safe_float(reward_components.get("foot_slip_speed_mean")),
        "foot_slip_speed_max": _safe_float(reward_components.get("foot_slip_speed_max")),
        "action_change_penalty": applied_reward_components["action_change_penalty"],
        "action_delta_mean_abs": _safe_float(reward_components.get("action_delta_mean_abs")),
        "support_degeneracy_penalty": applied_reward_components["support_degeneracy_penalty"],
        "consecutive_degenerate_support_steps": _safe_float(
            reward_components.get("consecutive_degenerate_support_steps")
        ),
        "foot_unused_penalty": applied_reward_components["foot_unused_penalty"],
        "foot_unused_steps_max": _safe_float(reward_components.get("foot_unused_steps_max")),
        "contact_quality_scale": _safe_float(reward_components.get("contact_quality_scale")),
    }


def _summarize_episode(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}

    duration_s = len(samples) * DT
    forward = np.asarray([sample["forward_progress_units"] for sample in samples], dtype=np.float64)
    forward_m = forward * UNIT_SCALE_M
    forward_deltas = np.diff(forward, prepend=forward[0])
    abs_forward_motion = float(np.sum(np.abs(forward_deltas)))
    net_forward_motion = float(forward[-1] - forward[0])
    net_abs_ratio = float(abs(net_forward_motion) / abs_forward_motion) if abs_forward_motion > 1e-9 else 0.0
    forward_speed = np.asarray([sample["forward_speed_units_s"] for sample in samples], dtype=np.float64)
    body_height = np.asarray([sample["body_height_units"] for sample in samples], dtype=np.float64)
    vertical_speed = np.asarray([sample["body_vertical_speed_units_s"] for sample in samples], dtype=np.float64)
    max_abs_tilt = np.asarray([sample["max_abs_tilt_deg"] for sample in samples], dtype=np.float64)
    angular_velocity = np.asarray([sample["angular_velocity_norm"] for sample in samples], dtype=np.float64)
    actions = np.asarray([sample["action"] for sample in samples], dtype=np.float64)
    contacts = np.asarray([sample["foot_contact"] for sample in samples], dtype=np.float64)
    grounded_leg_count = np.asarray([sample["grounded_leg_count"] for sample in samples], dtype=np.float64)
    foot_heights = np.asarray([sample["foot_heights_world_units"] for sample in samples], dtype=np.float64)
    foot_velocities = np.asarray([sample["foot_velocities_body_units_s"] for sample in samples], dtype=np.float64).reshape(
        len(samples),
        4,
        3,
    )
    foot_planar_speed = np.linalg.norm(foot_velocities[:, :, [0, 2]], axis=2)
    contact_planar_speed = foot_planar_speed[contacts > 0.5]
    support_counts = Counter(sample["support_pattern"] for sample in samples)
    com_support_norm = np.asarray(
        [sample["com_minus_support_centroid_norm_units"] for sample in samples],
        dtype=np.float64,
    )

    diagonal_a = np.logical_and(contacts[:, DIAGONAL_A[0]] > 0.5, contacts[:, DIAGONAL_A[1]] > 0.5)
    diagonal_b = np.logical_and(contacts[:, DIAGONAL_B[0]] > 0.5, contacts[:, DIAGONAL_B[1]] > 0.5)
    diagonal_a_only = np.logical_and(diagonal_a, np.logical_not(diagonal_b))
    diagonal_b_only = np.logical_and(diagonal_b, np.logical_not(diagonal_a))
    diagonal_state = np.where(diagonal_a_only, 1, np.where(diagonal_b_only, -1, 0))
    diagonal_nonzero = diagonal_state[diagonal_state != 0]
    diagonal_switches = _count_sign_changes(diagonal_nonzero.astype(np.float64))
    reward_breakdown = _reward_breakdown(samples)

    return {
        "episode": int(samples[0]["episode"]),
        "steps": int(len(samples)),
        "duration_s": float(duration_s),
        "done_reason": str(samples[-1]["done_reason"]),
        "reward_sum": reward_breakdown["reward_sum"],
        "reward_per_step": reward_breakdown["reward_per_step"],
        "reward_component_sum": reward_breakdown["reward_component_sum"],
        "reward_component_per_step": reward_breakdown["reward_component_per_step"],
        "reward_component_abs_share": reward_breakdown["reward_component_abs_share"],
        "reward_component_signed_share": reward_breakdown["reward_component_signed_share"],
        "reward_reconstructed_sum": reward_breakdown["reward_reconstructed_sum"],
        "reward_residual": reward_breakdown["reward_residual"],
        "net_forward_progress_units": net_forward_motion,
        "net_forward_progress_m": net_forward_motion * UNIT_SCALE_M,
        "absolute_forward_motion_units": abs_forward_motion,
        "absolute_forward_motion_m": abs_forward_motion * UNIT_SCALE_M,
        "net_to_absolute_forward_motion_ratio": net_abs_ratio,
        "forward_speed_mean_units_s": _mean(forward_speed),
        "forward_speed_std_units_s": _std(forward_speed),
        "forward_speed_sign_changes_per_s": float(_count_sign_changes(forward_speed) / max(duration_s, 1e-9)),
        "body_height_mean_units": _mean(body_height),
        "body_height_std_units": _std(body_height),
        "body_height_min_units": _min(body_height),
        "body_height_max_units": _max(body_height),
        "body_vertical_speed_abs_p95_units_s": _percentile(np.abs(vertical_speed), 95.0),
        "max_abs_tilt_mean_deg": _mean(max_abs_tilt),
        "max_abs_tilt_p95_deg": _percentile(max_abs_tilt, 95.0),
        "max_abs_tilt_max_deg": _max(max_abs_tilt),
        "angular_velocity_mean": _mean(angular_velocity),
        "angular_velocity_p95": _percentile(angular_velocity, 95.0),
        "action_mean_abs": float(np.mean(np.abs(actions))) if actions.size else 0.0,
        "action_switches_per_joint_s": _action_switch_rate(actions, duration_s),
        "joint_limit_penalty_sum": float(np.sum([sample["joint_limit_penalty"] for sample in samples])),
        "joint_limit_push_steps_max": float(np.max([sample["joint_limit_push_steps_max"] for sample in samples])),
        "grounded_leg_count_mean": _mean(grounded_leg_count),
        "grounded_leg_count_fraction": {
            str(count): float(np.mean(grounded_leg_count == count))
            for count in range(5)
        },
        "duty_factor": {
            leg_name: float(np.mean(contacts[:, leg_idx] > 0.5))
            for leg_idx, leg_name in enumerate(LEG_NAMES)
        },
        "stance_duration_mean_s": {
            leg_name: _mean(np.asarray(_run_durations_s(contacts[:, leg_idx] > 0.5, True), dtype=np.float64))
            for leg_idx, leg_name in enumerate(LEG_NAMES)
        },
        "swing_duration_mean_s": {
            leg_name: _mean(np.asarray(_run_durations_s(contacts[:, leg_idx] > 0.5, False), dtype=np.float64))
            for leg_idx, leg_name in enumerate(LEG_NAMES)
        },
        "contact_switches_per_leg_s": _contact_switches_per_second(contacts, duration_s),
        "support_pattern_fraction": {
            pattern: float(count / len(samples))
            for pattern, count in sorted(support_counts.items())
        },
        "diagonal_a_front_right_back_left_fraction": float(np.mean(diagonal_a_only)),
        "diagonal_b_front_left_back_right_fraction": float(np.mean(diagonal_b_only)),
        "diagonal_switches_per_s": float(diagonal_switches / max(duration_s, 1e-9)),
        "foot_height_min_units": {
            leg_name: float(np.min(foot_heights[:, leg_idx]))
            for leg_idx, leg_name in enumerate(LEG_NAMES)
        },
        "foot_height_p95_units": {
            leg_name: float(np.percentile(foot_heights[:, leg_idx], 95.0))
            for leg_idx, leg_name in enumerate(LEG_NAMES)
        },
        "swing_foot_height_p95_units": {
            leg_name: _percentile(foot_heights[contacts[:, leg_idx] < 0.5, leg_idx], 95.0)
            for leg_idx, leg_name in enumerate(LEG_NAMES)
        },
        "com_minus_support_centroid_norm_mean_units": _mean(com_support_norm),
        "com_minus_support_centroid_norm_p95_units": _percentile(com_support_norm, 95.0),
        "contact_foot_planar_speed_mean_units_s": _mean(contact_planar_speed),
        "contact_foot_planar_speed_p95_units_s": _percentile(contact_planar_speed, 95.0),
        "physics_hz": float(PHYSICS_HZ),
        "dt_s": float(DT),
        "unit_scale_m": float(UNIT_SCALE_M),
    }


def _summarize_global(episode_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not episode_summaries:
        return {}

    numeric_keys = [
        "steps",
        "duration_s",
        "reward_sum",
        "net_forward_progress_units",
        "net_forward_progress_m",
        "absolute_forward_motion_units",
        "net_to_absolute_forward_motion_ratio",
        "forward_speed_mean_units_s",
        "forward_speed_std_units_s",
        "forward_speed_sign_changes_per_s",
        "body_height_std_units",
        "max_abs_tilt_p95_deg",
        "angular_velocity_p95",
        "action_mean_abs",
        "action_switches_per_joint_s",
        "joint_limit_penalty_sum",
        "joint_limit_push_steps_max",
        "grounded_leg_count_mean",
        "com_minus_support_centroid_norm_p95_units",
        "diagonal_switches_per_s",
        "contact_foot_planar_speed_p95_units_s",
    ]
    summary: dict[str, Any] = {
        "episodes": len(episode_summaries),
        "physics_hz": float(PHYSICS_HZ),
        "render_fps": float(RENDER_FPS),
        "dt_s": float(DT),
        "unit_scale_m": float(UNIT_SCALE_M),
        "done_reasons": dict(Counter(item["done_reason"] for item in episode_summaries)),
    }
    for key in numeric_keys:
        values = np.asarray([item[key] for item in episode_summaries], dtype=np.float64)
        summary[f"{key}_mean"] = _mean(values)
        summary[f"{key}_min"] = _min(values)
        summary[f"{key}_max"] = _max(values)
    reward_component_sum = {
        key: float(np.sum([item["reward_component_sum"].get(key, 0.0) for item in episode_summaries]))
        for key in REWARD_COMPONENT_KEYS
    }
    total_steps = float(np.sum([item["steps"] for item in episode_summaries]))
    reward_reconstructed_sum = float(sum(reward_component_sum.values()))
    reward_sum = float(np.sum([item["reward_sum"] for item in episode_summaries]))
    reward_abs_total = float(sum(abs(value) for value in reward_component_sum.values()))
    summary["reward_sum_total"] = reward_sum
    summary["reward_component_sum"] = reward_component_sum
    summary["reward_component_per_step"] = {
        key: value / max(total_steps, 1e-9)
        for key, value in reward_component_sum.items()
    }
    summary["reward_component_abs_share"] = {
        key: (abs(value) / reward_abs_total if reward_abs_total > 1e-9 else 0.0)
        for key, value in reward_component_sum.items()
    }
    summary["reward_component_signed_share"] = {
        key: (value / reward_sum if abs(reward_sum) > 1e-9 else 0.0)
        for key, value in reward_component_sum.items()
    }
    summary["reward_reconstructed_sum"] = reward_reconstructed_sum
    summary["reward_residual"] = reward_sum - reward_reconstructed_sum
    return summary


def _format_dict(values: dict[str, Any], precision: int = 3) -> str:
    formatted = []
    for key, value in values.items():
        if isinstance(value, float):
            formatted.append(f"{key}={value:.{precision}f}")
        else:
            formatted.append(f"{key}={value}")
    return ", ".join(formatted)


def _format_percent_dict(values: dict[str, float], precision: int = 1) -> str:
    return ", ".join(
        f"{key}={value * 100.0:.{precision}f}%"
        for key, value in values.items()
    )


def _sorted_reward_keys(component_sum: dict[str, float]) -> list[str]:
    return sorted(
        REWARD_COMPONENT_KEYS,
        key=lambda key: abs(component_sum.get(key, 0.0)),
        reverse=True,
    )


def _reward_breakdown(samples: list[dict[str, Any]]) -> dict[str, Any]:
    reward_sum = float(np.sum([sample["reward"] for sample in samples]))
    component_sum = {
        key: float(np.sum([sample.get(key, 0.0) for sample in samples]))
        for key in REWARD_COMPONENT_KEYS
    }
    component_per_step = {
        key: value / max(len(samples), 1)
        for key, value in component_sum.items()
    }
    reconstructed_sum = float(sum(component_sum.values()))
    abs_total = float(sum(abs(value) for value in component_sum.values()))
    abs_share = {
        key: (abs(value) / abs_total if abs_total > 1e-9 else 0.0)
        for key, value in component_sum.items()
    }
    signed_share = {
        key: (value / reward_sum if abs(reward_sum) > 1e-9 else 0.0)
        for key, value in component_sum.items()
    }
    return {
        "reward_sum": reward_sum,
        "reward_per_step": reward_sum / max(len(samples), 1),
        "reward_component_sum": component_sum,
        "reward_component_per_step": component_per_step,
        "reward_component_abs_share": abs_share,
        "reward_component_signed_share": signed_share,
        "reward_reconstructed_sum": reconstructed_sum,
        "reward_residual": reward_sum - reconstructed_sum,
    }


def print_gait_eval_report(result: GaitEvalResult, json_path: str | None) -> None:
    global_summary = result.global_summary
    print("\n[GAIT DEBUG] Configuration")
    print(
        "[GAIT DEBUG]",
        f"episodes={global_summary.get('episodes', 0)}",
        f"physics_hz={global_summary.get('physics_hz', 0):.1f}",
        f"render_fps={global_summary.get('render_fps', 0):.1f}",
        f"dt_s={global_summary.get('dt_s', 0):.6f}",
        f"unit_scale_m={global_summary.get('unit_scale_m', 0):.3f}",
    )
    if json_path:
        print(f"[GAIT DEBUG] JSON: {json_path}")

    print("\n[GAIT DEBUG] Global Summary")
    print(f"[GAIT DEBUG] done_reasons: {global_summary.get('done_reasons', {})}")
    for key in (
        "reward_sum",
        "net_forward_progress_m",
        "absolute_forward_motion_units",
        "net_to_absolute_forward_motion_ratio",
        "forward_speed_sign_changes_per_s",
        "body_height_std_units",
        "max_abs_tilt_p95_deg",
        "angular_velocity_p95",
        "action_switches_per_joint_s",
        "grounded_leg_count_mean",
        "com_minus_support_centroid_norm_p95_units",
        "diagonal_switches_per_s",
        "contact_foot_planar_speed_p95_units_s",
        "joint_limit_penalty_sum",
    ):
        print(
            "[GAIT DEBUG]",
            f"{key}:",
            f"mean={global_summary.get(f'{key}_mean', 0.0):.4f}",
            f"min={global_summary.get(f'{key}_min', 0.0):.4f}",
            f"max={global_summary.get(f'{key}_max', 0.0):.4f}",
        )

    print("\n[GAIT DEBUG] Reward Breakdown")
    print(
        "[GAIT DEBUG]",
        f"observed_sum={global_summary.get('reward_sum_total', 0.0):.4f}",
        f"reconstructed_sum={global_summary.get('reward_reconstructed_sum', 0.0):.4f}",
        f"residual={global_summary.get('reward_residual', 0.0):.6f}",
    )
    reward_component_sum = global_summary.get("reward_component_sum", {})
    reward_component_per_step = global_summary.get("reward_component_per_step", {})
    reward_component_abs_share = global_summary.get("reward_component_abs_share", {})
    reward_component_signed_share = global_summary.get("reward_component_signed_share", {})
    for key in _sorted_reward_keys(reward_component_sum):
        print(
            "[GAIT DEBUG]",
            f"  {key}:",
            f"sum={reward_component_sum.get(key, 0.0):.4f}",
            f"per_step={reward_component_per_step.get(key, 0.0):.5f}",
            f"abs_responsibility={reward_component_abs_share.get(key, 0.0) * 100.0:.1f}%",
            f"signed_share={reward_component_signed_share.get(key, 0.0) * 100.0:.1f}%",
        )

    print("\n[GAIT DEBUG] Episodes")
    for episode in result.episodes:
        print(
            "[GAIT DEBUG]",
            f"ep={episode['episode']}",
            f"steps={episode['steps']}",
            f"duration_s={episode['duration_s']:.3f}",
            f"done={episode['done_reason']}",
            f"reward={episode['reward_sum']:.3f}",
            f"reward_step={episode['reward_per_step']:.4f}",
            f"net_m={episode['net_forward_progress_m']:.3f}",
            f"net_abs_ratio={episode['net_to_absolute_forward_motion_ratio']:.3f}",
            f"speed_sign_changes_s={episode['forward_speed_sign_changes_per_s']:.3f}",
            f"height_std={episode['body_height_std_units']:.3f}",
            f"tilt_p95={episode['max_abs_tilt_p95_deg']:.2f}",
            f"action_switch_s={episode['action_switches_per_joint_s']:.3f}",
        )
        print("[GAIT DEBUG]   duty:", _format_dict(episode["duty_factor"]))
        print("[GAIT DEBUG]   stance_s:", _format_dict(episode["stance_duration_mean_s"]))
        print("[GAIT DEBUG]   swing_s:", _format_dict(episode["swing_duration_mean_s"]))
        print("[GAIT DEBUG]   swing_clearance_p95_units:", _format_dict(episode["swing_foot_height_p95_units"]))
        print("[GAIT DEBUG]   support:", _format_dict(episode["support_pattern_fraction"]))
        ordered_episode_components = {
            key: episode["reward_component_sum"].get(key, 0.0)
            for key in _sorted_reward_keys(episode["reward_component_sum"])
        }
        ordered_episode_share = {
            key: episode["reward_component_abs_share"].get(key, 0.0)
            for key in ordered_episode_components
        }
        print("[GAIT DEBUG]   reward_components_sum:", _format_dict(ordered_episode_components, precision=4))
        print("[GAIT DEBUG]   reward_abs_responsibility:", _format_percent_dict(ordered_episode_share))

    print("\n[GAIT DEBUG] Reading Guide")
    print("[GAIT DEBUG] net_to_absolute_forward_motion_ratio close to 1 means little back-and-forth drift.")
    print("[GAIT DEBUG] duty_factor is per-leg stance fraction; diagonal fractions show FR+BL vs FL+BR support use.")
    print("[GAIT DEBUG] contact_foot_planar_speed is a foot-drag proxy measured while the foot is in contact.")
    print("[GAIT DEBUG] action_switches_per_joint_s measures command dithering rate normalized by simulated seconds.")
    print("[GAIT DEBUG] reward_abs_responsibility uses absolute component magnitudes, so it shows domination even when signs cancel.")


def save_gait_eval_json(result: GaitEvalResult, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "global_summary": result.global_summary,
        "episodes": result.episodes,
        "timeseries": result.timeseries,
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    return output_path


def run_gait_debug_eval(agent, env: QuadrupedEnv, render: bool = True) -> GaitEvalResult:
    all_samples: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    for episode_idx in range(DEBUG_GAIT_EVAL_EPISODES):
        state = env.reset_episode(pose_jitter=True)
        episode_samples: list[dict[str, Any]] = []
        camera_actions = [0] * 10
        reward = 0.0
        done = False
        step_time = 0.0
        stop_requested = False

        for step_idx in range(DEBUG_GAIT_EVAL_MAX_STEPS):
            if render:
                if not env.handle_events():
                    stop_requested = True
                    break
                keys = pygame.key.get_pressed()
                camera_actions = env.handle_camera_controls(keys)

            shoulders, elbows, action_info = agent.get_action(state=state, deterministic=False)
            if action_info is None:
                action_info = {}
            action = np.concatenate(
                [
                    np.asarray(shoulders, dtype=np.float32),
                    np.asarray(elbows, dtype=np.float32),
                ],
                dtype=np.float32,
            )
            state_value = action_info.get("value")
            next_state, reward, done, step_time = env.step(shoulders, elbows, camera_actions)
            sample = _sample_step(
                env=env,
                episode_idx=episode_idx,
                step_idx=step_idx,
                action=action,
                reward=reward,
                done=done,
                step_time=step_time,
                state_value=state_value,
            )
            episode_samples.append(sample)
            all_samples.append(sample)

            if render:
                env.render(reward, done, step_time, state_value)

            if done:
                break
            state = next_state

        if episode_samples:
            episode_summaries.append(_summarize_episode(episode_samples))
        if stop_requested:
            break

    result = GaitEvalResult(
        episodes=episode_summaries,
        global_summary=_summarize_global(episode_summaries),
        timeseries=all_samples,
    )
    json_path = save_gait_eval_json(result, DEBUG_GAIT_EVAL_JSON_PATH) if DEBUG_GAIT_EVAL_SAVE_JSON else None
    print_gait_eval_report(result, json_path=json_path)
    return result
