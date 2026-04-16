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
    DEBUG_GAIT_EVAL_PRINT_EPISODES,
    DEBUG_GAIT_EVAL_SAVE_JSON,
    DT,
    PHYSICS_HZ,
    RENDER_FPS,
    TASK_FORWARD_Z_SIGN,
    UNIT_SCALE_M,
)
from physics_env.envs.quadruped_env import QuadrupedEnv


LEG_NAMES = ("front_right", "front_left", "back_right", "back_left")
FRONT_LEGS = (0, 1)
REAR_LEGS = (2, 3)
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


def _dict_mean(dicts: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, float]:
    if not dicts:
        return {key: 0.0 for key in keys}
    return {
        key: float(np.mean([values.get(key, 0.0) for values in dicts]))
        for key in keys
    }


def _dict_sum(dicts: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, float]:
    if not dicts:
        return {key: 0.0 for key in keys}
    return {
        key: float(np.sum([values.get(key, 0.0) for values in dicts]))
        for key in keys
    }


def _positive_share(values: np.ndarray, mask: np.ndarray) -> float:
    positives = np.maximum(values, 0.0)
    total = float(np.sum(positives))
    if total <= 1e-9:
        return 0.0
    return float(np.sum(positives[mask]) / total)


def _component_share_by_leg(values: np.ndarray, positive_only: bool = False) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        totals = np.sum(values, axis=0)
    else:
        totals = values
    if positive_only:
        totals = np.maximum(totals, 0.0)
    denominator = float(np.sum(totals))
    if denominator <= 1e-9:
        return {leg_name: 0.0 for leg_name in LEG_NAMES}
    return {
        leg_name: float(totals[leg_idx] / denominator)
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
        "forward_progress_units": TASK_FORWARD_Z_SIGN * float(env.quadruped.position[2]),
        "forward_progress_m": TASK_FORWARD_Z_SIGN * float(env.quadruped.position[2]) * UNIT_SCALE_M,
        "forward_speed_units_s": TASK_FORWARD_Z_SIGN * float(env.quadruped.velocity[2]),
        "forward_speed_m_s": TASK_FORWARD_Z_SIGN * float(env.quadruped.velocity[2]) * UNIT_SCALE_M,
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
        "contact_normal_impulses_by_leg": env.quadruped.last_contact_normal_impulses_by_leg.astype(float).tolist(),
        "contact_tangent_impulses_by_leg": env.quadruped.last_contact_tangent_impulses_by_leg.astype(float).reshape(-1).tolist(),
        "contact_tangent_forward_impulses_by_leg": (
            env.quadruped.last_contact_tangent_forward_impulses_by_leg.astype(float).tolist()
        ),
        "contact_tangent_lateral_impulses_by_leg": (
            env.quadruped.last_contact_tangent_lateral_impulses_by_leg.astype(float).tolist()
        ),
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
    shoulder_actions = actions[:, :4] if actions.size else np.empty((0, 4), dtype=np.float64)
    elbow_actions = actions[:, 4:] if actions.size else np.empty((0, 4), dtype=np.float64)
    contacts = np.asarray([sample["foot_contact"] for sample in samples], dtype=np.float64)
    grounded_leg_count = np.asarray([sample["grounded_leg_count"] for sample in samples], dtype=np.float64)
    foot_heights = np.asarray([sample["foot_heights_world_units"] for sample in samples], dtype=np.float64)
    foot_velocities = np.asarray([sample["foot_velocities_body_units_s"] for sample in samples], dtype=np.float64).reshape(
        len(samples),
        4,
        3,
    )
    contact_normal_impulses = np.asarray(
        [sample["contact_normal_impulses_by_leg"] for sample in samples],
        dtype=np.float64,
    )
    contact_tangent_impulses = np.asarray(
        [sample["contact_tangent_impulses_by_leg"] for sample in samples],
        dtype=np.float64,
    ).reshape(len(samples), 4, 3)
    contact_forward_impulses = np.asarray(
        [sample["contact_tangent_forward_impulses_by_leg"] for sample in samples],
        dtype=np.float64,
    )
    contact_lateral_impulses = np.asarray(
        [sample["contact_tangent_lateral_impulses_by_leg"] for sample in samples],
        dtype=np.float64,
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
    front_pair = np.logical_and(contacts[:, FRONT_LEGS[0]] > 0.5, contacts[:, FRONT_LEGS[1]] > 0.5)
    rear_pair = np.logical_and(contacts[:, REAR_LEGS[0]] > 0.5, contacts[:, REAR_LEGS[1]] > 0.5)
    front_any = np.any(contacts[:, FRONT_LEGS] > 0.5, axis=1)
    rear_any = np.any(contacts[:, REAR_LEGS] > 0.5, axis=1)
    rear_pair_without_front = np.logical_and(rear_pair, np.logical_not(front_any))
    front_pair_without_rear = np.logical_and(front_pair, np.logical_not(rear_any))
    diagonal_any = np.logical_or(diagonal_a, diagonal_b)
    diagonal_only = np.logical_or(diagonal_a_only, diagonal_b_only)
    forward_accel = np.diff(forward_speed, prepend=forward_speed[0]) / max(DT, 1e-9)
    positive_forward_accel = forward_accel > 0.0
    positive_forward_progress_total = float(np.sum(np.maximum(forward_deltas, 0.0)))
    positive_forward_accel_total = float(np.sum(np.maximum(forward_accel, 0.0)))
    per_leg_positive_accel_share = {
        leg_name: _positive_share(forward_accel, contacts[:, leg_idx] > 0.5)
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_positive_progress_share = {
        leg_name: _positive_share(forward_deltas, contacts[:, leg_idx] > 0.5)
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_when_positive_accel_fraction = {
        leg_name: float(np.mean(contacts[positive_forward_accel, leg_idx] > 0.5))
        if np.any(positive_forward_accel) else 0.0
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_action_abs_mean = {
        leg_name: float(np.mean((np.abs(shoulder_actions[:, leg_idx]) + np.abs(elbow_actions[:, leg_idx])) * 0.5))
        if shoulder_actions.size and elbow_actions.size else 0.0
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_shoulder_action_abs_mean = {
        leg_name: float(np.mean(np.abs(shoulder_actions[:, leg_idx]))) if shoulder_actions.size else 0.0
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_elbow_action_abs_mean = {
        leg_name: float(np.mean(np.abs(elbow_actions[:, leg_idx]))) if elbow_actions.size else 0.0
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_action_switches_s = {}
    if actions.shape[0] >= 2 and duration_s > 0.0:
        for leg_idx, leg_name in enumerate(LEG_NAMES):
            leg_switches = np.count_nonzero(shoulder_actions[1:, leg_idx] != shoulder_actions[:-1, leg_idx])
            leg_switches += np.count_nonzero(elbow_actions[1:, leg_idx] != elbow_actions[:-1, leg_idx])
            per_leg_action_switches_s[leg_name] = float(leg_switches / (2.0 * duration_s))
    else:
        per_leg_action_switches_s = {leg_name: 0.0 for leg_name in LEG_NAMES}
    front_action_abs = float(np.sum([per_leg_action_abs_mean[LEG_NAMES[idx]] for idx in FRONT_LEGS]))
    rear_action_abs = float(np.sum([per_leg_action_abs_mean[LEG_NAMES[idx]] for idx in REAR_LEGS]))
    rear_action_abs_share = rear_action_abs / max(front_action_abs + rear_action_abs, 1e-9)
    front_positive_accel_share = float(sum(per_leg_positive_accel_share[LEG_NAMES[idx]] for idx in FRONT_LEGS))
    rear_positive_accel_share = float(sum(per_leg_positive_accel_share[LEG_NAMES[idx]] for idx in REAR_LEGS))
    front_positive_progress_share = float(sum(per_leg_positive_progress_share[LEG_NAMES[idx]] for idx in FRONT_LEGS))
    rear_positive_progress_share = float(sum(per_leg_positive_progress_share[LEG_NAMES[idx]] for idx in REAR_LEGS))
    contact_tangent_norm = np.linalg.norm(contact_tangent_impulses, axis=2)
    positive_contact_forward_impulses = np.maximum(contact_forward_impulses, 0.0)
    abs_contact_forward_impulses = np.abs(contact_forward_impulses)
    per_leg_contact_normal_impulse_sum = {
        leg_name: float(np.sum(contact_normal_impulses[:, leg_idx]))
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_contact_tangent_impulse_norm_sum = {
        leg_name: float(np.sum(contact_tangent_norm[:, leg_idx]))
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_contact_forward_impulse_sum = {
        leg_name: float(np.sum(contact_forward_impulses[:, leg_idx]))
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_contact_positive_forward_impulse_sum = {
        leg_name: float(np.sum(positive_contact_forward_impulses[:, leg_idx]))
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_contact_abs_forward_impulse_sum = {
        leg_name: float(np.sum(abs_contact_forward_impulses[:, leg_idx]))
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_contact_lateral_impulse_sum = {
        leg_name: float(np.sum(contact_lateral_impulses[:, leg_idx]))
        for leg_idx, leg_name in enumerate(LEG_NAMES)
    }
    per_leg_contact_positive_forward_impulse_share = _component_share_by_leg(
        contact_forward_impulses,
        positive_only=True,
    )
    per_leg_contact_abs_forward_impulse_share = _component_share_by_leg(abs_contact_forward_impulses)
    per_leg_contact_normal_impulse_share = _component_share_by_leg(contact_normal_impulses)
    front_contact_positive_forward_impulse_share = float(
        sum(per_leg_contact_positive_forward_impulse_share[LEG_NAMES[idx]] for idx in FRONT_LEGS)
    )
    rear_contact_positive_forward_impulse_share = float(
        sum(per_leg_contact_positive_forward_impulse_share[LEG_NAMES[idx]] for idx in REAR_LEGS)
    )
    front_contact_abs_forward_impulse_share = float(
        sum(per_leg_contact_abs_forward_impulse_share[LEG_NAMES[idx]] for idx in FRONT_LEGS)
    )
    rear_contact_abs_forward_impulse_share = float(
        sum(per_leg_contact_abs_forward_impulse_share[LEG_NAMES[idx]] for idx in REAR_LEGS)
    )
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
        "per_leg_positive_accel_share": per_leg_positive_accel_share,
        "per_leg_positive_progress_share": per_leg_positive_progress_share,
        "per_leg_when_positive_accel_fraction": per_leg_when_positive_accel_fraction,
        "per_leg_action_abs_mean": per_leg_action_abs_mean,
        "per_leg_shoulder_action_abs_mean": per_leg_shoulder_action_abs_mean,
        "per_leg_elbow_action_abs_mean": per_leg_elbow_action_abs_mean,
        "per_leg_action_switches_s": per_leg_action_switches_s,
        "rear_action_abs_share": rear_action_abs_share,
        "front_positive_accel_share": front_positive_accel_share,
        "rear_positive_accel_share": rear_positive_accel_share,
        "front_positive_progress_share": front_positive_progress_share,
        "rear_positive_progress_share": rear_positive_progress_share,
        "per_leg_contact_normal_impulse_sum": per_leg_contact_normal_impulse_sum,
        "per_leg_contact_tangent_impulse_norm_sum": per_leg_contact_tangent_impulse_norm_sum,
        "per_leg_contact_forward_impulse_sum": per_leg_contact_forward_impulse_sum,
        "per_leg_contact_positive_forward_impulse_sum": per_leg_contact_positive_forward_impulse_sum,
        "per_leg_contact_abs_forward_impulse_sum": per_leg_contact_abs_forward_impulse_sum,
        "per_leg_contact_lateral_impulse_sum": per_leg_contact_lateral_impulse_sum,
        "per_leg_contact_positive_forward_impulse_share": per_leg_contact_positive_forward_impulse_share,
        "per_leg_contact_abs_forward_impulse_share": per_leg_contact_abs_forward_impulse_share,
        "per_leg_contact_normal_impulse_share": per_leg_contact_normal_impulse_share,
        "front_contact_positive_forward_impulse_share": front_contact_positive_forward_impulse_share,
        "rear_contact_positive_forward_impulse_share": rear_contact_positive_forward_impulse_share,
        "front_contact_abs_forward_impulse_share": front_contact_abs_forward_impulse_share,
        "rear_contact_abs_forward_impulse_share": rear_contact_abs_forward_impulse_share,
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
        "diagonal_contact_fraction": float(np.mean(diagonal_any)),
        "diagonal_only_fraction": float(np.mean(diagonal_only)),
        "diagonal_switches_per_s": float(diagonal_switches / max(duration_s, 1e-9)),
        "front_pair_contact_fraction": float(np.mean(front_pair)),
        "rear_pair_contact_fraction": float(np.mean(rear_pair)),
        "front_pair_without_rear_fraction": float(np.mean(front_pair_without_rear)),
        "rear_pair_without_front_fraction": float(np.mean(rear_pair_without_front)),
        "rear_pair_vs_diagonal_contact_ratio": float(np.mean(rear_pair) / max(float(np.mean(diagonal_any)), 1e-9)),
        "positive_forward_accel_fraction": float(np.mean(positive_forward_accel)),
        "rear_pair_when_positive_accel_fraction": float(np.mean(rear_pair[positive_forward_accel]))
        if np.any(positive_forward_accel) else 0.0,
        "diagonal_when_positive_accel_fraction": float(np.mean(diagonal_any[positive_forward_accel]))
        if np.any(positive_forward_accel) else 0.0,
        "rear_pair_positive_accel_share": _positive_share(forward_accel, rear_pair),
        "rear_pair_without_front_positive_accel_share": _positive_share(forward_accel, rear_pair_without_front),
        "front_pair_positive_accel_share": _positive_share(forward_accel, front_pair),
        "diagonal_positive_accel_share": _positive_share(forward_accel, diagonal_any),
        "rear_pair_positive_progress_share": _positive_share(forward_deltas, rear_pair),
        "rear_pair_without_front_positive_progress_share": _positive_share(forward_deltas, rear_pair_without_front),
        "front_pair_positive_progress_share": _positive_share(forward_deltas, front_pair),
        "diagonal_positive_progress_share": _positive_share(forward_deltas, diagonal_any),
        "positive_forward_progress_units": positive_forward_progress_total,
        "positive_forward_accel_units_s2_sum": positive_forward_accel_total,
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
        "diagonal_contact_fraction",
        "diagonal_only_fraction",
        "diagonal_switches_per_s",
        "front_pair_contact_fraction",
        "rear_pair_contact_fraction",
        "front_pair_without_rear_fraction",
        "rear_pair_without_front_fraction",
        "rear_pair_vs_diagonal_contact_ratio",
        "positive_forward_accel_fraction",
        "rear_pair_when_positive_accel_fraction",
        "diagonal_when_positive_accel_fraction",
        "rear_pair_positive_accel_share",
        "rear_pair_without_front_positive_accel_share",
        "front_pair_positive_accel_share",
        "diagonal_positive_accel_share",
        "rear_pair_positive_progress_share",
        "rear_pair_without_front_positive_progress_share",
        "front_pair_positive_progress_share",
        "diagonal_positive_progress_share",
        "rear_action_abs_share",
        "front_positive_accel_share",
        "rear_positive_accel_share",
        "front_positive_progress_share",
        "rear_positive_progress_share",
        "front_contact_positive_forward_impulse_share",
        "rear_contact_positive_forward_impulse_share",
        "front_contact_abs_forward_impulse_share",
        "rear_contact_abs_forward_impulse_share",
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
    summary["duty_factor_mean"] = _dict_mean(
        [item["duty_factor"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_positive_accel_share_mean"] = _dict_mean(
        [item["per_leg_positive_accel_share"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_positive_progress_share_mean"] = _dict_mean(
        [item["per_leg_positive_progress_share"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_when_positive_accel_fraction_mean"] = _dict_mean(
        [item["per_leg_when_positive_accel_fraction"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_action_abs_mean"] = _dict_mean(
        [item["per_leg_action_abs_mean"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_shoulder_action_abs_mean"] = _dict_mean(
        [item["per_leg_shoulder_action_abs_mean"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_elbow_action_abs_mean"] = _dict_mean(
        [item["per_leg_elbow_action_abs_mean"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_action_switches_s_mean"] = _dict_mean(
        [item["per_leg_action_switches_s"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_normal_impulse_sum"] = _dict_sum(
        [item["per_leg_contact_normal_impulse_sum"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_tangent_impulse_norm_sum"] = _dict_sum(
        [item["per_leg_contact_tangent_impulse_norm_sum"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_forward_impulse_sum"] = _dict_sum(
        [item["per_leg_contact_forward_impulse_sum"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_positive_forward_impulse_sum"] = _dict_sum(
        [item["per_leg_contact_positive_forward_impulse_sum"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_abs_forward_impulse_sum"] = _dict_sum(
        [item["per_leg_contact_abs_forward_impulse_sum"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_lateral_impulse_sum"] = _dict_sum(
        [item["per_leg_contact_lateral_impulse_sum"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_positive_forward_impulse_share_mean"] = _dict_mean(
        [item["per_leg_contact_positive_forward_impulse_share"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_abs_forward_impulse_share_mean"] = _dict_mean(
        [item["per_leg_contact_abs_forward_impulse_share"] for item in episode_summaries],
        LEG_NAMES,
    )
    summary["per_leg_contact_normal_impulse_share_mean"] = _dict_mean(
        [item["per_leg_contact_normal_impulse_share"] for item in episode_summaries],
        LEG_NAMES,
    )
    support_pattern_sum = _dict_sum(
        [item["support_pattern_fraction"] for item in episode_summaries],
        tuple(
            sorted(
                {
                    pattern
                    for item in episode_summaries
                    for pattern in item["support_pattern_fraction"]
                }
            )
        ),
    )
    summary["support_pattern_fraction_mean"] = {
        key: value / max(len(episode_summaries), 1)
        for key, value in sorted(support_pattern_sum.items(), key=lambda item: item[1], reverse=True)
    }
    summary["top_support_patterns"] = dict(list(summary["support_pattern_fraction_mean"].items())[:8])
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
        "net_to_absolute_forward_motion_ratio",
        "forward_speed_sign_changes_per_s",
        "body_height_std_units",
        "max_abs_tilt_p95_deg",
        "angular_velocity_p95",
        "action_switches_per_joint_s",
        "grounded_leg_count_mean",
        "diagonal_switches_per_s",
        "contact_foot_planar_speed_p95_units_s",
    ):
        print(
            "[GAIT DEBUG]",
            f"{key}:",
            f"mean={global_summary.get(f'{key}_mean', 0.0):.4f}",
            f"min={global_summary.get(f'{key}_min', 0.0):.4f}",
            f"max={global_summary.get(f'{key}_max', 0.0):.4f}",
        )

    print("\n[GAIT DEBUG] Contact And Propulsion Proxy")
    print("[GAIT DEBUG] top_support:", _format_dict(global_summary.get("top_support_patterns", {}), precision=4))
    print(
        "[GAIT DEBUG] contact_pattern_mean:",
        _format_dict(
            {
                "diagonal": global_summary.get("diagonal_contact_fraction_mean", 0.0),
                "diagonal_only": global_summary.get("diagonal_only_fraction_mean", 0.0),
                "front_pair": global_summary.get("front_pair_contact_fraction_mean", 0.0),
                "rear_pair": global_summary.get("rear_pair_contact_fraction_mean", 0.0),
                "rear_pair_no_front": global_summary.get("rear_pair_without_front_fraction_mean", 0.0),
                "rear_pair_vs_diag": global_summary.get("rear_pair_vs_diagonal_contact_ratio_mean", 0.0),
            },
            precision=4,
        ),
    )
    print(
        "[GAIT DEBUG] positive_accel_share:",
        _format_dict(
            {
                "front_any_sum": global_summary.get("front_positive_accel_share_mean", 0.0),
                "rear_any_sum": global_summary.get("rear_positive_accel_share_mean", 0.0),
                "diagonal": global_summary.get("diagonal_positive_accel_share_mean", 0.0),
                "rear_pair": global_summary.get("rear_pair_positive_accel_share_mean", 0.0),
                "rear_pair_no_front": global_summary.get("rear_pair_without_front_positive_accel_share_mean", 0.0),
            },
            precision=4,
        ),
    )
    print(
        "[GAIT DEBUG] positive_progress_share:",
        _format_dict(
            {
                "front_any_sum": global_summary.get("front_positive_progress_share_mean", 0.0),
                "rear_any_sum": global_summary.get("rear_positive_progress_share_mean", 0.0),
                "diagonal": global_summary.get("diagonal_positive_progress_share_mean", 0.0),
                "rear_pair": global_summary.get("rear_pair_positive_progress_share_mean", 0.0),
                "rear_pair_no_front": global_summary.get("rear_pair_without_front_positive_progress_share_mean", 0.0),
            },
            precision=4,
        ),
    )

    print("\n[GAIT DEBUG] Per Leg Summary")
    print("[GAIT DEBUG] duty:", _format_dict(global_summary.get("duty_factor_mean", {}), precision=4))
    print(
        "[GAIT DEBUG] positive_accel_share_by_leg:",
        _format_dict(global_summary.get("per_leg_positive_accel_share_mean", {}), precision=4),
    )
    print(
        "[GAIT DEBUG] positive_progress_share_by_leg:",
        _format_dict(global_summary.get("per_leg_positive_progress_share_mean", {}), precision=4),
    )
    print(
        "[GAIT DEBUG] contact_when_positive_accel:",
        _format_dict(global_summary.get("per_leg_when_positive_accel_fraction_mean", {}), precision=4),
    )
    print("[GAIT DEBUG] action_abs:", _format_dict(global_summary.get("per_leg_action_abs_mean", {}), precision=4))
    print(
        "[GAIT DEBUG] shoulder_action_abs:",
        _format_dict(global_summary.get("per_leg_shoulder_action_abs_mean", {}), precision=4),
    )
    print(
        "[GAIT DEBUG] elbow_action_abs:",
        _format_dict(global_summary.get("per_leg_elbow_action_abs_mean", {}), precision=4),
    )
    print(
        "[GAIT DEBUG] action_switches_s:",
        _format_dict(global_summary.get("per_leg_action_switches_s_mean", {}), precision=4),
    )
    print(
        "[GAIT DEBUG]",
        f"rear_action_abs_share_mean={global_summary.get('rear_action_abs_share_mean', 0.0):.4f}",
    )

    print("\n[GAIT DEBUG] Contact Impulse Summary")
    print(
        "[GAIT DEBUG] normal_impulse_share:",
        _format_dict(global_summary.get("per_leg_contact_normal_impulse_share_mean", {}), precision=4),
    )
    print(
        "[GAIT DEBUG] positive_forward_impulse_share:",
        _format_dict(global_summary.get("per_leg_contact_positive_forward_impulse_share_mean", {}), precision=4),
    )
    print(
        "[GAIT DEBUG] abs_forward_impulse_share:",
        _format_dict(global_summary.get("per_leg_contact_abs_forward_impulse_share_mean", {}), precision=4),
    )
    print(
        "[GAIT DEBUG] forward_impulse_sum:",
        _format_dict(global_summary.get("per_leg_contact_forward_impulse_sum", {}), precision=4),
    )
    print(
        "[GAIT DEBUG]",
        f"front_positive_forward_impulse_share={global_summary.get('front_contact_positive_forward_impulse_share_mean', 0.0):.4f}",
        f"rear_positive_forward_impulse_share={global_summary.get('rear_contact_positive_forward_impulse_share_mean', 0.0):.4f}",
        f"front_abs_forward_impulse_share={global_summary.get('front_contact_abs_forward_impulse_share_mean', 0.0):.4f}",
        f"rear_abs_forward_impulse_share={global_summary.get('rear_contact_abs_forward_impulse_share_mean', 0.0):.4f}",
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

    if DEBUG_GAIT_EVAL_PRINT_EPISODES:
        print("\n[GAIT DEBUG] Episodes")
        for episode in result.episodes:
            print(
                "[GAIT DEBUG]",
                f"ep={episode['episode']}",
                f"steps={episode['steps']}",
                f"done={episode['done_reason']}",
                f"reward={episode['reward_sum']:.3f}",
                f"net_m={episode['net_forward_progress_m']:.3f}",
                f"net_abs_ratio={episode['net_to_absolute_forward_motion_ratio']:.3f}",
                f"tilt_p95={episode['max_abs_tilt_p95_deg']:.2f}",
                f"rear_action_share={episode['rear_action_abs_share']:.3f}",
            )
            print("[GAIT DEBUG]   duty:", _format_dict(episode["duty_factor"], precision=4))
            print(
                "[GAIT DEBUG]   propulsion:",
                _format_dict(
                    {
                        "diag_contact": episode["diagonal_contact_fraction"],
                        "rear_pair": episode["rear_pair_contact_fraction"],
                        "rear_pos_accel_sum": episode["rear_positive_accel_share"],
                        "front_pos_accel_sum": episode["front_positive_accel_share"],
                        "rear_pos_progress_sum": episode["rear_positive_progress_share"],
                        "front_pos_progress_sum": episode["front_positive_progress_share"],
                    },
                    precision=4,
                ),
            )
            print(
                "[GAIT DEBUG]   forward_impulse_share:",
                _format_dict(episode["per_leg_contact_positive_forward_impulse_share"], precision=4),
            )
    else:
        print("\n[GAIT DEBUG] Episodes: hidden; set DEBUG_GAIT_EVAL_PRINT_EPISODES=True for per-episode terminal details.")

    print("\n[GAIT DEBUG] Reading Guide")
    print("[GAIT DEBUG] net_to_absolute_forward_motion_ratio close to 1 means little back-and-forth drift.")
    print("[GAIT DEBUG] duty_factor is per-leg stance fraction; diagonal fractions show FR+BL vs FL+BR support use.")
    print("[GAIT DEBUG] contact_foot_planar_speed is a foot-drag proxy measured while the foot is in contact.")
    print("[GAIT DEBUG] action_switches_per_joint_s measures command dithering rate normalized by simulated seconds.")
    print("[GAIT DEBUG] reward_abs_responsibility uses absolute component magnitudes, so it shows domination even when signs cancel.")
    print("[GAIT DEBUG] propulsion_proxy uses body forward acceleration/progress crossed with contact patterns; it is not a contact-force measurement.")
    task_forward_label = "+world_z" if TASK_FORWARD_Z_SIGN > 0.0 else "-world_z"
    print(f"[GAIT DEBUG] contact_impulse is measured from solver impulses; positive_forward is along task forward ({task_forward_label}).")


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
        print(f"Running episode {episode_idx + 1}/{DEBUG_GAIT_EVAL_EPISODES}") # Stop deleting this line
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
