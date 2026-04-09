"""Bench runner for saved quadruped policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from physics_env.core.config import (
    ACTION_SIZE,
    ALPHA,
    GAMMA,
    PHYSICS_STEPS_PER_RENDER,
    RENDER_FPS,
    STATE_SIZE,
    set_seed,
)
from physics_env.envs.quadruped_env import QuadrupedEnv

from .metrics import BenchMetrics
from .scenarios import SCENARIOS


DEFAULT_POLICY_MODEL_PATH = "saved_models/quadruped_agent.pth"
TEST_RESET_SCENARIO = "env_reset"
POLICY_INITIAL_SCENARIOS = {
    TEST_RESET_SCENARIO: {
        "category": "learned_policy",
        "description": "QuadrupedEnv.reset_episode(pose_jitter=True) state used by test.py.",
    }
}


def _load_policy_agent(model_path: str | Path, device: str):
    from agent import QuadrupedAgent

    return QuadrupedAgent(
        device=device,
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=True,
        load_path=str(model_path),
    )


@dataclass
class PolicyBenchStats:
    samples: int = 0
    action_abs_sum: float = 0.0
    action_saturation_count: int = 0
    action_component_count: int = 0
    action_switch_count: int = 0
    action_switch_component_count: int = 0
    airborne_frames: int = 0
    max_abs_vertical_velocity: float = 0.0
    max_abs_shoulder_velocity: float = 0.0
    max_abs_elbow_velocity: float = 0.0
    entropy_sum: float = 0.0
    value_sum: float = 0.0
    entropy_samples: int = 0
    value_samples: int = 0
    first_done_reason: str | None = None
    final_done_reason: str = "running"
    _previous_action: np.ndarray | None = None

    def update(self, env, action: np.ndarray, action_info: dict, env_done: bool):
        self.samples += 1

        action = np.asarray(action, dtype=np.float32)
        self.action_abs_sum += float(np.abs(action).sum())
        self.action_saturation_count += int(np.count_nonzero(np.isclose(np.abs(action), 1.0)))
        self.action_component_count += int(action.size)

        if self._previous_action is not None:
            self.action_switch_count += int(np.count_nonzero(action != self._previous_action))
            self.action_switch_component_count += int(action.size)
        self._previous_action = action.copy()

        active_contacts = getattr(env.quadruped, "active_contact_indices", [])
        if len(active_contacts) == 0:
            self.airborne_frames += 1

        self.max_abs_vertical_velocity = max(
            self.max_abs_vertical_velocity,
            abs(float(env.quadruped.velocity[1])),
        )
        self.max_abs_shoulder_velocity = max(
            self.max_abs_shoulder_velocity,
            float(np.abs(env.quadruped.shoulder_velocities).max()),
        )
        self.max_abs_elbow_velocity = max(
            self.max_abs_elbow_velocity,
            float(np.abs(env.quadruped.elbow_velocities).max()),
        )

        if "entropy" in action_info:
            self.entropy_sum += float(action_info["entropy"])
            self.entropy_samples += 1
        if "value" in action_info:
            self.value_sum += float(action_info["value"])
            self.value_samples += 1

        self.final_done_reason = getattr(env, "last_done_reason", "running")
        if env_done and self.first_done_reason is None:
            self.first_done_reason = self.final_done_reason

    def to_dict(self) -> dict:
        return {
            "policy_mean_abs_action": self.action_abs_sum / max(self.action_component_count, 1),
            "policy_action_saturation_rate": self.action_saturation_count / max(self.action_component_count, 1),
            "policy_action_switch_rate": self.action_switch_count / max(self.action_switch_component_count, 1),
            "policy_airborne_frames": self.airborne_frames,
            "policy_airborne_ratio": self.airborne_frames / max(self.samples, 1),
            "policy_max_abs_vertical_velocity": self.max_abs_vertical_velocity,
            "policy_max_abs_shoulder_velocity": self.max_abs_shoulder_velocity,
            "policy_max_abs_elbow_velocity": self.max_abs_elbow_velocity,
            "policy_mean_entropy": self.entropy_sum / max(self.entropy_samples, 1),
            "policy_mean_value": self.value_sum / max(self.value_samples, 1),
            "policy_first_done_reason": self.first_done_reason,
            "policy_final_done_reason": self.final_done_reason,
        }


def run_policy_bench(
    model_path: str | Path = DEFAULT_POLICY_MODEL_PATH,
    initial_scenario: str = "settle",
    steps: int = 500,
    seed: int = 43,
    render: bool = False,
    deterministic: bool = True,
    device: str = "cpu",
    agent=None,
):
    if initial_scenario not in SCENARIOS and initial_scenario not in POLICY_INITIAL_SCENARIOS:
        available = ", ".join(sorted([*SCENARIOS, *POLICY_INITIAL_SCENARIOS]))
        raise ValueError(f"Unknown bench scenario '{initial_scenario}'. Available scenarios: {available}")

    set_seed(seed)
    policy_agent = agent if agent is not None else _load_policy_agent(model_path=model_path, device=device)
    scenario = SCENARIOS.get(initial_scenario)
    metrics = BenchMetrics(
        scenario=f"policy:{initial_scenario}",
        seed=seed,
        category="learned_policy",
        description=(
            POLICY_INITIAL_SCENARIOS[initial_scenario]["description"]
            if initial_scenario in POLICY_INITIAL_SCENARIOS
            else f"Saved policy rollout from initial scenario '{initial_scenario}'."
        ),
    )
    policy_stats = PolicyBenchStats()
    env = QuadrupedEnv(rendering=render, headless=not render, bench_mode=False)

    try:
        if initial_scenario == TEST_RESET_SCENARIO:
            state = env.reset_episode(pose_jitter=True)
        else:
            scenario.reset(env)
            env.reset_episode_state()
            state = env.get_state()

        running = True
        camera_actions = [0] * 10
        reward = 0.0
        env_done = False

        for step_idx in range(steps):
            if render and step_idx % PHYSICS_STEPS_PER_RENDER == 0:
                running = env.handle_events()
                if not running:
                    break
                keys = pygame.key.get_pressed()
                camera_actions = env.handle_camera_controls(keys)

            shoulders, elbows, action_info = policy_agent.get_action(state=state, deterministic=deterministic)
            if action_info is None:
                action_info = {}
            action = np.concatenate(
                [
                    np.asarray(shoulders, dtype=np.float32),
                    np.asarray(elbows, dtype=np.float32),
                ],
                dtype=np.float32,
            )

            next_state, reward, env_done, step_time = env.step(shoulders, elbows, camera_actions)
            policy_stats.update(env, action, action_info, env_done)
            metrics.update(env, reward, step_time, done=env_done, env_done=env_done)

            should_render = render and (
                (step_idx + 1) % PHYSICS_STEPS_PER_RENDER == 0 or step_idx == steps - 1 or env_done
            )
            if should_render:
                env.render(reward, env_done, step_time, state_value=action_info.get("value"))
                env.clock.tick(RENDER_FPS)

            if env_done or (scenario is not None and scenario.should_stop(env, step_idx, env_done)):
                break
            state = next_state

        payload = metrics.to_dict()
        payload.update(policy_stats.to_dict())
        payload.update(
            {
                "model_path": str(model_path) if agent is None else "<provided-agent>",
                "initial_scenario": initial_scenario,
                "deterministic_policy": deterministic,
                "final_forward_progress": payload["initial_position_z"] - payload["final_position_z"],
            }
        )
        return payload
    finally:
        pygame.quit()


def list_policy_initial_scenarios():
    descriptions = {
        name: {
            "category": scenario.category,
            "description": scenario.description,
        }
        for name, scenario in SCENARIOS.items()
    }
    descriptions.update(POLICY_INITIAL_SCENARIOS)
    return descriptions
