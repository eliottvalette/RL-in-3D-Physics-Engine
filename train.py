# train.py
import os
import time
import traceback

import numpy as np

from agent import QuadrupedAgent
from helpers_rl import save_models
from physics_env.core.config import (
    DEBUG_RL_TRAIN,
    EPISODES,
    EVAL_EPISODES,
    EVAL_INTERVAL,
    MAX_STEPS,
    PLOT_INTERVAL,
    ROLLOUT_STEPS,
    SAVE_INTERVAL,
)
from physics_env.envs.quadruped_env import QuadrupedEnv
from visualization import DataCollector


EVENT_KEYS = ("too_low", "too_high", "critical_tilt", "joint_limit_timeout")


def _new_event_flags():
    return {key: False for key in EVENT_KEYS}


def _update_event_flags(event_flags, done_reason):
    if done_reason in event_flags:
        event_flags[done_reason] = True


def _event_metrics(event_flags):
    clean_episode = not any(event_flags.values())
    return {
        "done_reason_too_low": 1.0 if event_flags["too_low"] else 0.0,
        "done_reason_too_high": 1.0 if event_flags["too_high"] else 0.0,
        "done_reason_critical_tilt": 1.0 if event_flags["critical_tilt"] else 0.0,
        "done_reason_joint_limit_timeout": 1.0 if event_flags["joint_limit_timeout"] else 0.0,
        "done_reason_max_steps": 1.0 if clean_episode else 0.0,
    }


def _aggregate_metric_dicts(metric_dicts):
    if not metric_dicts:
        return {}

    aggregated = {}
    all_keys = set()
    for metric_dict in metric_dicts:
        all_keys.update(metric_dict.keys())

    for key in all_keys:
        values = [metric_dict[key] for metric_dict in metric_dicts if metric_dict.get(key) is not None]
        if not values:
            continue
        aggregated[key] = float(np.mean(values))
    return aggregated


def run_episode(env: QuadrupedEnv, agent: QuadrupedAgent, rendering: bool, episode: int, render_every: int, data_collector: DataCollector):
    env.reset_episode()
    event_flags = _new_event_flags()
    episode_reward = 0.0
    locomotion_scales = []
    steps_count = 0

    for step in range(MAX_STEPS):
        state = env.get_state()
        shoulders, elbows, action_info = agent.get_action(state, deterministic=False)
        next_state, reward, terminated, step_time = env.step(shoulders, elbows)
        truncated = step == MAX_STEPS - 1

        agent.store_transition(
            state=state,
            action_info=action_info,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        data_collector.add_state(state)
        data_collector.add_metrics(env.last_reward_components.copy())

        episode_reward += reward
        steps_count += 1
        locomotion_scale = env.last_reward_components.get("locomotion_reward_scale")
        if locomotion_scale is not None:
            locomotion_scales.append(float(locomotion_scale))
        _update_event_flags(event_flags, env.last_done_reason)

        if rendering and (episode % render_every == 0):
            env.render(reward, terminated, step_time)

        if len(agent.rollout_buffer) >= ROLLOUT_STEPS or terminated or truncated:
            bootstrap_value = 0.0 if terminated else agent.evaluate_state(next_state)
            update_metrics = agent.update_from_rollout(last_value=bootstrap_value)
            if update_metrics is not None:
                data_collector.add_metrics(update_metrics)

        if terminated or truncated:
            break

    summary = {
        "steps_count": float(steps_count),
        "episode_reward": float(episode_reward),
        "forward_progress": float(max(0.0, -float(env.quadruped.position[2]))),
        "mean_locomotion_reward_scale": float(np.mean(locomotion_scales)) if locomotion_scales else None,
    }
    summary.update(_event_metrics(event_flags))

    if DEBUG_RL_TRAIN:
        print(f"[TRAIN] episode_reward={episode_reward:.3f} progress={summary['forward_progress']:.3f}")

    return summary


def run_evaluation_episode(agent: QuadrupedAgent, env: QuadrupedEnv):
    env.reset_episode()
    event_flags = _new_event_flags()
    episode_reward = 0.0
    locomotion_scales = []

    for _ in range(MAX_STEPS):
        state = env.get_state()
        shoulders, elbows, _ = agent.get_action(state, deterministic=True)
        _, reward, terminated, _ = env.step(shoulders, elbows)
        episode_reward += reward
        locomotion_scale = env.last_reward_components.get("locomotion_reward_scale")
        if locomotion_scale is not None:
            locomotion_scales.append(float(locomotion_scale))
        _update_event_flags(event_flags, env.last_done_reason)
        if terminated:
            break

    eval_metrics = {
        "eval_episode_reward": float(episode_reward),
        "eval_forward_progress": float(max(0.0, -float(env.quadruped.position[2]))),
        "eval_mean_locomotion_scale": float(np.mean(locomotion_scales)) if locomotion_scales else None,
        "eval_clean_episode": 1.0 if not any(event_flags.values()) else 0.0,
        "eval_too_low": 1.0 if event_flags["too_low"] else 0.0,
        "eval_too_high": 1.0 if event_flags["too_high"] else 0.0,
        "eval_critical_tilt": 1.0 if event_flags["critical_tilt"] else 0.0,
        "eval_joint_limit_timeout": 1.0 if event_flags["joint_limit_timeout"] else 0.0,
    }
    return eval_metrics


def main_training_loop(agent: QuadrupedAgent, episodes: int, rendering: bool, render_every: int):
    env = QuadrupedEnv(rendering=rendering)
    eval_env = QuadrupedEnv(rendering=False, headless=True)

    data_collector = DataCollector(
        save_interval=SAVE_INTERVAL,
        plot_interval=PLOT_INTERVAL,
        start_epsilon=0.0,
        epsilon_decay=1.0,
        epsilon_min=0.0,
    )

    episode = 0
    try:
        for episode in range(episodes):
            start_time = time.time()

            episode_summary = run_episode(env, agent, rendering, episode, render_every, data_collector)

            if EVAL_INTERVAL > 0 and (episode + 1) % EVAL_INTERVAL == 0:
                eval_summaries = [run_evaluation_episode(agent, eval_env) for _ in range(EVAL_EPISODES)]
                episode_summary.update(_aggregate_metric_dicts(eval_summaries))

            data_collector.add_metrics(episode_summary)
            data_collector.save_episode(episode)

            print(f"\n[TRAIN] Episode [{episode + 1}/{episodes}]")
            print(f"[TRAIN] Steps: {int(episode_summary['steps_count'])}")
            print(f"[TRAIN] Reward: {episode_summary['episode_reward']:.3f}")
            print(f"[TRAIN] Progress: {episode_summary['forward_progress']:.3f}")
            if "eval_episode_reward" in episode_summary:
                print(
                    "[TRAIN] Eval:",
                    f"reward={episode_summary['eval_episode_reward']:.3f}",
                    f"progress={episode_summary['eval_forward_progress']:.3f}",
                    f"clean={episode_summary['eval_clean_episode']:.2f}",
                )
            print(f"[TRAIN] Time taken: {time.time() - start_time:.2f} seconds")

        save_models(agent, episode)
        print("[TRAIN] Generating visualization...")
        data_collector.force_visualization()

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user; saving current model...")
        save_models(agent, episode)
        raise

    except Exception as exc:
        print(f"[TRAIN] An error occurred: {exc}")
        traceback.print_exc()
        save_models(agent, episode)
        print("[TRAIN] Generating visualization...")
        data_collector.force_visualization()
        raise

    finally:
        temp_dir = "temp_viz_json"
        if os.path.exists(temp_dir):
            for file_name in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file_name))
