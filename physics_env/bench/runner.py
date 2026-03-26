"""Bench runner for deterministic quadruped scenarios."""

import pygame

from physics_env.core.config import FPS, set_seed
from physics_env.envs.quadruped_env import QuadrupedEnv

from .metrics import BenchMetrics
from .scenarios import SCENARIOS


def run_bench(name: str = "settle", steps: int = 600, seed: int = 43, render: bool = False):
    if name not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown bench scenario '{name}'. Available scenarios: {available}")

    set_seed(seed)
    scenario = SCENARIOS[name]
    metrics = BenchMetrics(scenario=name, seed=seed)
    env = QuadrupedEnv(rendering=render, headless=not render)

    try:
        scenario.reset(env)
        running = True

        for step_idx in range(steps):
            camera_actions = [0] * 10

            if render:
                running = env.handle_events()
                if not running:
                    break
                keys = pygame.key.get_pressed()
                camera_actions = env.handle_camera_controls(keys)

            shoulders, elbows = scenario.actions(env, step_idx)
            _, reward, done, step_time = env.step(shoulders, elbows, camera_actions)
            metrics.update(env, reward, step_time, done)

            if render:
                env.render(reward, done, step_time)
                env.clock.tick(FPS)

            if done:
                break

        return metrics.to_dict()
    finally:
        pygame.quit()
