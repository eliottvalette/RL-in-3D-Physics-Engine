"""Bench runner for deterministic quadruped scenarios."""

import pygame

from physics_env.core.config import PHYSICS_STEPS_PER_RENDER, RENDER_FPS, set_seed
from physics_env.envs.quadruped_env import QuadrupedEnv

from .metrics import BenchMetrics
from .scenarios import SCENARIOS, scenario_descriptions


def run_bench(name: str = "settle", steps: int = 600, seed: int = 43, render: bool = False):
    if name not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown bench scenario '{name}'. Available scenarios: {available}")

    set_seed(seed)
    scenario = SCENARIOS[name]
    metrics = BenchMetrics(
        scenario=name,
        seed=seed,
        category=getattr(scenario, "category", "generic"),
        description=getattr(scenario, "description", ""),
    )
    env = QuadrupedEnv(rendering=render, headless=not render, bench_mode=True)

    try:
        scenario.reset(env)
        running = True
        camera_actions = [0] * 10

        for step_idx in range(steps):
            if render and step_idx % PHYSICS_STEPS_PER_RENDER == 0:
                running = env.handle_events()
                if not running:
                    break
                keys = pygame.key.get_pressed()
                camera_actions = env.handle_camera_controls(keys)

            shoulders, elbows = scenario.actions(env, step_idx)
            _, reward, env_done, step_time = env.step(shoulders, elbows, camera_actions)
            bench_done = env_done if getattr(scenario, "stop_on_env_done", False) else False
            metrics.update(env, reward, step_time, bench_done, env_done=env_done)

            should_render = render and (
                (step_idx + 1) % PHYSICS_STEPS_PER_RENDER == 0 or step_idx == steps - 1
            )
            if should_render:
                env.render(reward, env_done, step_time)
                env.clock.tick(RENDER_FPS)

            if scenario.should_stop(env, step_idx, env_done):
                break

        return metrics.to_dict()
    finally:
        pygame.quit()


def list_benches():
    return scenario_descriptions()
