"""Run a saved quadruped policy inside a bench initial condition."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_env.bench.policy_runner import DEFAULT_POLICY_MODEL_PATH, list_policy_initial_scenarios, run_policy_bench
from physics_env.bench.visualize import format_metrics


def main():
    parser = argparse.ArgumentParser(description="Run a saved quadruped policy from a bench initial condition.")
    parser.add_argument("--model", default=DEFAULT_POLICY_MODEL_PATH, help="Path to the saved agent checkpoint.")
    parser.add_argument("--initial", default="settle", help="Initial bench scenario name.")
    parser.add_argument("--steps", type=int, default=500, help="Maximum number of physics steps.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Torch device used for model inference.")
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy instead of using argmax.")
    parser.add_argument("--render", action="store_true", help="Render the rollout.")
    parser.add_argument("--list", action="store_true", help="List available initial scenarios and exit.")
    args = parser.parse_args()

    if args.list:
        for name, payload in sorted(list_policy_initial_scenarios().items()):
            print(f"{name}: [{payload['category']}] {payload['description']}")
        return

    metrics = run_policy_bench(
        model_path=args.model,
        initial_scenario=args.initial,
        steps=args.steps,
        seed=args.seed,
        render=args.render,
        deterministic=not args.stochastic,
        device=args.device,
    )
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
