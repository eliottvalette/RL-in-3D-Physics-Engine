"""Run deterministic physics benches without rendering."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_env.bench.runner import run_bench
from physics_env.bench.runner import list_benches
from physics_env.bench.visualize import format_metrics


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic quadruped bench headlessly.")
    parser.add_argument("--scenario", default="settle", help="Bench scenario name.")
    parser.add_argument("--steps", type=int, default=600, help="Maximum number of physics steps.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    parser.add_argument("--list", action="store_true", help="List available bench scenarios and exit.")
    args = parser.parse_args()

    if args.list:
        for name, payload in sorted(list_benches().items()):
            print(f"{name}: [{payload['category']}] {payload['description']}")
        return

    metrics = run_bench(name=args.scenario, steps=args.steps, seed=args.seed, render=False)
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
