import unittest
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from physics_env.bench.policy_runner import list_policy_initial_scenarios, run_policy_bench
from physics_env.bench.runner import list_benches, run_bench


ROOT = Path(__file__).resolve().parents[1]


class ZeroPolicy:
    def get_action(self, state, deterministic=False):
        del state, deterministic
        return (
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            {"entropy": 0.0, "value": 0.0},
        )


class BenchRunnerTest(unittest.TestCase):
    def assert_bench_clear(self, metrics, *, max_penetration, max_tilt, max_final_speed):
        self.assertFalse(metrics["env_done_seen"], metrics)
        self.assertFalse(metrics["done"], metrics)
        self.assertEqual(metrics["steps_executed"], 600)
        self.assertLess(metrics["max_ground_penetration"], max_penetration)
        self.assertLess(metrics["max_abs_tilt"], max_tilt)
        self.assertLess(metrics["final_speed"], max_final_speed)
        self.assertGreater(metrics["grounded_frames"], 0)
        self.assertGreater(metrics["final_contact_count"], 0)

        finite_keys = [
            "final_body_height",
            "final_position_x",
            "final_position_z",
            "max_speed",
            "max_angular_speed",
            "final_total_energy",
            "grounded_height_jitter_rms",
        ]
        for key in finite_keys:
            self.assertTrue(math.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}")

    def test_settle_bench_runs_headless(self):
        metrics = run_bench(name="settle", steps=5, seed=43, render=False)
        self.assertEqual(metrics["scenario"], "settle")
        self.assertEqual(metrics["seed"], 43)
        self.assertGreater(metrics["steps_executed"], 0)
        self.assertIn("max_ground_penetration", metrics)
        self.assertIn("grounded_height_jitter_rms", metrics)

    def test_bench_registry_exposes_multiple_scenarios(self):
        scenarios = list_benches()
        self.assertIn("drop_flat", scenarios)
        self.assertIn("demo_gait_animation", scenarios)
        self.assertIn("slide_x", scenarios)
        self.assertIn("single_leg_sweep", scenarios)
        self.assertEqual(scenarios["slide_x"]["category"], "contact_friction")

    def test_bench_headless_cli_lists_scenarios(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "apps" / "bench_headless.py"), "--list"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertIn("settle:", result.stdout)
        self.assertIn("demo_gait_animation:", result.stdout)
        self.assertIn("drop_flat:", result.stdout)
        self.assertIn("slide_x:", result.stdout)
        self.assertIn("front_legs_lifted:", result.stdout)

    def test_core_headless_benches_clear_600_steps(self):
        scenario_thresholds = {
            "demo_gait_animation": {"max_penetration": 0.02, "max_tilt": 0.30, "max_final_speed": 0.60},
            "settle": {"max_penetration": 0.02, "max_tilt": 0.08, "max_final_speed": 0.10},
            "drop_flat": {"max_penetration": 0.02, "max_tilt": 0.25, "max_final_speed": 0.15},
            "slide_x": {"max_penetration": 0.02, "max_tilt": 0.08, "max_final_speed": 0.10},
            "front_legs_lifted": {"max_penetration": 0.02, "max_tilt": 1.00, "max_final_speed": 0.15},
        }

        for scenario, thresholds in scenario_thresholds.items():
            with self.subTest(scenario=scenario):
                metrics = run_bench(name=scenario, steps=600, seed=43, render=False)
                self.assertEqual(metrics["scenario"], scenario)
                self.assert_bench_clear(metrics, **thresholds)

                if scenario == "front_legs_lifted":
                    self.assertIsNotNone(metrics["tip_time_s"])
                    self.assertGreater(metrics["final_body_height"], 2.4)
                else:
                    self.assertIsNone(metrics["tip_time_s"])
                    self.assertGreater(metrics["final_body_height"], 3.7)

    def test_policy_bench_runs_headless_with_injected_agent(self):
        metrics = run_policy_bench(
            agent=ZeroPolicy(),
            model_path="unused.pth",
            initial_scenario="settle",
            steps=5,
            seed=43,
            render=False,
        )

        self.assertEqual(metrics["scenario"], "policy:settle")
        self.assertEqual(metrics["category"], "learned_policy")
        self.assertEqual(metrics["model_path"], "<provided-agent>")
        self.assertGreater(metrics["steps_executed"], 0)
        self.assertIn("policy_airborne_ratio", metrics)
        self.assertIn("policy_action_saturation_rate", metrics)
        self.assertAlmostEqual(metrics["policy_mean_abs_action"], 0.0, places=6)

    def test_policy_bench_exposes_test_py_reset_initial_state(self):
        scenarios = list_policy_initial_scenarios()
        self.assertIn("env_reset", scenarios)

        metrics = run_policy_bench(
            agent=ZeroPolicy(),
            model_path="unused.pth",
            initial_scenario="env_reset",
            steps=1,
            seed=43,
            render=False,
        )

        self.assertEqual(metrics["scenario"], "policy:env_reset")
        self.assertAlmostEqual(metrics["initial_body_height"], 4.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
