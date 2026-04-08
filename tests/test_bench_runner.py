import unittest
import numpy as np

from physics_env.bench.policy_runner import list_policy_initial_scenarios, run_policy_bench
from physics_env.bench.runner import list_benches, run_bench


class ZeroPolicy:
    def get_action(self, state, deterministic=False):
        del state, deterministic
        return (
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            {"entropy": 0.0, "value": 0.0},
        )


class BenchRunnerTest(unittest.TestCase):
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
        self.assertIn("slide_x", scenarios)
        self.assertIn("single_leg_sweep", scenarios)
        self.assertEqual(scenarios["slide_x"]["category"], "contact_friction")

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
        self.assertGreater(metrics["initial_body_height"], 5.49)


if __name__ == "__main__":
    unittest.main()
