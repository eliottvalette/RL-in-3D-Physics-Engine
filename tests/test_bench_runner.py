import unittest

from physics_env.bench.runner import list_benches, run_bench


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


if __name__ == "__main__":
    unittest.main()
