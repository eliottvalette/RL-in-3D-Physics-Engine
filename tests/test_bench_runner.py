import unittest

from physics_env.bench.runner import run_bench


class BenchRunnerTest(unittest.TestCase):
    def test_settle_bench_runs_headless(self):
        metrics = run_bench(name="settle", steps=5, seed=43, render=False)
        self.assertEqual(metrics["scenario"], "settle")
        self.assertEqual(metrics["seed"], 43)
        self.assertGreater(metrics["steps_executed"], 0)
        self.assertIn("max_ground_penetration", metrics)


if __name__ == "__main__":
    unittest.main()
