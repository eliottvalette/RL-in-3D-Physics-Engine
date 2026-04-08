import json
import os
import tempfile
import unittest

import numpy as np

from visualization import DataCollector


class VisualizationStorageTest(unittest.TestCase):
    def test_metrics_json_keeps_only_plot_scalars_and_npz_keeps_distributions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = DataCollector(save_interval=1, plot_interval=10_000, output_dir=temp_dir)

            collector.add_state([1.0, 2.0, 3.0])
            collector.add_metrics(
                {
                    "episode_reward": 1.5,
                    "returns": [1.0, 2.0, 3.0],
                    "state_values": [0.25, 0.5],
                    "advantages": [-1.0, 2.0],
                }
            )
            collector.save_episode(0)

            metrics_path = os.path.join(temp_dir, "metrics_history.json")
            distributions_path = os.path.join(temp_dir, "value_distributions.npz")
            states_path = os.path.join(temp_dir, "episodes_states.json")

            self.assertTrue(os.path.exists(metrics_path))
            self.assertTrue(os.path.exists(distributions_path))
            self.assertFalse(os.path.exists(states_path))

            with open(metrics_path, "r") as metrics_file:
                metrics_history = json.load(metrics_file)

            self.assertEqual(list(metrics_history.keys()), ["0"])
            episode_metrics = metrics_history["0"]
            self.assertAlmostEqual(episode_metrics["episode_reward"], 1.5, places=6)
            self.assertAlmostEqual(episode_metrics["returns_mean"], 2.0, places=6)
            self.assertAlmostEqual(episode_metrics["state_value_mean"], 0.375, places=6)
            self.assertAlmostEqual(episode_metrics["advantage_mean"], 0.5, places=6)
            self.assertNotIn("returns", episode_metrics)
            self.assertNotIn("state_values", episode_metrics)
            self.assertNotIn("advantages", episode_metrics)

            with np.load(distributions_path) as distributions:
                np.testing.assert_allclose(distributions["returns"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
                np.testing.assert_allclose(distributions["state_values"], np.array([0.25, 0.5], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
