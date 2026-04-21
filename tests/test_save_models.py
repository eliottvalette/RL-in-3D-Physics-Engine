import tempfile
import unittest
from pathlib import Path

import torch

from helpers_rl import save_models


class _DummyAgent:
    def __init__(self):
        self.actor_model = torch.nn.Linear(1, 1)
        self.critic_model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.actor_model.parameters(), lr=0.1)
        self.critic_optimizer = torch.optim.SGD(self.critic_model.parameters(), lr=0.1)


class SaveModelsTest(unittest.TestCase):
    def test_save_models_overwrites_existing_latest_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            latest_path = models_dir / "quadruped_agent.pth"
            latest_path.write_bytes(b"existing model")

            save_models(_DummyAgent(), episode=1299, models_dir=str(models_dir))

            self.assertNotEqual(latest_path.read_bytes(), b"existing model")
            self.assertTrue((models_dir / "quadruped_agent_epoch_1299.pth").exists())

    def test_save_models_creates_latest_model_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            latest_path = models_dir / "quadruped_agent.pth"

            save_models(_DummyAgent(), episode=7, models_dir=str(models_dir))

            self.assertTrue(latest_path.exists())
            self.assertTrue((models_dir / "quadruped_agent_epoch_7.pth").exists())


if __name__ == "__main__":
    unittest.main()
