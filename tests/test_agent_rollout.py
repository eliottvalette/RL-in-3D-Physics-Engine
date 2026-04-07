import unittest

import numpy as np
import torch

from agent import QuadrupedAgent
from physics_env.core.config import ACTION_SIZE, ALPHA, GAMMA, STATE_SIZE


class AgentRolloutTest(unittest.TestCase):
    def _build_agent(self):
        return QuadrupedAgent(
            device="cpu",
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            gamma=GAMMA,
            learning_rate=ALPHA,
            load_model=False,
        )

    def test_deterministic_action_uses_argmax(self):
        agent = self._build_agent()
        for param in agent.actor_model.parameters():
            param.data.zero_()

        state = [0.0] * STATE_SIZE
        shoulders, elbows, action_info = agent.get_action(state, deterministic=True)

        self.assertEqual(action_info["action_idx"].tolist(), [0] * ACTION_SIZE)
        self.assertTrue(np.allclose(shoulders, -1.0))
        self.assertTrue(np.allclose(elbows, -1.0))

    def test_gae_bootstraps_truncated_rollout(self):
        agent = self._build_agent()
        agent.rollout_buffer.rewards = [1.0, 0.5]
        agent.rollout_buffer.values = [0.2, 0.1]
        agent.rollout_buffer.terminated = [0.0, 0.0]

        advantages, returns = agent._compute_gae(last_value=0.25)

        self.assertEqual(len(advantages), 2)
        self.assertEqual(len(returns), 2)
        self.assertGreater(returns[-1], 0.5)
        self.assertGreater(returns[0], returns[-1])

    def test_update_from_rollout_returns_metrics_and_clears_buffer(self):
        agent = self._build_agent()
        state_0 = [0.0] * STATE_SIZE
        state_1 = [0.1] * STATE_SIZE

        _, _, action_info_0 = agent.get_action(state_0, deterministic=False)
        _, _, action_info_1 = agent.get_action(state_1, deterministic=False)

        agent.store_transition(state_0, action_info_0, reward=0.8, terminated=False, truncated=False)
        agent.store_transition(state_1, action_info_1, reward=0.4, terminated=False, truncated=True)

        metrics = agent.update_from_rollout(last_value=0.25)

        self.assertIsNotNone(metrics)
        self.assertEqual(len(metrics["returns"]), 2)
        self.assertEqual(len(metrics["advantages"]), 2)
        self.assertEqual(len(metrics["state_values"]), 2)
        self.assertIn("entropy", metrics)
        self.assertIn("approx_kl", metrics)
        self.assertIn("ppo_epochs_completed", metrics)
        self.assertGreaterEqual(metrics["ppo_epochs_completed"], 1.0)
        self.assertEqual(len(agent.rollout_buffer), 0)


if __name__ == "__main__":
    unittest.main()
