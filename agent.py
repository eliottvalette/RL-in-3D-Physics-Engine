# agent.py
from dataclasses import dataclass, field
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QuadrupedActorModel, QuadrupedCriticModel
from physics_env.core.config import (
    CRITIC_LOSS_COEFF,
    DEBUG_RL_AGENT,
    ENTROPY_COEFF,
    GAE_LAMBDA,
    PPO_CLIP_EPS,
    PPO_EPOCHS,
    PPO_MINIBATCH_SIZE,
    PPO_TARGET_KL,
)


@dataclass
class RolloutBuffer:
    states: list[list[float]] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[float] = field(default_factory=list)
    truncated: list[float] = field(default_factory=list)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.terminated.clear()
        self.truncated.clear()

    def __len__(self):
        return len(self.states)


class QuadrupedAgent:
    """On-policy actor-critic agent with categorical actions and GAE."""

    def __init__(self, device, state_size, action_size, gamma, learning_rate, load_model=False, load_path=None):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.gae_lambda = GAE_LAMBDA
        self.entropy_coeff = ENTROPY_COEFF
        self.critic_loss_coeff = CRITIC_LOSS_COEFF
        self.ppo_clip_eps = PPO_CLIP_EPS
        self.ppo_epochs = PPO_EPOCHS
        self.ppo_minibatch_size = PPO_MINIBATCH_SIZE
        self.ppo_target_kl = PPO_TARGET_KL

        self.actor_model = QuadrupedActorModel(input_dim=state_size, output_dim=action_size * 3).to(device)
        self.critic_model = QuadrupedCriticModel(input_dim=state_size).to(device)
        self.optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
        self.rollout_buffer = RolloutBuffer()

        if load_model:
            self.load(load_path)

    def load(self, load_path):
        if not isinstance(load_path, str):
            raise TypeError(f"[AGENT] load_path doit etre une chaine de caracteres (recu: {type(load_path)})")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"[AGENT] Le fichier {load_path} n'existe pas")

        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            self.actor_model.load_state_dict(checkpoint["actor_state_dict"])
            self.critic_model.load_state_dict(checkpoint["critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            print(f"[AGENT] Modele charge avec succes: {load_path}")
        except Exception as exc:
            raise RuntimeError(f"[AGENT] Erreur lors du chargement du modele: {str(exc)}") from exc

    def _state_tensor(self, state):
        if not isinstance(state, (list, np.ndarray)):
            raise TypeError(f"[AGENT] state doit etre une liste ou un numpy array (recu: {type(state)})")
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_action(self, state, deterministic=False):
        state_t = self._state_tensor(state)

        with torch.no_grad():
            logits = self.actor_model(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action_idx = torch.argmax(logits, dim=-1)
            else:
                action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx).sum(-1)
            entropy = dist.entropy().sum(-1)
            value = self.critic_model(state_t).squeeze(-1)

        action_idx = action_idx.squeeze(0)
        action_vec = (action_idx - 1).to(torch.float32)
        actions_np = action_vec.cpu().numpy()
        shoulders = actions_np[:4]
        elbows = actions_np[4:]

        if DEBUG_RL_AGENT:
            print(f"[AGENT] deterministic={deterministic}")
            print(f"[AGENT] action_idx={action_idx}")
            print(f"[AGENT] log_prob={log_prob.item():.4f} entropy={entropy.item():.4f} value={value.item():.4f}")

        action_info = {
            "action_idx": action_idx.detach().cpu(),
            "log_prob": float(log_prob.item()),
            "entropy": float(entropy.item()),
            "value": float(value.item()),
        }
        return shoulders, elbows, action_info

    def evaluate_state(self, state):
        with torch.no_grad():
            return float(self.critic_model(self._state_tensor(state)).squeeze(-1).item())

    def store_transition(self, state, action_info, reward, terminated, truncated):
        self.rollout_buffer.states.append(list(state))
        self.rollout_buffer.actions.append(action_info["action_idx"].clone().to(torch.int64))
        self.rollout_buffer.log_probs.append(float(action_info["log_prob"]))
        self.rollout_buffer.values.append(float(action_info["value"]))
        self.rollout_buffer.rewards.append(float(reward))
        self.rollout_buffer.terminated.append(float(terminated))
        self.rollout_buffer.truncated.append(float(truncated))

    def remember(self, state, action_idx, reward, done, next_state):
        del action_idx, next_state
        self.store_transition(
            state=state,
            action_info={"action_idx": torch.zeros(self.action_size, dtype=torch.int64), "log_prob": 0.0, "value": 0.0},
            reward=reward,
            terminated=done,
            truncated=False,
        )

    def _compute_gae(self, last_value):
        rewards = np.asarray(self.rollout_buffer.rewards, dtype=np.float32)
        values = np.asarray(self.rollout_buffer.values + [float(last_value)], dtype=np.float32)
        terminated = np.asarray(self.rollout_buffer.terminated, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            next_non_terminal = 1.0 - terminated[idx]
            delta = rewards[idx] + self.gamma * values[idx + 1] * next_non_terminal - values[idx]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[idx] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def update_from_rollout(self, last_value):
        if len(self.rollout_buffer) == 0:
            return None

        advantages, returns = self._compute_gae(last_value)

        states_tensor = torch.tensor(self.rollout_buffer.states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.stack(self.rollout_buffer.actions).to(self.device)
        old_log_probs_tensor = torch.tensor(self.rollout_buffer.log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-8)

        actor_losses = []
        critic_losses = []
        entropies = []
        approx_kls = []
        completed_epochs = 0
        rollout_len = len(self.rollout_buffer)
        minibatch_size = max(1, min(self.ppo_minibatch_size, rollout_len))

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(rollout_len, device=self.device)
            stop_epoch = False

            for start in range(0, rollout_len, minibatch_size):
                batch_indices = permutation[start : start + minibatch_size]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                logits = self.actor_model(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1)
                state_values = self.critic_model(batch_states).squeeze(-1)

                ratios = torch.exp(log_probs - batch_old_log_probs)
                unclipped_objective = ratios * batch_advantages.detach()
                clipped_ratios = torch.clamp(
                    ratios,
                    1.0 - self.ppo_clip_eps,
                    1.0 + self.ppo_clip_eps,
                )
                clipped_objective = clipped_ratios * batch_advantages.detach()

                actor_loss = -torch.min(unclipped_objective, clipped_objective).mean()
                actor_loss -= self.entropy_coeff * entropy.mean()
                critic_loss = F.mse_loss(state_values, batch_returns.detach())

                self.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.critic_loss_coeff * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.mean().item()))
                approx_kls.append(float(approx_kl.item()))

                if approx_kl.item() > self.ppo_target_kl:
                    stop_epoch = True
                    break

            completed_epochs += 1
            if stop_epoch:
                break

        with torch.no_grad():
            final_logits = self.actor_model(states_tensor)
            final_dist = torch.distributions.Categorical(logits=final_logits)
            final_log_probs = final_dist.log_prob(actions_tensor).sum(-1)
            final_entropy = final_dist.entropy().sum(-1)
            state_values = self.critic_model(states_tensor).squeeze(-1)
            final_approx_kl = (old_log_probs_tensor - final_log_probs).mean()

        rollout_rewards = np.asarray(self.rollout_buffer.rewards, dtype=np.float32)
        actor_loss_mean = float(np.mean(actor_losses)) if actor_losses else None
        critic_loss_mean = float(np.mean(critic_losses)) if critic_losses else None
        entropy_mean = float(np.mean(entropies)) if entropies else float(final_entropy.mean().item())
        approx_kl_mean = float(np.mean(approx_kls)) if approx_kls else float(final_approx_kl.item())
        metrics = {
            "reward_norm_mean": float(np.mean(rollout_rewards)) if len(rollout_rewards) > 0 else None,
            "critic_loss": critic_loss_mean,
            "actor_loss": actor_loss_mean,
            "entropy": entropy_mean,
            "approx_kl": approx_kl_mean,
            "final_approx_kl": float(final_approx_kl.item()),
            "ppo_epochs_completed": float(completed_epochs),
            "total_loss": (
                float(actor_loss_mean + self.critic_loss_coeff * critic_loss_mean)
                if actor_loss_mean is not None and critic_loss_mean is not None
                else None
            ),
            "returns": returns.tolist(),
            "advantages": advantages.tolist(),
            "state_values": state_values.detach().cpu().numpy().tolist(),
            "rollout_reward_sum": float(np.sum(rollout_rewards)),
            "rollout_len": float(len(self.rollout_buffer)),
            "bootstrap_value": float(last_value),
        }
        self.rollout_buffer.clear()
        return metrics

    def train_model(self, epsilon=None, batch_size=32):
        del epsilon, batch_size
        return None
