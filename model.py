# model.py

import random as rd

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_env.core.config import DEBUG_RL_MODEL


class QuadrupedActorModel(nn.Module):
    """Policy network producing per-joint categorical logits."""

    def __init__(self, input_dim, output_dim, dim_feedforward=512):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.seq_1 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
        )

        self.seq_2 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
        )

        self.seq_3 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
        )

        self.action_head = nn.Sequential(
            nn.Linear(dim_feedforward, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, state):
        x_1 = self.seq_1(state)
        x_2 = self.seq_2(x_1)
        x_3 = self.seq_3(x_2)
        action_logits = self.action_head(x_3).view(-1, self.output_dim // 3, 3)

        if DEBUG_RL_MODEL and rd.random() < 0.01:
            probs = F.softmax(action_logits, dim=-1)
            print(
                "[MODEL] [ACTOR] logits",
                f"mean={action_logits.mean():.3f}",
                f"std={action_logits.std():.3f}",
                f"min={action_logits.min():.3f}",
                f"max={action_logits.max():.3f}",
            )
            print(f"[MODEL] [ACTOR] probs[0]: {probs[0]}")

        return action_logits

    def get_policy(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class QuadrupedCriticModel(nn.Module):
    """Value network estimating V(s)."""

    def __init__(self, input_dim, dim_feedforward=512):
        super().__init__()

        self.seq_1 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
        )

        self.seq_2 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
        )

        self.seq_3 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
        )

        self.V_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, input_tensor):
        x_1 = self.seq_1(input_tensor)
        x_2 = self.seq_2(x_1)
        x_3 = self.seq_3(x_2)

        if DEBUG_RL_MODEL and rd.random() < 0.02:
            print(
                "[MODEL] [CRITIC]",
                f"mean={x_3.mean():.3f}",
                f"std={x_3.std():.3f}",
                f"min={x_3.min():.3f}",
                f"max={x_3.max():.3f}",
            )

        return self.V_head(x_3)
