# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd
from physics_env.core.config import DEBUG_RL_MODEL

class QuadrupedActorModel(nn.Module):
    """
    Calcule la politique π_θ(a | s).

    Le réseau construit un vecteur de caractéristiques partagé h = shared_layers(state),
    puis produit des logits pour toutes les combinaisons d'actions possibles.

    La sortie est une distribution catégorielle sur toutes les combinaisons d'actions possibles.
    """
    def __init__(self, input_dim, output_dim, dim_feedforward=512):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.seq_1 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.seq_2 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.seq_3 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )
        self.action_head = nn.Sequential(
            nn.Linear(dim_feedforward, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

    def forward(self, state, deterministic: bool = False):
        """
        Args
        ----
        state : (batch, state_dim)  — entrée normalisée
        deterministic : bool        — True → argmax (évaluation)

        Returns
        -------
        action :   (batch, 8)        ∈ {-1,0,1}
        probs :    (batch, 8, 3)     pour debug éventuel
        """
        x_1 = self.seq_1(state)
        x_2 = self.seq_2(x_1)
        x_3 = self.seq_3(x_2)
        action_logits  = self.action_head(x_3)                          # (B, 24)
        action_logits = action_logits.view(-1, self.output_dim // 3, 3) # (B, 8, 3)

        # soft‑max par artic. (dim=-1)
        probs   = F.softmax(action_logits, dim=-1)       # (B, 8, 3)

        # distribution catégorielle indépendante pour chaque joint
        dist    = torch.distributions.Categorical(probs=probs)

        # échantillonnage / argmax
        action_idx = dist.sample()   # (B, 8)

        # map [0,1,2] → [-1,0,+1]
        actions   = (action_idx - 1).float()

        if DEBUG_RL_MODEL and rd.random() < 0.01:
            print(f"[MODEL] [ACTOR] probs mean={action_logits.mean():.3f}, std={action_logits.std():.3f}, min={action_logits.min():.3f}, max={action_logits.max():.3f}")
            print(f"[MODEL] [ACTOR] action_logits : {action_logits[0]}")
            print(f"[MODEL] [ACTOR] actions : {actions[0]}")

        return actions, probs
    
class QuadrupedCriticModel(nn.Module):
    """
    Réseau Q duel pour les actions composites :
        • branche partagée  → h
        • tête V(s)         → (batch,1)
        • tête A(s,a)       → (batch, num_actions) - une pour chaque action
        • Q(s,a)=V+A-mean(A)
    """
    def __init__(self, input_dim, dim_feedforward=512):
        super().__init__()

        self.seq_1 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.seq_2 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.seq_3 = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward)
        )

        self.V_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, 1)

        )

    def forward(self, input_tensor):
        """
        Here V(s) estimates the value of the state, it's an estimation of how much the situation is favorable (in terms of future expected rewards)
        """
        x_1 = self.seq_1(input_tensor)
        x_2 = self.seq_2(x_1)
        x_3 = self.seq_3(x_2)
        
        if DEBUG_RL_MODEL and rd.random() < 0.02 :
            print(f"[MODEL] [CRITIC] x_3 : mean = {x_3.mean()}, std = {x_3.std()}, min = {x_3.min()}, max = {x_3.max()}")

        V = self.V_head(x_3)

        if DEBUG_RL_MODEL and rd.random() < 0.02:
            print(f"[MODEL] [CRITIC] V : mean = {V.mean()}, std = {V.std()}, min = {V.min()}, max = {V.max()}")

        return V



