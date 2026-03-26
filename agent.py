# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import QuadrupedActorModel, QuadrupedCriticModel
from physics_env.core.config import DEBUG_RL_AGENT
import torch.nn.functional as F
from collections import deque
import random
import time
import os

class QuadrupedAgent:
    """
    Wrapper de haut niveau qui couple un **Acteur** (politique π_θ) et un **Critique Duel**
    (Q_ϕ & V_ϕ).
    Caractéristiques principales
    ------------ 
      • *Boucle d'apprentissage*  
        1. L'acteur produit π_θ(a | s) et sélectionne les actions (ε-greedy).  
        2. Le critique produit Q(s,·) et V(s) → Cible TD  
           *td* = r + γ maxₐ′ Q(s′, a′).  
        3. Pertes  
           - **Acteur** : -log π_θ · Avantage  (A = Q - V) - β H[π]  
           - **Critique**: MSE(Q(s,a), td)  
        4. Deux optimiseurs Adam indépendants mettent à jour θ et ϕ.
    """
    def __init__(self, device,state_size, action_size, gamma, learning_rate, load_model=False, load_path=None):
        """
        Initialisation de l'agent
        :param state_size: Taille du vecteur d'état
        :param action_size: Nombre d'actions possibles
        :param gamma: Facteur d'actualisation pour les récompenses futures
        :param learning_rate: Taux d'apprentissage
        :param load_model: Si True, charge un modèle existant
        :param load_path: Chemin vers le modèle à charger
        """

        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coeff = 0.1
        self.value_loss_coeff = 0.15
        self.invalid_action_loss_coeff = 15
        self.policy_loss_coeff = 0.8
        self.reward_norm_coeff = 4.0
        self.target_match_loss_coeff = 0.2
        self.critic_loss_coeff = 0.015

        # Utilisation du modèle Transformer qui attend une séquence d'inputs
        self.actor_model = QuadrupedActorModel(input_dim=state_size, output_dim=action_size * 3).to(device)
        self.critic_model = QuadrupedCriticModel(input_dim=state_size).to(device)
        self.critic_target = QuadrupedCriticModel(input_dim=state_size).to(device)
        self.critic_target.load_state_dict(self.critic_model.state_dict())
        self.optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate * 0.1)
        self.memory = deque(maxlen=1_000)  # Buffer de replay
        self.polyak_tau = 0.995

        if load_model:
            self.load(load_path)
        
    def load(self, load_path):
        """
        Charge un modèle sauvegardé
        """
        if not isinstance(load_path, str):
            raise TypeError(f"[AGENT] load_path doit être une chaîne de caractères (reçu: {type(load_path)})")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"[AGENT] Le fichier {load_path} n'existe pas")
        
        try:
            checkpoint = torch.load(load_path)
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_model.load_state_dict(checkpoint['critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"[AGENT] Modèle chargé avec succès: {load_path}")
        except Exception as e:
            raise RuntimeError(f"[AGENT] Erreur lors du chargement du modèle: {str(e)}")

    def get_action(self, state, epsilon=0.0):
        """
        Sélectionne une action continue selon la politique (tanh ∈ [‑1, 1]), avec exploration :
        - epsilon : full random sur [-1,1]
        - sinon : μ + bruit gaussien, puis tanh
        Retourne (shoulder_actions, elbow_actions, action_idx) où action_idx est le vecteur d'indices (8,) dans {0,1,2}.
        """
        import numpy as np
        if not isinstance(state, (list, np.ndarray)):
            raise TypeError(f"[AGENT] state doit être une liste ou un numpy array (reçu: {type(state)})")
        
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, state)

        if np.random.rand() < epsilon:  # FULL RANDOM
            action_idx = torch.randint(0, 3, (self.action_size,), device=self.device)  # (8,) 0/1/2
            if DEBUG_RL_AGENT:
                print(f"[AGENT] random action_idx:", action_idx)

        else:  # POLICY
            with torch.no_grad():
                action_discrete, _ = self.actor_model(state_t)  # (-1,0,1) (1,8)
            action_idx = (action_discrete.squeeze(0) + 1).long()  # → (8,) 0/1/2
            if DEBUG_RL_AGENT:
                print(f"[AGENT] policy action_idx:", action_idx)

        # --- mapping pour l’environnement ---
        action_vec = (action_idx - 1).float()  # (-1,0,1)
        actions_np = action_vec.cpu().numpy()  # (8,)
        shoulder_actions = actions_np[:4]
        elbow_actions = actions_np[4:]

        if DEBUG_RL_AGENT:
            print(f"[AGENT] actions_idx : {action_idx}")
            print(f"[AGENT] shoulder_actions : {shoulder_actions}")
            print(f"[AGENT] elbow_actions : {elbow_actions}")

        # on stocke des INDICES int64 (shape (8,))
        return shoulder_actions, elbow_actions, action_idx

    def remember(self, state, action_idx, reward, done, next_state):
        """
        Stocke une transition dans la mémoire de replay, cette transition sera utilisée pour l'entrainement du model
        """
        self.memory.append((state, action_idx, reward, done, next_state))

    def train_model(self, epsilon, batch_size=32):
        """
        Une étape d'optimisation sur un mini-batch.

        Workflow
        --------
            1.  Échantillonne `batch_size` transitions du buffer de replay choisi
                (court = on-policy, long = off-policy).  
            2.  Calcule
                    π_θ(a|s)                        # Réseau acteur
                    Q_ϕ(s, ·), V_ϕ(s)               # Réseau critique 
                    Q_target(s′, ·)                 # Réseau critique cible pour TD  
                    td_target = r + γ·maxₐ′ Q_target(s′, a′)  
                    advantage = Q(s,a) − V(s)  
            3.  Pertes  
                    critic_loss = Huber(Q(s,a), td_target)  
                    actor_loss  = −E[log π(a|s) · advantage] − β entropy  
            4.  Rétropropager et mettre à jour les deux optimiseurs.
            5.  Mettre à jour le réseau cible avec un lissage de Polyak.
        """
        if len(self.memory) < batch_size:
            if DEBUG_RL_AGENT:
                print('Pas assez de données pour entraîner:', len(self.memory))
            return {
                'reward_norm_mean': None,
                'critic_loss': None,
                'actor_loss': None,
                'entropy': None,
                'total_loss': None,
                'epsilon': epsilon
            }

        batch = random.sample(self.memory, batch_size)
        state, action_idx_b, rewards, dones, next_state = zip(*batch)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        states_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action_idx_tensor = torch.stack(action_idx_b).to(self.device)

        # Critic forward
        state_values = self.critic_model(states_tensor).squeeze(1)
        next_state_values = self.critic_target(next_states_tensor).squeeze(1).detach()

        # TD target et advantage
        td_targets = rewards_tensor + self.gamma * next_state_values * (1 - dones_tensor)
        advantages = td_targets - state_values.detach()

        # Critic loss
        critic_loss = F.mse_loss(state_values, td_targets)

        # Actor loss
        _, probs = self.actor_model(states_tensor)                      # (B, 8, 3)
        dist     = torch.distributions.Categorical(probs=probs)         # (B, 8, 3)  Dist contient 8 variables aléatoires, une pour chaque articulation, de Loi de probabilité définie par le actor pour un state précis
        log_probs_actions = dist.log_prob(action_idx_tensor).sum(-1)    # Somme des logs de probabilités données aux actions réellement jouées 
        entropy      = dist.entropy().sum(-1).mean()

        actor_loss = -(advantages * log_probs_actions).mean() \
                     - self.entropy_coeff * entropy

        # Optim Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Optim Actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak update
        with torch.no_grad():
            for param, target_param in zip(self.critic_model.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(self.polyak_tau)
                target_param.data.add_((1 - self.polyak_tau) * param.data)

        metrics = {
            'reward_norm_mean': rewards_tensor.mean().item(),
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy': entropy.item(),
            'total_loss': actor_loss.item() + critic_loss.item(),
            'epsilon': epsilon,
            'td_targets': td_targets.detach().cpu().numpy().tolist(),
            'state_values': state_values.detach().cpu().numpy().tolist(),
            'action_idx': action_idx_tensor.detach().cpu().numpy().tolist(),
        }
        return metrics
