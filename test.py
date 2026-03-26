# train.py
import os
import numpy as np
import time
import traceback
import pygame
from pygame.locals import *
from visualization import DataCollector
from agent import QuadrupedAgent
from physics_env.envs.quadruped_env import QuadrupedEnv
from typing import List, Tuple
from physics_env.core.config import *
import torch

def test_agent(agent: QuadrupedAgent, env: QuadrupedEnv):
    """
    Exécute un épisode infini du jeu de quadruped, dans lequel
    L'utilisateur peut se déplacer dans l'environnement et voir 
    L'Agent Agir.
    """

    env.quadruped.reset()
    env.circles_passed.clear()
    env.prev_potential = None
    env.consecutive_steps_below_critical_height = 0
    env.prev_radius = None

    #### Boucle principale du jeu ####
    running = True
    while running:
        # Handle pygame events for camera control
        running = env.handle_events()
        
        # Get keyboard state for camera controls
        keys = pygame.key.get_pressed()
        camera_actions = env.handle_camera_controls(keys)

        if keys[K_SPACE]:
            env.quadruped.reset()
            env.circles_passed.clear()
            env.prev_potential = None
            env.consecutive_steps_below_critical_height = 0
            env.prev_radius = None
        
        # Récupération de l'état actuel
        state = env.get_state()

        # Prédiction avec une inférence classique du modèle
        shoulders, elbows, _ = agent.get_action(state = state, epsilon = 0.0)

        # Prédiction de la valeur de l'état
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_value = agent.critic_model.forward(state_tensor).item()

        # Exécuter l'action dans l'environnement avec les actions de caméra
        _, reward, done, step_time = env.step(shoulders, elbows, camera_actions)
        
        # Rendu graphique
        env.render(reward, done, step_time, state_value)

        if done:
            env.quadruped.reset()
            env.circles_passed.clear()
            env.prev_potential = None
            env.consecutive_steps_below_critical_height = 0
            env.prev_radius = None

if __name__ == "__main__":
    agent = QuadrupedAgent(
        device='cpu',
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=True,
        load_path=f"saved_models/quadruped_agent.pth",
    )
    env = QuadrupedEnv(rendering=True)
    test_agent(agent, env)
