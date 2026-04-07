import pygame
from pygame.locals import *
from agent import QuadrupedAgent
from physics_env.envs.quadruped_env import QuadrupedEnv
from physics_env.core.config import *
import torch

def test_agent(agent: QuadrupedAgent, env: QuadrupedEnv):
    """
    Exécute un épisode infini du jeu de quadruped, dans lequel
    L'utilisateur peut se déplacer dans l'environnement et voir 
    L'Agent Agir.
    """

    env.reset_episode()

    #### Boucle principale du jeu ####
    running = True
    while running:
        # Handle pygame events for camera control
        running = env.handle_events()
        
        # Get keyboard state for camera controls
        keys = pygame.key.get_pressed()
        camera_actions = env.handle_camera_controls(keys)

        if keys[K_SPACE]:
            env.reset_episode()
        
        # Récupération de l'état actuel
        state = env.get_state()

        # Prédiction avec une inférence classique du modèle
        shoulders, elbows, _ = agent.get_action(state=state, deterministic=True)

        # Prédiction de la valeur de l'état
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_value = agent.critic_model.forward(state_tensor).item()

        # Exécuter l'action dans l'environnement avec les actions de caméra
        _, reward, done, step_time = env.step(shoulders, elbows, camera_actions)
        
        # Rendu graphique
        env.render(reward, done, step_time, state_value)

        if done:
            env.reset_episode()


def build_test_agent():
    return QuadrupedAgent(
        device='cpu',
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=True,
        load_path="saved_models/quadruped_agent.pth",
    )


def main():
    agent = build_test_agent()
    env = QuadrupedEnv(rendering=True)
    test_agent(agent, env)

if __name__ == "__main__":
    main()
