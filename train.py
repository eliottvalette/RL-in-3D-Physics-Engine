# train.py
import os
import numpy as np
import time
import traceback
from visualization import DataCollector
from agent import QuadrupedAgent
from physics_env.envs.quadruped_env import QuadrupedEnv
from typing import List, Tuple
from physics_env.core.config import EPISODES, EPS_DECAY, START_EPS, EPS_MIN, DEBUG_RL_TRAIN, SAVE_INTERVAL, PLOT_INTERVAL, MAX_STEPS
from helpers_rl import save_models

def run_episode(env: QuadrupedEnv, agent: QuadrupedAgent, epsilon: float, rendering: bool, episode: int, render_every: int, data_collector: DataCollector) -> Tuple[List[float], List[dict]]:
    """
    Exécute un épisode complet du jeu de quadruped.
    
    Args:
        env (QuadrupedEnv): L'environnement de jeu
        epsilon (float): Paramètre d'exploration
        rendering (bool): Active/désactive le rendu graphique
        episode (int): Numéro de l'épisode en cours
        render_every (int): Fréquence de mise à jour du rendu
        data_collector (DataCollector): Collecteur de données pour la visualisation

    Returns:
        Tuple[List[float], List[dict]]: Récompenses finales et métriques d'entraînement
    """

    env.quadruped.reset()
    env.circles_passed.clear()
    env.prev_potential = None  # Réinitialiser le potentiel pour le nouvel épisode
    env.consecutive_steps_below_critical_height = 0
    env.prev_radius = None

    if DEBUG_RL_TRAIN:
        print(f"[TRAIN] Début de la main")

    #### Boucle principale du jeu ####
    steps_count = 0
    for step in range(MAX_STEPS):
        # Récupération de l'état actuel
        state = env.get_state()

        # Prédiction avec une inférence classique du modèle
        shoulders, elbows, action_vec = agent.get_action(state, epsilon)
        
        if DEBUG_RL_TRAIN:
            print(f"[TRAIN] state : {state}")
            print(f"[TRAIN] shoulder_actions : {shoulders}")
            print(f"[TRAIN] elbow_actions : {elbows}")

        # Exécuter l'action dans l'environnement
        next_state, reward, done, step_time = env.step(shoulders, elbows)
        data_collector.add_metrics(env.last_reward_components.copy())

        # Stocker l'expérience
        agent.remember(state, action_vec, reward, done, next_state)

        data_collector.add_state(state)
        steps_count += 1
    
        # Rendu graphique si activé
        if rendering and (episode % render_every == 0):
            env.render(reward, done, step_time)

        # Training mid-episode car ils sont très longs
        if step % 5 == 0:
            metrics = agent.train_model(epsilon)
            data_collector.add_metrics(metrics)
        
        if done:
            break

    print(f"\n[TRAIN] === Résultats de l'épisode [{episode + 1}/{EPISODES}] ===")
    print(f"[TRAIN] Nombre de steps: {steps_count}")

    metrics = agent.train_model(epsilon)
    final_done_reason = env.last_done_reason if done else "max_steps"
    # Ajouter le nombre de steps aux métriques
    metrics['steps_count'] = steps_count
    metrics['done_reason_too_high'] = 1.0 if final_done_reason == "too_high" else 0.0
    metrics['done_reason_critical_tilt'] = 1.0 if final_done_reason == "critical_tilt" else 0.0
    metrics['done_reason_joint_limit_timeout'] = 1.0 if final_done_reason == "joint_limit_timeout" else 0.0
    metrics['done_reason_max_steps'] = 1.0 if final_done_reason == "max_steps" else 0.0
    data_collector.add_metrics(metrics)
    data_collector.save_episode(episode)

def main_training_loop(agent: QuadrupedAgent, episodes: int, rendering: bool, render_every: int):
    """
    Boucle principale d'entraînement des agents.
    
    Args:
        agent (QuadrupedAgent): Agent à entraîner
        episodes (int): Nombre total d'épisodes d'entraînement
        rendering (bool): Active/désactive le rendu graphique
        render_every (int): Fréquence de mise à jour du rendu graphique
    """
    # Initialisation des historiques et de l'environnement
    env = QuadrupedEnv(rendering=rendering)
    env.clock.tick(240)
    
    # Configuration du collecteur de données
    data_collector = DataCollector(
        save_interval=SAVE_INTERVAL,
        plot_interval=PLOT_INTERVAL,
        start_epsilon=START_EPS,
        epsilon_decay=EPS_DECAY,
        epsilon_min=EPS_MIN
    )
    
    try:
        for episode in range(episodes):
            start_time = time.time()

            # Décroissance d'epsilon
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, EPS_MIN, START_EPS)
            epsilon = np.clip(START_EPS * EPS_DECAY ** episode, EPS_MIN, START_EPS)
            
            # Exécuter l'épisode et obtenir les résultats incluant les métriques
            run_episode(env, agent, epsilon, rendering, episode, render_every, data_collector)
            
            # Afficher les informations de l'épisode
            print(f"\n[TRAIN] Episode [{episode + 1}/{episodes}]")
            print(f"[TRAIN] Randomness: {epsilon*100:.3f}%")
            print(f"[TRAIN] Time taken: {time.time() - start_time:.2f} seconds")
            
        # Save models at end of training
        if episode == episodes - 1 :
            save_models(agent, episode)
            print("[TRAIN] Generating visualization...")
            data_collector.force_visualization()

    except Exception as e:
        print(f"[TRAIN] An error occurred: {e}")
        traceback.print_exc()
        save_models(agent, episode)
        print("[TRAIN] Generating visualization...")
        data_collector.force_visualization()

        # delete temp_viz_json files
        for file in os.listdir("temp_viz_json"):
            os.remove(os.path.join("temp_viz_json", file))
        
    finally:
        save_models(agent, episode)
        print("[TRAIN] Generating visualization...")
        data_collector.force_visualization()

        # delete temp_viz_json files
        for file in os.listdir("temp_viz_json"):
            os.remove(os.path.join("temp_viz_json", file))
