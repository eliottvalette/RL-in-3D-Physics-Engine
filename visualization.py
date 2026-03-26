# visualization.py
import matplotlib
matplotlib.use('Agg')  # Use Agg backend that doesn't require GUI
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
import os
import json
import pandas as pd
from datetime import datetime
from physics_env.core.config import DEBUG_RL_VIZ, START_EPS, EPS_DECAY, EPS_MIN, PLOT_INTERVAL, SAVE_INTERVAL

# Remove PLAYERS
# PLAYERS = ['Player_0', 'Player_1', 'Player_2']

class DataCollector:
    def __init__(self, save_interval, plot_interval, start_epsilon, epsilon_decay, epsilon_min, output_dir="temp_viz_json"):
        """
        Initialise le collecteur de données.
        
        Args:
            save_interval (int): Nombre d'épisodes à regrouper avant de sauvegarder
            plot_interval (int): Intervalle pour le tracé
            output_dir (str): Répertoire pour enregistrer les fichiers JSON
        """
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.output_dir = output_dir
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.current_episode_states = []
        self.batch_episode_states = [] # Contient un liste de current_episode_states qui seront ajouté toutes les save_interval dans le fichier json
        self.batch_episode_metrics = [] # Contient les métriques d'entraînement pour chaque épisode
        self.current_episode_metrics = [] # Liste des métriques pour l'épisode courant
        
        # Créer le répertoire pour les JSON s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Créer le dossier visualizations si il n'existe pas
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")

        # Créer le dossier daté pour les visualisations
        self.viz_dir = "visualizations" # On save dans le dossier visualizations, pas le peine de créer un dossier daté
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)

        # Supprimer les fichiers JSON existants dans le répertoire
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

        self.visualizer = Visualizer(
            output_dir=output_dir, 
            viz_dir=self.viz_dir,
            start_epsilon=start_epsilon, 
            epsilon_decay=epsilon_decay, 
            epsilon_min=epsilon_min,
            plot_interval=plot_interval, 
            save_interval=save_interval
        )
    
    def add_state(self, state):
        """
        Ajoute un état à l'épisode courant. Convertit les numpy arrays en listes pour la sérialisation JSON.
        """
        if isinstance(state, np.ndarray):
            self.current_episode_states.append(state.tolist())
        else:
            self.current_episode_states.append(state.copy() if hasattr(state, 'copy') else state)
    
    def add_metrics(self, episode_metrics):
        """
        Ajoute les métriques d'une étape à la liste de l'épisode courant.
        Args:
            episode_metrics (dict): Dictionnaire des métriques pour une étape
        """
        self.current_episode_metrics.append(episode_metrics)
    
    def save_episode(self, episode_num):
        """
        Sauvegarde les états et métriques de l'épisode courant dans des fichiers JSON.
        
        Args:
            episode_num (int): Numéro de l'épisode
        """
        # Ajouter l'épisode courant au batch
        self.batch_episode_states.append(self.current_episode_states)

        # Calculer la moyenne des métriques de l'épisode
        # Récupérer toutes les clés présentes dans les dicts
        all_keys = set()
        for m in self.current_episode_metrics:
            all_keys.update(m.keys())
        mean_metrics = {}
        LIST_KEYS = ["td_targets", "state_values", "action_idx"]
        for key in all_keys:
            values = [m[key] for m in self.current_episode_metrics if key in m and m[key] is not None]
            if key in LIST_KEYS:
                # S'assurer que chaque valeur est une liste
                concat = []
                for v in values:
                    if not isinstance(v, list):
                        v = [v]
                    concat.extend(v)
                mean_metrics[key] = concat
            else:
                mean_metrics[key] = float(np.mean(values)) if values else None
        self.batch_episode_metrics.append(mean_metrics)
        self.current_episode_metrics = []
        
        # Vérifier si on a atteint l'intervalle de sauvegarde
        if len(self.batch_episode_states) >= self.save_interval:
            # Sauvegarder les états
            states_filename = os.path.join(self.output_dir, "episodes_states.json")
            
            # Créer le fichier avec un dictionnaire vide s'il n'existe pas
            if not os.path.exists(states_filename):
                with open(states_filename, 'w') as f:
                    f.write("{}")
            
            # Ajouter les nouveaux épisodes au fichier JSON
            for i, episode_states in enumerate(self.batch_episode_states):
                episode_idx = episode_num - len(self.batch_episode_states) + i + 1
                self._append_to_json(states_filename, str(episode_idx), episode_states)
            
            # Sauvegarder les métriques
            metrics_filename = os.path.join(self.output_dir, "metrics_history.json")
            
            # Créer le fichier avec un dictionnaire vide s'il n'existe pas
            if not os.path.exists(metrics_filename):
                with open(metrics_filename, 'w') as f:
                    f.write("{}")
            
            # Ajouter les nouvelles métriques au fichier JSON
            for i, episode_metrics in enumerate(self.batch_episode_metrics):
                episode_idx = episode_num - len(self.batch_episode_metrics) + i + 1
                self._append_to_json(metrics_filename, str(episode_idx), episode_metrics)

            # Reset batches
            self.batch_episode_states = []
            self.batch_episode_metrics = []
        
        # Reset current episode states
        self.current_episode_states = []

        if episode_num % (self.plot_interval) == (self.plot_interval) - 1:
            # Load Jsons
            metrics_path = os.path.join(self.output_dir, "metrics_history.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                self.visualizer.plot_metrics(metrics_data)
                plt.close('all')

    def _append_to_json(self, file_path, key, data):
        """
        Ajoute une nouvelle entrée à un fichier JSON existant en modifiant le fichier en place.
        
        Args:
            file_path (str): Chemin du fichier JSON
            key (str): Clé de la nouvelle entrée
            data: Données à ajouter
        """
        with open(file_path, 'r+') as f:
            f.seek(0)
            content = f.read().strip()
            # On part du principe que le contenu est un dict valide
            # Supprimer le dernier caractère (qui doit être "}")
            f.seek(0, os.SEEK_END)
            pos = f.tell() - 1
            f.seek(pos)
            last_char = f.read(1)
            if last_char != '}':
                raise ValueError(f"Fichier JSON mal formé: {file_path}")
            
            # Vérifier si le dict est vide
            if content == "{}":
                new_content = f'"{key}": {json.dumps(data)}'
            else:
                new_content = f', "{key}": {json.dumps(data)}'
            
            # Réécriture : on tronque le dernier "}" et on y ajoute notre contenu + "}"
            f.seek(pos)
            f.truncate()
            f.write(new_content + "}")
    
    def force_visualization(self):
        """
        Force la génération de toutes les visualisations
        """
        metrics_path = os.path.join(self.output_dir, "metrics_history.json")
        if not os.path.exists(metrics_path):
            print("[VIZ] No metrics file found, skipping visualization.")
            return
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        self.visualizer.plot_metrics(metrics_data)
        self.visualizer.plot_losses(metrics_data)
        self.visualizer.plot_state_value_distributions(metrics_data)
        self.visualizer.plot_steps_per_episode(metrics_data)
        plt.close('all')

class Visualizer:
    """
    Visualise les métriques d'entraînement RL (quadruped)
    """
    def __init__(self, start_epsilon, epsilon_decay, epsilon_min, plot_interval, save_interval, output_dir="temp_viz_json", viz_dir=None):
        self.output_dir = output_dir
        self.viz_dir = viz_dir or "visualizations"
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.plot_interval = plot_interval
        self.save_interval = save_interval
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def plot_metrics(self, metrics_data, dpi=500):
        """
        Génère des visualisations des métriques d'entraînement RL à partir du fichier metrics_history.json
        """
        fig = plt.figure(figsize=(24, 20))
        pastel_colors = ['#003049', '#006DAA', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']
        metrics_to_plot = [
            ('reward_norm_mean', 'Reward Normalisée Moyenne', None, None),
            ('critic_loss', 'Critic Loss (MSE entre Q-Value et TD-Target)', None, None),
            ('actor_loss', 'Actor Loss (Log-Prob * Advantage)', None, None),
            ('entropy', 'Entropie', None, None),
            ('total_loss', 'Actor Loss + Critic Loss', None, None),
            ('epsilon', 'Epsilon (Exploration)', 1, 0),
        ]
        for idx, (metric_name, display_name, y_max, y_min) in enumerate(metrics_to_plot):
            ax = plt.subplot(2, 3, idx + 1)
            episodes = []
            values = []
            for episode_num, episode_metrics in metrics_data.items():
                if metric_name in episode_metrics and episode_metrics[metric_name] is not None:
                    episodes.append(int(episode_num))
                    values.append(float(episode_metrics[metric_name]))
            window = self.plot_interval * 3
            if len(values) > 0:
                rolling_avg = pd.Series(values).rolling(window=window, min_periods=1).mean()
                ax.plot(episodes, rolling_avg, label='Agent', color=pastel_colors[0], linewidth=2)
            ax.set_title(f'Evolution de {display_name}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(display_name)
            ax.legend()
            ax.set_ylim(y_min, y_max)
            ax.set_facecolor('#F0F0F0')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.rcParams.update({
            'figure.facecolor': '#FFFFFF',
            'axes.facecolor': '#F8F9FA',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 2,
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'RL_metrics.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()
        self.plot_losses(metrics_data, dpi)
        self.plot_state_value_distributions(metrics_data, dpi)
        self.plot_steps_per_episode(metrics_data, dpi)

    def plot_losses(self, metrics_data, dpi=500):
        """
        Trace actor_loss et critic_loss sur le même graphique.
        """
        episodes = []
        actor_losses = []
        critic_losses = []
        for episode_num, episode_metrics in metrics_data.items():
            if 'actor_loss' in episode_metrics and episode_metrics['actor_loss'] is not None:
                episodes.append(int(episode_num))
                actor_losses.append(float(episode_metrics['actor_loss']))
            else:
                actor_losses.append(np.nan)
            if 'critic_loss' in episode_metrics and episode_metrics['critic_loss'] is not None:
                critic_losses.append(float(episode_metrics['critic_loss']))
            else:
                critic_losses.append(np.nan)
        plt.figure(figsize=(12, 7))
        plt.plot(episodes, pd.Series(actor_losses).rolling(window=10, min_periods=1).mean(), label='Actor Loss', color='#D62828')
        plt.plot(episodes, pd.Series(critic_losses).rolling(window=10, min_periods=1).mean(), label='Critic Loss', color='#003049')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Actor & Critic Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'losses.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_state_value_distributions(self, metrics_data, dpi=500):
        """
        Histogrammes des td_targets (true state value) et state_values (prédits par le critic).
        Affiche deux sous-graphiques côte à côte : un pour td_targets, un pour state_values.
        """
        all_td_targets = []
        all_state_values = []
        for episode_metrics in metrics_data.values():
            if 'td_targets' in episode_metrics and episode_metrics['td_targets'] is not None:
                all_td_targets.extend(episode_metrics['td_targets'])
            if 'state_values' in episode_metrics and episode_metrics['state_values'] is not None:
                all_state_values.extend(episode_metrics['state_values'])

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharey=True)
        
        sns.histplot(all_td_targets, color='#F77F00', label='TD Targets (True State Value)', kde=True, stat='density', bins=40, alpha=0.6, ax=axes[0])
        axes[0].set_xlabel('TD Target Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution des TD Targets')
        axes[0].legend()
        axes[0].set_xlim(min(all_td_targets+all_state_values), max(all_td_targets+all_state_values))

        sns.histplot(all_state_values, color='#006DAA', label='State Values (Critic)', kde=True, stat='density', bins=40, alpha=0.6, ax=axes[1])
        axes[1].set_xlabel('State Value (Critic)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution des State Values (Critic)')
        axes[1].legend()
        axes[1].set_xlim(min(all_td_targets+all_state_values), max(all_td_targets+all_state_values))

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'state_value_distributions.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_steps_per_episode(self, metrics_data, dpi=500):
        """
        Graphique dédié au nombre de steps par épisode avec une ligne de tendance.
        """
        episodes = []
        steps_counts = []
        
        for episode_num, episode_metrics in metrics_data.items():
            if 'steps_count' in episode_metrics and episode_metrics['steps_count'] is not None:
                episodes.append(int(episode_num))
                steps_counts.append(float(episode_metrics['steps_count']))
        
        if not steps_counts:
            print("[VIZ] Aucune donnée de steps_count trouvée")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Graphique principal avec les points
        plt.plot(episodes, steps_counts, 'o-', color='#006DAA', alpha=0.6, linewidth=1, markersize=3, label='Steps par épisode')
        
        # Ligne de tendance (moyenne mobile)
        window = min(20, len(steps_counts) // 4)  # Fenêtre adaptative
        if len(steps_counts) > window:
            rolling_avg = pd.Series(steps_counts).rolling(window=window, min_periods=1).mean()
            plt.plot(episodes, rolling_avg, color='#D62828', linewidth=3, label=f'Moyenne mobile ({window} épisodes)')
        
        # Ligne horizontale pour la moyenne globale
        mean_steps = np.mean(steps_counts)
        plt.axhline(y=mean_steps, color='#F77F00', linestyle='--', linewidth=2, label=f'Moyenne globale: {mean_steps:.1f}')
        
        plt.xlabel('Episode')
        plt.ylabel('Nombre de Steps')
        plt.title('Évolution du Nombre de Steps par Épisode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ajouter des statistiques en texte
        stats_text = f'Min: {min(steps_counts):.0f} | Max: {max(steps_counts):.0f} | Écart-type: {np.std(steps_counts):.1f}'
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'steps_per_episode.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    visualizer = Visualizer(start_epsilon=START_EPS, epsilon_decay=EPS_DECAY, epsilon_min=EPS_MIN, plot_interval=PLOT_INTERVAL, save_interval=SAVE_INTERVAL)
    # On les json une seule fois
    states_path = os.path.join(visualizer.output_dir, "episodes_states.json")
    metrics_path = os.path.join(visualizer.output_dir, "metrics_history.json")

    with open(states_path, 'r') as f:
        states_data = json.load(f)
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)

    visualizer.plot_metrics(metrics_data)
    visualizer.plot_losses(metrics_data)
    visualizer.plot_state_value_distributions(metrics_data)
    visualizer.plot_steps_per_episode(metrics_data)

    plt.close('all')
