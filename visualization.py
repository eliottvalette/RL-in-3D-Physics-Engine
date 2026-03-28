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
        LIST_KEYS = ["returns", "advantages", "state_values"]
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
        plt.style.use("dark_background")
        sns.set_theme(
            style="darkgrid",
            rc={
                "axes.facecolor": "#0e1117",
                "figure.facecolor": "#0e1117",
                "grid.color": "#222a",
                "axes.edgecolor": "#555",
                "text.color": "#dddddd",
                "axes.labelcolor": "#dddddd",
                "xtick.color": "#bbbbbb",
                "ytick.color": "#bbbbbb",
            },
        )
        self.palette = {
            "cyan": "#7BDFF2",
            "blue": "#60A5FA",
            "green": "#80ED99",
            "yellow": "#FFD166",
            "orange": "#F77F00",
            "red": "#FF6B6B",
            "magenta": "#C77DFF",
            "teal": "#2EC4B6",
        }

    def _metric_value_from_episode(self, episode_metrics, metric_name):
        if metric_name == "positive_reward_sum":
            keys = ("distance_reward", "z_speed_reward", "sparse_reward")
            values = [episode_metrics.get(key) for key in keys]
            if any(value is not None for value in values):
                return float(sum(value or 0.0 for value in values))
            return None

        if metric_name == "penalty_sum":
            joint_limit_penalty = episode_metrics.get("joint_limit_penalty")
            if joint_limit_penalty is None:
                joint_limit_penalty = episode_metrics.get("gait_reward")
            values = [
                episode_metrics.get("tilt_penalty"),
                joint_limit_penalty,
                episode_metrics.get("angular_velocity_penalty", episode_metrics.get("pitch_rate_penalty")),
                episode_metrics.get("action_smoothness_penalty"),
                episode_metrics.get("height_penalty"),
            ]
            if any(value is not None for value in values):
                return float(sum(value or 0.0 for value in values))
            return None

        if metric_name == "net_reward_sum":
            keys = (
                "distance_reward",
                "z_speed_reward",
                "sparse_reward",
                "tilt_penalty",
                "joint_limit_penalty",
                "angular_velocity_penalty",
                "action_smoothness_penalty",
                "height_penalty",
                "terminal_event_reward",
            )
            values = [episode_metrics.get(key) for key in keys]
            if episode_metrics.get("joint_limit_penalty") is None and episode_metrics.get("gait_reward") is not None:
                values[4] = episode_metrics.get("gait_reward")
            if any(value is not None for value in values):
                return float(sum(value or 0.0 for value in values))
            return None

        if metric_name == "joint_limit_penalty":
            value = episode_metrics.get("joint_limit_penalty")
            if value is None:
                value = episode_metrics.get("gait_reward")
            if value is None:
                return None
            return float(value)

        if metric_name == "returns_mean":
            values = episode_metrics.get("returns")
            if values:
                return float(np.mean(values))
            return None

        if metric_name == "state_value_mean":
            values = episode_metrics.get("state_values")
            if values:
                return float(np.mean(values))
            return None

        if metric_name == "advantage_mean":
            values = episode_metrics.get("advantages")
            if values:
                return float(np.mean(values))
            return None

        value = episode_metrics.get(metric_name)
        if value is None:
            return None
        return float(value)

    def _sorted_metric_series(self, metrics_data, metric_name):
        episodes = []
        values = []
        for episode_num in sorted(metrics_data.keys(), key=lambda value: int(value)):
            episode_metrics = metrics_data[episode_num]
            metric_value = self._metric_value_from_episode(episode_metrics, metric_name)
            if metric_value is not None:
                episodes.append(int(episode_num))
                values.append(metric_value)
        return np.array(episodes, dtype=np.int32), np.array(values, dtype=np.float64)

    def _rolling_average(self, values, window=None):
        if len(values) == 0:
            return np.array([], dtype=np.float64)
        if window is None:
            window = max(5, min(40, len(values) // 20 or 1))
        return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()

    def _robust_ylim(self, values, lower_q=2.0, upper_q=98.0, pad_ratio=0.12):
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None

        low = float(np.percentile(finite, lower_q))
        high = float(np.percentile(finite, upper_q))
        if np.isclose(low, high):
            spread = max(1.0, abs(low) * 0.1)
            return low - spread, high + spread

        pad = (high - low) * pad_ratio
        return low - pad, high + pad

    def _combined_robust_xlim(self, *arrays, lower_q=1.0, upper_q=99.0, pad_ratio=0.08):
        finite_arrays = [np.asarray(arr, dtype=np.float64) for arr in arrays if len(arr) > 0]
        if not finite_arrays:
            return None
        merged = np.concatenate(finite_arrays)
        merged = merged[np.isfinite(merged)]
        if merged.size == 0:
            return None

        low = float(np.percentile(merged, lower_q))
        high = float(np.percentile(merged, upper_q))
        if np.isclose(low, high):
            spread = max(1.0, abs(low) * 0.1)
            return low - spread, high + spread

        pad = (high - low) * pad_ratio
        return low - pad, high + pad

    def _style_axis(self, ax, title, xlabel="Episode", ylabel=None):
        ax.set_title(title, fontsize=13, pad=10, color="#F5F7FA")
        ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28, linewidth=0.8)
        ax.set_facecolor("#0e1117")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _plot_timeseries(self, ax, episodes, values, title, ylabel, color, robust_ylim=True):
        if len(values) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#888888", transform=ax.transAxes)
            self._style_axis(ax, title, ylabel=ylabel)
            return

        smooth = self._rolling_average(values)
        ax.plot(episodes, values, color=color, alpha=0.20, linewidth=1.0)
        ax.plot(episodes, smooth, color=color, linewidth=2.4)
        ax.scatter(episodes[-1], smooth[-1], color=color, s=18, zorder=3)

        if robust_ylim:
            ylim = self._robust_ylim(values)
            if ylim is not None:
                ax.set_ylim(*ylim)

        self._style_axis(ax, title, ylabel=ylabel)
        ax.text(
            0.015,
            0.92,
            f"last={values[-1]:.3f}\nmean={np.mean(values):.3f}",
            transform=ax.transAxes,
            fontsize=9,
            color="#C9D1D9",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#151B23", edgecolor="#333A45", alpha=0.9),
        )

    def _plot_grouped_series(self, ax, metrics_data, title, ylabel, series_specs, y_limits=None, robust_ylim=True):
        has_data = False
        all_values = []
        for metric_name, label, color in series_specs:
            episodes, values = self._sorted_metric_series(metrics_data, metric_name)
            if len(values) == 0:
                continue
            has_data = True
            smooth = self._rolling_average(values)
            ax.plot(episodes, values, color=color, alpha=0.16, linewidth=0.9)
            ax.plot(episodes, smooth, color=color, linewidth=2.2, label=label)
            all_values.extend(values.tolist())

        self._style_axis(ax, title, ylabel=ylabel)
        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#888888", transform=ax.transAxes)
            return

        if y_limits is not None:
            ax.set_ylim(*y_limits)
        elif robust_ylim:
            ylim = self._robust_ylim(all_values)
            if ylim is not None:
                ax.set_ylim(*ylim)

        ax.legend(frameon=True, facecolor="#151B23", edgecolor="#333A45", fontsize=9, ncol=2)

    def plot_metrics(self, metrics_data, dpi=500):
        """
        Dashboard principal mono-curve 4x3.
        """
        fig, axes = plt.subplots(3, 4, figsize=(24, 16))
        axes = axes.flatten()

        mono_specs = [
            ("reward_norm_mean", "Rollout Reward Mean", "Reward", self.palette["cyan"]),
            ("steps_count", "Steps / Episode", "Steps", self.palette["blue"]),
            ("entropy", "Policy Entropy", "Value", self.palette["green"]),
            ("rollout_len", "Rollout Length", "Steps", self.palette["magenta"]),
            ("critic_loss", "Critic Loss", "Loss", self.palette["blue"]),
            ("actor_loss", "Actor Loss", "Loss", self.palette["red"]),
            ("episode_reward", "Episode Reward", "Reward", self.palette["cyan"]),
            ("forward_progress", "Forward Progress", "Distance", self.palette["orange"]),
            ("distance_reward", "Progress Reward", "Reward", self.palette["orange"]),
            ("z_speed_reward", "Forward Speed Reward", "Reward", self.palette["teal"]),
            ("sparse_reward", "Checkpoint Reward", "Reward", self.palette["yellow"]),
            ("locomotion_reward_scale", "Locomotion Scale", "Scale", self.palette["green"]),
            ("joint_limit_penalty", "Joint Limit Penalty", "Penalty", self.palette["magenta"]),
            ("returns_mean", "Returns Mean", "Value", self.palette["yellow"]),
        ]

        for ax, (metric_name, title, ylabel, color) in zip(axes, mono_specs):
            episodes, values = self._sorted_metric_series(metrics_data, metric_name)
            self._plot_timeseries(ax, episodes, values, title, ylabel, color)

        fig.suptitle("RL Training Dashboard", fontsize=22, color="#F5F7FA", y=0.992)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.savefig(os.path.join(self.viz_dir, 'RL_metrics.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()
        self.plot_metrics_grouped(metrics_data, dpi)
        self.plot_losses(metrics_data, dpi)
        self.plot_state_value_distributions(metrics_data, dpi)
        self.plot_steps_per_episode(metrics_data, dpi)

    def plot_metrics_grouped(self, metrics_data, dpi=500):
        """
        Dashboard groupé par familles de signaux.
        """
        fig, axes = plt.subplots(3, 2, figsize=(22, 18))
        axes = axes.flatten()

        self._plot_grouped_series(
            axes[0],
            metrics_data,
            "Reward Overview",
            "Reward",
            [
                ("reward_norm_mean", "Normalized Reward", self.palette["cyan"]),
                ("positive_reward_sum", "Positive Reward Sum", self.palette["green"]),
                ("distance_reward", "Progress", self.palette["orange"]),
                ("z_speed_reward", "Forward Speed", self.palette["teal"]),
                ("sparse_reward", "Checkpoint", self.palette["yellow"]),
                ("episode_reward", "Episode Reward", self.palette["blue"]),
            ],
        )
        self._plot_grouped_series(
            axes[1],
            metrics_data,
            "Pose Gates",
            "Scale",
            [
                ("tilt_reward_scale", "Tilt Scale", self.palette["orange"]),
                ("height_reward_scale", "Height Scale", self.palette["yellow"]),
                ("locomotion_reward_scale", "Pose Scale", self.palette["green"]),
                ("mean_locomotion_reward_scale", "Episode Mean Pose Scale", self.palette["blue"]),
            ],
        )
        self._plot_grouped_series(
            axes[2],
            metrics_data,
            "Penalty Overview",
            "Penalty",
            [
                ("penalty_sum", "Penalty Sum", self.palette["red"]),
                ("tilt_penalty", "Tilt Penalty", self.palette["orange"]),
                ("joint_limit_penalty", "Joint Limit", self.palette["magenta"]),
                ("angular_velocity_penalty", "Angular Velocity", self.palette["blue"]),
                ("action_smoothness_penalty", "Action Smoothness", self.palette["teal"]),
            ],
        )
        self._plot_grouped_series(
            axes[3],
            metrics_data,
            "Evaluation",
            "Value",
            [
                ("eval_episode_reward", "Eval Reward", self.palette["cyan"]),
                ("eval_forward_progress", "Eval Progress", self.palette["orange"]),
                ("eval_mean_locomotion_scale", "Eval Pose Scale", self.palette["green"]),
                ("eval_clean_episode", "Eval Clean", self.palette["yellow"]),
            ],
        )

        steps_episodes, steps_values = self._sorted_metric_series(metrics_data, "steps_count")
        if len(steps_values) > 0:
            smooth_steps = self._rolling_average(steps_values, window=min(30, max(10, len(steps_values) // 15)))
            axes[4].scatter(steps_episodes, steps_values, color=self.palette["blue"], alpha=0.30, s=12)
            axes[4].plot(steps_episodes, smooth_steps, color=self.palette["blue"], linewidth=2.6, label="Steps")
            axes[4].axhline(np.mean(steps_values), color=self.palette["yellow"], linestyle="--", linewidth=1.5, label="Mean")
            ylim = self._robust_ylim(steps_values, lower_q=1.0, upper_q=99.0, pad_ratio=0.08)
            if ylim is not None:
                axes[4].set_ylim(*ylim)
            self._style_axis(axes[4], "Episode Length", ylabel="Steps")
            axes[4].legend(frameon=True, facecolor="#151B23", edgecolor="#333A45", fontsize=9)
        else:
            self._style_axis(axes[4], "Episode Length", ylabel="Steps")
            axes[4].text(0.5, 0.5, "No data", ha="center", va="center", color="#888888", transform=axes[4].transAxes)

        self._plot_grouped_series(
            axes[5],
            metrics_data,
            "Done Rates",
            "Rate",
            [
                ("done_reason_critical_tilt", "Critical Tilt", self.palette["red"]),
                ("done_reason_joint_limit_timeout", "Joint Limit", self.palette["magenta"]),
                ("done_reason_too_high", "Too High", self.palette["yellow"]),
                ("done_reason_max_steps", "Clean Episode", self.palette["green"]),
            ],
            y_limits=(-0.02, 1.02),
            robust_ylim=False,
        )

        fig.suptitle("RL Training Dashboard", fontsize=22, color="#F5F7FA", y=0.992)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.savefig(os.path.join(self.viz_dir, 'RL_metrics_group.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_losses(self, metrics_data, dpi=500):
        """
        Trace les losses en full-scale + zoom robuste.
        """
        actor_episodes, actor_losses = self._sorted_metric_series(metrics_data, "actor_loss")
        critic_episodes, critic_losses = self._sorted_metric_series(metrics_data, "critic_loss")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (episodes, values, name, color) in enumerate([
            (actor_episodes, actor_losses, "Actor Loss", self.palette["red"]),
            (critic_episodes, critic_losses, "Critic Loss", self.palette["blue"]),
        ]):
            raw_ax = axes[idx * 2]
            zoom_ax = axes[idx * 2 + 1]
            smooth = self._rolling_average(values)

            raw_ax.plot(episodes, values, color=color, alpha=0.25, linewidth=1.0)
            raw_ax.plot(episodes, smooth, color=color, linewidth=2.5)
            if len(values) > 0:
                linthresh = max(1e-2, np.nanpercentile(np.abs(values), 60) * 0.05)
                raw_ax.set_yscale("symlog", linthresh=linthresh)
            self._style_axis(raw_ax, f"{name} Full Scale", ylabel="Loss")

            zoom_ax.plot(episodes, values, color=color, alpha=0.18, linewidth=1.0)
            zoom_ax.plot(episodes, smooth, color=color, linewidth=2.5)
            ylim = self._robust_ylim(values)
            if ylim is not None:
                zoom_ax.set_ylim(*ylim)
            self._style_axis(zoom_ax, f"{name} Zoomed", ylabel="Loss")

        fig.suptitle("Actor / Critic Losses", fontsize=20, color="#F5F7FA", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.savefig(os.path.join(self.viz_dir, 'losses.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_state_value_distributions(self, metrics_data, dpi=500):
        """
        Histogrammes dark des TD targets et state values avec stats de résumé.
        """
        all_returns = []
        all_state_values = []
        for episode_metrics in metrics_data.values():
            if 'returns' in episode_metrics and episode_metrics['returns'] is not None:
                all_returns.extend(episode_metrics['returns'])
            if 'state_values' in episode_metrics and episode_metrics['state_values'] is not None:
                all_state_values.extend(episode_metrics['state_values'])

        if not all_returns or not all_state_values:
            print("[VIZ] Not enough value data found, skipping value distribution plots.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        xlim = self._combined_robust_xlim(all_returns, all_state_values)
        plots = [
            (axes[0], np.asarray(all_returns, dtype=np.float64), "Returns", self.palette["orange"]),
            (axes[1], np.asarray(all_state_values, dtype=np.float64), "Critic State Values", self.palette["blue"]),
        ]

        for ax, values, title, color in plots:
            sns.histplot(values, color=color, kde=True, stat='density', bins=50, alpha=0.55, ax=ax)
            ax.axvline(np.mean(values), color="#F5F7FA", linestyle="--", linewidth=1.3, alpha=0.8)
            ax.axvline(np.median(values), color="#BBBBBB", linestyle=":", linewidth=1.3, alpha=0.8)
            if xlim is not None:
                ax.set_xlim(*xlim)
            self._style_axis(ax, title, xlabel="Value", ylabel="Density")
            ax.text(
                0.015,
                0.87,
                f"mean={np.mean(values):.2f}\nmedian={np.median(values):.2f}\nstd={np.std(values):.2f}",
                transform=ax.transAxes,
                fontsize=9,
                color="#C9D1D9",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#151B23", edgecolor="#333A45", alpha=0.9),
            )

        fig.suptitle("Value Distributions", fontsize=20, color="#F5F7FA", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.savefig(os.path.join(self.viz_dir, 'state_value_distributions.jpg'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_steps_per_episode(self, metrics_data, dpi=500):
        """
        Graphique dark du nombre de steps par épisode avec tendance lisible.
        """
        episodes = []
        steps_counts = []
        for episode_num in sorted(metrics_data.keys(), key=lambda value: int(value)):
            episode_metrics = metrics_data[episode_num]
            if 'steps_count' in episode_metrics and episode_metrics['steps_count'] is not None:
                episodes.append(int(episode_num))
                steps_counts.append(float(episode_metrics['steps_count']))

        if not steps_counts:
            print("[VIZ] Aucune donnée de steps_count trouvée")
            return

        episodes_array = np.asarray(episodes, dtype=np.int32)
        steps_array = np.asarray(steps_counts, dtype=np.float64)
        rolling_avg = self._rolling_average(steps_array, window=min(30, max(10, len(steps_array) // 15)))

        plt.figure(figsize=(16, 9))
        plt.scatter(episodes_array, steps_array, color=self.palette["blue"], alpha=0.35, s=12, label="Episode Steps")
        plt.plot(episodes_array, rolling_avg, color=self.palette["red"], linewidth=2.8, label="Rolling Mean")
        plt.axhline(np.mean(steps_array), color=self.palette["yellow"], linestyle="--", linewidth=1.8, label=f"Mean = {np.mean(steps_array):.1f}")

        ylim = self._robust_ylim(steps_array, lower_q=1.0, upper_q=99.0, pad_ratio=0.08)
        if ylim is not None:
            plt.ylim(*ylim)

        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps per Episode')
        plt.legend(frameon=True, facecolor="#151B23", edgecolor="#333A45")
        plt.grid(True, alpha=0.28)
        plt.figtext(
            0.5,
            0.02,
            f"min={np.min(steps_array):.0f} | max={np.max(steps_array):.0f} | std={np.std(steps_array):.1f}",
            ha='center',
            fontsize=10,
            color="#D9E2EC",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#151B23", edgecolor="#333A45", alpha=0.95),
        )

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

    plt.close('all')
