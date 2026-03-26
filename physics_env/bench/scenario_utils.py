"""Shared utilities for deterministic quadruped bench scenarios."""

from __future__ import annotations

import numpy as np


def reset_env_state(env):
    env.quadruped.reset()
    env.quadruped.prev_vertices = None
    env.circles_passed.clear()
    env.prev_potential = None
    env.consecutive_steps_below_critical_height = 0
    env.consecutive_steps_above_critical_height = 0
    env.prev_radius = None


def sync_quadruped_state(env):
    env.quadruped.sync_orientation_from_euler()
    env.quadruped._needs_update = True
    env.quadruped.rotated_vertices = env.quadruped.get_vertices()
    env.quadruped.snapshot_local_geometry()


def set_base_state(env, position=None, rotation=None, velocity=None, angular_velocity=None):
    if position is not None:
        env.quadruped.position = np.array(position, dtype=np.float64)
    if rotation is not None:
        env.quadruped.rotation = np.array(rotation, dtype=np.float64)
    if velocity is not None:
        env.quadruped.velocity = np.array(velocity, dtype=np.float64)
    if angular_velocity is not None:
        env.quadruped.angular_velocity = np.array(angular_velocity, dtype=np.float64)
    sync_quadruped_state(env)


def set_joint_pose(env, shoulders=None, elbows=None):
    if shoulders is not None:
        env.quadruped.shoulder_angles = np.array(shoulders, dtype=np.float64)
        env.quadruped.shoulder_velocities = np.zeros(4, dtype=np.float64)
    if elbows is not None:
        env.quadruped.elbow_angles = np.array(elbows, dtype=np.float64)
        env.quadruped.elbow_velocities = np.zeros(4, dtype=np.float64)
    sync_quadruped_state(env)


def align_lowest_vertex_to_ground(env, clearance=0.0):
    sync_quadruped_state(env)
    min_y = float(env.quadruped.rotated_vertices[:, 1].min())
    env.quadruped.position[1] += clearance - min_y
    sync_quadruped_state(env)


def zero_actions():
    return np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)


class BenchScenario:
    name = "unnamed"
    category = "generic"
    description = ""
    stop_on_env_done = False

    def reset(self, env):
        raise NotImplementedError

    def actions(self, env, step_idx):
        del env, step_idx
        return zero_actions()

    def should_stop(self, env, step_idx, done):
        del env, step_idx
        return self.stop_on_env_done and done
