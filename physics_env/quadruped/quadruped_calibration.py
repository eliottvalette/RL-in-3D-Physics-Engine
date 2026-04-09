import numpy as np


# Python-side copy of the validated box rig geometry from `three_rendering`.
# The effective physics geometry is derived by a single uniform scale so the
# whole rig can be resized coherently without touching individual dimensions.
SOURCE_RIG_SCALE = 0.6632184350604978
PHYSICS_RIG_SCALE = SOURCE_RIG_SCALE / 1.1


def _vec3(values):
    return np.array(values, dtype=np.float64)


def _scaled_vec3(values):
    return PHYSICS_RIG_SCALE * _vec3(values)


# Local root convention:
# - root remains the quadruped "body height" reference used by the env
# - geometry is scaled from the viewer rig but kept fully local to Python
BODY_CENTER_OFFSET = _scaled_vec3([0.0, -0.05, 0.15])
BODY_SIZE = _scaled_vec3([5.7461, 1.938, 12.0739])  # x, y, z

UPPER_LEG_SIZE = _scaled_vec3([1.3, 3.9, 1.6753])  # x, y, z
LOWER_LEG_SIZE = _scaled_vec3([0.752, 4.9, 1.1237])  # x, y, z


# Quadruped leg order in Python:
# 0 = front_right, 1 = front_left, 2 = back_right, 3 = back_left
LEG_ORDER = ("front_right", "front_left", "back_right", "back_left")

SHOULDER_POSITIONS = (
    _scaled_vec3([2.77, 0.55, 4.95]),
    _scaled_vec3([-2.77, 0.55, 4.95]),
    _scaled_vec3([2.77, 0.55, -5.3]),
    _scaled_vec3([-2.77, 0.55, -5.3]),
)

UPPER_LEG_CENTER_OFFSETS = (
    _scaled_vec3([0.7572, -2.4212, 0.0]),
    _scaled_vec3([-0.7572, -2.4212, 0.0]),
    _scaled_vec3([0.7572, -2.4212, 0.0]),
    _scaled_vec3([-0.7572, -2.4212, 0.0]),
)

ELBOW_OFFSETS = (
    _scaled_vec3([0.7572, -4.2, -0.15]),
    _scaled_vec3([-0.7572, -4.2, -0.15]),
    _scaled_vec3([0.7572, -4.2, -0.15]),
    _scaled_vec3([-0.7572, -4.2, -0.15]),
)

LOWER_LEG_CENTER_OFFSETS = (
    _scaled_vec3([1.1, -2.4, -0.2]),
    _scaled_vec3([-1.1, -2.4, -0.2]),
    _scaled_vec3([1.1, -2.4, 0.0]),
    _scaled_vec3([-1.1, -2.4, 0.0]),
)


# Python-side rest pose starts from neutral joint angles.
INITIAL_SHOULDER_ANGLES = np.array([-0.6, -0.6, -0.6, -0.6], dtype=np.float64)
INITIAL_ELBOW_ANGLES = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
