# quadruped_points.py
import numpy as np

from ..core.config import units_to_meters
from .quadruped_calibration import (
    BODY_CENTER_OFFSET,
    BODY_SIZE,
    ELBOW_OFFSETS,
    LOWER_LEG_CENTER_OFFSETS,
    LOWER_LEG_SIZE,
    SHOULDER_POSITIONS,
    UPPER_LEG_CENTER_OFFSETS,
    UPPER_LEG_SIZE,
)


BODY_LENGTH = float(BODY_SIZE[0])
BODY_WIDTH = float(BODY_SIZE[2])
BODY_HEIGHT = float(BODY_SIZE[1])
UPPER_LEG_LENGTH = float(UPPER_LEG_SIZE[0])
UPPER_LEG_WIDTH = float(UPPER_LEG_SIZE[2])
UPPER_LEG_HEIGHT = float(UPPER_LEG_SIZE[1])
LOWER_LEG_LENGTH = float(LOWER_LEG_SIZE[0])
LOWER_LEG_WIDTH = float(LOWER_LEG_SIZE[2])
LOWER_LEG_HEIGHT = float(LOWER_LEG_SIZE[1])
Y_OFFSET = float(BODY_CENTER_OFFSET[1] - 0.5 * BODY_SIZE[1])

BODY_LENGTH_M = units_to_meters(BODY_LENGTH)
BODY_WIDTH_M = units_to_meters(BODY_WIDTH)
BODY_HEIGHT_M = units_to_meters(BODY_HEIGHT)
UPPER_LEG_LENGTH_M = units_to_meters(UPPER_LEG_LENGTH)
UPPER_LEG_WIDTH_M = units_to_meters(UPPER_LEG_WIDTH)
UPPER_LEG_HEIGHT_M = units_to_meters(UPPER_LEG_HEIGHT)
LOWER_LEG_LENGTH_M = units_to_meters(LOWER_LEG_LENGTH)
LOWER_LEG_WIDTH_M = units_to_meters(LOWER_LEG_WIDTH)
LOWER_LEG_HEIGHT_M = units_to_meters(LOWER_LEG_HEIGHT)
Y_OFFSET_M = units_to_meters(Y_OFFSET)


def create_box_vertices(center, size):
    """Create 8 axis-aligned vertices for a cuboid centered at `center`."""
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    half_x, half_y, half_z = size / 2.0
    cx, cy, cz = center

    return [
        np.array([cx - half_x, cy - half_y, cz - half_z], dtype=np.float64),
        np.array([cx - half_x, cy - half_y, cz + half_z], dtype=np.float64),
        np.array([cx + half_x, cy - half_y, cz - half_z], dtype=np.float64),
        np.array([cx + half_x, cy - half_y, cz + half_z], dtype=np.float64),
        np.array([cx - half_x, cy + half_y, cz - half_z], dtype=np.float64),
        np.array([cx - half_x, cy + half_y, cz + half_z], dtype=np.float64),
        np.array([cx + half_x, cy + half_y, cz - half_z], dtype=np.float64),
        np.array([cx + half_x, cy + half_y, cz + half_z], dtype=np.float64),
    ]


def create_body(_length=None, _width=None, _height=None, _y_offset=None):
    return create_box_vertices(BODY_CENTER_OFFSET, BODY_SIZE)


def create_upper_legs(_body_vertices=None, _leg_length=None, _leg_width=None, _leg_height=None):
    upper_legs = []
    for shoulder_position, upper_center_offset in zip(SHOULDER_POSITIONS, UPPER_LEG_CENTER_OFFSETS):
        upper_center = shoulder_position + upper_center_offset
        upper_legs.append(create_box_vertices(upper_center, UPPER_LEG_SIZE))
    return upper_legs


def calculate_shoulder_positions(_upper_legs=None):
    return [position.copy() for position in SHOULDER_POSITIONS]


def calculate_elbow_positions(_upper_legs=None):
    return [
        (shoulder_position + elbow_offset).copy()
        for shoulder_position, elbow_offset in zip(SHOULDER_POSITIONS, ELBOW_OFFSETS)
    ]


def create_lower_legs(_upper_legs=None, _leg_length=None, _leg_width=None, _leg_height=None):
    lower_legs = []
    elbow_positions = calculate_elbow_positions()
    for elbow_position, lower_center_offset in zip(elbow_positions, LOWER_LEG_CENTER_OFFSETS):
        lower_center = elbow_position + lower_center_offset
        lower_legs.append(create_box_vertices(lower_center, LOWER_LEG_SIZE))
    return lower_legs


def create_quadruped_vertices(
    body_length=BODY_LENGTH,
    body_width=BODY_WIDTH,
    body_height=BODY_HEIGHT,
    upper_leg_length=UPPER_LEG_LENGTH,
    upper_leg_width=UPPER_LEG_WIDTH,
    upper_leg_height=UPPER_LEG_HEIGHT,
    lower_leg_length=LOWER_LEG_LENGTH,
    lower_leg_width=LOWER_LEG_WIDTH,
    lower_leg_height=LOWER_LEG_HEIGHT,
    y_offset=Y_OFFSET,
):
    del body_length, body_width, body_height
    del upper_leg_length, upper_leg_width, upper_leg_height
    del lower_leg_length, lower_leg_width, lower_leg_height
    del y_offset

    body_vertices = create_body()
    upper_legs = create_upper_legs()
    shoulder_positions = calculate_shoulder_positions()
    elbow_positions = calculate_elbow_positions()
    lower_legs = create_lower_legs()

    return {
        "body": body_vertices,
        "upper_legs": upper_legs,
        "lower_legs": lower_legs,
        "shoulder_positions": shoulder_positions,
        "elbow_positions": elbow_positions,
        "all_parts": [body_vertices] + upper_legs + lower_legs,
    }


def get_quadruped_vertices():
    vertices_dict = create_quadruped_vertices()
    all_vertices = []

    for part in vertices_dict["all_parts"]:
        all_vertices.extend(part)

    return all_vertices

