# update_quad.py
import numpy as np

from ..core.config import CONTACT_BIAS, CONTACT_MANIFOLD_MIN_XZ_SPACING, CONTACT_POSITION_CORRECTION
from ..core.config import CONTACT_SLOP, CONTACT_THRESHOLD_BASE, CONTACT_THRESHOLD_MULTIPLIER, DEBUG_CONTACT
from ..core.config import DT, GRAVITY, MAX_ANGULAR_VELOCITY, MAX_CONTACT_POINTS
from ..core.config import FRICTION, MAX_VELOCITY, NORMAL_SOLVER_ITERATIONS
from .quadruped import Quadruped
from ..core.helpers import limit_vector


GROUND_NORMAL = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _select_contact_indices(vertices: np.ndarray, contact_threshold: float) -> np.ndarray:
    candidate_indices = np.flatnonzero(vertices[:, 1] <= contact_threshold)
    if candidate_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    sorted_indices = candidate_indices[np.argsort(vertices[candidate_indices, 1])]
    selected: list[int] = []

    for idx in sorted_indices:
        idx = int(idx)
        point_xz = vertices[idx, [0, 2]]
        if all(
            np.linalg.norm(point_xz - vertices[selected_idx, [0, 2]]) >= CONTACT_MANIFOLD_MIN_XZ_SPACING
            for selected_idx in selected
        ):
            selected.append(idx)
        if len(selected) >= MAX_CONTACT_POINTS:
            break

    if len(selected) < min(MAX_CONTACT_POINTS, sorted_indices.size):
        for idx in sorted_indices:
            idx = int(idx)
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= MAX_CONTACT_POINTS:
                break

    return np.array(selected, dtype=np.int64)


def _point_velocity(quadruped: Quadruped, relative_position: np.ndarray) -> np.ndarray:
    return quadruped.velocity + np.cross(quadruped.angular_velocity, relative_position)


def _solve_contact_constraints(
    quadruped: Quadruped,
    current_vertices: np.ndarray,
    mass: float,
    inertia_diag: np.ndarray,
    contact_threshold: float,
) -> np.ndarray:
    contact_indices = _select_contact_indices(current_vertices, contact_threshold)
    if contact_indices.size == 0:
        return current_vertices

    accumulated_normal_impulses = np.zeros(contact_indices.shape[0], dtype=np.float64)
    accumulated_tangent_impulses = np.zeros((contact_indices.shape[0], 3), dtype=np.float64)
    max_penetration = 0.0

    for _ in range(NORMAL_SOLVER_ITERATIONS):
        for local_idx, vertex_idx in enumerate(contact_indices):
            point = current_vertices[vertex_idx]
            penetration = max(0.0, -float(point[1]))
            max_penetration = max(max_penetration, penetration)

            relative_position = point - quadruped.position
            vertex_velocity = _point_velocity(quadruped, relative_position)
            normal_velocity = float(np.dot(vertex_velocity, GROUND_NORMAL))

            if penetration <= 0.0 and normal_velocity >= 0.0:
                continue

            bias = CONTACT_BIAS * max(0.0, penetration - CONTACT_SLOP) / DT
            r_cross_n = np.cross(relative_position, GROUND_NORMAL)
            inv_inertia_term = np.divide(r_cross_n, inertia_diag, out=np.zeros_like(r_cross_n), where=inertia_diag != 0.0)
            denom = (1.0 / mass) + float(np.dot(GROUND_NORMAL, np.cross(inv_inertia_term, relative_position)))

            if denom <= 1e-9:
                continue

            impulse_delta = -(normal_velocity + bias) / denom
            new_accumulated = max(0.0, accumulated_normal_impulses[local_idx] + impulse_delta)
            impulse_delta = new_accumulated - accumulated_normal_impulses[local_idx]
            if impulse_delta <= 0.0:
                new_accumulated = accumulated_normal_impulses[local_idx]
            else:
                accumulated_normal_impulses[local_idx] = new_accumulated
                impulse = GROUND_NORMAL * impulse_delta

                quadruped.velocity += impulse / mass
                angular_impulse = np.cross(relative_position, impulse)
                quadruped.angular_velocity += np.divide(
                    angular_impulse,
                    inertia_diag,
                    out=np.zeros_like(angular_impulse),
                    where=inertia_diag != 0.0,
                )

            if accumulated_normal_impulses[local_idx] <= 0.0:
                continue

            vertex_velocity = _point_velocity(quadruped, relative_position)
            tangent_velocity = vertex_velocity - GROUND_NORMAL * float(np.dot(vertex_velocity, GROUND_NORMAL))
            tangent_speed = float(np.linalg.norm(tangent_velocity))

            if tangent_speed <= 1e-6:
                continue

            tangent_direction = tangent_velocity / tangent_speed
            r_cross_t = np.cross(relative_position, tangent_direction)
            inv_inertia_term_t = np.divide(r_cross_t, inertia_diag, out=np.zeros_like(r_cross_t), where=inertia_diag != 0.0)
            tangent_denom = (1.0 / mass) + float(np.dot(tangent_direction, np.cross(inv_inertia_term_t, relative_position)))

            if tangent_denom <= 1e-9:
                continue

            tangent_impulse_delta_mag = -tangent_speed / tangent_denom
            previous_tangent_impulse = accumulated_tangent_impulses[local_idx]
            candidate_tangent_impulse = previous_tangent_impulse + tangent_direction * tangent_impulse_delta_mag
            friction_limit = FRICTION * accumulated_normal_impulses[local_idx]
            candidate_norm = float(np.linalg.norm(candidate_tangent_impulse))

            if candidate_norm > friction_limit and candidate_norm > 1e-9:
                candidate_tangent_impulse *= friction_limit / candidate_norm

            tangent_impulse_delta = candidate_tangent_impulse - previous_tangent_impulse
            if np.linalg.norm(tangent_impulse_delta) <= 1e-9:
                continue

            accumulated_tangent_impulses[local_idx] = candidate_tangent_impulse
            quadruped.velocity += tangent_impulse_delta / mass
            tangent_angular_impulse = np.cross(relative_position, tangent_impulse_delta)
            quadruped.angular_velocity += np.divide(
                tangent_angular_impulse,
                inertia_diag,
                out=np.zeros_like(tangent_angular_impulse),
                where=inertia_diag != 0.0,
            )

    if max_penetration > CONTACT_SLOP:
        position_correction = CONTACT_POSITION_CORRECTION * (max_penetration - CONTACT_SLOP)
        quadruped.position[1] += position_correction
        quadruped._needs_update = True
        current_vertices = quadruped.get_vertices()

    if DEBUG_CONTACT:
        print(
            f"[CONTACT N] points={contact_indices.size} max_pen={max_penetration:.5f} "
            f"vel={quadruped.velocity} ang={quadruped.angular_velocity}"
        )

    return current_vertices


def update_quadruped(quadruped: Quadruped):
    """
    Met a jour le quadruped avec un solveur de contact normal plus coherent.

    Cette passe se concentre sur :
    - contacts de repos via une marge de contact
    - solveur normal sequentiel par point de contact
    - correction de penetration moins grossiere

    La friction tangentielle reste pour l'instant la meme heuristique.
    """

    quadruped._needs_update = True
    current_vertices = quadruped.get_vertices()

    mass = quadruped.mass
    inertia_diag = quadruped.I_body.copy()

    quadruped.velocity += GRAVITY * DT
    quadruped.position += quadruped.velocity * DT
    quadruped.rotation = (quadruped.rotation + quadruped.angular_velocity * DT) % (2 * np.pi)

    quadruped._needs_update = True
    current_vertices = quadruped.get_vertices()

    contact_threshold = max(
        CONTACT_THRESHOLD_BASE,
        abs(float(quadruped.velocity[1])) * DT * CONTACT_THRESHOLD_MULTIPLIER,
    )

    current_vertices = _solve_contact_constraints(
        quadruped=quadruped,
        current_vertices=current_vertices,
        mass=mass,
        inertia_diag=inertia_diag,
        contact_threshold=contact_threshold,
    )

    quadruped.velocity = limit_vector(quadruped.velocity, MAX_VELOCITY)
    quadruped.angular_velocity = limit_vector(quadruped.angular_velocity, MAX_ANGULAR_VELOCITY)

    quadruped._needs_update = True
    quadruped.prev_vertices = quadruped.get_vertices().copy()

    if DEBUG_CONTACT:
        print(f"[VELOCITY] {quadruped.velocity}, [ANGULAR] {quadruped.angular_velocity}")
        print(f"[POSITION] {quadruped.position}, [ROTATION] {quadruped.rotation}")
        print("------------------------------------------------------------------------------------------------\n")
