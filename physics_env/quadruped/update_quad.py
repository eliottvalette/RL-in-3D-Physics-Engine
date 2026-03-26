# update_functions.py
import numpy as np


from ..core.config import DT, GRAVITY, SLIP_THRESHOLD, CONTACT_THRESHOLD_BASE, CONTACT_THRESHOLD_MULTIPLIER
from ..core.config import MAX_VELOCITY, MAX_ANGULAR_VELOCITY, MAX_IMPULSE, MAX_AVERAGE_IMPULSE, DEBUG_CONTACT, STATIC_FRICTION_CAP
from .quadruped import Quadruped
from ..core.helpers import limit_vector, batch_cross


def update_quadruped(quadruped: Quadruped):
    """
    Collision avancée entre le quadruped et le sol avec gestion améliorée des rebonds.
    Corrections physiques appliquées :
    - Correction de pénétration conservatrice
    - Critères de contact dynamiques
    - Amortissement réaliste avec restitution et friction
    - Limitation de vitesse douce
    - Rotation stable
    """

    # Mettre à jour les sommets du cube
    quadruped._needs_update = True
    current_vertices = quadruped.get_vertices()
    prev_vertices = quadruped.prev_vertices if quadruped.prev_vertices is not None else current_vertices

    
    # --- paramètres corps -----------------------------
    mass = quadruped.mass          # ≈ 4.4 kg
    I    = quadruped.I_body.copy() # (3,)

    # Appliquer la gravité au centre de masse
    quadruped.velocity += GRAVITY * DT

    # Mise à jour de la position et de la rotation
    quadruped.position += quadruped.velocity * DT
    quadruped.rotation = (quadruped.rotation + quadruped.angular_velocity * DT) % (2 * np.pi)
    
    # Recalculer les sommets après mise à jour
    current_vertices = quadruped.get_vertices()
    
    # Critères de contact dynamiques
    contact_threshold = max(CONTACT_THRESHOLD_BASE, abs(quadruped.velocity[1]) * DT * CONTACT_THRESHOLD_MULTIPLIER)

    # Limitation de vitesse douce
    quadruped.velocity = limit_vector(quadruped.velocity, MAX_VELOCITY)
    quadruped.angular_velocity = limit_vector(quadruped.angular_velocity, MAX_ANGULAR_VELOCITY)

    # Calculer la pénétration maximale sur tous les sommets
    penetrations = []

    # --- on sépare maintenant vertical (normal) et tangentiel ---
    collision_impulses_normal = []
    collision_angular_impulses_normal = []
    collision_impulses_tangent = []
    collision_angular_impulses_tangent = []
    
    # Filtrer les vertices en collision avec le sol
    collision_vertices = [vertex for vertex in current_vertices if vertex[1] < 0]
    
    if collision_vertices:
        # Préparer les données pour calculs vectorisés
        collision_vertices = np.array(collision_vertices)
        relative_positions = collision_vertices - quadruped.position
        
        # Calculer les vitesses des vertices (translation + rotation)
        # Utiliser batch_cross pour optimiser
        angular_contributions = batch_cross(
            np.full((len(collision_vertices), 3), quadruped.angular_velocity),
            relative_positions
        )
        vertex_velocities = quadruped.velocity + angular_contributions
        
        # Filtrer les vertices qui descendent
        descending_mask = vertex_velocities[:, 1] < 0
        if np.any(descending_mask):
            descending_vertices = collision_vertices[descending_mask]
            descending_relative_positions = relative_positions[descending_mask]
            descending_velocities = vertex_velocities[descending_mask]
            
            # Enregistrer les pénétrations
            penetrations.extend(-descending_vertices[:, 1])
            
            # Normal du sol pour tous les vertices
            normals = np.full((len(descending_vertices), 3), [0, 1, 0])
            
            # Calculer les vitesses relatives
            relative_velocities = np.sum(descending_velocities * normals, axis=1)
            
            # Calculer r_cross_n pour tous les vertices
            r_cross_n = batch_cross(descending_relative_positions, normals)
            
            # Calculer les dénominateurs pour tous les vertices
            r_cross_n_div_I = np.divide(r_cross_n, I, out=np.zeros_like(r_cross_n), where=I!=0)
            r_cross_n_div_I_cross_r = batch_cross(r_cross_n_div_I, descending_relative_positions)
            dot_terms = np.sum(normals * r_cross_n_div_I_cross_r, axis=1)
            denoms = (1/mass) + dot_terms
            
            # Calculer les impulsions scalaires
            valid_denoms = denoms != 0
            if np.any(valid_denoms):
                scalar_impulses = -relative_velocities[valid_denoms] / denoms[valid_denoms]
                scalar_impulses = np.clip(scalar_impulses, -MAX_IMPULSE, MAX_IMPULSE)
                
                # Calculer les impulsions normales
                normal_impulses = (scalar_impulses[:, np.newaxis] * normals[valid_denoms]) / mass
                collision_impulses_normal.extend(normal_impulses)
                
                # Calculer les impulsions angulaires
                angular_impulses = np.divide(
                    batch_cross(descending_relative_positions[valid_denoms], 
                               scalar_impulses[:, np.newaxis] * normals[valid_denoms]),
                    I, out=np.zeros((len(scalar_impulses), 3)), where=I!=0
                )
                collision_angular_impulses_normal.extend(angular_impulses)
    
    # Appliquer la correction de position une seule fois
    if penetrations:
        mean_penetration = np.mean(penetrations)
        if DEBUG_CONTACT:
            print(f"[CORRECTION] Correction de position appliquée: +{mean_penetration * 0.2:.5f}")
        quadruped.position[1] += mean_penetration * 0.2

    # --- Moyenne et application des impulsions verticales ---
    if collision_impulses_normal:
        avg_imp_n = limit_vector(np.mean(collision_impulses_normal, axis=0), MAX_AVERAGE_IMPULSE)
        avg_ang_n = limit_vector(np.mean(collision_angular_impulses_normal, axis=0), MAX_AVERAGE_IMPULSE)
        if DEBUG_CONTACT:
            print(f"[IMPULSES N] Moyenne impulsion normale: {avg_imp_n}, angulaire: {avg_ang_n}")
        quadruped.velocity += avg_imp_n
        quadruped.angular_velocity += avg_ang_n

    # --- Moyenne et application des impulsions tangentielles ---
    if collision_impulses_tangent:
        avg_imp_t = limit_vector(np.mean(collision_impulses_tangent, axis=0), MAX_AVERAGE_IMPULSE)   # mets un plafond dédié si besoin
        avg_ang_t = limit_vector(np.mean(collision_angular_impulses_tangent, axis=0), MAX_AVERAGE_IMPULSE)
        if DEBUG_CONTACT:
            print(f"[IMPULSES T] Moyenne impulsion tangentielle: {avg_imp_t}, angulaire: {avg_ang_t}")
        quadruped.velocity += avg_imp_t
        quadruped.angular_velocity += avg_ang_t

    # --- Ajout : traction latérale basée sur t‑1 ---
    traction_imp, traction_ang = [], []
    
    # Convertir en arrays pour calculs vectorisés
    prev_vertices_array = np.array(prev_vertices)
    current_vertices_array = np.array(current_vertices)
    
    # Identifier les points au sol
    previous_on_ground = prev_vertices_array[:, 1] <= contact_threshold
    current_on_ground = current_vertices_array[:, 1] <= contact_threshold
    both_on_ground = previous_on_ground & current_on_ground
    
    if np.any(both_on_ground):
        # Extraire les vertices concernés
        ground_prev = prev_vertices_array[both_on_ground].copy()
        ground_current = current_vertices_array[both_on_ground].copy()
        
        # Normaliser la hauteur (considérer que les points restent au sol)
        ground_prev[:, 1] = 0
        ground_current[:, 1] = 0
        
        # Calculer les déplacements
        deltas = ground_current - ground_prev
        deltas[:, 1] = 0.0  # composante tangentielle
        
        # Filtrer par dead zone
        delta_norms = np.linalg.norm(deltas, axis=1)
        significant_movement = delta_norms >= SLIP_THRESHOLD * DT
        
        if np.any(significant_movement):
            significant_deltas = deltas[significant_movement]
            significant_current = ground_current[significant_movement]
            
            # Calculer les impulsions nécessaires
            J_needed = -mass * significant_deltas / DT  # N·s
            J_cap = STATIC_FRICTION_CAP * DT  # adhérence max
            J_clipped = np.clip(J_needed, -J_cap, J_cap)
            
            # Impulsions linéaires
            traction_imp.extend(J_clipped / mass)
            
            # Impulsions angulaires (utiliser batch_cross)
            r_vectors = significant_current - quadruped.position
            angular_impulses = np.divide(
                batch_cross(r_vectors, J_clipped),
                I, out=np.zeros_like(r_vectors), where=I!=0
            )
            traction_ang.extend(angular_impulses)

    if traction_imp:
        if DEBUG_CONTACT:
            print(f"[TRACTION] Moyenne traction linéaire: {np.mean(traction_imp, axis=0)}, angulaire: {np.mean(traction_ang, axis=0)}")
        quadruped.velocity += limit_vector(np.mean(traction_imp, axis=0), MAX_AVERAGE_IMPULSE)
        quadruped.angular_velocity += limit_vector(np.mean(traction_ang, axis=0), MAX_AVERAGE_IMPULSE)

    # Sauvegarder les vertices actuels pour la prochaine itération
    quadruped.prev_vertices = current_vertices.copy()

    if DEBUG_CONTACT:
        print(f"[VELOCITY] Velocity: {quadruped.velocity}, Angular Velocity: {quadruped.angular_velocity}")
        print(f"[POSITION] Position: {quadruped.position}, Rotation: {quadruped.rotation}")
        print("------------------------------------------------------------------------------------------------\n")
