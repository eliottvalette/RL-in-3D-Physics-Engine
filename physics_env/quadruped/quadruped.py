# quadruped.py
import numpy as np
import pygame
from ..core.config import *
from ..rendering.camera import Camera3D
import copy
import math

# --- MASSES (kg) ---------------------------------------------------
BODY_MASS        = 3.0        # tronc
UPPER_LEG_MASS   = 0.35       # ×4
LOWER_LEG_MASS   = 0.15       # ×4

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800

class Quadruped:
    def __init__(self, position, vertices, vertices_dict, rotation = np.array([0.0, 0.0, 0.0]), velocity = np.array([0.0, 0.0, 0.0]), color = (255, 255, 255)):
        self.initial_position = position.copy()
        self.initial_velocity = velocity.copy()
        self.initial_rotation = rotation.copy()
        self.initial_angular_velocity = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_rotation = rotation.copy()
        self.initial_vertices = vertices.copy()
        self.initial_vertices_dict = vertices_dict.copy()
        self.color = color

        self.position = position # position en x, y, z du centre du quadruped
        self.velocity = velocity
        self.rotation = rotation # rotation en radians
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vertices = vertices.copy()
        self.vertices_dict = vertices_dict.copy()
        
        # Get shoulder and elbow positions from vertices_dict if available
        self.shoulder_positions = vertices_dict.get('shoulder_positions', []) if vertices_dict else []
        self.elbow_positions = vertices_dict.get('elbow_positions', []) if vertices_dict else []

        # Articulations (épaules et coudes) - angles en radians
        # Épaules: 0=Front Right, 1=Front Left, 2=Back Right, 3=Back Left
        self.shoulder_angles = np.array([0.0, 0.0, 0.0, 0.0])
        self.shoulder_velocities = np.array([0.0, 0.0, 0.0, 0.0])
        # Coudes: 0=Front Right, 1=Front Left, 2=Back Right, 3=Back Left
        self.elbow_angles = np.array([0.0, 0.0, 0.0, 0.0])
        self.elbow_velocities = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Angles initiaux
        self.initial_shoulder_angles = self.shoulder_angles.copy()
        self.initial_elbow_angles = self.elbow_angles.copy()
        self.initial_shoulder_velocities = self.shoulder_velocities.copy()
        self.initial_elbow_velocities = self.elbow_velocities.copy()
        self.prev_vertices = None
        self.active_contact_indices = np.empty(0, dtype=np.int64)
        self._needs_update = True
        self.rotated_vertices = None
        self.local_center_of_mass = self._compute_local_center_of_mass()
        
        # --- masse & inertie réalistes --------------------------
        self.mass, self.I_body = self._compute_mass_inertia()
        self.rotated_vertices = self.get_vertices()
        self.motor_delay =  4

        # Danger zones
        self.too_high = False
        self.too_low = False
        self.steps_since_too_high = 0
        self.steps_since_too_low = 0
    
    def reset_random(self):
        self.position = self.initial_position.copy()
        self.vertices = self.initial_vertices.copy()
        self.velocity = self.initial_velocity.copy() + np.random.rand(3)
        self.rotation = self.initial_rotation.copy() + np.random.rand(3)
        self.angular_velocity = self.initial_angular_velocity.copy() + np.random.rand(3)
        self.shoulder_angles = self.initial_shoulder_angles.copy()
        self.elbow_angles = self.initial_elbow_angles.copy()
        self.shoulder_velocities = self.initial_shoulder_velocities.copy()
        self.elbow_velocities = self.initial_elbow_velocities.copy()
        self.prev_vertices = None
        self.active_contact_indices = np.empty(0, dtype=np.int64)
        self._needs_update = True
        self.rotated_vertices = self.get_vertices()
    
    def reset(self):
        self.position = self.initial_position.copy()
        self.vertices = self.initial_vertices.copy()
        self.velocity = self.initial_velocity.copy()
        self.rotation = self.initial_rotation.copy()
        self.angular_velocity = self.initial_angular_velocity.copy()
        self.shoulder_angles = self.initial_shoulder_angles.copy()
        self.elbow_angles = self.initial_elbow_angles.copy()
        self.shoulder_velocities = self.initial_shoulder_velocities.copy()
        self.elbow_velocities = self.initial_elbow_velocities.copy()
        self.prev_vertices = None
        self.active_contact_indices = np.empty(0, dtype=np.int64)
        self._needs_update = True
        self.rotated_vertices = self.get_vertices()

    def get_rotation_matrix(self):
        rot_x, rot_y, rot_z = self.rotation
        cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
        cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
        cos_z, sin_z = math.cos(rot_z), math.sin(rot_z)

        rx = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        ry = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        rz = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        return rz @ ry @ rx

    def get_world_center_of_mass_offset(self):
        return self.get_rotation_matrix() @ self.local_center_of_mass

    def get_world_center_of_mass(self):
        return self.position + self.get_world_center_of_mass_offset()

    def get_center_of_mass_velocity(self):
        return self.velocity + np.cross(self.angular_velocity, self.get_world_center_of_mass_offset())

    def get_world_inverse_inertia(self):
        rotation_matrix = self.get_rotation_matrix()
        body_inverse_inertia = np.diag(np.divide(1.0, self.I_body, out=np.zeros_like(self.I_body), where=self.I_body != 0.0))
        return rotation_matrix @ body_inverse_inertia @ rotation_matrix.T

    def get_world_inertia(self):
        rotation_matrix = self.get_rotation_matrix()
        body_inertia = np.diag(self.I_body)
        return rotation_matrix @ body_inertia @ rotation_matrix.T
    
    def get_vertices(self):
        """Retourne les vertices. Recalcule uniquement si nécessaire."""
        if not self._needs_update and self.rotated_vertices is not None:
            return self.rotated_vertices

        # ---------- pré‑calculs optimisés --------------
        # Précalcul des sinus/cosinus pour les articulations
        cos_sh = np.cos(self.shoulder_angles)
        sin_sh = np.sin(self.shoulder_angles)
        cos_el = np.cos(self.elbow_angles)
        sin_el = np.sin(self.elbow_angles)

        R_global = self.get_rotation_matrix()

        # Précalcul des matrices de rotation des articulations pour chaque jambe
        shoulder_rotations = []
        elbow_rotations = []
        
        for leg_idx in range(4):
            # Matrice de rotation d'épaule (rotation autour de l'axe X)
            R_shoulder = np.array([
                [1, 0, 0],
                [0, cos_sh[leg_idx], -sin_sh[leg_idx]],
                [0, sin_sh[leg_idx], cos_sh[leg_idx]]
            ])
            shoulder_rotations.append(R_shoulder)
            
            # Matrice de rotation de coude (rotation autour de l'axe X)
            R_elbow = np.array([
                [1, 0, 0],
                [0, cos_el[leg_idx], -sin_el[leg_idx]],
                [0, sin_el[leg_idx], cos_el[leg_idx]]
            ])
            elbow_rotations.append(R_elbow)

        # Allocation mémoire optimisée avec numpy array
        result = np.empty((len(self.vertices), 3))
        
        for i, vertex in enumerate(self.vertices):
            part_index = i // 8
            v = vertex.copy()

            if part_index == 0:
                # Body - pas de transformation locale
                pass

            elif 1 <= part_index <= 4:
                # Upper legs (épaules)
                leg_index = part_index - 1
                shoulder_center = self.shoulder_positions[leg_index]
                relative_pos = v - shoulder_center
                
                # Application de la rotation d'épaule
                v = shoulder_center + shoulder_rotations[leg_index] @ relative_pos

            elif 5 <= part_index <= 8:
                # Lower legs (coudes)
                leg_index = part_index - 5
                shoulder_center = self.shoulder_positions[leg_index]
                relative_pos = v - shoulder_center
                
                # Application de la rotation d'épaule
                v = shoulder_center + shoulder_rotations[leg_index] @ relative_pos

                # Calcul de la position du coude transformée
                elbow_center_original = self.elbow_positions[leg_index]
                elbow_relative_pos = elbow_center_original - shoulder_center
                elbow_center_transformed = shoulder_center + shoulder_rotations[leg_index] @ elbow_relative_pos

                # Application de la rotation de coude
                relative_pos = v - elbow_center_transformed
                v = elbow_center_transformed + elbow_rotations[leg_index] @ relative_pos

            # Rotation globale et translation en une seule opération
            v = R_global @ v + self.position
            result[i] = v

        self.rotated_vertices = result
        self._needs_update = False
        
        return result
    
    def get_vertices_dict(self):
        return self.vertices_dict

    def set_shoulder_angle(self, leg_index, angle):
        capped_angle = max(-math.pi/2, min(math.pi/2, angle))
        self.shoulder_angles[leg_index] = capped_angle
        self._needs_update = True
    
    def set_elbow_angle(self, leg_index, angle):
        capped_angle = max(-math.pi/2, min(math.pi/2, angle))
        self.elbow_angles[leg_index] = capped_angle
        self._needs_update = True
    
    def adjust_shoulder_angle(self, leg_index, delta_angle):
        random_noise = np.random.randn() * 0.01 - 0.005
        if delta_angle > 0 :
            self.shoulder_velocities[leg_index] = np.clip(self.shoulder_velocities[leg_index] + (delta_angle / self.motor_delay) + random_noise, 0, delta_angle)
        elif delta_angle < 0:
            self.shoulder_velocities[leg_index] = np.clip(self.shoulder_velocities[leg_index] + (delta_angle / self.motor_delay) + random_noise, delta_angle, 0)
        else:
            self.shoulder_velocities[leg_index] = 0

        new_angle = self.shoulder_angles[leg_index] + self.shoulder_velocities[leg_index]
        capped_angle = max(-math.pi/2, min(math.pi/2, new_angle))
        self.shoulder_angles[leg_index] = capped_angle

        if capped_angle in [math.pi/2, -math.pi/2]:
            self.shoulder_velocities[leg_index] = 0
        self._needs_update = True
    
    def adjust_elbow_angle(self, leg_index, delta_angle):
        if delta_angle > 0:
            self.elbow_velocities[leg_index] = np.clip(self.elbow_velocities[leg_index] + (delta_angle / self.motor_delay), 0, delta_angle)
        elif delta_angle < 0:   
            self.elbow_velocities[leg_index] = np.clip(self.elbow_velocities[leg_index] + (delta_angle / self.motor_delay), delta_angle, 0)
        else:
            self.elbow_velocities[leg_index] = 0
        new_angle = self.elbow_angles[leg_index] + self.elbow_velocities[leg_index]
        capped_angle = max(-math.pi/2, min(math.pi/2, new_angle))
        self.elbow_angles[leg_index] = capped_angle
        self._needs_update = True
    
    def get_state(self):
        """
        Retourne l'état étendu du quadruped.
        
        """
        # infos de base
        base = np.concatenate([
            self.position,
            self.velocity,
            self.rotation,
            self.shoulder_angles,
            self.shoulder_velocities,
            self.elbow_angles,
            self.elbow_velocities
        ])

        # 1. Utiliser les vertices déjà calculés
        vertices = self.rotated_vertices 
        
        # 2. Les min/max X Y Z du Body
        body_vertices = vertices[0:8]
        body_xs = [v[0] for v in body_vertices]
        body_ys = [v[1] for v in body_vertices]
        body_zs = [v[2] for v in body_vertices]
        body_min_x, body_max_x = min(body_xs), max(body_xs)
        body_min_y, body_max_y = min(body_ys), max(body_ys)
        body_min_z, body_max_z = min(body_zs), max(body_zs)

        body_limits = np.array([body_min_x, body_max_x, body_min_y, body_max_y, body_min_z, body_max_z])

        # 3. min/max Y pour chaque patte (FR, FL, BR, BL)
        min_max_y = []
        for leg_idx in range(4):
            upper_start = (1 + leg_idx) * 8        # bloc upper‑leg
            lower_start = (5 + leg_idx) * 8        # bloc lower‑leg

            # on met les 16 sommets dans une seule liste
            leg_vertices = (
                vertices[upper_start : upper_start + 8] +
                vertices[lower_start : lower_start + 8]
            )
            ys = [v[1] for v in leg_vertices]
            min_max_y.extend([min(ys), max(ys)])
        
        # 4. est-ce que les angles sont capés ? pour chaque angle, un vecteur de taille 2, [a, b], a = 1 si l'angle est capé à pi/2, 0 sinon, b = 1 si l'angle est capé à -pi/2, 0 sinon
        cap_shoulder = []
        cap_elbow = []
        for angle in self.shoulder_angles:
            cap_shoulder.append([1 if angle >= math.pi/2 * 0.9 else 0, 1 if angle <= -math.pi/2 * 0.9 else 0])
        for angle in self.elbow_angles:
            cap_elbow.append([1 if angle >= math.pi/2 * 0.9 else 0, 1 if angle <= -math.pi/2 * 0.9 else 0])
        cap_shoulder = np.array(cap_shoulder).flatten()
        cap_elbow = np.array(cap_elbow).flatten()

        # Est-ce que le quadruped est dans les danger zones ?
        too_high = np.array([self.too_high], dtype=np.float32)
        too_low = np.array([self.too_low], dtype=np.float32)
        steps_since_too_high = np.array([self.steps_since_too_high / 50], dtype=np.float32)
        steps_since_too_low = np.array([self.steps_since_too_low / 20], dtype=np.float32)

        # 5. état final
        state = np.concatenate([base, body_limits, np.array(min_max_y, dtype=np.float32), cap_shoulder, cap_elbow, too_high, too_low, steps_since_too_high, steps_since_too_low])
        return state.tolist()

    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le quadruped 3D avec projection et profondeur (arêtes seulement)"""        
        # Utiliser les vertices déjà calculés
        vertices = self.rotated_vertices
        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:  # projected peut etre None si le point est derrière la caméra
                projected_vertices.append(projected)
        
        if len(projected_vertices) < 8:
            return  # Le quadruped est partiellement hors champ de vision, on ne dessine pas
        
        # Définir les couleurs pour chaque partie
        colors = [
            (255, 255, 255),  # Body (white)
            (0, 0, 255),      # Upper leg 0 - Front right (blue)
            (255, 0, 0),      # Upper leg 1 - Front left (red)
            (0, 255, 0),      # Upper leg 2 - Back right (green)
            (255, 255, 255),  # Upper leg 3 - Back left (white)
            (0, 0, 255),      # Lower leg 0 - Front right (blue)
            (255, 0, 0),      # Lower leg 1 - Front left (red)
            (0, 255, 0),      # Lower leg 2 - Back right (green)
            (255, 255, 255)   # Lower leg 3 - Back left (white)
        ]
        
        # Dessiner chaque partie du quadruped (body + 8 legs)
        # Chaque partie a 8 vertices, donc on dessine par groupes de 8
        parts_per_leg = 8
        total_parts = len(projected_vertices) // parts_per_leg
        
        for part_idx in range(total_parts):
            start_idx = part_idx * parts_per_leg
            end_idx = start_idx + parts_per_leg
            
            if end_idx <= len(projected_vertices):
                part_vertices = projected_vertices[start_idx:end_idx]
                part_color = colors[part_idx] if part_idx < len(colors) else self.color
                
                # Définir les arêtes pour chaque partie (cube)
                edges = [
                    (0, 1), (1, 3), (3, 2), (2, 0),  # Face avant
                    (4, 5), (5, 7), (7, 6), (6, 4),  # Face arrière
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
                ]
                
                # Dessiner les arêtes de cette partie
                for edge in edges:
                    if edge[0] < len(part_vertices) and edge[1] < len(part_vertices):
                        start = part_vertices[edge[0]][:2]
                        end = part_vertices[edge[1]][:2]
                        
                        # Vérifier que les coordonnées sont valides
                        if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                            0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                            pygame.draw.line(screen, part_color, start, end, 2)
    
    def draw_premium(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le quadruped 3D avec faces pleines et dégradé de gris basé sur la profondeur"""
        # Utiliser les vertices déjà calculés
        vertices = self.rotated_vertices
        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:  # projected peut etre None si le point est derrière la caméra
                projected_vertices.append(projected)
        
        if len(projected_vertices) < 8:
            return  # Le quadruped est partiellement hors champ de vision, on ne dessine pas
        
        # Définir les faces pour chaque cube (6 faces par cube)
        # Chaque face est définie par 4 indices de vertices
        cube_faces = [
            [0, 1, 3, 2],  # Face avant (bottom)
            [4, 5, 7, 6],  # Face arrière (top)
            [0, 1, 5, 4],  # Face gauche
            [2, 3, 7, 6],  # Face droite
            [0, 2, 6, 4],  # Face inférieure
            [1, 3, 7, 5]   # Face supérieure
        ]
        
        # Dessiner chaque partie du quadruped (body + 8 legs)
        # Chaque partie a 8 vertices, donc on dessine par groupes de 8
        parts_per_leg = 8
        total_parts = len(projected_vertices) // parts_per_leg
        
        # Trier les faces par profondeur pour le rendu correct (painter's algorithm)
        all_faces = []
        
        for part_idx in range(total_parts):
            start_idx = part_idx * parts_per_leg
            end_idx = start_idx + parts_per_leg
            
            if end_idx <= len(projected_vertices):
                part_vertices = projected_vertices[start_idx:end_idx]
                
                # Calculer le centre de profondeur de cette partie pour le tri
                part_depth = sum(v[2] for v in part_vertices) / len(part_vertices)
                
                # Créer les faces pour cette partie
                for face_indices in cube_faces:
                    face_vertices = [part_vertices[i] for i in face_indices if i < len(part_vertices)]
                    
                    if len(face_vertices) == 4:
                        # Calculer la profondeur moyenne de la face
                        face_depth = sum(v[2] for v in face_vertices) / len(face_vertices)
                        
                        # Calculer la normale de la face pour déterminer si elle est visible
                        # Simplification : on dessine toutes les faces pour l'instant
                        
                        all_faces.append({
                            'vertices': face_vertices,
                            'depth': face_depth,
                            'part_idx': part_idx
                        })
        
        # Trier les faces par profondeur (les plus éloignées en premier)
        all_faces.sort(key=lambda face: face['depth'], reverse=True)
        
        # Dessiner les faces triées
        for face in all_faces:
            face_vertices = face['vertices']
            depth = face['depth']
            part_idx = face['part_idx']
            
            # Calculer la couleur basée sur la profondeur (dégradé de gris)
            # Plus la profondeur est grande, plus la couleur est sombre
            base_intensity = 200  # Gris clair de base
            depth_factor = max(0, min(1, depth / 50))  # Normaliser la profondeur
            intensity = int(base_intensity * (1 - depth_factor * 0.7))  # Réduire jusqu'à 30% de l'intensité
            
            color = (intensity, intensity, intensity)
            
            # Convertir les vertices en points 2D pour pygame
            points_2d = [(int(v[0]), int(v[1])) for v in face_vertices]
            
            # Vérifier que tous les points sont dans les limites de l'écran
            valid_points = True
            for x, y in points_2d:
                if not (0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT):
                    valid_points = False
                    break
            
            if valid_points and len(points_2d) >= 3:
                # Dessiner la face pleine
                pygame.draw.polygon(screen, color, points_2d)
                
                # Optionnel : dessiner les contours des faces pour plus de définition
                pygame.draw.polygon(screen, (50, 50, 50), points_2d, 1)

    # ------------------------------------------------------------------
    #  Inertie ≈ somme des solides élémentaires : 1 parallélépipède + 8 barreaux
    # ------------------------------------------------------------------
    def _compute_mass_inertia(self):
        """
        Retourne (masse_totale, I_body) :
        I_body = np.array([Ixx, Iyy, Izz]) dans le repère du corps (non tourné)
        """
        # 0‑‑7 = tronc
        body_vs = self.initial_vertices[0:8]
        xs = [v[0] for v in body_vs]
        ys = [v[1] for v in body_vs]
        zs = [v[2] for v in body_vs]
        a, b, c = (max(xs) - min(xs),
                   max(ys) - min(ys),
                   max(zs) - min(zs))               # côtés du pavé

        I_body = np.zeros(3)

        # --- 1) Tronc plein (parallélépipède) -------------------
        I_body += np.array([
            BODY_MASS * (b**2 + c**2) / 12,
            BODY_MASS * (a**2 + c**2) / 12,
            BODY_MASS * (a**2 + b**2) / 12
        ])

        # --- 2) Quatre upper‑legs (barreaux verticals) ----------
        # On approxime chaque barre comme un cylindre / bâton fin (axe y)
        leg_len_u = 0.7 * b                  # ≈ 70 % hauteur tronc
        I_bar_u   = UPPER_LEG_MASS * (leg_len_u**2) / 12
        I_body += 4 * np.array([I_bar_u, 0, I_bar_u])   # autour de l'axe y

        # --- 3) Quatre lower‑legs (barreaux verticaux) ----------
        leg_len_l = 0.9 * b
        I_bar_l   = LOWER_LEG_MASS * (leg_len_l**2) / 12
        I_body += 4 * np.array([I_bar_l, 0, I_bar_l])

        # totale :
        mass_tot = BODY_MASS + 4 * (UPPER_LEG_MASS + LOWER_LEG_MASS)

        return mass_tot, I_body

    def _compute_local_center_of_mass(self):
        masses = [BODY_MASS] + [UPPER_LEG_MASS] * 4 + [LOWER_LEG_MASS] * 4
        part_centers = []
        for part_idx in range(len(self.vertices) // 8):
            part_vertices = np.array(self.vertices[part_idx * 8:(part_idx + 1) * 8], dtype=np.float64)
            part_centers.append(part_vertices.mean(axis=0))
        weighted_sum = sum(mass * center for mass, center in zip(masses, part_centers))
        return weighted_sum / sum(masses)
