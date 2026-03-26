# cube.py
import numpy as np
import pygame
from ..core.config import *
from ..rendering.camera import Camera3D
import math

class Cube3D:
    def __init__(self, position, x_length = 1, y_length = 1, z_length = 1, rotation = np.array([0.0, 0.0, 0.0]), velocity = np.array([0.0, 0.0, 0.0]), color = (255, 255, 255)):
        self.initial_position = position.copy()
        self.initial_velocity = velocity.copy()
        self.initial_rotation = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_angular_velocity = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_rotation = rotation.copy()

        self.position = position # position en x, y, z
        self.velocity = velocity
        self.rotation = rotation # rotation en radians
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.x_length = float(x_length)
        self.y_length = float(y_length)
        self.z_length = float(z_length)
        self.color = color

        self.rotated_vertices = self.get_vertices()
    
    def reset(self):
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy() * np.random.rand(3)
        self.rotation = self.initial_rotation.copy() * np.random.rand(3)
        self.angular_velocity = self.initial_angular_velocity.copy() * np.random.rand(3)
    
    def get_face_center(self, face_index: int) -> np.array:
        """
        Retourne le centre de la face donnée par son index parmi les 6 faces
        Prend en compte la rotation du cube

        L'ordre des faces est :
        0: face supérieure (+y)
        1: face droite (+x)
        2: face inférieure (-y)
        3: face gauche (-x)
        4: face avant (+z)
        5: face arrière (-z)
        """
        # Définir le centre de la face dans le repère local du cube
        if face_index == 0:
            local_center = np.array([0, self.y_length / 2, 0])
        elif face_index == 1:
            local_center = np.array([self.x_length / 2, 0, 0])
        elif face_index == 2:
            local_center = np.array([0, -self.y_length / 2, 0])
        elif face_index == 3:
            local_center = np.array([-self.x_length / 2, 0, 0])
        elif face_index == 4:
            local_center = np.array([0, 0, self.z_length / 2])
        elif face_index == 5:
            local_center = np.array([0, 0, -self.z_length / 2])
        else:
            raise ValueError(f"Face index must be between 0 and 5, got {face_index}")
        
        # Appliquer les rotations
        # Rotation autour de l'axe X (pitch)
        cos_x = math.cos(self.rotation[0])
        sin_x = math.sin(self.rotation[0])
        y1 = local_center[1] * cos_x - local_center[2] * sin_x
        z1 = local_center[1] * sin_x + local_center[2] * cos_x
        rotated_center = np.array([local_center[0], y1, z1])
        
        # Rotation autour de l'axe Y (yaw)
        cos_y = math.cos(self.rotation[1])
        sin_y = math.sin(self.rotation[1])
        x2 = rotated_center[0] * cos_y + rotated_center[2] * sin_y
        z2 = -rotated_center[0] * sin_y + rotated_center[2] * cos_y
        rotated_center = np.array([x2, rotated_center[1], z2])
        
        # Rotation autour de l'axe Z (roll)
        cos_z = math.cos(self.rotation[2])
        sin_z = math.sin(self.rotation[2])
        x3 = rotated_center[0] * cos_z - rotated_center[1] * sin_z
        y3 = rotated_center[0] * sin_z + rotated_center[1] * cos_z
        rotated_center = np.array([x3, y3, rotated_center[2]])
        
        # Ajouter la position du cube
        return self.position + rotated_center

    def get_corner_position(self, corner_index: int) -> np.array:
        """
        Retourne la position du coin donné par son index (0 à 7) dans le repère monde.
        L'ordre des coins est :
        0: (-x, -y, -z)
        1: (+x, -y, -z)
        2: (-x, +y, -z)
        3: (+x, +y, -z)
        4: (-x, -y, +z)
        5: (+x, -y, +z)
        6: (-x, +y, +z)
        7: (+x, +y, +z)
        """
        return self.get_vertices()[corner_index]
         

    def get_large_bounding_box(self, camera: Camera3D):
        """
        Retourne le grand pavé droit englobant le cube a partir des sommets du cube

        Ce pavé doit aura une rotation nulle dans le plan orthonormé x, y, z
        et une position qui correspond au centre du cube
        """
        # Calculer la bounding box 3D
        x_min = min(vertex[0] for vertex in self.rotated_vertices)
        y_min = min(vertex[1] for vertex in self.rotated_vertices)
        z_min = min(vertex[2] for vertex in self.rotated_vertices)
        x_max = max(vertex[0] for vertex in self.rotated_vertices)
        y_max = max(vertex[1] for vertex in self.rotated_vertices)
        z_max = max(vertex[2] for vertex in self.rotated_vertices)
        
        # On détermine les 8 sommets du pavé droit
        vertices = []
        for x in [x_min, x_max]:
            for y in [y_min, y_max]:
                for z in [z_min, z_max]:
                    vertices.append(np.array([x, y, z]))
        
        # On les projette sur le plan x, z
        projected_vertices = []
        for vertex in vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:
                projected_vertices.append(projected)
        return vertices, projected_vertices
        
    
    def get_vertices(self):
        """Retourne les 8 sommets du cube dans le repère monde"""
        # Calculer les 8 sommets du cube
        vertices = []
        for x in [-self.x_length/2, self.x_length/2]:
            for y in [-self.y_length/2, self.y_length/2]:
                for z in [-self.z_length/2, self.z_length/2]:
                    vertex = np.array([x, y, z])
                    vertices.append(vertex)
        
        # Appliquer les rotations aux sommets
        rotated_vertices = []
        for vertex in vertices:
            # Rotation autour de l'axe X (pitch)
            cos_x = math.cos(self.rotation[0])
            sin_x = math.sin(self.rotation[0])
            y1 = vertex[1] * cos_x - vertex[2] * sin_x
            z1 = vertex[1] * sin_x + vertex[2] * cos_x
            rotated_vertex = np.array([vertex[0], y1, z1])
            
            # Rotation autour de l'axe Y (yaw)
            cos_y = math.cos(self.rotation[1])
            sin_y = math.sin(self.rotation[1])
            x2 = rotated_vertex[0] * cos_y + rotated_vertex[2] * sin_y
            z2 = -rotated_vertex[0] * sin_y + rotated_vertex[2] * cos_y
            rotated_vertex = np.array([x2, rotated_vertex[1], z2])
            
            # Rotation autour de l'axe Z (roll)
            cos_z = math.cos(self.rotation[2])
            sin_z = math.sin(self.rotation[2])
            x3 = rotated_vertex[0] * cos_z - rotated_vertex[1] * sin_z
            y3 = rotated_vertex[0] * sin_z + rotated_vertex[1] * cos_z
            rotated_vertex = np.array([x3, y3, rotated_vertex[2]])
            
            # Ajouter la position du cube
            final_vertex = self.position + rotated_vertex
            rotated_vertices.append(final_vertex)
        
        return rotated_vertices

    def get_vertices_and_intermediates(self, points_per_edge: int):
        """Retourne les 8 sommets du cube et les points intermédiaires entre les sommets"""
        # Obtenir les 8 sommets de base
        base_vertices = self.get_vertices()
        
        # Définir les arêtes du cube (paires d'indices de sommets)
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Face avant
            (4, 5), (5, 7), (7, 6), (6, 4),  # Face arrière
            (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
        ]
        
        # Liste pour stocker tous les points (sommets + intermédiaires)
        all_points = base_vertices.copy()
        
        # Ajouter les points intermédiaires sur chaque arête
        for start_idx, end_idx in edges:
            start_vertex = base_vertices[start_idx]
            end_vertex = base_vertices[end_idx]
            
            # Calculer les points intermédiaires
            for i in range(1, points_per_edge):
                t = i / points_per_edge  # Paramètre d'interpolation (0 à 1)
                intermediate_point = start_vertex * (1 - t) + end_vertex * t
                all_points.append(intermediate_point)
        
        return all_points

    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le cube 3D avec projection et profondeur"""        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in self.rotated_vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:  # projected peut etre None si le point est derrière la caméra
                projected_vertices.append(projected)
        
        if len(projected_vertices) < 8:
            return  # Le cube est partiellement hors champ de vision, on ne dessine pas les faces
        
        # Dessiner les faces du cube (simplifié - juste les arêtes)
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Face avant
            (4, 5), (5, 7), (7, 6), (6, 4),  # Face arrière
            (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
        ]
        
        # Dessiner les arêtes
        for edge in edges:
            if edge[0] < len(projected_vertices) and edge[1] < len(projected_vertices):
                start = projected_vertices[edge[0]][:2]
                end = projected_vertices[edge[1]][:2]
                
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, self.color, start, end, 2)
        
    def draw_bounding_box(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le grand rectangle englobant le cube"""
        _, projected_large_vertices = self.get_large_bounding_box(camera)

        # Dessiner les faces du cube (simplifié - juste les arêtes)
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Face avant
            (4, 5), (5, 7), (7, 6), (6, 4),  # Face arrière
            (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
        ]
        
        # Dessiner les arêtes
        for edge in edges:
            if edge[0] < len(projected_large_vertices) and edge[1] < len(projected_large_vertices):
                start = projected_large_vertices[edge[0]][:2]
                end = projected_large_vertices[edge[1]][:2]
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, WHITE, start, end, 2)

        
