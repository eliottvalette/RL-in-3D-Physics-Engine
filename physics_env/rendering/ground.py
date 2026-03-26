# ground.py
import numpy as np
import pygame
from ..core.config import *
from .camera import Camera3D

class Ground:
    def __init__(self, size=20):
        self.size = size
        self._3d_world_points = []
        
    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le sol en 3D"""
        for x in range(-self.size, self.size + 1):
            for z in range(-self.size, self.size + 1):
                # Créer un point au sol
                point_3d = np.array([x, 0, z])
                self._3d_world_points.append(point_3d)
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    # Couleur basée sur la profondeur
                    depth = projected[2]
                    color_intensity = max(0, min(255, 255 - depth * 10))
                    color = (color_intensity, color_intensity, color_intensity)
                    
                    # Dessiner un petit carré
                    size = np.clip(int(10 / depth), 1, 10)
                    rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                    # Vérifier que le rectangle est dans les limites de l'écran
                    if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                        rect.right > 0 and rect.bottom > 0):
                        pygame.draw.rect(screen, color, rect) 
    
    def draw_premium(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le sol en 3D avec faces pleines et dégradé de gris basé sur la profondeur"""
        # Créer une grille de points pour former des faces
        grid_size = self.size * 2 + 1
        
        # Collecter toutes les faces pour le tri par profondeur
        all_faces = []
        
        # Parcourir la grille pour créer des faces carrées
        for x in range(-self.size, self.size):
            for z in range(-self.size, self.size):
                # Créer les 4 coins d'une face carrée
                corners_3d = [
                    np.array([x, 0, z]),           # Coin inférieur gauche
                    np.array([x + 1, 0, z]),       # Coin inférieur droit
                    np.array([x + 1, 0, z + 1]),   # Coin supérieur droit
                    np.array([x, 0, z + 1])        # Coin supérieur gauche
                ]
                
                # Projeter les 4 coins
                corners_2d = []
                valid_face = True
                
                for corner in corners_3d:
                    projected = camera.project_3d_to_2d(corner)
                    if projected:
                        corners_2d.append(projected)
                    else:
                        valid_face = False
                        break
                
                if valid_face and len(corners_2d) == 4:
                    # Calculer la profondeur moyenne de la face
                    face_depth = sum(corner[2] for corner in corners_2d) / 4
                    
                    # Vérifier que tous les points sont dans les limites de l'écran
                    points_in_screen = True
                    for corner in corners_2d:
                        if not (0 <= corner[0] < WINDOW_WIDTH and 0 <= corner[1] < WINDOW_HEIGHT):
                            points_in_screen = False
                            break
                    
                    if points_in_screen:
                        all_faces.append({
                            'corners': corners_2d,
                            'depth': face_depth
                        })
        
        # Trier les faces par profondeur (les plus éloignées en premier)
        all_faces.sort(key=lambda face: face['depth'], reverse=True)
        
        # Dessiner les faces triées
        for face in all_faces:
            corners_2d = face['corners']
            depth = face['depth']
            
            # Calculer la couleur basée sur la profondeur (dégradé de gris)
            # Plus la profondeur est grande, plus la couleur est sombre
            base_intensity = 180  # Gris clair de base
            depth_factor = max(0, min(1, depth / 100))  # Normaliser la profondeur
            intensity = int(base_intensity * (1 - depth_factor * 0.8))  # Réduire jusqu'à 20% de l'intensité
            
            color = (intensity, intensity, intensity)
            
            # Convertir les corners en points 2D pour pygame
            points_2d = [(int(corner[0]), int(corner[1])) for corner in corners_2d]
            
            # Dessiner la face pleine
            pygame.draw.polygon(screen, color, points_2d)
            
            # Optionnel : dessiner les contours des faces pour plus de définition
            # Utiliser une couleur plus sombre pour les contours
            contour_color = (max(0, intensity - 40), max(0, intensity - 40), max(0, intensity - 40))
            pygame.draw.polygon(screen, contour_color, points_2d, 1)

    def draw_axes(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine les axes 3D pour référence"""
        origin = np.array([0, 0, 0])
        axes = [
            (np.array([5, 0, 0]), RED),    # X
            (np.array([0, 5, 0]), GREEN),  # Y  
            (np.array([0, 0, 5]), BLUE)    # Z
        ]
        
        for axis_end, color in axes:
            start_proj = camera.project_3d_to_2d(origin)
            end_proj = camera.project_3d_to_2d(axis_end)
            
            if start_proj and end_proj:
                start = start_proj[:2]
                end = end_proj[:2]
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, color, start, end, 1)

class FloorAndWall:
    def __init__(self, size=10):
        self.size = size
        self._3d_world_points = []
    
    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """
        Dessine le sol puis le mur en 3D
        Le sol est un carré de 20x20, à hauteur y=0
        Le mur est un carré de 20x10, en x=10. 
        Le mur est vertical, normal à l'axe X (c'est-à-dire que sa normale pointe selon +X).
        """
        for x in range(-self.size, self.size + 1):
            for z in range(-self.size, self.size + 1):
                # Créer un point au sol
                point_3d = np.array([x, 0, z])
                self._3d_world_points.append(point_3d)
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    # Couleur basée sur la profondeur
                    depth = projected[2]
                    color_intensity = max(0, min(255, 255 - depth * 10))
                    color = (color_intensity, color_intensity, color_intensity)
                    
                    # Dessiner un petit carré
                    size = np.clip(int(10 / depth), 1, 10)
                    rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                    # Vérifier que le rectangle est dans les limites de l'écran
                    if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                        rect.right > 0 and rect.bottom > 0):
                        pygame.draw.rect(screen, color, rect) 
        
        for z in range(-self.size, self.size + 1):
            for y in range(0, self.size + 1):
                point_3d = np.array([self.size, y, z])
                self._3d_world_points.append(point_3d)
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    # Couleur basée sur la profondeur
                    depth = projected[2]
                    color_intensity = max(0, min(255, 255 - depth * 10))
                    color = (color_intensity, color_intensity, color_intensity)
                    
                    # Dessiner un petit carré
                    size = np.clip(int(10 / depth), 1, 10)
                    rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                    # Vérifier que le rectangle est dans les limites de l'écran
                    if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                        rect.right > 0 and rect.bottom > 0):
                        pygame.draw.rect(screen, color, rect) 
    
    def draw_axes(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine les axes 3D pour référence"""
        origin = np.array([0, 0, 0])
        axes = [
            (np.array([5, 0, 0]), RED),    # X
            (np.array([0, 5, 0]), GREEN),  # Y  
            (np.array([0, 0, 5]), BLUE)    # Z
        ]
        
        for axis_end, color in axes:
            start_proj = camera.project_3d_to_2d(origin)
            end_proj = camera.project_3d_to_2d(axis_end)
            
            if start_proj and end_proj:
                start = start_proj[:2]
                end = end_proj[:2]
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, color, start, end, 1)

class FloorAndRamp:
    def __init__(self, size=20, ramp_angle=45):
        self.size = size
        self.ramp_angle = ramp_angle
        self._3d_world_points = []

    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """
        Dans un espace de 20x20, on a la moitié 20x10 qui est plate, 
        et l'autre moitié 20x10 qui est une rampe.
        """
        for x in range(-self.size, 1): # partie plate
            for z in range(-self.size, self.size + 1):
                point_3d = np.array([x, 0, z])
                self._3d_world_points.append(point_3d)
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    # Couleur basée sur la profondeur
                    depth = projected[2]
                    color_intensity = max(0, min(255, 255 - depth * 10))
                    color = (color_intensity, color_intensity, color_intensity)
                    
                    # Dessiner un petit carré
                    size = np.clip(int(10 / depth), 1, 10)
                    rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                    # Vérifier que le rectangle est dans les limites de l'écran
                    if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                        rect.right > 0 and rect.bottom > 0):
                        pygame.draw.rect(screen, color, rect) 
        
        for x in range(0, self.size + 1): # partie rampe
            for z in range(-self.size, self.size + 1):
                # Calculer la hauteur de la rampe basée sur l'angle
                ramp_height = (x / self.size) * self.size * np.tan(np.radians(self.ramp_angle))
                point_3d = np.array([x, ramp_height, z])
                self._3d_world_points.append(point_3d)
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    # Couleur basée sur la profondeur
                    depth = projected[2]
                    color_intensity = max(0, min(255, 255 - depth * 10))
                    color = (color_intensity, color_intensity, color_intensity)
                    
                    # Dessiner un petit carré
                    size = np.clip(int(10 / depth), 1, 10)
                    rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                    # Vérifier que le rectangle est dans les limites de l'écran
                    if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                        rect.right > 0 and rect.bottom > 0):
                        pygame.draw.rect(screen, color, rect)
    
    def draw_axes(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine les axes 3D pour référence"""
        origin = np.array([0, 0, 0])
        axes = [
            (np.array([5, 0, 0]), RED),    # X
            (np.array([0, 5, 0]), GREEN),  # Y  
            (np.array([0, 0, 5]), BLUE)    # Z
        ]
        
        for axis_end, color in axes:
            start_proj = camera.project_3d_to_2d(origin)
            end_proj = camera.project_3d_to_2d(axis_end)
            
            if start_proj and end_proj:
                start = start_proj[:2]
                end = end_proj[:2]
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, color, start, end, 1)


class Staircase:
    def __init__(self, size=20, num_steps=10, step_width=1.0, step_height=1.0, step_depth=1.0, start_x=0, start_z=0):
        self.size = size
        self.num_steps = num_steps
        self.step_width = step_width
        self.step_height = step_height
        self.step_depth = step_depth
        self.start_x = start_x
        self.start_z = start_z
        self._3d_world_points = []
        
        # Pre-calculate step coordinates since they don't change
        self.step_coordinates_flat = self._calculate_step_coordinates_flat()
        self.step_coordinates_vertical = self._calculate_step_coordinates_vertical()

    def _calculate_step_coordinates_flat(self):
        """
        Calcule les coordonnées des marches d'escalier pour les collisions.
        Chaque marche est définie par:
        - y: hauteur de la marche
        - step_points: liste de points 4 points (x,z) définissant la marche (rectangle parallèle au sol)
        """
        steps = {}
        # Partie droite plate
        y = 0
        x_min, x_max = -self.size, self.size
        z_min, z_max = -self.size, 1 # 1 pour la contremarche de la première marche

        steps[y] = [(x_max, z_min), (x_max, z_max), (x_min, z_max), (x_min, z_min)]

        # Partie gauche avec escalier
        for step in range(1, self.num_steps +1):
            y = step * self.step_height
            z_min, z_max = step * self.step_depth, (step + 1) * self.step_depth
            x_min, x_max = -self.size, self.size

            steps[y] = [(x_max, z_min), (x_max, z_max), (x_min, z_max), (x_min, z_min)]
        
        return steps

    def _calculate_step_coordinates_vertical(self):
        """
        Calcule les coordonnées de la contremarche verticale entre les marches d'escalier pour les collisions.
        Chaque contremarche est définie par:
        - y: hauteur de la marche dont elle est la contremarche
        - step_points: liste de points 4 points (x, y, z) définissant la contremarche (rectangle vertical)
        La contremarche est sur le plan normal à z (position z fixe).
        """

        vertical_steps = {}
        # Pas de contremarche pour la partie droite plate

        # Partie gauche avec escalier
        for step in range(self.num_steps):  # -1 car pas de contremarche après la dernière marche
            step_y = step * self.step_height
            step_z = step * self.step_depth + self.step_depth  # Position z de la contremarche
            x_min, x_max = -self.size, self.size
            y_min, y_max = step_y, step_y + self.step_height

            # Les 4 coins de la contremarche verticale (rectangle vertical)
            vertical_steps[step_y] = [
                (x_max, y_min, step_z),  # Coin supérieur droit
                (x_max, y_max, step_z),  # Coin inférieur droit  
                (x_min, y_max, step_z),  # Coin inférieur gauche
                (x_min, y_min, step_z)   # Coin supérieur gauche
            ]
        
        return vertical_steps


    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """
        Dessine la rampe en 3D
        Je veux sur un carré de 40x40, la partie droite 40x20 soit juste plate, 
        et la partie gauche 40x20 un escalier.
        """
        for x in range(-self.size, self.size + 1):  # -20 à 20 en x
            for z in range(-self.size, 1):  # -20 à 0 en z
                point_3d = np.array([x, 0, z])
                self._3d_world_points.append(point_3d)
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    depth = projected[2]
                    color_intensity = max(0, min(255, 255 - depth * 10))
                    color = (color_intensity, color_intensity, color_intensity)
                    
                    size = np.clip(int(10 / depth), 1, 10)
                    rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                    if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                        rect.right > 0 and rect.bottom > 0):
                        pygame.draw.rect(screen, color, rect)
        
        # Partie gauche avec escalier
        for step in range(self.num_steps):
            step_y = step * self.step_height
            step_z_start = step * self.step_depth
            
            # Dessiner la marche horizontale
            for x in range(-self.size, self.size + 1):  # -20 à 20 en x
                for z in range(int(step_z_start), int(step_z_start + self.step_depth)): 
                    point_3d = np.array([x, step_y, z])
                    self._3d_world_points.append(point_3d)
                    projected = camera.project_3d_to_2d(point_3d)
                    
                    if projected:
                        depth = projected[2]
                        color_intensity = max(0, min(255, 255 - depth * 10))
                        color = (color_intensity, color_intensity, color_intensity)
                        
                        size = np.clip(int(10 / depth), 1, 10)
                        rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                        if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                            rect.right > 0 and rect.bottom > 0):
                            pygame.draw.rect(screen, color, rect)
            
            # Dessiner la contremarche verticale (si ce n'est pas la dernière marche)
            if step < self.num_steps - 1:
                for x in range(-self.size, self.size + 1):
                    for y in range(int(step_y), int(step_y + self.step_height)):
                        point_3d = np.array([x, y, step_z_start + self.step_depth])
                        self._3d_world_points.append(point_3d)
                        projected = camera.project_3d_to_2d(point_3d)
                        
                        if projected:
                            depth = projected[2]
                            color_intensity = max(0, min(255, 255 - depth * 10))
                            color = (color_intensity, color_intensity, color_intensity)
                            
                            size = np.clip(int(10 / depth), 1, 10)
                            rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                            if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                                rect.right > 0 and rect.bottom > 0):
                                pygame.draw.rect(screen, color, rect)
    
    def draw_axes(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine les axes 3D pour référence"""
        origin = np.array([0, 0, 0])
        axes = [
            (np.array([5, 0, 0]), RED),    # X
            (np.array([0, 5, 0]), GREEN),  # Y  
            (np.array([0, 0, 5]), BLUE)    # Z
        ]
        
        for axis_end, color in axes:
            start_proj = camera.project_3d_to_2d(origin)
            end_proj = camera.project_3d_to_2d(axis_end)
            
            if start_proj and end_proj:
                start = start_proj[:2]
                end = end_proj[:2]
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, color, start, end, 1)

    def draw_step_coordinates_flat(self, screen: pygame.Surface, camera: Camera3D):
        """
        Dessine les coordonnées des marches d'escalier pour vérification
        """
        
        for i, (y, step_points) in enumerate(self.step_coordinates_flat.items()):
            # Dessiner les 4 coins de chaque marche
            for j, (x, z) in enumerate(step_points):
                point_3d = np.array([x, y, z])
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    # Afficher le numéro de la marche et du point
                    font = pygame.font.Font(None, 16)
                    text = f"{i}-{j}"
                    text_surface = font.render(text, True, WHITE)
                    screen.blit(text_surface, (projected[0] + 10, projected[1] - 10))
            
            # Dessiner les arêtes de la marche (connecter les points)
            for j in range(4):
                x1, z1 = step_points[j]
                x2, z2 = step_points[(j + 1) % 4]
                
                point1_3d = np.array([x1, y, z1])
                point2_3d = np.array([x2, y, z2])
                
                proj1 = camera.project_3d_to_2d(point1_3d)
                proj2 = camera.project_3d_to_2d(point2_3d)
                
                if proj1 and proj2:
                    pygame.draw.line(screen, GREEN, 
                                   (int(proj1[0]), int(proj1[1])), 
                                   (int(proj2[0]), int(proj2[1])), 2)
    
    def draw_step_coordinates_vertical(self, screen: pygame.Surface, camera: Camera3D):
        """
        Dessine les coordonnées des contremarches verticales pour vérification
        """
        
        for i, (y, step_points) in enumerate(self.step_coordinates_vertical.items()):
            # Dessiner les 4 coins de chaque contremarche
            for j, (x, y_point, z) in enumerate(step_points):
                point_3d = np.array([x, y_point, z])
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    # Afficher le numéro de la contremarche et du point
                    font = pygame.font.Font(None, 16)
                    text = f"V{i}-{j}"
                    text_surface = font.render(text, True, RED)
                    screen.blit(text_surface, (projected[0] + 10, projected[1] - 10))
            
            # Dessiner les arêtes de la contremarche (connecter les points)
            for j in range(4):
                x1, y1, z1 = step_points[j]
                x2, y2, z2 = step_points[(j + 1) % 4]
                
                point1_3d = np.array([x1, y1, z1])
                point2_3d = np.array([x2, y2, z2])
                
                proj1 = camera.project_3d_to_2d(point1_3d)
                proj2 = camera.project_3d_to_2d(point2_3d)
                
                if proj1 and proj2:
                    pygame.draw.line(screen, RED, 
                                   (int(proj1[0]), int(proj1[1])), 
                                   (int(proj2[0]), int(proj2[1])), 2)
        
