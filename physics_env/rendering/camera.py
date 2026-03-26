# camera.py
import math
import numpy as np
from ..core.config import *

class Camera3D:
    def __init__(self):
        self.position = np.array([-9.5, 8.8, -8.6])
        self.rotation = np.array([-0.5, 1.0, -0.5])  # pitch, yaw, roll (pitch = on leve et on baisse la tete, yaw = on tourne la tete à droite et à gauche, roll = on tourne le corps à droite et à gauche)
        self.fov = math.pi / 2  # 90 degrés
        self.near = 0.1
        
    def project_3d_to_2d(self, point_3d):
        """
        Projette un point 3D en coordonnées 2D d'écran
        Input : point_3d : [x, y, z] dans un repère 3D
        Output : [x, y, z] où x et y sont les coordonnées 2D et z est la profondeur (distance entre le point et la caméra)
        """
        # Translation relative à la caméra
        relative_pos = point_3d - self.position
        
        # Rotation de la caméra (simplifiée - seulement yaw et pitch)
        cos_yaw = math.cos(self.rotation[1])
        sin_yaw = math.sin(self.rotation[1])
        cos_pitch = math.cos(self.rotation[0])
        sin_pitch = math.sin(self.rotation[0])
        
        # Rotation Y (yaw)
        x = relative_pos[0] * cos_yaw - relative_pos[2] * sin_yaw
        y = relative_pos[1]
        z = relative_pos[0] * sin_yaw + relative_pos[2] * cos_yaw
        
        # Rotation X (pitch)
        y2 = y * cos_pitch - z * sin_pitch
        z2 = y * sin_pitch + z * cos_pitch

        if z2 <= self.near:  # Éviter la division par zéro
            return None # Si le point est derrière la caméra, on ne le projette pas
            
        scale = 400 / z2  # Facteur d'échelle
        
        # Limiter le facteur d'échelle pour éviter les coordonnées énormes
        if scale > 1000:
            return None
            
        screen_x = WINDOW_WIDTH // 2 + x * scale
        screen_y = WINDOW_HEIGHT // 2 - y2 * scale
        
        # Vérifier que les coordonnées ne sont pas infinies ou NaN
        if (math.isnan(screen_x) or math.isnan(screen_y) or 
            math.isinf(screen_x) or math.isinf(screen_y)):
            return None
            
        # Vérifier que les coordonnées sont dans des limites raisonnables
        if abs(screen_x) > 10000 or abs(screen_y) > 10000:
            return None
        
        return (int(screen_x), int(screen_y), z2)
    
    def get_depth(self, point_3d):
        """Retourne la profondeur d'un point pour le tri"""
        relative_pos = point_3d - self.position
        return math.sqrt(np.dot(relative_pos, relative_pos))
    
    # Déplacement de la caméra en fonction de sa rotation
    def go_straight(self, speed):
        self.position[2] += speed * math.cos(self.rotation[1])
        self.position[0] += speed * math.sin(self.rotation[1])
    
    def go_backward(self, speed):
        self.position[2] -= speed * math.cos(self.rotation[1])
        self.position[0] -= speed * math.sin(self.rotation[1])
    
    def go_left(self, speed):
        self.position[0] -= speed * math.cos(self.rotation[1])
        self.position[2] += speed * math.sin(self.rotation[1])
    
    def go_right(self, speed):
        self.position[0] += speed * math.cos(self.rotation[1])
        self.position[2] -= speed * math.sin(self.rotation[1])

    def go_up(self, speed):
        self.position[1] += speed

    def go_down(self, speed):
        self.position[1] -= speed

    def yaw_left(self, speed):
        self.rotation[1] -= speed

    def yaw_right(self, speed):
        self.rotation[1] += speed

    def pitch_up(self, speed):
        self.rotation[0] -= speed

    def pitch_down(self, speed):
        self.rotation[0] += speed

    def update_camera(self, action_idx, camera_speed=0.1, rotation_speed=0.02):
        """
        Update camera position or rotation based on action index:
        0: z (forward)
        1: s (backward)
        2: q (left)
        3: d (right)
        4: e (up)
        5: a (down)
        6: left (yaw left)
        7: right (yaw right)
        8: up (pitch up)
        9: down (pitch down)
        """
        if action_idx == 0:
            self.go_straight(camera_speed)
        elif action_idx == 1:
            self.go_backward(camera_speed)
        elif action_idx == 2:
            self.go_left(camera_speed)
        elif action_idx == 3:
            self.go_right(camera_speed)
        elif action_idx == 4:
            self.go_up(camera_speed)
        elif action_idx == 5:
            self.go_down(camera_speed)
        elif action_idx == 6:
            self.yaw_left(rotation_speed)
        elif action_idx == 7:
            self.yaw_right(rotation_speed)
        elif action_idx == 8:
            self.pitch_up(rotation_speed)
        elif action_idx == 9:
            self.pitch_down(rotation_speed)
