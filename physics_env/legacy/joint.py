# joint.py
import pygame
import numpy as np
import math
from .cube import Cube3D
from ..rendering.camera import Camera3D
from ..core.config import *

class Joint:
    def __init__(self, object_1: Cube3D, object_2: Cube3D, face_1: int = -1, face_2: int = -1, corner_1: int = -1, corner_2: int = -1, initial_angle: float = 0.0, color: tuple[int, int, int] = (0, 255, 0)):
        """
        object_1 : Cube3D
        object_2 : Cube3D
        face_1 : int <- index de la face sur object_1
        face_2 : int <- index de la face sur object_2
        initial_angle : float <- angle initial du joint en radians (0 = ouvert plat)
        """
        self.object_1 = object_1
        self.object_2 = object_2
        self.face_1 = face_1
        self.face_2 = face_2
        self.corner_1 = corner_1
        self.corner_2 = corner_2
        self.angle = initial_angle  # Angle du joint en radians
        self.color = color

        self.joint_position_1 = None
        self.joint_position_2 = None
        self.joint_position = None

        self.using_face_1 = face_1 != -1
        self.using_face_2 = face_2 != -1
        
        # Calculer les positions initiales
        self.update_positions()

    def update_positions(self):
        """Met à jour les positions des objets pour respecter l'angle du joint"""
        # Position du joint (point de connexion toujours sur l'objet 1)
        joint_position = self.object_1.get_face_center(self.face_1) if self.using_face_1 else self.object_1.get_corner_position(self.corner_1)
        
        # Calculer la direction de l'objet 2 suivant l'angle du joint
        object_2_direction = np.array([
            math.cos(self.angle),
            math.sin(self.angle),
            0.0
        ])
        
        # Calculer la nouvelle position de l'objet 2
        # L'objet 2 doit être positionné pour que sa face de jointure touche aussi le joint
        object_2_offset = object_2_direction * (self.object_2.x_length / 2)
        self.object_2.position = joint_position + object_2_offset
        
        # Faire tourner l'objet 2 selon l'angle du joint
        # La rotation se fait autour de l'axe Z (normal au plan XY)
        self.object_2.rotation[2] = self.angle
        
        # Mettre à jour les positions des points de joint
        self.joint_position_1 = self.object_1.get_face_center(self.face_1) if self.using_face_1 else self.object_1.get_corner_position(self.corner_1)
        self.joint_position_2 = self.object_2.get_face_center(self.face_2) if self.using_face_2 else self.object_2.get_corner_position(self.corner_2)
        self.joint_position = (self.joint_position_1 + self.joint_position_2) / 2

    def set_angle(self, angle):
        """Définit l'angle du joint et met à jour les positions"""
        self.angle = angle
        self.update_positions()

    def update(self):
        """Met à jour le joint (appelé chaque frame)"""
        # Forcer les objets à respecter l'angle du joint
        self.update_positions()
    
    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """
        Dessine le joint sur l'écran, 
        On dessine un carré de 4x4 pixels en joint_position,
        et on trace les lignes qui relient le joint à object_1 et object_2
        """
        projected_position_1 = camera.project_3d_to_2d(self.joint_position_1)
        projected_position_2 = camera.project_3d_to_2d(self.joint_position_2)
        projected_joint_position = camera.project_3d_to_2d(self.joint_position)

        if projected_position_1 and projected_position_2 and projected_joint_position :
            # Dessiner le carré de 4x4 pixels en joint_position
            pygame.draw.rect(screen, self.color, (projected_joint_position[0] - 2, projected_joint_position[1] - 2, 4, 4))

            # Dessiner le lien entre le point d'ancrage entre object_1 et object_2
            pygame.draw.line(screen, self.color, (projected_position_1[0], projected_position_1[1]), (projected_position_2[0], projected_position_2[1]), 1)
            
