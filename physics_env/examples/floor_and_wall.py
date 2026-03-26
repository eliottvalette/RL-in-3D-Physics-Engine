# floor_and_wall.py
import pygame
import numpy as np
import random as rd
import math
from pygame.locals import *
from ..core.config import *
from ..rendering.camera import Camera3D
from ..legacy.cube import Cube3D
from ..rendering.ground import FloorAndWall
from ..legacy.update_functions import *
    

# --- Initialisation Pygame ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Moteur Physique 3D - Pygame")
clock = pygame.time.Clock()

# --- Objets du monde ---
camera = Camera3D()
cube = Cube3D(
        position=np.array([1.0, 8.0, 1.0]),
        x_length=5.0,
        y_length=2.0,
        z_length=3.0,
        rotation=np.array([4.0, 1.0, 1.0]),
        velocity=np.array([8.0, 4.0, 2.0])
    )
floor_and_wall = FloorAndWall(size=10)

# --- Contrôles caméra ---
camera_speed = 0.1
rotation_speed = 0.02

# --- Boucle principale ---
running = True
while running:
    # --- Gestion des événements ---
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
    
    # --- Contrôles caméra ---
    keys = pygame.key.get_pressed()
    
    # Mouvement caméra
    if keys[K_z]: # Z = Avancer
        camera.go_straight(camera_speed)
    if keys[K_s]: # S = Reculer
        camera.go_backward(camera_speed)
    if keys[K_q]: # Q = Gauche
        camera.go_left(camera_speed)
    if keys[K_d]: # D = Droite
        camera.go_right(camera_speed)
    if keys[K_a]: # A = Monter
        camera.position[1] += camera_speed
    if keys[K_e]: # E = Descendre
        camera.position[1] -= camera_speed
    
    # Rotation caméra
    if keys[K_LEFT]:
        camera.rotation[1] -= rotation_speed
    if keys[K_RIGHT]:
        camera.rotation[1] += rotation_speed
    if keys[K_UP]:
        camera.rotation[0] += rotation_speed
    if keys[K_DOWN]:
        camera.rotation[0] -= rotation_speed

    if keys[K_SPACE]:
        cube.reset()
    
    # --- Mise à jour physique ---
    update_ground_and_wall_complex(cube, floor_level=0, wall_distance=floor_and_wall.size)
    
    # --- Rendu ---
    screen.fill(BLACK)
    
    # Dessiner le monde 3D
    floor_and_wall.draw(screen, camera)
    floor_and_wall.draw_axes(screen, camera)
    cube.draw(screen, camera)
    
    # --- Interface utilisateur ---
    font = pygame.font.Font(None, 24)
    
    # Informations de position
    pos_text = f"Position: ({cube.position[0]:.2f}, {cube.position[1]:.2f}, {cube.position[2]:.2f})"
    vel_text = f"Vitesse: ({cube.velocity[0]:.2f}, {cube.velocity[1]:.2f}, {cube.velocity[2]:.2f})"
    cam_text = f"Caméra: ({camera.position[0]:.1f}, {camera.position[1]:.1f}, {camera.position[2]:.1f})"
    angular_vel_text = f"Rotation: ({cube.angular_velocity[0]:.2f}, {cube.angular_velocity[1]:.2f}, {cube.angular_velocity[2]:.2f})"

    pos_surface = font.render(pos_text, True, WHITE)
    vel_surface = font.render(vel_text, True, WHITE)
    cam_surface = font.render(cam_text, True, WHITE)
    angular_vel_surface = font.render(angular_vel_text, True, WHITE)
    
    screen.blit(pos_surface, (10, 10))
    screen.blit(vel_surface, (10, 35))
    screen.blit(cam_surface, (10, 60))
    screen.blit(angular_vel_surface, (10, 85))
    
    # Instructions
    instructions = [
        "Contrôles:",
        "ZQSD - Déplacer caméra",
        "AE - Monter/Descendre caméra", 
        "Flèches - Rotation caméra",
        "Espace - Reset cube",
        "Échap - Quitter"
    ]
    
    for i, instruction in enumerate(instructions):
        inst_surface = font.render(instruction, True, GRAY)
        screen.blit(inst_surface, (10, WINDOW_HEIGHT - 120 + i * 20))
    
    # --- Mise à jour écran ---
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
