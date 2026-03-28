# config.py
import numpy as np
import random as rd
import torch

# --- Configuration ---
PHYSICS_HZ = 120
RENDER_FPS = 60
FPS = RENDER_FPS  # alias legacy pour les viewers/exemples
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800

# --- Couleurs ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

# --- Physique ---
# Convention d'unites:
# 4.0 unites de longueur pour le body correspondent a 0.80 m.
# Donc 1 unite moteur = 0.20 m dans le monde physique.
UNIT_SCALE_M = 0.20

GRAVITY = np.array([0, -9.81, 0])
DT = 1 / PHYSICS_HZ
PHYSICS_STEPS_PER_RENDER = max(1, int(round(PHYSICS_HZ / RENDER_FPS)))

# --- Stabilisation dynamique ------------------------------------
G_STAB = 0.6          # gain d’amortissement (0‑1)
CONTACT_VERTICES_MAX = 32   # 4 lower‑legs × 8 sommets chacune

# --- Constantes de collision ---
RESTITUTION = 0.2    # Coefficient de restitution (0 = inélastique, 1 = parfaitement élastique)
FRICTION = 1.0       # Adherence cible proche d'un contact caoutchouc / beton sec
CONTACT_THRESHOLD_BASE = 0.05  # 0.05 unites ~= 1 cm
CONTACT_THRESHOLD_MULTIPLIER = 1.5  # Multiplicateur pour le seuil dynamique
CONTACT_SLOP = 0.01  # marge de penetration toleree ~= 2 mm
CONTACT_BIAS = 0.20  # vitesse de correction de penetration
CONTACT_POSITION_CORRECTION = 0.80
NORMAL_SOLVER_ITERATIONS = 8
MAX_CONTACT_POINTS = 8
CONTACT_MANIFOLD_MIN_XZ_SPACING = 0.35

# --- Limites de vitesse ---
MAX_VELOCITY = 10.0
MAX_ANGULAR_VELOCITY = 5.0
MAX_IMPULSE = 5.0
MAX_AVERAGE_IMPULSE = 2.0

# --- Angles d'articulations ---
SHOULDER_DELTA = 0.025
ELBOW_DELTA = 0.025
# 1.0 = comportement moteur actuel (prod), 0.0 = moteur plus assisté:
# réponse plus directe et freinage/ralentissement plus rapide.
MOTOR_DIFFICULTY = 1.0
MOTOR_RESPONSE_GAIN = 0.25
MOTOR_IDLE_BRAKE_GAIN = 0.35
MOTOR_VELOCITY_DAMPING = 0.08
MOTOR_STOP_EPS = 1e-4
MOTOR_RESPONSE_DIFFICULTY_BOOST = 0.45
MOTOR_BRAKE_DIFFICULTY_BOOST = 0.45
MOTOR_DAMPING_DIFFICULTY_BOOST = 0.22

# --- Contact / Friction ----------------------------------------------------
# 0.02 unites/s ~= 4 mm/s avec l'echelle ci-dessus.
SLIP_THRESHOLD = 0.02
STATIC_FRICTION_CAP  = 50.0     # impulsion maximale transmise au quadruped
TILT_DEADZONE = np.deg2rad(8.0)
TILT_NO_REWARD_ANGLE = np.deg2rad(10.0)
TILT_PENALTY_COEF = 6.0
CRITICAL_TILT_ANGLE = np.deg2rad(20.0)
MAX_CONSECUTIVE_CRITICAL_TILT_STEPS = 45
JOINT_LIMIT_THRESHOLD = np.pi / 2 * 0.9
MAX_CONSECUTIVE_JOINT_LIMIT_STEPS = 200
TERMINAL_PENALTY_TOO_HIGH = -4.0
TERMINAL_PENALTY_CRITICAL_TILT = -3.0
TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT = -2.0
TERMINAL_BONUS_MAX_STEPS = 0.5
PROGRESS_REWARD_COEF = 20.0
FORWARD_SPEED_REWARD_COEF = 0.75
JOINT_LIMIT_PROGRESS_PENALTY_COEF = 1.5
PITCH_RATE_PENALTY_COEF = 0.08
HEIGHT_PENALTY = -0.5
MIN_BODY_HEIGHT = 4.5
MAX_BODY_HEIGHT = 5.5
HEIGHT_REWARD_DECAY_MARGIN = 0.5
STABILITY_BONUS = 0.30
STABILITY_TILT_THRESHOLD = np.deg2rad(6.0)

# ----- Debug Physics Simulation --------------------------------------
DEBUG_CONTACT = False       

# ----- Debug RL Training --------------------------------------
DEBUG_RL_TRAIN = False
DEBUG_RL_MODEL = False
DEBUG_RL_AGENT = False
DEBUG_RL_VIZ   = False

# ----- RL Training Config --------------------------------------
EPISODES  = 2_000
MAX_STEPS = 500

START_EPS = 0.69
EPS_DECAY = 0.995
EPS_MIN   = 0.01

PLOT_INTERVAL = 30
SAVE_INTERVAL = 30

GAMMA = 0.99
ALPHA = 0.001
STATE_SIZE = 67
ACTION_SIZE = 8

RENDERING = True
PROFILING = False


def units_to_meters(value):
    return value * UNIT_SCALE_M


def meters_to_units(value):
    return value / UNIT_SCALE_M

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
