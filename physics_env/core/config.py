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

# --- Physique ---
# Convention d'unites:
# 4.0 unites de longueur pour le body correspondent a 0.80 m.
# Donc 1 unite moteur = 0.20 m dans le monde physique.
UNIT_SCALE_M = 0.20
TASK_FORWARD_Z_SIGN = 1.0  # +Z is the red/front side in the box renderer.
TASK_FORWARD_WORLD = np.array([0.0, 0.0, TASK_FORWARD_Z_SIGN], dtype=np.float64)

GRAVITY = np.array([0, -9.81, 0])
DT = 1 / PHYSICS_HZ
PHYSICS_STEPS_PER_RENDER = max(1, int(round(PHYSICS_HZ / RENDER_FPS)))

# --- Constantes de collision ---
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

# --- Angles d'articulations ---
SHOULDER_DELTA = 0.025
ELBOW_DELTA = 0.025
SHOULDER_ANGLE_MIN = -np.pi / 2
SHOULDER_ANGLE_MAX = np.pi / 2
ELBOW_ANGLE_MIN = -np.pi
ELBOW_ANGLE_MAX = 0

# Limites "near joint limit" pretes pour des bornes distinctes et asymetriques.
SHOULDER_JOINT_LIMIT_THRESHOLD_LOW = SHOULDER_ANGLE_MIN * 0.9
SHOULDER_JOINT_LIMIT_THRESHOLD_HIGH = SHOULDER_ANGLE_MAX * 0.9
ELBOW_JOINT_LIMIT_THRESHOLD_LOW = ELBOW_ANGLE_MIN * 0.9
ELBOW_JOINT_LIMIT_THRESHOLD_HIGH = ELBOW_ANGLE_MAX * 0.9

# 1.0 = comportement moteur actuel (prod), 0.0 = moteur plus assisté:
# réponse plus directe et freinage/ralentissement plus rapide.
MOTOR_DIFFICULTY = 1.0
RESPONSE_COEFF = 1
MOTOR_RESPONSE_GAIN = 0.04 * RESPONSE_COEFF
MOTOR_IDLE_BRAKE_GAIN = 0.02 * RESPONSE_COEFF
MOTOR_VELOCITY_DAMPING = 0.01 * RESPONSE_COEFF
MOTOR_STOP_EPS = 1e-4
MOTOR_RESPONSE_DIFFICULTY_BOOST = 0.45
MOTOR_BRAKE_DIFFICULTY_BOOST = 0.45
MOTOR_DAMPING_DIFFICULTY_BOOST = 0.22

# --- Reset pose randomization --------------------------------------
RESET_JOINT_ANGLE_JITTER = np.deg2rad(9.0)
RESET_VERTICAL_AXIS_ROTATION_JITTER = np.deg2rad(3.0)

# --- Contraintes RL ----------------------------------------------------
CRITICAL_TILT_ANGLE = np.deg2rad(20.0)
TILT_SOFT_REWARD_MARGIN = np.deg2rad(8.0)
JOINT_LIMIT_ANGLE_EPS = 1e-3
JOINT_LIMIT_STUCK_GRACE_STEPS = 8
JOINT_LIMIT_STUCK_EXP_RATE = 3.0
MAX_CONSECUTIVE_JOINT_LIMIT_STEPS = 200
TERMINAL_PENALTY_TOO_HIGH = TERMINAL_PENALTY_TOO_LOW = TERMINAL_PENALTY_CRITICAL_TILT = TERMINAL_PENALTY_JOINT_LIMIT_TIMEOUT = TERMINAL_PENALTY_AIRBORNE = -10
PROGRESS_REWARD_COEF = 1.0
CONTACT_GRACE_STEPS = 5
MAX_AIRBORNE_STEPS = 80
ANGULAR_VELOCITY_PENALTY_COEF = 0.005
JOINT_LIMIT_PENALTY_COEF = 0.25
FOOT_SLIP_SPEED_THRESHOLD = 0.75
FOOT_SLIP_PENALTY_COEF = 0.02
FOOT_SLIP_MAX_PENALTY = 0.02
ACTION_CHANGE_PENALTY_COEF = 0.003
SUPPORT_DEGENERACY_GRACE_STEPS = 12
SUPPORT_DEGENERACY_WINDOW_STEPS = 120
SUPPORT_DEGENERACY_MAX_PENALTY = 0.02
FOOT_UNUSED_GRACE_STEPS = 120
FOOT_UNUSED_WINDOW_STEPS = 240
FOOT_UNUSED_MAX_PENALTY = 0.025
CONTACT_QUALITY_MIN_LEGS = 2
CONTACT_QUALITY_SCALE_FLOOR = 0.25
MIN_BODY_HEIGHT = 3.2
MAX_BODY_HEIGHT = 5.5
HEIGHT_SOFT_REWARD_MARGIN = 0.25

# ----- Debug Physics Simulation --------------------------------------
DEBUG_CONTACT = False       

# ----- Debug RL Training --------------------------------------
DEBUG_RL_TRAIN = False
DEBUG_RL_MODEL = False
DEBUG_RL_AGENT = False

# ----- Debug Gait Evaluation --------------------------------------
DEBUG_GAIT_EVAL = True
DEBUG_GAIT_EVAL_EPISODES = 10
DEBUG_GAIT_EVAL_MAX_STEPS = 750
DEBUG_GAIT_EVAL_SAVE_JSON = True
DEBUG_GAIT_EVAL_JSON_PATH = "visualizations/debug_gait_eval.json"
DEBUG_GAIT_EVAL_PRINT_EPISODES = False

# ----- RL Training Config --------------------------------------
EPISODES  = 2_000
MAX_STEPS = 750
ROLLOUT_STEPS = 256
GAE_LAMBDA = 0.998
ENTROPY_COEFF = 0.02
CRITIC_LOSS_COEFF = 0.5
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 64
PPO_TARGET_KL = 0.02
EVAL_INTERVAL = 10
EVAL_EPISODES = 1

PLOT_INTERVAL = 100
SAVE_INTERVAL = 100
PLOT_DPI = 200

GAMMA = 0.999
ALPHA = 3e-4
STATE_SIZE = 102
ACTION_SIZE = 8

RENDERING = False
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
