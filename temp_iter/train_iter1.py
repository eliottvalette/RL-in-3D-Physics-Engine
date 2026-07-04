"""Iter1 retrain: continue from baseline checkpoint with both reward scales OFF.

Run with:
    REWARD_TARGET_PROGRESS_SCALE_ENABLED=0 REWARD_TRACTION_SCALE_ENABLED=0 \
    python temp_iter/train_iter1.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import QuadrupedAgent
from physics_env.core.config import (
    set_seed, GAMMA, ALPHA, STATE_SIZE, ACTION_SIZE,
    REWARD_TARGET_PROGRESS_SCALE_ENABLED, REWARD_TRACTION_SCALE_ENABLED,
)
from train import main_training_loop


def main():
    print(f"[ITER1] target_progress_scale_enabled={REWARD_TARGET_PROGRESS_SCALE_ENABLED}")
    print(f"[ITER1] traction_scale_enabled={REWARD_TRACTION_SCALE_ENABLED}")
    if REWARD_TARGET_PROGRESS_SCALE_ENABLED or REWARD_TRACTION_SCALE_ENABLED:
        raise RuntimeError("Expected both flags OFF for iter1; check env vars")

    set_seed(43)
    agent = QuadrupedAgent(
        state_size=STATE_SIZE,
        device="cpu",
        action_size=ACTION_SIZE,
        gamma=GAMMA,
        learning_rate=ALPHA,
        load_model=True,
        load_path="saved_models/quadruped_agent.pth",
    )
    main_training_loop(agent, episodes=300, rendering=False, render_every=10**9)


if __name__ == "__main__":
    main()
