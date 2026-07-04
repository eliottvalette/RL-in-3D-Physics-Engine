"""Headless gait eval — run 10 episodes, save JSON."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from agent import QuadrupedAgent
from physics_env.envs.quadruped_env import QuadrupedEnv
from physics_env.diagnostics.gait_eval import run_gait_debug_eval
from physics_env.core.config import (
    set_seed, GAMMA, ALPHA, STATE_SIZE, ACTION_SIZE,
    REWARD_TARGET_PROGRESS_SCALE_ENABLED, REWARD_TRACTION_SCALE_ENABLED,
)


def main():
    if len(sys.argv) != 3:
        print("usage: python temp_iter/eval_headless.py <checkpoint.pth> <output_json>")
        sys.exit(1)
    ckpt = sys.argv[1]
    out_json = sys.argv[2]
    print(f"[EVAL] target_progress={REWARD_TARGET_PROGRESS_SCALE_ENABLED} traction={REWARD_TRACTION_SCALE_ENABLED}")
    set_seed(43)
    agent = QuadrupedAgent(
        state_size=STATE_SIZE, device="cpu", action_size=ACTION_SIZE,
        gamma=GAMMA, learning_rate=ALPHA,
        load_model=True, load_path=ckpt,
    )
    env = QuadrupedEnv(rendering=False, headless=True)
    run_gait_debug_eval(agent, env, render=False)
    shutil.copy("visualizations/debug_gait_eval.json", out_json)
    print(f"[EVAL] saved -> {out_json}")


if __name__ == "__main__":
    main()
