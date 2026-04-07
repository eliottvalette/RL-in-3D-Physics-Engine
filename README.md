# 3D Physics Engine

3D physics sandbox for a quadruped robot, with a reinforcement learning loop for forward locomotion.

The project currently contains:

- Pygame-based 3D rendering and manual control.
- A custom quadruped body, joint, contact, and friction simulation.
- An on-policy actor-critic training loop using a PPO-style update.
- Deterministic physics benches for regression checks.

## Run Manual Scenes

```bash
python -m physics_env.examples.first_scene
python -m physics_env.examples.floor_and_wall
python -m physics_env.examples.floor_and_ramp
python -m physics_env.examples.staircase
python -m physics_env.examples.tilted_cube
```

## Run The Quadruped Environment

```bash
python -m physics_env.envs.quadruped_env
```

Manual controls:

- `ZQSD`: move camera
- `AE`: move camera up/down
- Arrow keys: rotate camera
- `R/F/T/G/Y/H/U/J`: shoulder joints
- `1-8`: elbow joints
- `Space`: reset quadruped with randomization
- `B`: reset quadruped without randomization
- `P`: print current state
- `Esc`: quit

## Train The Agent

```bash
python main.py
```

`main.py` builds a `QuadrupedAgent` and starts `main_training_loop`.

Current important behavior:

- `main.py` currently uses `load_model=True`, so training resumes from `saved_models/quadruped_agent.pth`.
- Set `load_model=False` in `main.py` to train from scratch.
- `Ctrl+C` triggers the `KeyboardInterrupt` fallback and saves the current model snapshot.
- `saved_models/quadruped_agent_epoch_<episode>.pth` is written on save.
- `saved_models/quadruped_agent.pth` is created only if it does not already exist; it is not overwritten.

## Watch A Trained Agent

```bash
python test.py
```

This loads `saved_models/quadruped_agent.pth` and runs deterministic actions from the actor.

## Current RL Setup

The agent is an on-policy actor-critic agent:

- Actor: categorical policy over 8 joints.
- Each joint has 3 discrete actions: `-1`, `0`, `+1`.
- Critic: scalar value function `V(s)`.
- Rollout buffer stores states, actions, old log-probs, values, rewards, and terminal flags.
- GAE is used for advantages.
- PPO-style update uses clipped policy ratios, minibatches, multiple epochs, entropy bonus, critic loss, gradient clipping, and an approximate KL early stop.

Current hyperparameters live in `physics_env/core/config.py`.

## Task And Reward Contract

The locomotion task is defined as:

```text
move forward while keeping a valid posture
```

Forward progress is measured as:

```text
forward = -position_z
progress_delta = forward_now - forward_previous
```

Reward contract:

```text
if posture is invalid:
    reward = terminal_penalty
    done = True
else:
    reward = progress_delta
```

There is no active velocity reward, sparse checkpoint reward, dense tilt penalty, action smoothness penalty, or hand-tuned penalty soup in the current reward.

Terminal failure reasons:

- `too_low`
- `too_high`
- `critical_tilt`
- `joint_limit_timeout`

## Caveats

Training is still experimental. Stochastic rollouts can show forward progress before deterministic evaluation becomes robust.

There is no epsilon-greedy exploration currently active; exploration comes from sampling the categorical policy and from the entropy term.

Observation normalization is not currently implemented.
