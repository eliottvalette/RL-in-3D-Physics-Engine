# Run all the physics environments
```bash
clear
python -m physics_env.examples.first_scene
python -m physics_env.examples.floor_and_wall
python -m physics_env.examples.floor_and_ramp
python -m physics_env.examples.staircase
python -m physics_env.examples.tilted_cube
```

# Run the quadruped environment
```bash
clear
python -m physics_env.envs.quadruped_env
```

# Run the training
```bash
clear
python main.py
```

# Test the trained model
```bash
clear
python test.py
python apps/play_policy.py
```

# Run deterministic benches
```bash
clear
python apps/bench_headless.py --scenario settle --steps 600
python apps/bench_viewer.py --scenario settle --steps 3000
```

# Generate the code to send to the chat
```bash
clear
python files_to_send.py
```

# Visualize the training profile
```bash
clear
snakeviz profiling/training_profile.prof
```

# Visualize the physics engine profile
```bash
clear
snakeviz profiling/physics_engine_only.prof
```
