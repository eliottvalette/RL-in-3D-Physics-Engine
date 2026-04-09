# Update the last best model
```bash
clear
rm -rf saved_models/quadruped_agent.pth
mv saved_models/quadruped_agent_epoch_1999.pth saved_models/quadruped_agent.pth
```

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
```

# Run deterministic benches
```bash
clear
python apps/bench_headless.py --list
python apps/bench_headless.py --scenario settle --steps 600
python apps/bench_headless.py --scenario drop_flat --steps 600
python apps/bench_headless.py --scenario slide_x --steps 600
python apps/bench_headless.py --scenario front_legs_lifted --steps 600
python apps/bench_viewer.py --scenario settle --steps 3000
python apps/bench_viewer.py --scenario front_legs_lifted --steps 3000
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
