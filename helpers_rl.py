# helpers_rl.py
import os
import torch
import json

def save_models(agent, episode, models_dir="saved_models"):
    """
    Save the trained models for each player that has one.
    
    Args:
        players: List of players
        episode: Current episode number
        models_dir: Directory where models should be saved
    """
    print("\nSaving models...")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    if not hasattr(agent, 'actor_model') and not hasattr(agent, 'critic_model'):
        raise ValueError("Agent has no actor or critic model")
    
    
    # Create a dictionary containing both models' state dicts
    checkpoint = {
        'actor_state_dict': agent.actor_model.state_dict(),
        'critic_state_dict': agent.critic_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict()
    }
    
    epoch_path = f"{models_dir}/quadruped_agent_epoch_{episode}.pth"
    latest_path = f"{models_dir}/quadruped_agent.pth"

    torch.save(checkpoint, epoch_path)
    try:
        with open(latest_path, "xb") as latest_file:
            torch.save(checkpoint, latest_file)
    except FileExistsError:
        print(f"Latest model already exists, leaving untouched: {latest_path}")
    print("Models saved successfully!")

def save_metrics(metrics_history, output_dir):
    """
    Save training metrics history to a JSON file.
    
    Args:
        metrics_history: Dictionary containing training metrics
        output_dir: Directory where metrics should be saved
    """
    metrics_path = os.path.join(output_dir, "metrics_history.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"\nTraining completed and metrics saved to '{metrics_path}'")
