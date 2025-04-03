import gymnasium as gym
from stable_baselines3 import DQN, PPO
from environment.custom_env2 import AmbulanceEnv
import time
import numpy as np
import random
from typing import Dict, Tuple, Optional

ACTION_MEANINGS = {
    "steering": {
        0: "Straight",
        1: "Soft Left",
        2: "Hard Left", 
        3: "Soft Right",
        4: "Hard Right"
    },
    "throttle": {
        0: "Brake",
        1: "Maintain",
        2: "Accelerate"
    }
}

def action_idx_to_dict(action_idx: int) -> Tuple[Dict, str]:
    """Convert action index to steering and throttle components with descriptions."""
    if isinstance(action_idx, np.ndarray):
        action_idx = action_idx.item()
        
    steering = action_idx // 3  # 3 throttle options
    throttle = action_idx % 3
    return {
        "steering": steering,
        "throttle": throttle
    }, f"{ACTION_MEANINGS['steering'][steering]} + {ACTION_MEANINGS['throttle'][throttle]}"



def visualize_model(model_path: str, num_episodes: int = 3) -> None:
    """Visualize a trained model's performance."""
    env = AmbulanceEnv(grid_size=(10, 10), render_mode='human')
    
    try:
        
        print("\n=== Loading Trained Model ===")
        model = DQN.load(model_path)
        
        for episode in range(num_episodes):
            obs_dict, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            patients_delivered = 0  # Track actual deliveries
            
            print(f"\nEpisode {episode + 1} (Trained Model)")
            print(f"Initial Target: {obs_dict['destination']}")
            print(f"Patients remaining: {env.total_patients - env.patients_picked}")
            
            while not done:
                action_idx, _ = model.predict(obs_dict, deterministic=True)
                action, action_str = action_idx_to_dict(action_idx)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Update delivery count when patient is dropped at hospital
                if "DELIVERED" in info.get("status", ""):
                    patients_delivered += 1
                
                env.render()
                print(f"Step {steps:03d}: {action_str} | "
                      f"Pos: {next_obs['position']} | "
                      f"Speed: {env.current_speed:.1f} | "
                      f"Reward: {reward:+6.1f} (Total: {total_reward:6.1f}) | "
                      f"Status: {info.get('status', '')}")
                
                obs_dict = next_obs
                time.sleep(0.05)
            
            print(f"\nEpisode completed in {steps} steps")
            print(f"Total reward: {total_reward:.1f}")
            print(f"Patients delivered: {patients_delivered}/{env.total_patients}")  # Use actual deliveries count
            if env.mission_complete:
                print("MISSION SUCCESS - All patients delivered!")
            time.sleep(0.1)
    
    finally:
        env.close()

if __name__ == "__main__":
    visualize_model(
        model_path="models/dqn/lr5e-4.zip",
        num_episodes=3
    )

