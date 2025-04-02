import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from environment.custom_env2 import AmbulanceEnv
from typing import Dict, Any
from stable_baselines3.common.callbacks import CallbackList


# 1. Environment Setup with Action Wrapper
class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steering_actions = 5  # 0-4
        self.throttle_actions = 3  # 0-2
        self.action_space = gym.spaces.Discrete(self.steering_actions * self.throttle_actions)

    def action(self, act):
        steering = act // self.throttle_actions
        throttle = act % self.throttle_actions
        return {"steering": steering, "throttle": throttle}

def make_env(render_mode=None):
    env = AmbulanceEnv(grid_size=(10, 10), render_mode=render_mode)
    env = FlattenActionWrapper(env)
    env = Monitor(env)  # Wrap environment to monitor rewards
    return env

# 2. Callback for Tracking Progress and Saving Models
class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log rewards and lengths
            if len(self.model.ep_info_buffer) > 0:
                self.episode_rewards.append(self.model.ep_info_buffer[-1]['r'])
                self.episode_lengths.append(self.model.ep_info_buffer[-1]['l'])

                # Save best model
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model with mean reward: {mean_reward:.2f}")
                    self.model.save(self.save_path)

            # Log training loss
            if len(self.model.logger.name_to_value) > 0 and 'train/loss' in self.model.logger.name_to_value:
                self.losses.append(self.model.logger.name_to_value['train/loss'])

        return True
    
hyperparams = {
    
      'advanced': {
          'learning_rate': 5e-4,
          'gamma': 0.99,
          'buffer_size': 100000,
          'batch_size': 32,
          'exploration_fraction': 0.2,
          'exploration_initial_eps': 1.0,
          'exploration_final_eps': 0.05,
          'target_update_interval': 10000,
          'learning_starts': 10000,
          'train_freq': (4, "step")
      }

}

# 4. Training Function with Model Saving

def train_and_evaluate(config: Dict[str, Any],
                       config_name: str,
                       total_timesteps: int = 750000):
    print(f"\nTraining with configuration: {config_name}")
    print(config)

    # Create log directory
    log_dir = f"logs_{config_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Creating environment
    env = make_env()

    # Creating eval environment
    eval_env = make_env()

    # Network architecture
    policy_kwargs = dict(
      net_arch=[128, 128],
      activation_fn=torch.nn.ReLU,
      normalize_images=False
    )

    # Create and train model
    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        exploration_fraction=config['exploration_fraction'],
        exploration_initial_eps=config['exploration_initial_eps'],
        exploration_final_eps=config['exploration_final_eps'],
        target_update_interval=config['target_update_interval'],
        learning_starts=config['learning_starts'], 
        verbose=1,
        tensorboard_log=log_dir,
        device='auto'  # Explicit device setting
    )


    reward_callback = RewardCallback(check_freq=1000, log_dir=log_dir)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=log_dir,
                                 log_path=log_dir,
                                 eval_freq=10000,
                                 n_eval_episodes=5,
                                 deterministic=True)
    model.learn(total_timesteps=total_timesteps,
                callback=CallbackList([reward_callback, eval_callback]),
                tb_log_name=config_name)

    # Save final model
    model.save(os.path.join(log_dir, "final_model"))

    return {
        'model': model,
        'episode_rewards': reward_callback.episode_rewards,
        'losses': reward_callback.losses,
        'config_name': config_name,
        'log_dir': log_dir
    }

def plot_rewards(results_dict: Dict[str, Any], save_path: str = None):
    plt.figure(figsize=(12, 6))
    for config_name, result in results_dict.items():
        rewards = result['episode_rewards']
        episodes = range(1, len(rewards)+1)
        plt.plot(episodes, rewards, label=config_name)

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Performance Across Configurations')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(os.path.join(save_path, 'rewards_comparison.png'))
    plt.show()

def plot_losses(results_dict: Dict[str, Any], save_path: str = None):
    plt.figure(figsize=(12, 6))
    for config_name, result in results_dict.items():
        losses = result['losses']
        steps = range(1, len(losses)+1)
        plt.plot(steps, losses, label=config_name)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Across Configurations')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(os.path.join(save_path, 'loss_comparison.png'))
    plt.show()

def plot_individual_training(result: Dict[str, Any]):
    """Plot rewards and losses for a single training run"""
    plt.figure(figsize=(12, 5))

    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(result['episode_rewards'])
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f"{result['config_name']} - Episode Rewards")
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(result['losses'])
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f"{result['config_name']} - Training Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(result['log_dir'], 'training_metrics.png'))
    plt.show()
    
# 6. Convergence Analysis
def analyze_convergence(results_dict: Dict[str, Any],
                        window_size=10,
                        threshold=0.05):
    convergence_data = {}

    for config_name, result in results_dict.items():
        rewards = result['episode_rewards']

        # Calculate moving average
        moving_avg = np.convolve(rewards,
                                 np.ones(window_size)/window_size,
                                 mode='valid')

        # Find when performance stabilizes
        stable_episode = None
        for i in range(len(moving_avg)-window_size):
            window = moving_avg[i:i+window_size]
            if np.std(window) < threshold * np.mean(window):
                stable_episode = i + window_size
                break

        # Calculate final performance metrics
        final_reward = np.mean(rewards[-window_size:])if len(rewards) >= window_size else np.mean(rewards)
        max_reward = np.max(rewards)

        convergence_data[config_name] = {
            'stable_episode': stable_episode,
            'final_reward': final_reward,
            'max_reward': max_reward,
            'total_episodes': len(rewards)
        }

    return convergence_data

def main():
    # Train with different configurations
    results = {}
    for config_name, config in hyperparams.items():
        results[config_name] = train_and_evaluate(config,
                                                  config_name,
                                                  total_timesteps=750000)
        plot_individual_training(results[config_name])  # Plot individual training curves

    # Generate comparison plots
    plot_rewards(results, save_path="comparison_plots")
    plot_losses(results, save_path="comparison_plots")

    # Analyze convergence
    convergence = analyze_convergence(results)
    print("\nConvergence Analysis:")
    for config, data in convergence.items():
        print(f"\nConfiguration: {config}")
        print(f"Episodes to stable performance: {data['stable_episode']if data['stable_episode'] else 'Did not stabilize'}")
        print(f"Final average reward (last 10 eps): {data['final_reward']:.2f}")
        print(f"Maximum reward achieved: {data['max_reward']:.2f}")
        print(f"Total episodes completed: {data['total_episodes']}")

if __name__ == "__main__":
    os.makedirs("comparison_plots", exist_ok=True)
    main()
