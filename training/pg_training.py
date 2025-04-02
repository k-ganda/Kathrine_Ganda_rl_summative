import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from typing import Dict, Any
from stable_baselines3.common.callbacks import CallbackList
from environment.custom_env2 import AmbulanceEnv

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
    
ppo_hyperparams = {
    'high_entropy': {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.1,             
        'max_grad_norm': 0.5,
    },
}

def train_and_evaluate_ppo(config: Dict[str, Any], config_name: str, total_timesteps: int = 750000):
    print(f"\nTraining PPO with configuration: {config_name}")

    log_dir = f"ppo_logs_{config_name}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env()
    eval_env = make_env()

    # Network architecture
    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=torch.nn.ReLU,
        normalize_images=False
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device='auto',
        **config  
    )

    # Custom callback to track entropy
    class EntropyCallback(BaseCallback):
        def __init__(self, check_freq: int = 1000):
            super().__init__()
            self.check_freq = check_freq
            self.entropies = []

        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:
                entropy = float(self.model.logger.name_to_value.get("train/entropy_loss", 0))
                self.entropies.append(entropy)
            return True

    entropy_callback = EntropyCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([entropy_callback, eval_callback]),
        tb_log_name=config_name
    )

    model.save(os.path.join(log_dir, "final_model"))

    return {
        'model': model,
        'episode_rewards': [ep['r'] for ep in model.ep_info_buffer], # PPO stores rewards here
        'entropies': entropy_callback.entropies,
        'config_name': config_name,
        'log_dir': log_dir
    }
    
# 5. Analysis Functions
def plot_entropy(results_dict: Dict[str, Any], save_path: str = None):
    plt.figure(figsize=(12, 6))
    for config_name, result in results_dict.items():
        entropies = result['entropies']
        steps = np.arange(len(entropies)) * 10000  
        plt.plot(steps, entropies, label=config_name)

    plt.xlabel('Training Steps')
    plt.ylabel('Policy Entropy')
    plt.title('Policy Exploration Across Configurations')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(os.path.join(save_path, 'entropy_comparison.png'))
    plt.show()

def plot_individual_training_ppo(result: Dict[str, Any]):
    """PPO-specific training plots"""
    plt.figure(figsize=(12, 5))

    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(result['episode_rewards'])
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f"{result['config_name']} - Episode Rewards")
    plt.grid(True)

    # Entropy plot
    plt.subplot(1, 2, 2)
    plt.plot(result['entropies'])
    plt.xlabel('Training Steps (x10000)')
    plt.ylabel('Policy Entropy')
    plt.title(f"{result['config_name']} - Exploration")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(result['log_dir'], 'ppo_training_metrics.png'))
    plt.show()
    
def analyze_ppo_convergence(results_dict: Dict[str, Any],
                          reward_window: int = 10,
                          entropy_window: int = 100,
                          reward_threshold: float = 0.1,
                          entropy_threshold: float = 0.01) -> Dict[str, Dict[str, Any]]:
    """
    Analyzes PPO convergence with the same structure as your DQN version but adds entropy metrics.
    """
    convergence_data = {}

    for config_name, result in results_dict.items():
        rewards = np.array(result['episode_rewards'])
        entropies = np.array(result['entropies'])

        # Skip if no data
        if len(rewards) == 0 or len(entropies) == 0:
            convergence_data[config_name] = {
                'stable_episode': None,
                'final_reward': None,
                'max_reward': None,
                'entropy_stable_step': None,
                'final_entropy': None,
                'total_episodes': 0,
                'total_steps': 0
            }
            continue

        # --- Reward Stability ---
        if len(rewards) >= reward_window:
            reward_ma = np.convolve(rewards, np.ones(reward_window)/reward_window, mode='valid')

            stable_episode = None
            for i in range(len(reward_ma) - reward_window):
                window = reward_ma[i:i+reward_window]
                if np.std(window)/np.mean(window) < reward_threshold:
                    stable_episode = i + reward_window
                    break
        else:
            stable_episode = None

        # --- Entropy Decay ---
        if len(entropies) >= entropy_window:
            entropy_ma = np.convolve(entropies, np.ones(entropy_window)/entropy_window, mode='valid')

            entropy_stable_step = None
            for i in range(len(entropy_ma) - entropy_window):
                window = entropy_ma[i:i+entropy_window]
                if abs(np.mean(window[-10:]) - np.mean(window[:10])) < entropy_threshold:
                    entropy_stable_step = (i + entropy_window) * 10000 
                    break
        else:
            entropy_stable_step = None

        convergence_data[config_name] = {
            'stable_episode': stable_episode,
            'final_reward': np.mean(rewards[-min(reward_window, len(rewards)):]),
            'max_reward': np.max(rewards) if len(rewards) > 0 else None,
            'entropy_stable_step': entropy_stable_step,
            'final_entropy': np.mean(entropies[-min(entropy_window, len(entropies)):]) if len(entropies) > 0 else None,
            'total_episodes': len(rewards),
            'total_steps': len(entropies) * 10000  
        }

    return convergence_data

def main():
    # Train with different PPO configurations
    results = {}
    for config_name, config in ppo_hyperparams.items():
        results[config_name] = train_and_evaluate_ppo(
            config,
            config_name,
            total_timesteps=750000
        )
        plot_individual_training_ppo(results[config_name])  # PPO-specific plots

    # Generate comparison plots
    #plot_rewards(results, save_path="ppo_comparison_plots")
    plot_entropy(results, save_path="ppo_comparison_plots")  

    # Analyze and print convergence
    convergence = analyze_ppo_convergence(results)
    print("\n=== PPO Convergence Analysis ===")
    for config, data in convergence.items():
        print(f"\nConfiguration: {config}")
        print(f"Episodes to reward stability: {data['stable_episode'] or 'Did not stabilize'}")
        print(f"Steps to entropy stability: {data['entropy_stable_step'] or 'Did not stabilize'}")
        print(f"Final average reward (last 10 eps): {data['final_reward']:.2f}")
        print(f"Maximum reward achieved: {data['max_reward']:.2f}")
        print(f"Final policy entropy: {data['final_entropy']:.4f}")
        print(f"Total episodes completed: {data['total_episodes']}")
        print(f"Total training steps: {data['total_steps']}")

if __name__ == "__main__":
    os.makedirs("ppo_comparison_plots", exist_ok=True)
    main()
