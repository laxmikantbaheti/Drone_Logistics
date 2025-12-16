import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# SB3 Imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor

# [OPTIONAL] Use this if your environment runs forever
from gymnasium.wrappers import TimeLimit

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment wrapper
from ddls_src.rl_interface.rl_scenario import LogisticRLScenario
# # ==============================================================================
# # [FIX] MONKEY PATCH FOR PYTORCH SIMPLEX ERROR
# # ==============================================================================
import torch
from sb3_contrib.common.maskable import distributions
#
# # 1. Save the original constructor so we can call it later
# original_init = distributions.MaskableCategorical.__init__
# torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")

# 2. Define a "Safe" constructor that sanitizes input
def safe_init(self, probs=None, logits=None, validate_args=None):
    if probs is not None:
        # A. Fix tiny negative values (e.g. -1e-20) caused by float precision
        if (probs < 0).any():
            probs = torch.clamp(probs, min=1e-8)

        # B. Force re-normalization so sum is EXACTLY 1.0
        #    This satisfies the ((value.sum(-1) - 1).abs() < 1e-6) check
        probs = probs / probs.sum(-1, keepdim=True)

    # 3. Call the original PyTorch/SB3 logic with clean data
    # original_init(self, probs=probs, logits=logits, validate_args=validate_args)


# 4. Apply the patch
# distributions.MaskableCategorical.__init__ = safe_init


# ==============================================================================

# --- 1. Custom Callback for Metrics ONLY ---
class LogisticsStatsCallback(BaseCallback):
    """
    Custom callback to extract specific logistics metrics.
    """

    def __init__(self, verbose=0):
        super(LogisticsStatsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_makespans = []
        self.episode_real_durations = []
        self.episode_start_time = time.time()
        self.current_episode_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Accumulate reward
        self.current_episode_reward += self.locals["rewards"][0]

        # Check if the episode is done
        if self.locals["dones"][0]:
            self.episode_count += 1
            real_duration = time.time() - self.episode_start_time
            infos = self.locals["infos"][0]
            sim_makespan = infos.get("current_time", 0.0)

            self.episode_rewards.append(self.current_episode_reward)
            self.episode_makespans.append(sim_makespan)
            self.episode_real_durations.append(real_duration)

            if self.verbose > 0:
                print(f"--- Episode {self.episode_count} Finished ---")
                print(f"    Total Reward: {self.current_episode_reward:.2f}")
                print(f"    Makespan (Sim): {sim_makespan:.2f}s")
                print(f"    Duration (Real): {real_duration:.4f}s")

            # Reset trackers
            self.current_episode_reward = 0.0
            self.episode_start_time = time.time()

        return True


# --- 2. Helper to expose the mask to SB3 ---
def mask_fn(env: LogisticRLScenario) -> np.ndarray:
    return env.unwrapped._get_agent_mask()


# --- 3. Main Training Execution ---
def run_ppo_simulation():
    print("=========================================================")
    print("===   Running LogisticRLScenario with MaskablePPO     ===")
    print("=========================================================")

    # 1. Define Configuration
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'large_instance.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {
                "file_path": config_file_path
            }
        },
    }

    # 2. Initialize Environment
    env = LogisticRLScenario(sim_config, visualize=False)

    # [CRITICAL] Set Max Episode Steps
    # We set this to 5000. It is crucial that n_steps below is > 5000.
    MAX_STEPS = 500000
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)

    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    # [CRITICAL MODIFICATION] - Episodic Update Configuration
    # We want to collect a FULL episode before updating.
    EPISODE_BUFFER_SIZE = 400  # Must be > MAX_STEPS (5000)

    # 3. Define Model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,

        # --- Episodic Update Settings ---
        learning_rate=5e-3,
        n_steps=EPISODE_BUFFER_SIZE,  # Wait for ~5120 steps before training
        batch_size=128,  # Standard mini-batch size (or set to 5120 for full-batch)
        n_epochs=5,  # Train on this episode data 10 times
        gamma=0.99,
        gae_lambda=1.0,  # 1.0 = Monte Carlo (No bootstrapping)
        ent_coef=0.01,
        device="cuda"# Encourages exploration (helpful for sparse rewards)
    )

    # 4. Training Configuration
    # Increase this for real training (e.g., 500)
    NUM_EPISODES = 700

    # Total timesteps must be enough to cover NUM_EPISODES * MAX_STEPS
    LARGE_TIMESTEPS = NUM_EPISODES * (EPISODE_BUFFER_SIZE + 100)

    print(f"\n--- Starting Training for {NUM_EPISODES} episodes ---")

    # Callbacks
    stats_callback = LogisticsStatsCallback(verbose=1)
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=NUM_EPISODES, verbose=1)
    callbacks = CallbackList([stats_callback, stop_callback])

    # Start learning
    model.learn(total_timesteps=LARGE_TIMESTEPS, callback=callbacks)

    # 5. Final Summary
    print("\n=========================================================")
    print(f"=== Training Summary ({len(stats_callback.episode_rewards)} Episodes Completed) ===")
    if len(stats_callback.episode_rewards) > 0:
        print(f"  Mean Total Reward: {np.mean(stats_callback.episode_rewards):.2f}")
        print(f"  Mean Makespan:     {np.mean(stats_callback.episode_makespans):.2f}s")
        print(f"  Mean Real Duration:{np.mean(stats_callback.episode_real_durations):.4f}s")
    print("=========================================================")

    # 6. Plotting Results
    episode_indices = list(range(1, len(stats_callback.episode_rewards) + 1))
    plot_results(
        episode_indices,
        stats_callback.episode_rewards,
        stats_callback.episode_makespans,
        stats_callback.episode_real_durations
    )


def plot_results(episodes, total_rewards, makespans, real_durations):
    if not episodes:
        print("No episode data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, total_rewards, color='b', linewidth=1, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance: Total Reward per Episode')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, makespans, color='r', linewidth=1, label='Delivery Makespan')
    plt.xlabel('Episode')
    plt.ylabel('Delivery Time (s)')
    plt.title('Operational Efficiency: Delivery Makespan per Episode')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, real_durations, color='purple', linewidth=1, label='Computation Time')
    plt.xlabel('Episode')
    plt.ylabel('Real-World Time (s)')
    plt.title('Computational Performance: Real-Time Duration per Episode')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_ppo_simulation()