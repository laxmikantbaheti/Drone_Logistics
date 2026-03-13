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
import torch.nn as nn

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment wrapper
from ddls_src.rl_interface.rl_scenario import LogisticRLScenario

# ==============================================================================
# [OPTIONAL] MONKEY PATCH FOR PYTORCH SIMPLEX ERROR (left as-is)
# ==============================================================================
import torch
from sb3_contrib.common.maskable import distributions

# torch.set_default_device("cuda")


def safe_init(self, probs=None, logits=None, validate_args=None):
    if probs is not None:
        if (probs < 0).any():
            probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(-1, keepdim=True)
    # original_init(self, probs=probs, logits=logits, validate_args=validate_args)


# distributions.MaskableCategorical.__init__ = safe_init

# ==============================================================================
# --- 1. Custom Callback for Metrics ONLY ---
# ==============================================================================
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
        self.current_episode_reward += self.locals["rewards"][0]

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

    # ----------------------------------------------------------------------
    # 1) Define Configuration (UPDATED TO USE VRPDBenchmarkDataGenerator)
    # ----------------------------------------------------------------------
    script_path = os.path.dirname(os.path.realpath(__file__))

    # Change this to match where your VRP-D instances are stored
    vrp_instance_path = os.path.join(
        script_path,
        "..",
        "scenarios",
        "vrp_d_instances",
        "VRP-D",
        "A-n32-k5-20.vrp"
    )
    vrp_instance_path = os.path.normpath(vrp_instance_path)

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            # IMPORTANT: must match your generator factory registry name
            "generator_type": "vrpd",
            "generator_config": {
                "instance_path": vrp_instance_path,

                # Keep these custom for your delivery model
                "num_drones": 6,
                "num_microhubs": 2,
                "bbox": (0, 0, 100, 100),
                "std_dev_scale": 4.0,

                # Capacity-derived configs inside generator
                "drone_capacity_ratio": 0.2,

                # Speeds (if your sim uses them)
                "truck_speed": 1.0,
                "drone_speed": 1.0,

                # Optional
                "seed": 42,

                # If you want to override trucks instead of using -k# from filename:
                # "num_trucks": 5,
            }
        },
    }

    print("\n--- Scenario Config ---")
    print("Generator:", sim_config["data_loader_config"]["generator_type"])
    print("Instance:", sim_config["data_loader_config"]["generator_config"]["instance_path"])

    # ----------------------------------------------------------------------
    # 2) Initialize Environment
    # ----------------------------------------------------------------------
    env = LogisticRLScenario(sim_config, visualize=False, custom_log=1)

    MAX_STEPS = 5_000_000
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)

    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    # ----------------------------------------------------------------------
    # 3) Define Model
    # ----------------------------------------------------------------------
    EPISODE_BUFFER_SIZE = 1000  # NOTE: you had a comment saying must be > MAX_STEPS; it's not.
                              # Keep as-is since you didn't ask to change training behavior.

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-3,
        n_steps=EPISODE_BUFFER_SIZE,
        batch_size=50,
        n_epochs=10,
        gamma=1,
        gae_lambda=0.99,
        ent_coef=0.00,
    )

    # ----------------------------------------------------------------------
    # 4) Training Configuration
    # ----------------------------------------------------------------------
    NUM_EPISODES = 1500
    LARGE_TIMESTEPS = NUM_EPISODES * (EPISODE_BUFFER_SIZE + 100)

    print(f"\n--- Starting Training for {NUM_EPISODES} episodes ---")

    stats_callback = LogisticsStatsCallback(verbose=1)
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=NUM_EPISODES, verbose=1)
    callbacks = CallbackList([stats_callback, stop_callback])

    model.learn(total_timesteps=LARGE_TIMESTEPS, callback=callbacks)

    # ----------------------------------------------------------------------
    # 5) Final Summary
    # ----------------------------------------------------------------------
    print("\n=========================================================")
    print(f"=== Training Summary ({len(stats_callback.episode_rewards)} Episodes Completed) ===")
    if len(stats_callback.episode_rewards) > 0:
        print(f"  Mean Total Reward: {np.mean(stats_callback.episode_rewards):.2f}")
        print(f"  Mean Makespan:     {np.mean(stats_callback.episode_makespans):.2f}s")
        print(f"  Mean Real Duration:{np.mean(stats_callback.episode_real_durations):.4f}s")
    print("=========================================================")

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
