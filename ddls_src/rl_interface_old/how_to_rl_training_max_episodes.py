import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
# [OPTIONAL] Use this if your environment runs forever
from gymnasium.wrappers import TimeLimit
# SB3 Imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment wrapper
from ddls_src.rl_interface_old.rl_scenario import LogisticRLScenario


# --- 1. Custom Callback for Metrics ONLY (Removed stopping logic) ---
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

        return True  # Always return True, let StopTrainingOnMaxEpisodes handle stopping


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
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data_mh_matrix.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 300.0,
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {
                "file_path": config_file_path
            }
        },
    }

    # 2. Initialize Environment
    env = LogisticRLScenario(sim_config, visualize=False)

    # [CRITICAL CHECK] Wraps env to ensure it returns DONE after N steps if the simulation doesn't.
    # Adjust max_episode_steps to a value slightly higher than your expected makespan
    # If your environment handles this internally, you can remove this wrapper.
    env = TimeLimit(env, max_episode_steps=5000)

    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    # 3. Define Model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-3,
        gamma=0.99,
    )

    # 4. Training Configuration
    NUM_EPISODES = 1
    LARGE_TIMESTEPS = 1_000_000  # Keep this high

    print(f"\n--- Starting Training for {NUM_EPISODES} episodes ---")

    # [MODIFICATION] Create the Callback List
    # 1. Stats callback (logs data)
    stats_callback = LogisticsStatsCallback(verbose=1)
    # 2. Stop callback (forces stop after N episodes)
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=NUM_EPISODES, verbose=1)

    # Combine them
    callbacks = CallbackList([stats_callback, stop_callback])

    # Start learning
    model.learn(total_timesteps=LARGE_TIMESTEPS, callback=callbacks)

    # 5. Final Summary
    # Note: We access stats_callback explicitly to get the data
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