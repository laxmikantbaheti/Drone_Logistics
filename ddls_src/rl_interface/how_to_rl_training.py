import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# SB3 Imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment wrapper
from ddls_src.rl_interface.rl_scenario import LogisticRLScenario


# --- 1. Custom Callback for Metrics & Plotting ---
class LogisticsStatsCallback(BaseCallback):
    """
    Custom callback to extract specific logistics metrics (Makespan, Real Duration)
    at the end of each episode, replicating the manual logging you had before.
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
        # Accumulate reward for the current step (SB3 vec_envs return arrays)
        self.current_episode_reward += self.locals["rewards"][0]

        # Check if the episode is done
        if self.locals["dones"][0]:
            self.episode_count += 1

            # Calculate durations
            real_duration = time.time() - self.episode_start_time

            # Extract info to get simulation time (makespan)
            infos = self.locals["infos"][0]
            sim_makespan = infos.get("current_time", 0.0)

            # Store metrics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_makespans.append(sim_makespan)
            self.episode_real_durations.append(real_duration)

            # Log to console (similar to your previous script)
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
    """
    Bridge function to allow SB3 to access the environment's agent mask.
    """
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
    # We create the environment
    env = LogisticRLScenario(sim_config, visualize=False)

    # Wrap it with Monitor to help SB3 track internal stats
    env = Monitor(env)

    # Wrap it with ActionMasker so the agent knows which actions are valid
    env = ActionMasker(env, mask_fn)

    # 3. Define Model (MaskablePPO)
    # MlpPolicy is suitable for your vector observation space
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-3,
        gamma=0.99,
        # batch_size=64,  # Adjust based on memory
        # ent_coef=0.01   # Entropy coefficient for exploration
    )

    # 4. Training
    NUM_EPISODES = 1000
    # Approximate timesteps (assuming avg 20 steps per episode, adjust as needed)
    # Or you can just train for a large number and let the loop handle it
    TOTAL_TIMESTEPS = NUM_EPISODES * 100

    print(f"\n--- Starting Training for {TOTAL_TIMESTEPS} timesteps ---")

    # Initialize our custom callback
    callback = LogisticsStatsCallback(verbose=1)

    # Start the learning process
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # 5. Final Summary
    print("\n=========================================================")
    print(f"=== Training Summary ({len(callback.episode_rewards)} Episodes Completed) ===")
    if len(callback.episode_rewards) > 0:
        print(f"  Mean Total Reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"  Mean Makespan:     {np.mean(callback.episode_makespans):.2f}s")
        print(f"  Mean Real Duration:{np.mean(callback.episode_real_durations):.4f}s")
    print("=========================================================")

    # 6. Plotting Results
    episode_indices = list(range(1, len(callback.episode_rewards) + 1))
    plot_results(
        episode_indices,
        callback.episode_rewards,
        callback.episode_makespans,
        callback.episode_real_durations
    )


def plot_results(episodes, total_rewards, makespans, real_durations):
    """
    Generates THREE separate figures for Total Reward, Makespan, and Real Duration.
    """
    if not episodes:
        print("No episode data to plot.")
        return

    # --- Figure 1: Total Reward per Episode ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, total_rewards, color='b', linewidth=1, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance: Total Reward per Episode')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Figure 2: Delivery Makespan (Simulation Time) ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, makespans, color='r', linewidth=1, label='Delivery Makespan')
    plt.xlabel('Episode')
    plt.ylabel('Delivery Time (s)')
    plt.title('Operational Efficiency: Delivery Makespan per Episode')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Figure 3: Real-World Computational Duration ---
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