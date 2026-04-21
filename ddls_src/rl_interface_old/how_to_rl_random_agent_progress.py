import math  # Added for potential mathematical operations
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment wrapper
from ddls_src.rl_interface_old.rl_scenario import LogisticRLScenario


def random_policy(mask):
    """
    A simple policy that picks a random valid action from the mask.
    """
    valid_indices = np.where(mask)[0]
    if len(valid_indices) > 0:
        return np.random.choice(valid_indices)
    return 0


def format_time(seconds):
    """Converts seconds into HH:MM:SS format."""
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    seconds = s % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def display_progress_bar(episode, total_episodes, global_start_real_time, last_makespan):
    """
    Displays an over-the-top, floating console progress bar.
    """
    percent = (episode / total_episodes)
    bar_length = 50
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    elapsed_time = time.time() - global_start_real_time

    # Simple ETA based on average episode time
    if episode > 0:
        avg_time_per_ep = elapsed_time / episode
        remaining_time = avg_time_per_ep * (total_episodes - episode)
    else:
        remaining_time = 0

    progress_line = (
        f"\r🚀 TRAINING PROGRESS | {bar} | {percent * 100:6.2f}% "
        f"| EPISODE {episode}/{total_episodes} "
        f"| ELAPSED: {format_time(elapsed_time)} "
        f"| ETA: {format_time(remaining_time)} "
        f"| LAST MS: {last_makespan:.0f}s"
    )
    sys.stdout.write(progress_line)
    sys.stdout.flush()


def run_multi_episode_simulation():
    print("=========================================================")
    print("===   Running Multi-Episode LogisticRLScenario        ===")
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
    # Note: Assuming LogisticRLScenario is accessible and configured correctly.
    env = LogisticRLScenario(sim_config, visualize=False)

    # 3. Define Training Parameters
    NUM_EPISODES = 10000

    # --- Storage for Plotting ---
    episode_indices = []
    all_total_rewards = []
    all_makespans = []  # Simulation time (logic time)
    all_real_durations = []  # Computational time (wall-clock time)
    # ----------------------------

    # Progress bar initialization
    global_start_real_time = time.time()
    last_makespan = 0.0

    # 4. Main Episode Loop
    for episode in range(1, NUM_EPISODES + 1):

        obs = env.reset(seed=episode)

        done = False
        step_count = 0
        episode_reward = 0

        # Capture info for final time
        info = {}

        start_real_time = time.time()

        # Inner Step Loop
        while not done:
            step_count += 1

            # A. Get Valid Actions
            current_mask = env._get_agent_mask()

            # B. Select Action (Policy)
            action = random_policy(current_mask)

            # C. Step
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward

            # The verbose in-step print statement has been removed to ensure the progress bar is not overwritten.

        # Episode Complete
        real_duration = time.time() - start_real_time
        delivery_makespan = info.get('current_time', 0.0)
        last_makespan = delivery_makespan

        # --- Store Metric Data ---
        episode_indices.append(episode)
        all_total_rewards.append(episode_reward)
        all_makespans.append(delivery_makespan)
        all_real_durations.append(real_duration)
        # -------------------------

        # --- DISPLAY PROGRESS BAR ---
        display_progress_bar(episode, NUM_EPISODES, global_start_real_time, last_makespan)

        # Print detailed summary only every 100 episodes or at the end
        if episode % 100 == 0 or episode == NUM_EPISODES:
            sys.stdout.write('\n')  # Move to a new line before printing summary
            print(f"--- Episode {episode} Finished ---")
            print(f"    Steps: {step_count}")
            print(f"    Total Reward: {episode_reward:.2f}")
            print(f"    Makespan (Sim): {delivery_makespan:.2f}s")
            print(f"    Duration (Real): {real_duration:.4f}s")

    # 5. Final Summary
    sys.stdout.write('\n')  # Final new line after the loop
    print("\n=========================================================")
    print(f"=== Training Summary ({NUM_EPISODES} Episodes) ===")
    print(f"  Mean Total Reward: {np.mean(all_total_rewards):.2f}")
    print(f"  Mean Makespan:     {np.mean(all_makespans):.2f}s")
    print(f"  Mean Real Duration:{np.mean(all_real_durations):.4f}s")
    print("=========================================================")

    env.close()

    # 6. Plotting Results
    plot_results(episode_indices, all_total_rewards, all_makespans, all_real_durations)


def plot_results(episodes, total_rewards, makespans, real_durations):
    """
    Generates THREE separate figures for Total Reward, Makespan, and Real Duration.
    """

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
    plt.plot(episodes, makespans, color='r', linewidth=1, label='Makespan (Sim Time)')
    plt.xlabel('Episode')
    plt.ylabel('Simulation Time (s)')
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
    run_multi_episode_simulation()