import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment wrapper
from ddls_src.rl_interface.rl_scenario import LogisticRLScenario


def random_policy(mask):
    """
    A simple policy that picks a random valid action from the mask.
    """
    valid_indices = np.where(mask)[0]
    if len(valid_indices) > 0:
        return np.random.choice(valid_indices)
    return 0


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
    env = LogisticRLScenario(sim_config, visualize=False)

    # 3. Define Training Parameters
    NUM_EPISODES = 1

    # --- Storage for Plotting ---
    episode_indices = []
    all_total_rewards = []
    all_makespans = []  # Simulation time (logic time)
    all_real_durations = []  # Computational time (wall-clock time)
    # ----------------------------

    # 4. Main Episode Loop
    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n--- Starting Episode {episode}/{NUM_EPISODES} ---")

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

            if step_count % 10 == 0:
                print(f"Ep {episode} | Step {step_count} | Time {info['current_time']:.0f}s | Reward {episode_reward}")

        # Episode Complete
        real_duration = time.time() - start_real_time
        delivery_makespan = info['current_time']

        # --- Store Metric Data ---
        episode_indices.append(episode)
        all_total_rewards.append(episode_reward)
        all_makespans.append(delivery_makespan)
        all_real_durations.append(real_duration)  # <--- Added Real Duration
        # -------------------------

        print(f"--- Episode {episode} Finished ---")
        print(f"    Steps: {step_count}")
        print(f"    Total Reward: {episode_reward}")
        print(f"    Makespan (Sim): {delivery_makespan:.2f}s")
        print(f"    Duration (Real): {real_duration:.4f}s")

    # 5. Final Summary
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