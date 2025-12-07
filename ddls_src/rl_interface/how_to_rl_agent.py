import os
import sys
import numpy as np
import time

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment wrapper
# (Make sure this matches the filename where you saved the class above)
from ddls_src.rl_interface.rl_scenario import LogisticRLScenario


def random_policy(mask):
    """
    A simple policy that picks a random valid action from the mask.
    """
    valid_indices = np.where(mask)[0]
    if len(valid_indices) > 0:
        return np.random.choice(valid_indices)
    return 0  # Should not happen if mask logic is correct


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
    # We initialize it ONCE. reset() handles the episode restarts.
    env = LogisticRLScenario(sim_config, visualize=False)

    # 3. Define Training Parameters
    NUM_EPISODES = 100
    all_episode_rewards = []
    all_episode_steps = []

    # 4. Main Episode Loop
    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n--- Starting Episode {episode}/{NUM_EPISODES} ---")

        # Reset environment for the new episode
        obs = env.reset(seed=episode)  # Optional: Pass seed for reproducibility

        done = False
        step_count = 0
        episode_reward = 0

        start_time = time.time()

        # Inner Step Loop
        while not done:
            step_count += 1

            # A. Get Valid Actions
            # For the first step we query env, for subsequent steps we could use 'info'
            current_mask = env._get_agent_mask()

            # B. Select Action (Policy)
            action = random_policy(current_mask)

            # C. Step
            obs, reward, done, info = env.step(action)

            episode_reward += reward

            # Optional: Print progress every N steps
            if step_count % 10 == 0:
                print(f"Ep {episode} | Step {step_count} | Time {info['current_time']:.0f}s | Reward {episode_reward}")

        # Episode Complete
        duration = time.time() - start_time
        all_episode_rewards.append(episode_reward)
        all_episode_steps.append(step_count)

        print(f"--- Episode {episode} Finished ---")
        print(f"    Steps: {step_count}")
        print(f"    Total Reward: {episode_reward}")
        print(f"    Real-world Duration: {duration:.2f}s")

    # 5. Final Summary
    print("\n=========================================================")
    print(f"=== Training Summary ({NUM_EPISODES} Episodes) ===")
    print(f"  Average Reward: {np.mean(all_episode_rewards):.2f}")
    print(f"  Average Steps:  {np.mean(all_episode_steps):.2f}")
    print("=========================================================")

    env.close()


if __name__ == "__main__":
    run_multi_episode_simulation()