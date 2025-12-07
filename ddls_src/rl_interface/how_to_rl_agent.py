import os
import sys
import numpy as np
import time

# Ensure the project root is in the python path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment wrapper
# Ensure your wrapper file is named 'logistic_rl_scenario.py'
from ddls_src.rl_interface.rl_scenario import LogisticRLScenario


def random_policy(mask):
    """
    A simple policy that picks a random valid action from the mask.
    In a real training scenario, this would be your PPO/DQN model.
    """
    # np.where returns a tuple, we take the first element (the array of indices)
    valid_indices = np.where(mask)[0]

    if len(valid_indices) > 0:
        return np.random.choice(valid_indices)

    # Fallback: This case should ideally not happen if the environment
    # logic correctly finds a state with available agent actions.
    return 0


def run_main_simulation():
    print("=========================================================")
    print("===   Running LogisticRLScenario (Step-Logic Demo)    ===")
    print("=========================================================")

    # 1. Define the simulation configuration
    script_path = os.path.dirname(os.path.realpath(__file__))
    # Point to the new matrix-specific data file
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data_mh_matrix.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "movement_mode":"matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 300.0,  # 5 minutes per internal step
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {
                "file_path": config_file_path
            }
        },
        }


    # 2. Initialize Environment
    # Set visualize=True to see the matplotlib plots
    env = LogisticRLScenario(sim_config, visualize=False)

    # 3. Reset Environment
    # Note: With your logic, this might return t=0. The first step() will then
    # fast-forward to the first actual decision point.
    obs = env.reset()

    done = False
    step_count = 0
    total_reward = 0

    print(f"Initial Observation: {obs}")

    # 4. Main RL Loop
    while not done:
        step_count += 1

        # --- A. Get Valid Actions (Masking) ---
        # We access the mask from the system (or info dict if available from previous step)
        # For the very first step, we query the env directly.
        current_mask = env._get_agent_mask()  # Using helper method for demo purposes

        # --- B. Select Action ---
        action = random_policy(current_mask)

        # --- C. Step Environment ---
        # This call enters your 'while' loop logic:
        # It will Auto-Execute -> Advance Time -> Repeat until Agent Action is needed.
        obs, reward, done, info = env.step(action)

        total_reward += reward

        # --- D. Logging ---
        # print(f"\n[Step {step_count}]")
        # print(f"  Agent Action Selected: {action}")
        # print(f"  System Time Reached:   {info['current_time']:.1f}s")
        # print(f"  Reward Received:       {reward}")
        # print(f"  Observation (Orders):  Total={obs[0]}, Delivered={obs[1]}")

        # Optional: Slow down to watch visualization
        # time.sleep(0.5)

    print("\n=========================================================")
    print(f"=== Simulation Finished. Total Steps: {step_count}, Reward: {total_reward} ===")
    print("=========================================================")

    env.close()


if __name__ == "__main__":
    run_main_simulation()