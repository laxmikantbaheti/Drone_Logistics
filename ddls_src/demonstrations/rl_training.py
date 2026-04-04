import numpy as np
import os
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.config.automatic_logic_maps import AUTOMATIC_LOGIC_CONFIG
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.rl_interface.gym_env import LogisticsEnv


def run_rl_agent():
    print("--- Initializing RL Environment ---")

    # 1. Configure the Simulation
    # We disable high-level logic so the Agent has full control
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.ASSIGN_ORDER_TO_TRUCK] = False
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.ASSIGN_ORDER_TO_DRONE] = False
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.TRUCK_TO_NODE] = True  # Keep routing auto for now
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.LOAD_TRUCK_ACTION] = True  # Keep loading auto
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.UNLOAD_TRUCK_ACTION] = True

    # Path to your data config
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data_mh_matrix.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 30.0,
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {"file_path": config_file_path}
        },
        "new_order_config": {
            "arrival_schedule": {"900.0": 1}
        }
    }

    # 2. Create System and Wrapper
    system = LogisticsSystem(config=sim_config, p_logging=False)
    env = LogisticsEnv(system)

    print("--- Starting Training Loop (Random Agent) ---")

    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # 3. Agent Logic
        # (Replace this with PPO/DQN predict later)

        # A. Get Mask
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]

        # B. Pick Action
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            # Fallback (shouldn't happen if mask is correct)
            print("Warning: No valid actions found!")
            break

        # C. Step Environment
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

        # Logging
        action_tuple = system._reverse_action_map.get(action)
        if action_tuple[0] != SimulationAction.NO_OPERATION:
            print(f"Step {steps}: Action {action} {action_tuple} | Reward: {reward:.2f}")
        else:
            # Don't spam log with NO_OPs
            pass

    print(f"--- Episode Finished ---")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward}")


if __name__ == "__main__":
    run_rl_agent()