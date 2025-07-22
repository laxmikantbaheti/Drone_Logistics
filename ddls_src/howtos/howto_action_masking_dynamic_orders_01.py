import numpy as np
from typing import Dict, Any
import os  # <-- Import the os module

# MLPro Imports
from mlpro.bf.systems import Action

# Local Imports
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.actions.action_enums import SimulationAction


def run_dynamic_masking_demo():
    """
    Demonstrates the full, refactored framework with event-driven dynamic
    action space and constraint updates.
    """
    print("======================================================")
    print("=== Howto: Dynamic Action Masking Demonstration  ===")
    print("======================================================")

    # 1. Define the main simulation configuration
    # This config includes a schedule for the OrderGenerator

    # --- FIX: Build a robust, absolute path to the config file ---
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data.json')
    config_file_path = os.path.normpath(config_file_path)  # Clean up path (e.g., ..\)

    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 300.0,  # 5 minutes per step
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {
                "file_path": config_file_path
            }
        },
        "new_order_config": {
            # At time=900s (cycle 3), 1 new order will arrive.
            # At time=1800s (cycle 6), 2 new orders will arrive.
            "arrival_schedule": {"900.0": 1, "1800.0": 2}
        }
    }

    # 2. Instantiate the main LogisticsSystem
    # This will set up all entities, managers, and the event-driven mapper
    print("\n[Step 1] Initializing the LogisticsSystem...")
    logistics_system = LogisticsSystem(p_id='logsys_001',
                                       p_visualize=False,
                                       p_logging=True,
                                       config=sim_config)
    print("  - LogisticsSystem and all sub-components initialized.")

    # 3. Simulation Loop
    num_cycles = 100  # <--- UPDATED to 100 cycles
    print(f"\n[Step 2] Starting simulation loop for {num_cycles} cycles...")

    for i in range(num_cycles):
        current_time = logistics_system.time_manager.get_current_time()
        print(f"\n\n------------------- SIMULATION CYCLE {i + 1} (Time: {current_time}s) -------------------")

        # Get the current state and mask
        current_state = logistics_system.get_state()
        system_mask = logistics_system.get_current_mask()

        print("\n*** Current High-Level State ***")
        print(f"  - Total Orders: {current_state.get_value('total_orders')}")
        print(f"  - Delivered Orders: {current_state.get_value('delivered_orders')}")
        print("********************************")

        # Find a valid action for the agent to take
        valid_system_actions = np.where(system_mask)[0]

        print(f"\n  - Action Masker reports {len(valid_system_actions)} valid system actions.")

        if len(valid_system_actions) > 0:
            # For this demo, the agent just picks the first valid action
            action_to_take_idx = valid_system_actions[0]
            action_obj = Action(p_action_space=logistics_system.get_action_space(), p_values=[action_to_take_idx])

            action_tuple = logistics_system.action_manager._reverse_action_map.get(action_to_take_idx)
            print(f"  - Agent chose to act: {action_tuple}")

            # Simulate one step
            logistics_system.simulate_reaction(p_state=current_state, p_action=action_obj)

        else:
            print("  - Agent has no valid actions. Simulating a 'NO_OPERATION' step.")
            # Create a NO_OP action to advance time
            no_op_idx = logistics_system.action_map.get((SimulationAction.NO_OPERATION,))
            action_obj = Action(p_action_space=logistics_system.get_action_space(), p_values=[no_op_idx])
            logistics_system.simulate_reaction(p_state=current_state, p_action=action_obj)

    print("\n=============================================")
    print("========= Demonstration Complete ==========")
    print("=============================================")


if __name__ == "__main__":
    run_dynamic_masking_demo()
