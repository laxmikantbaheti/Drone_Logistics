import numpy as np
from typing import Dict, Any
import os

# MLPro Imports
from mlpro.bf.systems import Action

# Local Imports
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.actions.action_enums import SimulationAction


def run_logistics_system_demo():
    """
    Demonstrates the full, refactored framework by running the top-level LogisticsSystem.
    This script initializes the complete simulation and runs it for a set number of cycles,
    showing how the system state and action masks evolve.
    """
    print("======================================================")
    print("=== Howto: Running the Full LogisticsSystem        ===")
    print("======================================================")

    # 1. Define the main simulation configuration
    # This config includes a schedule for the OrderGenerator
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data.json')
    config_file_path = os.path.normpath(config_file_path)

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
    print("\n[Step 1] Initializing the LogisticsSystem...")
    logistics_system = LogisticsSystem(p_id='logsys_001',
                                       p_visualize=False,
                                       p_logging=True,
                                       config=sim_config)
    print("  - LogisticsSystem and all sub-components initialized.")

    # 3. Simulation Loop
    num_cycles = 15
    print(f"\n[Step 2] Starting simulation loop for {num_cycles} cycles...")

    for i in range(num_cycles):
        current_time = logistics_system.time_manager.get_current_time()
        print(f"\n\n------------------- SIMULATION CYCLE {i + 1} (Time: {current_time}s) -------------------")

        # Get the current state and mask directly from the system
        current_state = logistics_system.get_state()
        system_mask = logistics_system.get_current_mask()

        print("\n*** Current High-Level State ***")
        print(f"  - Total Orders: {current_state.get_value('total_orders')}")
        print(f"  - Delivered Orders: {current_state.get_value('delivered_orders')}")
        print("********************************")

        # Find a valid action for the agent to take
        valid_system_actions = np.where(system_mask)[0]

        print(f"\n  - ActionMasker reports {len(valid_system_actions)} valid system actions.")

        action_to_take_idx = -1
        if len(valid_system_actions) > 0:
            # For this demo, the agent just picks the first valid action
            action_to_take_idx = valid_system_actions[0]
            # Reverse map lookup requires the full action manager, which is now internal
            # We'll just print the index for now.
            print(f"  - Agent chose to act with action index: {action_to_take_idx}")
        else:
            print("  - Agent has no valid actions. Simulating a 'NO_OPERATION' step.")
            # Get the index for the NO_OPERATION action
            action_to_take_idx = logistics_system.action_map.get((SimulationAction.NO_OPERATION,))

        # Create the action object and simulate one step
        action_obj = Action(p_action_space=logistics_system.get_action_space(), p_values=[action_to_take_idx])
        logistics_system.simulate_reaction(p_state=current_state, p_action=action_obj)

    print("\n=============================================")
    print("========= Demonstration Complete ==========")
    print("=============================================")


if __name__ == "__main__":
    run_logistics_system_demo()
