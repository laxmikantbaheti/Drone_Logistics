import os
import numpy as np

# Local Imports
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.config.automatic_logic_maps import AUTOMATIC_LOGIC_CONFIG
from ddls_src.actions.base import SimulationAction
from ddls_src.core.basics import LogisticsAction


def run_system_directly_demo():
    """
    A 'howto' script that demonstrates how to run the LogisticsSystem directly,
    without using a Scenario class. This script manages the simulation loop itself.
    """
    print("======================================================")
    print("=== Howto: Running the LogisticsSystem Directly    ===")
    print("======================================================")

    # 1. Configure the automatic logic for this experiment
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.ASSIGN_ORDER_TO_TRUCK] = False
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.TRUCK_TO_NODE] = True
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.LOAD_TRUCK_ACTION] = True
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.UNLOAD_TRUCK_ACTION] = True

    # 2. Define the main simulation configuration
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 300.0,
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {"file_path": config_file_path}
        },
        "new_order_config": {
            "arrival_schedule": {"900.0": 1, "1800.0": 2}
        }
    }

    # 3. Instantiate the LogisticsSystem directly
    print("\n--- Initializing LogisticsSystem ---")
    logistics_system = LogisticsSystem(p_id='logsys_001',
                                       p_visualize=False,
                                       p_logging=False,
                                       config=sim_config)
    print("--- Initialization Complete ---")

    # 4. The Main Simulation Loop
    num_cycles = 20
    print(f"\n--- Starting Manual Simulation Loop for {num_cycles} cycles ---")

    for i in range(num_cycles):
        current_time = logistics_system.time_manager.get_current_time()
        print(f"\n\n------------------- SIMULATION CYCLE {i + 1} (Time: {current_time}s) -------------------")

        # Get the current state and action mask from the system
        current_state = logistics_system.get_state()
        system_mask = logistics_system.get_current_mask()

        print("\n*** Current High-Level State ***")
        print(f"  - Total Orders: {current_state.get_value(current_state.get_related_set().get_dim_by_name("total_orders").get_id())}")
        print(f"  - Delivered Orders: {current_state.get_value(current_state.get_related_set().get_dim_by_name('delivered_orders').get_id())}")
        print("********************************")

        # Simple "dummy agent" logic: pick the first valid action
        valid_actions = np.where(system_mask)[0]
        action_to_take_idx = -1

        if len(valid_actions) > 0:
            action_to_take_idx = valid_actions[0]
            action_tuple = logistics_system._reverse_action_map.get(action_to_take_idx)
            print(f"\n  - Policy chose action: {action_tuple}")
        else:
            print("\n  - Policy has no valid actions, choosing NO_OPERATION.")
            action_to_take_idx = logistics_system.action_map.get((SimulationAction.NO_OPERATION,))

        # Create the action object
        action_obj = LogisticsAction(p_action_space=logistics_system.get_action_space(), p_values=[action_to_take_idx])

        # Call simulate_reaction to advance the simulation by one step
        logistics_system.simulate_reaction(p_state=current_state, p_action=action_obj)

    print("\n--- Simulation Loop Finished ---")


if __name__ == "__main__":
    run_system_directly_demo()
