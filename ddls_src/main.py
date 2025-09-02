# In file: ddls_src/main.py

import json
import os
import random
import numpy as np

# Import the main simulation class and the SimulationAction enum
from ddls_src.core.logistics_simulation import LogisticsSimulation
from ddls_src.actions.action_enums import SimulationAction


def run_simulation_with_visualization():
    """
    Main execution script for the logistics simulation with live visualization.
    """
    # Define paths to configuration files
    config_file_path = os.path.join(os.path.dirname(__file__), 'config', 'default_simulation_config.json')

    # Load main simulation configuration
    with open(config_file_path, 'r') as f:
        sim_config = json.load(f)
    print(f"Loaded simulation config from: {config_file_path}")

    # Instantiate the LogisticsSimulation
    simulation = LogisticsSimulation(sim_config)

    # Initialize the simulation (loads entities, sets up managers, etc.)
    simulation.initialize_simulation()

    # --- Simulation Loop ---
    print("\n--- Starting Simulation Loop with Live Visualization ---")
    current_time = simulation.time_manager.get_current_time()
    simulation_end_time = sim_config.get("simulation_end_time", 3600)

    # Prepare the shared dictionary for plotting data
    figure_data = {}

    # Initialize the plot with the static network data
    simulation.network.initialize_plot_data(figure_data)

    step_count = 0
    while current_time < simulation_end_time:
        step_count += 1
        print(f"\n--- Simulation Step {step_count} (Time: {current_time:.2f}s) ---")

        # --- Decision Loop Phase ---
        simulation.start_decision_loop()

        while simulation.is_decision_loop_active():
            # --- Dummy Agent Logic ---
            current_mask = simulation.get_current_mask()
            valid_action_indices = np.where(current_mask)[0]

            if len(valid_action_indices) == 0:
                chosen_action_index = simulation.action_map.get((SimulationAction.NO_OPERATION,))
                if chosen_action_index is None:
                    simulation._decision_loop_active = False
                    continue
            else:
                chosen_action_index = random.choice(valid_action_indices)

            action_executed = simulation.process_agent_micro_action(chosen_action_index)
            action_tuple = simulation.action_manager._reverse_action_map.get(chosen_action_index, "N/A")
            print(f"  - Agent chose index {chosen_action_index} ({action_tuple}). Executed: {action_executed}")

        # --- Progression Phase ---
        raw_outcomes = simulation.advance_main_timestep()
        current_time = simulation.time_manager.get_current_time()
        print(f"Simulation advanced to time: {current_time:.2f}s. Outcomes: {raw_outcomes}")

        # --- Update Plotting ---
        # Populate figure_data with the latest dynamic information
        simulation.get_current_global_state().update_plot_data(figure_data)

        # Call the network's update function to redraw the plot
        simulation.network.update_plot_data(figure_data)

    print("\n--- Simulation Finished ---")
    input("Press Enter to close the plot...")  # Keep the plot window open


if __name__ == "__main__":
    run_simulation_with_visualization()