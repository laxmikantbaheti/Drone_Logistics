import json
import os
import random  # For dummy agent actions
import numpy as np  # For np.where in dummy agent

# Import the main simulation class and the SimulationAction enum
from ddls_src.core.logistics_simulation import LogisticsSimulation
from ddls_src.actions.action_enums import SimulationAction


# --- Placeholder for your plotting framework integration ---
# This is where you would define or import your plotting functions
# that consume the `figure_data` dictionary.
def initialize_plotting_framework(figure_data: dict):
    """
    Placeholder function to initialize your external plotting framework.
    This function would set up the initial figure based on the data.
    """
    print("\n--- Initializing Plotting Framework ---")
    print(f"Initial figure data keys: {figure_data.keys()}")
    # Example: Call your plotting library's figure creation and initial data setting
    # fig = my_plotting_lib.create_figure()
    # my_plotting_lib.draw_network(fig, figure_data['network_nodes'], figure_data['network_edges'])
    # my_plotting_lib.draw_static_elements(fig, figure_data['node_details'], figure_data['edge_details'])
    # You might store the figure object globally or return it for later updates
    print("Plotting framework initialized with static data.")


def update_plotting_framework(figure_data: dict):
    """
    Placeholder function to update your external plotting framework.
    This function would refresh the plot with dynamic data.
    """
    print(f"\n--- Updating Plotting Framework at time {figure_data.get('current_time', 'N/A')} ---")
    # Example: Update existing plot elements
    # my_plotting_lib.update_vehicle_positions(figure_data['vehicle_positions'])
    # my_plotting_lib.update_parcel_locations(figure_data['parcel_locations'])
    # my_plotting_lib.update_edge_traffic(figure_data['network_edges_dynamic'])
    # my_plotting_lib.update_dashboard_metrics(figure_data['supply_chain_overview'], figure_data['fleet_overview'])
    # my_plotting_lib.refresh_plot(fig)
    print("Plotting framework updated with dynamic data.")


# --- Main Simulation Execution ---
if __name__ == "__main__":
    # Define paths to configuration files
    # Ensure this path is correct relative to where you run main.py
    config_file_path = os.path.join(os.path.dirname(__file__), 'config', 'default_simulation_config.json')

    # Load main simulation configuration
    with open(config_file_path, 'r') as f:
        sim_config = json.load(f)
    print(f"Loaded simulation config from: {config_file_path}")

    # Instantiate the LogisticsSimulation
    # The DataLoader config is now directly part of sim_config
    simulation = LogisticsSimulation(sim_config)

    # Initialize the simulation (this will load entities via DataLoader/ScenarioGenerator, set up managers, etc.)
    simulation.initialize_simulation()

    # --- Simulation Loop ---
    print("\n--- Starting Simulation Loop ---")
    current_time = simulation.time_manager.get_current_time()
    simulation_end_time = sim_config.get("simulation_end_time", 3600)  # Default to 1 hour if not in config
    main_timestep_duration = sim_config.get("main_timestep_duration", 300)

    # Prepare initial plotting data
    figure_data = {}
    # Call initialize_plot_data on relevant components. GlobalState's call will cascade to entities.
    simulation.get_current_global_state().initialize_plot_data(figure_data)
    simulation.data_loader.initialize_plot_data(figure_data)
    simulation.scenario_generator.initialize_plot_data(figure_data)
    simulation.network.initialize_plot_data(figure_data)  # Network's plotting is separate from GlobalState's
    simulation.supply_chain_manager.initialize_plot_data(figure_data)
    simulation.resource_manager.initialize_plot_data(
        figure_data)  # This will call Fleet/MicroHubs managers' init_plot_data
    simulation.network_manager.initialize_plot_data(figure_data)

    initialize_plotting_framework(figure_data)  # Call your framework's initializer

    step_count = 0
    while current_time < simulation_end_time:
        step_count += 1
        print(f"\n--- Simulation Step {step_count} (Time: {current_time:.2f}s) ---")

        # --- Decision Loop Phase (Agent's turn to act) ---
        simulation.start_decision_loop()
        micro_action_count = 0
        max_micro_actions_per_step = 5  # Limit micro-actions to prevent infinite loops for dummy agent

        while simulation.is_decision_loop_active() and micro_action_count < max_micro_actions_per_step:
            micro_action_count += 1

            # --- Dummy Agent Logic ---
            # This is a very simple random agent that picks a valid action.
            current_mask = simulation.get_current_mask()
            valid_action_indices = np.where(current_mask)[0]

            if len(valid_action_indices) == 0:
                print("  No valid actions available for dummy agent. Forcing NO_OPERATION.")
                # Force NO_OPERATION if no other valid actions
                no_op_index = simulation.action_map.get((SimulationAction.NO_OPERATION,))
                if no_op_index is not None:
                    simulation.process_agent_micro_action(no_op_index)
                else:
                    # Fallback if NO_OPERATION isn't mapped (should not happen if action_mapping is correct)
                    simulation._decision_loop_active = False  # Force end
                break  # Exit micro-action loop

            # Choose a random valid action
            chosen_action_index = random.choice(valid_action_indices)

            # Optionally, force NO_OPERATION after a few actions to terminate the loop
            # This makes the decision loop finite for the dummy agent.
            if micro_action_count >= max_micro_actions_per_step:
                no_op_index = simulation.action_map.get((SimulationAction.NO_OPERATION,))
                if no_op_index is not None:
                    chosen_action_index = no_op_index
                else:
                    print("Warning: NO_OPERATION not found in action map, cannot force end decision loop.")
                    simulation._decision_loop_active = False  # Force end

            action_executed = simulation.process_agent_micro_action(chosen_action_index)
            print(
                f"  Micro-Action {micro_action_count}: Agent chose index {chosen_action_index} ({simulation.action_manager._reverse_action_map.get(chosen_action_index)}). Executed: {action_executed}")

            # If the action was invalid, the agent might try again.
            # For this simple loop, we just continue.
            if not action_executed:
                print(f"  Action was invalid. Dummy agent will try another valid action (if available).")
                # In a real RL loop, the agent would get a negative reward or observation indicating invalidity
                # and re-select. For this basic run, we'll just let it try again or eventually hit NO_OP.

        # Ensure decision loop is actually inactive before advancing main timestep
        if simulation.is_decision_loop_active():
            print("Warning: Decision loop still active after max micro-actions. Forcing end.")
            no_op_index = simulation.action_map.get((SimulationAction.NO_OPERATION,))
            if no_op_index is not None:
                simulation.process_agent_micro_action(no_op_index)
            else:
                simulation._decision_loop_active = False

        # --- Progression Phase ---
        raw_outcomes = simulation.advance_main_timestep()
        current_time = simulation.time_manager.get_current_time()
        print(f"Simulation advanced to time: {current_time:.2f}s. Outcomes: {raw_outcomes}")

        # --- Update Plotting ---
        # Update GlobalState's current_time for plotting consistency
        figure_data['current_time'] = current_time
        simulation.get_current_global_state().update_plot_data(figure_data)
        # Call update_plot_data on other managers/generators if they have dynamic elements
        simulation.data_loader.update_plot_data(figure_data)  # Usually no dynamic updates
        simulation.scenario_generator.update_plot_data(figure_data)  # Usually no dynamic updates
        simulation.network.update_plot_data(figure_data)  # Network's plotting is separate from GlobalState's
        simulation.supply_chain_manager.update_plot_data(figure_data)
        simulation.resource_manager.update_plot_data(
            figure_data)  # This will call Fleet/MicroHubs managers' update_plot_data
        simulation.network_manager.update_plot_data(figure_data)

        update_plotting_framework(figure_data)  # Call your framework's updater

        # Optional: Add a small delay for visualization if running quickly
        # import time
        # time.sleep(0.1)

    print("\n--- Simulation Finished ---")
    print(f"Total simulation time: {current_time:.2f}s")
    print(f"Total steps: {step_count}")

