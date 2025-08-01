import os

# Local Imports
from ddls_src.scenarios.assignment_scenario import AssignmentScenario
from ddls_src.config.automatic_logic_maps import AUTOMATIC_LOGIC_CONFIG
from ddls_src.actions.action_enums import SimulationAction

def run_assignment_demo():
    """
    A clean script that demonstrates the "assignment-only" research design.
    It configures the automatic logic and runs the self-contained AssignmentScenario.
    """
    print("======================================================")
    print("=== Howto: Running an Assignment-Only Agent        ===")
    print("======================================================")

    # 1. Configure the automatic logic for this experiment
    # The agent is responsible for assignments, so we set those to False.
    # The environment will handle everything else automatically.
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.ASSIGN_ORDER_TO_TRUCK] = False
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.ASSIGN_ORDER_TO_DRONE] = False
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.TRUCK_TO_NODE] = True
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.LOAD_TRUCK_ACTION] = True
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.UNLOAD_TRUCK_ACTION] = True
    # ... other actions can be configured as needed

    # 2. Define the main simulation configuration
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 300.0,
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": { "file_path": config_file_path }
        },
        "new_order_config": {
            "arrival_schedule": { "900.0": 1, "1800.0": 2 }
        }
    }

    # 3. Instantiate and run the Scenario
    scenario = AssignmentScenario(p_cycle_limit=20,
                                  p_visualize=False,
                                  p_logging=True,
                                  config=sim_config)

    print("\n--- Starting Scenario Run ---")
    scenario.run()
    print("\n--- Scenario Finished ---")


if __name__ == "__main__":
    run_assignment_demo()
