import os
from ddls_src.scenarios.scenario import LogisticsScenario

def run_matrix_scenario_demo():
    """
    Demonstrates and validates the distance matrix-based movement functionality
    by running the standard LogisticsScenario with a specific configuration.
    """
    print("======================================================")
    print("=== Howto: Running a Distance Matrix Scenario      ===")
    print("======================================================")

    # 1. Define the simulation configuration
    script_path = os.path.dirname(os.path.realpath(__file__))
    # Point to the new matrix-specific data file
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data_mh_matrix.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,  # 1-second steps for clear validation
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {"file_path": config_file_path}
        }
    }

    # 2. Instantiate and run the Scenario
    # The DummyAgent inside the scenario will automatically pick valid actions.
    # We can observe the console output to validate the timer-based movement.
    scenario = LogisticsScenario(p_cycle_limit=150000,
                                 p_visualize=False,
                                 p_logging=True, # Set to True to see detailed logs
                                 config=sim_config)

    print("\n--- Starting Scenario Run ---")
    scenario.run()
    print("\n--- Scenario Finished ---")

    print("\n=============================================")
    print("=========   Validation Complete   =========")
    print("=============================================")
    print("Check the console logs to see the truck's status change from 'en_route' to 'idle' after the timer expires.")


if __name__ == "__main__":
    run_matrix_scenario_demo()