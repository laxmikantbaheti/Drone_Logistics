import os

# Local Imports
from ddls_src.scenarios.scenario import LogisticsScenario

def run_framework_demo():
    """
    A clean, professional script that demonstrates the full, refactored framework.
    It configures and runs the self-contained LogisticsScenario.
    """
    print("======================================================")
    print("=== Howto: Running the Full Logistics Framework    ===")
    print("======================================================")

    # 1. Define the main simulation configuration
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

    # 2. Instantiate the Scenario, passing the configuration to it.
    # The scenario will handle the setup of the LogisticsSystem internally.
    scenario = LogisticsScenario(p_cycle_limit=15,
                                 p_visualize=False,
                                 p_logging=True,
                                 config=sim_config)

    # 3. Run the scenario
    # The .run() method is inherited from MLPro's ScenarioBase and starts the simulation loop.
    print("\n--- Starting Scenario Run ---")
    scenario.run()
    print("\n--- Scenario Finished ---")


if __name__ == "__main__":
    run_framework_demo()
