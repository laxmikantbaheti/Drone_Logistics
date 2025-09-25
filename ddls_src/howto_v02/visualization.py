# In file: ddls_src/howtos/howto_visual_scenario.py

import os
from ddls_src.scenarios.scenario import LogisticsScenario

def run_visual_demo():
    """
    A clean script that demonstrates the full framework with live visualization.
    """
    print("======================================================")
    print("=== Howto: Running the Visual Logistics Scenario   ===")
    print("======================================================")

    # 1. Define the simulation configuration
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 30.0,
        "simulation_end_time": 72000.0, # Run for 2 hours
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {"file_path": config_file_path}
        },
        # "new_order_config": {
        #     "arrival_schedule": { "900.0": 2, "1800.0": 3, "3600.0": 4 }
        # }
    }

    # 2. Instantiate and run the Scenario with visualization enabled
    scenario = LogisticsScenario(p_cycle_limit=100000,
                                 p_visualize=True, # <-- IMPORTANT
                                 p_logging=False,
                                 config=sim_config)

    print("\n--- Starting Scenario Run ---")
    scenario.run()
    print("\n--- Scenario Finished ---")

if __name__ == "__main__":
    run_visual_demo()