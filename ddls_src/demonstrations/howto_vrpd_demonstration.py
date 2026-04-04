import os
from datetime import datetime, timedelta

# from dateutil.tz import datetime_ambiguous

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
    # config_file_path = os.path.join(script_path, '..', 'config', 'large_instance.json')
    # config_file_path = os.path.normpath(config_file_path)
    # Change this to match where your VRP-D instances are stored
    vrp_instance_path = os.path.join(
        script_path,
        "..",
        "scenarios",
        "vrp_d_instances",
        "VRP-D",
        "A-n32-k5-20.vrp"
    )
    vrp_instance_path = os.path.normpath(vrp_instance_path)
    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            # IMPORTANT: must match your generator factory registry name
            "generator_type": "vrpd",
            "generator_config": {
                "instance_path": vrp_instance_path,

                # Keep these custom for your delivery model
                "num_drones": 6,
                "num_microhubs": 2,
                "bbox": (0, 0, 100, 100),
                "std_dev_scale": 4.0,

                # Capacity-derived configs inside generator
                "drone_capacity_ratio": 0.2,

                # Speeds (if your sim uses them)
                "truck_speed": 1.0,
                "drone_speed": 1.0,

                # Optional
                "seed": 42,

                # If you want to override trucks instead of using -k# from filename:
                # "num_trucks": 5,
            }
        },
    }
    print("\n--- Scenario Config ---")
    print("Generator:", sim_config["data_loader_config"]["generator_type"])
    print("Instance:", sim_config["data_loader_config"]["generator_config"]["instance_path"])

    # 2. Instantiate and run the Scenario
    # The DummyAgent inside the scenario will automatically pick valid actions.
    # We can observe the console output to validate the timer-based movement.
    scenario = LogisticsScenario(p_cycle_limit=150000,
                                 p_visualize=False,
                                 p_logging=False, # Set to True to see detailed logs
                                 config=sim_config,
                                 custom_log = True)

    print("\n--- Starting Scenario Run ---")
    start = datetime.now()
    print("start---",start)
    scenario.run()
    print("\n--- Scenario Finished ---")
    end = datetime.now()
    print("end ---",end)
    print(end-start)
    print(scenario._system.global_state.current_time)
    print("\n=============================================")
    print("=========   Validation Complete   =========")
    print("=============================================")
    print("Check the console logs to see the truck's status change from 'en_route' to 'idle' after the timer expires.")


if __name__ == "__main__":
    run_matrix_scenario_demo()