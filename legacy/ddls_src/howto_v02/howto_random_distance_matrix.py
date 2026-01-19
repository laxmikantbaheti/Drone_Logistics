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
        "movement_mode": "matrix",  # <-- IMPORTANT: Tell the Network class to use the matrix
        "initial_time": 0.0,
        "main_timestep_duration": 30.0,
        "simulation_end_time": 3600.0,
        "data_loader_config": {
            "generator_type": "distance_matrix",  # <-- Specify the new generator type
            "generator_config": {
                # Config for the DistanceMatrixDataGenerator
                "base_scale_factor": 50,
                "num_nodes": 20,  # Optionally define exact number of nodes
                # "distance_matrix": { ... }, # Optionally provide your own matrix
                "max_travel_time": 1800.0,  # Used if generating matrix randomly (max seconds)
                "scaling_factors": {  # Used to determine ratios of depots, customers etc.
                    "depots": 0.1,
                    "customers": 1.5,
                    "micro_hubs": 0.2,
                    "trucks": 0.2,
                    "drones": 0.3,
                    "initial_orders": 2.0
                },
                # ... other vehicle/order parameters ...
                "truck_payload_range": [5, 15],
                "sla_min_hours": 0.5,
                "sla_max_hours": 1.5,
            }
        },
        # ... other config like new_order_config ...
    }

    # 2. Instantiate and run the Scenario
    # The DummyAgent inside the scenario will automatically pick valid actions.
    # We can observe the console output to validate the timer-based movement.
    scenario = LogisticsScenario(p_cycle_limit=150000,
                                 p_visualize=False,
                                 p_logging=False, # Set to True to see detailed logs
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