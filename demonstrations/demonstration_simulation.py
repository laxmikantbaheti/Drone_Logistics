import os
from datetime import datetime
from ddls_src.scenarios.scenario import LogisticsScenario
import random


def run_scenario_demo():
    seed = 111
    random.seed(seed)

    print("==================================================================")
    print("=== Howto: Running Simplified Distance Matrix Scenario         ===")
    print("==================================================================")

    # 1. Simplified simulation configuration with direct entity counts and ranges
    sim_config = {
        "seed": 111,
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            "generator_type": "distance_matrix",
            "generator_config": {
                # Direct Entity Counts (No Scale Factors)
                "num_depots": 3,
                "num_customers": 10,
                "num_micro_hubs": 2,
                "num_trucks": 5,
                "num_initial_orders": 150,

                # Area and Ranges
                "area_x_range": (0.0, 100.0),
                "area_y_range": (0.0, 100.0),
                "truck_payload_range": [8, 16],
                "drone_payload_range": [1, 3]
            }
        }
    }

    # 2. Instantiate and run the Scenario
    scenario = LogisticsScenario(
        p_cycle_limit=250000,
        p_logging=False,
        config=sim_config,
        custom_log=False,
        plot_resuts=True,
    )

    print("\n--- Starting Scenario Run ---")
    start = datetime.now()
    print("Start time:", start)

    scenario.run()

    print("\n--- Scenario Finished ---")
    print(f"Total Truck Distance: {scenario._system.total_truck_distance:.2f}")
    print(f"Total Drone Distance: {scenario._system.total_drone_distance:.2f}")
    end = datetime.now()
    print("End time:", end)
    print("Total Execution Elapsed:", end - start)
    print(f"Constraint Latency: {sum(scenario._system.global_state.constraint_latency_per_step)}")
    print(f"Steps: {scenario.get_cycle_id()}")
    print("\n=============================================")
    print("=========   Validation Complete   =========")
    print("=============================================")


if __name__ == "__main__":
    run_scenario_demo()