import os
from datetime import datetime
from ddls_src.scenarios.scenario import LogisticsScenario
import random

def run_scenario_demo():
    # Set the seed for full reproducibility
    seed = 42
    random.seed(seed)
    """
    Demonstrates and validates the distance matrix-based movement functionality
    on a large-scale instance (>= 50 nodes) generated entirely in-memory.
    """
    print("==================================================================")
    print("=== Howto: Running Large Distance Matrix Scenario (50+ Nodes)  ===")
    print("==================================================================")

    # 1. Define the simulation configuration with direct in-memory generator parameters
    sim_config = {
        "seed": 45,
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            "generator_type": "distance_matrix",
            "generator_config": {
                "base_scale_factor": 10,
                "num_nodes": 60,  # Explicitly configured for >= 50 nodes
                "area_x_range": (0.0, 200.0),
                "area_y_range": (0.0, 200.0),
                "scaling_factors": {
                    "nodes": 6.0,
                    "depots": 0.3,       # ~3 depots
                    "customers": 4.5,    # ~45 customers
                    "micro_hubs": 0.6,   # ~6 micro-hubs (and 6 paired drones)
                    "trucks": 0.5,       # ~5 trucks
                    "initial_orders": 3.5 # ~35 orders
                },
                "truck_payload_range": [8, 16],
                "drone_payload_range": [1, 3],
                "truck_speed_range": [40.0, 80.0],
                "drone_speed_range": [25.0, 50.0],
                "initial_fuel_range": [100.0, 200.0],
                "initial_battery_range": [0.85, 1.0],
                "sla_min_hours": 1.5,
                "sla_max_hours": 6.0,
                "priority_distribution": {1: 0.6, 2: 0.3, 3: 0.1},
                "truck_fuel_consumption_rate": 0.08,
                "drone_battery_drain_rate_flying": 0.004,
                "drone_battery_drain_rate_idle": 0.0008,
                "drone_battery_charge_rate": 0.02,
                "drone_eligible_order_ratio": 0.45
            }
        }
    }

    # 2. Instantiate and run the Scenario
    scenario = LogisticsScenario(
        p_cycle_limit=250000,
        p_visualize=False,
        p_logging=False,
        config=sim_config,
        custom_log=True
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
    print("\n=============================================")
    print("=========   Validation Complete   =========")
    print("=============================================")


if __name__ == "__main__":
    run_scenario_demo()