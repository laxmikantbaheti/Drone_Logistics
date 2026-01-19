import os

# Local Imports
from ddls_src.scenarios.scenario import LogisticsScenario
from ddls_src.config.automatic_logic_maps import AUTOMATIC_LOGIC_CONFIG
from ddls_src.actions.action_enums import SimulationAction

def run_framework_demo():
    """
    A clean, professional script that demonstrates the full, refactored framework.
    It configures and runs the self-contained LogisticsScenario.
    This version now uses the RandomDataGenerator to create a larger, more complex scenario.
    """
    print("======================================================")
    print("=== Howto: Running the Full Logistics Framework    ===")
    print("======================================================")

    # 1. Configure the automatic logic for this experiment
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.ASSIGN_ORDER_TO_TRUCK] = False
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.TRUCK_TO_NODE] = True
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.LOAD_TRUCK_ACTION] = True
    AUTOMATIC_LOGIC_CONFIG[SimulationAction.UNLOAD_TRUCK_ACTION] = True

    # 2. Define the main simulation configuration for a larger, random scenario
    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 300.0,
        "data_loader_config": {
            "generator_type": "random", # <-- Use the random generator
            "generator_config": {
                "base_scale_factor": 50, # <-- Increased for a much larger scenario
                "scaling_factors": {
                    "nodes": 4.0,        # <-- More nodes
                    "depots": 0.1,
                    "customers": 1.5,
                    "micro_hubs": 0.2,
                    "trucks": 0.5,       # <-- More trucks
                    "drones": 0.4,
                    "initial_orders": 5.0 # <-- More initial orders
                },
                "grid_size": [100.0, 100.0],
            }
        },
        "new_order_config": {
            "arrival_schedule": { "900.0": 5, "1800.0": 10 } # More dynamic orders
        }
    }

    # 3. Instantiate the Scenario, passing the configuration to it.
    scenario = LogisticsScenario(p_cycle_limit=200, # <-- Increased cycle limit for larger scenario
                                 p_visualize=False,
                                 p_logging=False,
                                 config=sim_config)

    # 4. Run the scenario
    print("\n--- Starting Scenario Run ---")
    scenario.run()
    print("\n--- Scenario Finished ---")


if __name__ == "__main__":
    run_framework_demo()
