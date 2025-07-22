# Import the main system and the scenario runner
from ...core.logistics_system import LogisticsSystem
from ..scenarios.scenario import LogisticsScenario

# This 'howto' script demonstrates the full refactored architecture.
# Note: This script assumes that all the refactored entity and manager classes
# are correctly implemented and imported by the LogisticsSystem.

def run_howto():
    """
    Sets up and runs the logistics simulation with the assignment-only agent design.
    """
    print("Starting 'howto_run_assignment_agent.py'...")

    # 1. Define the main simulation configuration
    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 300.0, # 5 minutes per step
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {
                # Make sure this path is correct relative to where you run the script
                "file_path": "ddls_src/config/initial_entity_data.json"
            }
        }
    }

    # 2. Instantiate the main LogisticsSystem
    # The LogisticsSystem now contains the entire simulation engine.
    logistics_system = LogisticsSystem(p_id='logsys_001',
                                       p_visualize=False,
                                       p_logging=True,
                                       config=sim_config)

    # 3. Instantiate the Scenario to run the system
    # The scenario will use a simple random policy to generate actions.
    scenario = LogisticsScenario(p_system=logistics_system,
                                 p_cycle_limit=10, # Run for 10 simulation steps
                                 p_visualize=False,
                                 p_logging=True)

    # 4. Run the scenario
    print("\nRunning Scenario...")
    scenario.run()
    print("\nScenario finished.")

if __name__ == "__main__":
    run_howto()
