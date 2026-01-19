import sys
import os
import json

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = "C:\\Users\\SHAIK RIFSHU\\PycharmProjects\\Drone_Logistics\\ddls_src"
sys.path.insert(0, project_root)

# Import necessary classes
from ddls_src.core.global_state import GlobalState
from ddls_src.managers.resource_manager.base import ResourceManager
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator

# Import entity classes for type hinting and verification
from ddls_src.entities.node import Node
from ddls_src.entities.edge import Edge
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.micro_hub import MicroHub
from ddls_src.actions.action_enums import SimulationAction  # For ResourceSpecificType enums


def setup_initial_simulation_state_for_manager_demo():
    """
    Helper function to set up a basic initial simulation state for manager demonstrations.
    This mimics the initialization process in LogisticsSimulation.
    """
    print("\n--- Setting Up Initial Simulation State for ResourceManager Demo ---")

    # Define paths to configuration files (relative to project_root)
    config_dir = os.path.join(project_root, 'config')
    initial_entity_data_path = os.path.join(config_dir, 'initial_entity_data.json')

    # Ensure initial_entity_data.json is always created/overwritten with the correct dummy data
    print(f"Overwriting dummy {initial_entity_data_path} for demonstration consistency.")
    dummy_json_data = {
        "nodes": [
            {"id": 0, "coords": [0.0, 0.0], "type": "depot", "is_loadable": True, "is_unloadable": True,
             "is_charging_station": True},
            {"id": 1, "coords": [10.0, 5.0], "type": "customer", "is_loadable": False, "is_unloadable": True,
             "is_charging_station": False},
            {"id": 3, "coords": [5.0, 20.0], "type": "micro_hub", "is_loadable": True, "is_unloadable": True,
             "is_charging_station": True, "num_charging_slots": 2}
        ],
        "edges": [
            {"id": 0, "start_node_id": 0, "end_node_id": 1, "length": 11.18, "base_travel_time": 670.8},
            {"id": 1, "start_node_id": 1, "end_node_id": 0, "length": 11.18, "base_travel_time": 670.8},
            {"id": 2, "start_node_id": 0, "end_node_id": 3, "length": 20.62, "base_travel_time": 1237.2}
        ],
        "trucks": [
            {"id": 101, "start_node_id": 0, "max_payload_capacity": 5, "max_speed": 60.0, "initial_fuel": 100.0,
             "fuel_consumption_rate": 0.1},
            {"id": 102, "start_node_id": 0, "max_payload_capacity": 5, "max_speed": 60.0, "initial_fuel": 100.0,
             "fuel_consumption_rate": 0.1}
        ],
        "drones": [
            {"id": 201, "start_node_id": 0, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 1.0,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01},
            {"id": 202, "start_node_id": 3, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 0.8,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01}
        ],
        "micro_hubs": [],  # Micro-hubs are handled within the 'nodes' list now
        "orders": [],  # Not directly relevant for ResourceManager demo
        "initial_time": 0.0
    }
    with open(initial_entity_data_path, 'w') as f:  # Use 'w' mode to overwrite
        json.dump(dummy_json_data, f, indent=4)

    # Configuration for DataLoader to use JsonFileDataGenerator
    data_loader_config = {
        "generator_type": "json_file",
        "generator_config": {
            "file_path": initial_entity_data_path  # Path to the JSON file
        }
    }

    data_loader = DataLoader(data_loader_config)
    raw_data = data_loader.load_initial_simulation_data()

    scenario_generator = ScenarioGenerator(raw_data)
    initial_entities = scenario_generator.build_entities()

    global_state = GlobalState(initial_entities)

    print("Initial GlobalState setup complete for manager demo.")
    return global_state


def demonstrate_resource_manager_functionality():
    """
    Demonstrates the functionalities of the ResourceManager class.
    """
    print("--- Demonstrating ResourceManager Class Functionality ---")

    global_state = setup_initial_simulation_state_for_manager_demo()
    resource_manager = ResourceManager(global_state)

    # Get initial entities for reference
    truck1 = global_state.get_entity("truck", 101)
    truck2 = global_state.get_entity("truck", 102)
    drone1 = global_state.get_entity("drone", 201)
    micro_hub3 = global_state.get_entity("micro_hub", 3)

    print("\nInitial State:")
    print(f"  Truck 101 Status: {truck1.status}")
    print(f"  Truck 102 Status: {truck2.status}")
    print(f"  Drone 201 Status: {drone1.status}")
    print(f"  MicroHub 3 Operational Status: {micro_hub3.operational_status}")
    print(f"  MicroHub 3 Launches Blocked: {micro_hub3.is_blocked_for_launches}")
    print(f"  MicroHub 3 Package Transfer Unavailable: {micro_hub3.is_package_transfer_unavailable}")

    # 1. Flag Vehicle for Maintenance
    print("\n1. Flagging Vehicle 101 (Truck) for Maintenance:")
    resource_manager.flag_vehicle_for_maintenance(101)
    print(f"  Truck 101 Status: {truck1.status}")

    print("\n   Attempting to flag an already maintained vehicle (should fail):")
    resource_manager.flag_vehicle_for_maintenance(101)
    print(f"  Truck 101 Status: {truck1.status}")

    # 2. Release Vehicle from Maintenance
    print("\n2. Releasing Vehicle 101 (Truck) from Maintenance:")
    resource_manager.release_vehicle_from_maintenance(101)
    print(f"  Truck 101 Status: {truck1.status}")

    print("\n   Flagging Drone 201 for Maintenance:")
    resource_manager.flag_vehicle_for_maintenance(201)
    print(f"  Drone 201 Status: {drone1.status}")
    resource_manager.release_vehicle_from_maintenance(201)
    print(f"  Drone 201 Status: {drone1.status}")

    # 3. Flag Unavailability of Service at MicroHub
    print("\n3. Flagging Service Unavailability at MicroHub 3:")
    # First, activate MicroHub 3 for context
    micro_hub3.activate()
    print(f"  MicroHub 3 activated. Status: {micro_hub3.operational_status}")

    resource_manager.flag_unavailability_of_service_at_micro_hub(3, SimulationAction.RESOURCE_LAUNCHES)
    print(f"  MicroHub 3 Launches Blocked: {micro_hub3.is_blocked_for_launches}")

    resource_manager.flag_unavailability_of_service_at_micro_hub(3, SimulationAction.RESOURCE_PACKAGE_SORTING_SERVICE)
    print(f"  MicroHub 3 Package Transfer Unavailable: {micro_hub3.is_package_transfer_unavailable}")

    # 4. Release Unavailability of Service at MicroHub
    print("\n4. Releasing Service Unavailability at MicroHub 3:")
    resource_manager.release_unavailability_of_service_at_micro_hub(3, SimulationAction.RESOURCE_LAUNCHES)
    print(f"  MicroHub 3 Launches Blocked: {micro_hub3.is_blocked_for_launches}")

    resource_manager.release_unavailability_of_service_at_micro_hub(3,
                                                                    SimulationAction.RESOURCE_PACKAGE_SORTING_SERVICE)
    print(f"  MicroHub 3 Package Transfer Unavailable: {micro_hub3.is_package_transfer_unavailable}")

    print("\n--- ResourceManager Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_resource_manager_functionality()

