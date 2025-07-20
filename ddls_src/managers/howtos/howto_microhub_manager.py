import sys
import os
import json

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = "C:\\Users\\SHAIK RIFSHU\\PycharmProjects\\Drone_Logistics\\ddls_src"
sys.path.insert(0, project_root)

# Import necessary classes
from ddls_src.core.global_state import GlobalState
from ddls_src.managers.resource_manager.micro_hub_manager import MicroHubsManager
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator

# Import entity classes for type hinting and verification
from ddls_src.entities.node import Node
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.micro_hub import MicroHub


def setup_initial_simulation_state_for_manager_demo():
    """
    Helper function to set up a basic initial simulation state for manager demonstrations.
    This mimics the initialization process in LogisticsSimulation.
    """
    print("\n--- Setting Up Initial Simulation State for MicroHubsManager Demo ---")

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
             "is_charging_station": True, "num_charging_slots": 2},
            {"id": 7, "coords": [35.0, 15.0], "type": "micro_hub", "is_loadable": True, "is_unloadable": True,
             "is_charging_station": True, "num_charging_slots": 1}
        ],
        "edges": [
            {"id": 0, "start_node_id": 0, "end_node_id": 1, "length": 11.18, "base_travel_time": 670.8},
            {"id": 1, "start_node_id": 1, "end_node_id": 0, "length": 11.18, "base_travel_time": 670.8},
            {"id": 2, "start_node_id": 0, "end_node_id": 3, "length": 20.62, "base_travel_time": 1237.2}
        ],
        "trucks": [],
        "drones": [
            {"id": 201, "start_node_id": 0, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 0.5,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01},
            {"id": 202, "start_node_id": 3, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 0.2,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01}
        ],
        "micro_hubs": [],  # Micro-hubs are handled within the 'nodes' list now
        "orders": [],  # Not directly relevant for this demo
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


def demonstrate_micro_hubs_manager_functionality():
    """
    Demonstrates the functionalities of the MicroHubsManager class.
    """
    print("--- Demonstrating MicroHubsManager Class Functionality ---")

    global_state = setup_initial_simulation_state_for_manager_demo()
    micro_hubs_manager = MicroHubsManager(global_state)

    # Get initial entities for reference
    micro_hub3 = global_state.get_entity("micro_hub", 3)
    micro_hub7 = global_state.get_entity("micro_hub", 7)
    drone1 = global_state.get_entity("drone", 201)
    drone2 = global_state.get_entity("drone", 202)

    print("\nInitial State:")
    print(f"  MicroHub 3 Status: {micro_hub3.operational_status}, Slots: {micro_hub3.charging_slots}")
    print(f"  MicroHub 7 Status: {micro_hub7.operational_status}, Slots: {micro_hub7.charging_slots}")
    print(f"  Drone 201 Status: {drone1.status}, Battery: {drone1.battery_level * 100:.1f}%")
    print(f"  Drone 202 Status: {drone2.status}, Battery: {drone2.battery_level * 100:.1f}%")

    # 1. Activate MicroHub
    print("\n1. Activating MicroHub 3:")
    micro_hubs_manager.activate_micro_hub(3)
    print(f"  MicroHub 3 Status: {micro_hub3.operational_status}")
    print(f"  Attempting to activate already active hub (should fail):")
    micro_hubs_manager.activate_micro_hub(3)

    # 2. Add Drone to Charging Queue (assign slot)
    print("\n2. Adding Drone 202 to Charging Queue at MicroHub 3:")
    # Ensure Drone 202 is at MicroHub 3
    drone2.current_node_id = 3
    drone2.set_status("idle")  # Ensure it's not already charging

    micro_hubs_manager.add_to_charging_queue(3, 202)
    print(f"  MicroHub 3 Slots: {micro_hub3.charging_slots}")
    print(f"  Drone 202 Status: {drone2.status}")
    print(f"  MicroHub 3 Available Slots: {micro_hub3.get_available_charging_slots()}")

    print("\n   Attempting to add Drone 201 (not at hub) to MicroHub 3 (should fail):")
    micro_hubs_manager.add_to_charging_queue(3, 201)
    print(f"  Drone 201 Status: {drone1.status}")

    print("\n   Attempting to add Drone 202 again (already charging, should fail):")
    micro_hubs_manager.add_to_charging_queue(3, 202)

    # Fill up MicroHub 3's slots
    drone3 = Drone(id=203, start_node_id=3, max_payload_capacity=1, max_speed=30.0,
                   initial_battery=0.1, battery_drain_rate_flying=0.005, battery_drain_rate_idle=0.001,
                   battery_charge_rate=0.01)
    global_state.add_entity(drone3)
    drone3.current_node_id = 3
    drone3.set_status("idle")
    micro_hubs_manager.add_to_charging_queue(3, 203)
    print(f"\n   Added Drone 203 to MicroHub 3. Slots: {micro_hub3.charging_slots}")
    print(f"  MicroHub 3 Available Slots: {micro_hub3.get_available_charging_slots()}")
    print("\n   Attempting to add another drone (slots full, should fail):")
    micro_hubs_manager.add_to_charging_queue(3, drone1.id)  # Drone1 not at hub, but also slots full

    # 3. Deactivate MicroHub
    print("\n3. Deactivating MicroHub 3 (should release charging drones):")
    micro_hubs_manager.deactivate_micro_hub(3)
    print(f"  MicroHub 3 Status: {micro_hub3.operational_status}")
    print(f"  MicroHub 3 Slots: {micro_hub3.charging_slots}")  # Should be all None
    print(f"  Drone 202 Status: {drone2.status} (should be idle)")
    print(f"  Drone 203 Status: {drone3.status} (should be idle)")

    # 4. Attempt to add to charging queue of inactive hub
    print("\n4. Attempting to add Drone 201 to inactive MicroHub 3 (should fail):")
    drone1.current_node_id = 3  # Temporarily move drone1 for test
    drone1.set_status("idle")
    micro_hubs_manager.add_to_charging_queue(3, 201)
    print(f"  Drone 201 Status: {drone1.status}")
    drone1.current_node_id = 0  # Reset drone1 location

    # 5. Activate and Deactivate another MicroHub
    print("\n5. Activating and Deactivating MicroHub 7:")
    micro_hubs_manager.activate_micro_hub(7)
    print(f"  MicroHub 7 Status: {micro_hub7.operational_status}")
    micro_hubs_manager.deactivate_micro_hub(7)
    print(f"  MicroHub 7 Status: {micro_hub7.operational_status}")

    print("\n--- MicroHubsManager Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_micro_hubs_manager_functionality()

