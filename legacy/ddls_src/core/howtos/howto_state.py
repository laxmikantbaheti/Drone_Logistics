import sys
import os
import json

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = "C:\\Users\\SHAIK RIFSHU\\PycharmProjects\\Drone_Logistics\\ddls_src"
sys.path.insert(0, project_root)

# Import necessary classes
from ddls_src.core.global_state import GlobalState
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator

# Import entity classes for instantiation
from ddls_src.entities.node import Node
from ddls_src.entities.edge import Edge
from ddls_src.entities.order import Order
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.micro_hub import MicroHub


def setup_initial_entities_for_global_state_demo():
    """
    Helper function to get a set of instantiated entities for GlobalState.
    This uses DataLoader and ScenarioGenerator, mimicking the simulation setup.
    """
    print("\n--- Setting Up Initial Entities for GlobalState Demo ---")

    # Define paths to configuration files (relative to project_root)
    config_dir = os.path.join(project_root, 'config')
    initial_entity_data_path = os.path.join(config_dir, 'initial_entity_data.json')

    # Ensure initial_entity_data.json is always created/overwritten with the correct dummy data
    print(f"Overwriting dummy {initial_entity_data_path} for demonstration consistency.")
    dummy_json_data = {
        "nodes": [
            {"id": 0, "coords": [0.0, 0.0], "type": "depot", "is_loadable": True, "is_unloadable": True,
             "is_charging_station": True, "packages_held": [1001]},
            {"id": 1, "coords": [10.0, 5.0], "type": "customer", "is_loadable": False, "is_unloadable": True,
             "is_charging_station": False},
            {"id": 3, "coords": [5.0, 20.0], "type": "micro_hub", "is_loadable": True, "is_unloadable": True,
             "is_charging_station": True, "num_charging_slots": 2, "packages_held": [1002]}
        ],
        "edges": [
            {"id": 0, "start_node_id": 0, "end_node_id": 1, "length": 11.18, "base_travel_time": 670.8},
            {"id": 1, "start_node_id": 1, "end_node_id": 0, "length": 11.18, "base_travel_time": 670.8}
        ],
        "trucks": [
            {"id": 101, "start_node_id": 0, "max_payload_capacity": 5, "max_speed": 60.0, "initial_fuel": 100.0,
             "fuel_consumption_rate": 0.1}
        ],
        "drones": [
            {"id": 201, "start_node_id": 0, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 1.0,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01}
        ],
        "micro_hubs": [],  # Handled within 'nodes' list now
        "orders": [
            {"id": 1001, "customer_node_id": 1, "time_received": 0.0, "SLA_deadline": 1800.0, "priority": 1},
            {"id": 1002, "customer_node_id": 3, "time_received": 0.0, "SLA_deadline": 2400.0, "priority": 2}
        ],
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

    print("Initial entities setup complete.")
    return initial_entities


def demonstrate_global_state_functionality():
    """
    Demonstrates the core functionalities of the GlobalState class.
    """
    print("--- Demonstrating GlobalState Class Functionality ---")

    initial_entities = setup_initial_entities_for_global_state_demo()

    # 1. Initialize GlobalState
    print("\n1. Initializing GlobalState:")
    global_state = GlobalState(initial_entities)
    print(
        f"  GlobalState initialized with {len(global_state.nodes)} nodes, {len(global_state.trucks)} trucks, {len(global_state.orders)} orders.")
    print(f"  Current time: {global_state.current_time}")

    # 2. Get Entity by Type and ID
    print("\n2. Getting Entities by Type and ID:")
    node0 = global_state.get_entity("node", 0)
    print(f"  Node 0: ID={node0.id}, Type={node0.type}, Coords={node0.coords}")

    truck101 = global_state.get_entity("truck", 101)
    print(f"  Truck 101: ID={truck101.id}, Status={truck101.status}, Current Node={truck101.current_node_id}")

    order1001 = global_state.get_entity("order", 1001)
    print(f"  Order 1001: ID={order1001.id}, Status={order1001.status}")

    # Attempt to get non-existent entity
    try:
        global_state.get_entity("order", 9999)
    except KeyError as e:
        print(f"  Attempted to get non-existent order: {e}")

    # 3. Get All Entities of a Type
    print("\n3. Getting All Entities of a Type:")
    all_drones = global_state.get_all_entities_by_type("drone")
    print(f"  All Drones (IDs): {list(all_drones.keys())}")

    all_micro_hubs = global_state.get_all_entities_by_type("micro_hub")
    print(f"  All MicroHubs (IDs): {list(all_micro_hubs.keys())}")

    # 4. Add and Remove Entities
    print("\n4. Adding and Removing Entities:")
    new_node = Node(id=10, coords=(50.0, 50.0), type="new_junction")
    global_state.add_entity(new_node)
    print(f"  Added Node 10. Total nodes: {len(global_state.nodes)}")
    print(f"  Node 10 type: {global_state.get_entity('node', 10).type}")

    global_state.remove_entity("node", 10)
    print(f"  Removed Node 10. Total nodes: {len(global_state.nodes)}")
    try:
        global_state.get_entity("node", 10)
    except KeyError as e:
        print(f"  Attempted to get removed node: {e}")

    # 5. Update Entity Attribute (using specific entity methods, which GlobalState facilitates)
    print("\n5. Updating Entity Attributes:")
    print(f"  Initial Truck 101 status: {truck101.status}")
    truck101.set_status("en_route")
    print(f"  Updated Truck 101 status: {truck101.status}")

    print(f"  Initial Node 0 packages: {node0.get_packages()}")
    node0.add_package(1003)  # Add another package to Node 0
    print(f"  Updated Node 0 packages: {node0.get_packages()}")

    # 6. Specific Getters
    print("\n6. Using Specific Getters:")
    print(f"  Truck 101 location node ID: {global_state.get_truck_location(101)}")
    print(f"  Is Node 0 loadable? {global_state.is_node_loadable(0)}")
    print(f"  Order 1001 status: {global_state.get_order_status(1001)}")
    print(f"  Vehicle 201 (Drone) status: {global_state.get_vehicle_status(201)}")
    print(f"  Drone 201 battery level: {global_state.get_drone_battery_level(201)}")
    print(f"  MicroHub 3 status: {global_state.get_micro_hub_status(3)}")
    print(f"  Packages at Node 0: {global_state.get_packages_at_node(0)}")

    print("\n--- GlobalState Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_global_state_functionality()

