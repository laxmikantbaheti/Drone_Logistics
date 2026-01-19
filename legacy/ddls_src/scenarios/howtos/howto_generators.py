import sys
import os
import json

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = "C:\\Users\\SHAIK RIFSHU\\PycharmProjects\\Drone_Logistics\\ddls_src"
sys.path.insert(0, project_root)

# Import classes from data_resources and scenario modules
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.random_generator import RandomDataGenerator
from ddls_src.scenarios.generators.json_file_data_generator import JsonFileDataGenerator
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator

# Define paths to configuration files (relative to project_root)
config_dir = os.path.join(project_root, 'config')
initial_entity_data_path = os.path.join(config_dir, 'initial_entity_data.json')

# Ensure initial_entity_data.json exists for the JSON example
# We'll create a dummy one if it doesn't exist for demonstration purposes
if not os.path.exists(initial_entity_data_path):
    print(f"Creating dummy {initial_entity_data_path} for demonstration.")
    dummy_json_data = {
        "nodes": [
            {"id": 0, "coords": [0.0, 0.0], "type": "depot", "is_loadable": True, "is_unloadable": True,
             "is_charging_station": True},
            {"id": 1, "coords": [10.0, 5.0], "type": "customer", "is_loadable": False, "is_unloadable": True,
             "is_charging_station": False}
        ],
        "edges": [
            {"id": 0, "start_node_id": 0, "end_node_id": 1, "length": 11.18, "base_travel_time": 670.8},
            {"id": 1, "start_node_id": 1, "end_node_id": 0, "length": 11.18, "base_travel_time": 670.8}
        ],
        "trucks": [
            {"id": 101, "start_node_id": 0, "max_payload_capacity": 5, "max_speed": 60.0, "initial_fuel": 100.0,
             "fuel_consumption_rate": 0.1}
        ],
        "drones": [],
        "micro_hubs": [],
        "orders": [
            {"id": 1001, "customer_node_id": 1, "time_received": 0.0, "SLA_deadline": 1800.0, "priority": 1}
        ],
        "initial_time": 0.0
    }
    with open(initial_entity_data_path, 'w') as f:
        json.dump(dummy_json_data, f, indent=4)


def demonstrate_json_data_loading():
    print("\n--- Demonstrating JSON File Data Loading ---")
    print(f"Attempting to load data from: {initial_entity_data_path}")

    # Configuration for DataLoader to use JsonFileDataGenerator
    data_loader_config = {
        "generator_type": "json_file",
        "generator_config": {
            "file_path": initial_entity_data_path  # Path to the JSON file
        }
    }

    data_loader = DataLoader(data_loader_config)
    raw_data = data_loader.load_initial_simulation_data()

    print("\nRaw data loaded (first 2 nodes and orders):")
    for i, node_data in enumerate(raw_data.get('nodes', [])):
        if i >= 2: break
        print(f"  Node: {node_data}")
    for i, order_data in enumerate(raw_data.get('orders', [])):
        if i >= 2: break
        print(f"  Order: {order_data}")

    # Now use ScenarioGenerator to build entities from this raw data
    scenario_generator = ScenarioGenerator(raw_data)
    initial_entities = scenario_generator.build_entities()

    print("\nEntities instantiated from JSON data:")
    print(f"  Number of Nodes: {len(initial_entities['nodes'])}")
    print(f"  Number of Edges: {len(initial_entities['edges'])}")
    print(f"  Number of Trucks: {len(initial_entities['trucks'])}")
    print(f"  Number of Orders: {len(initial_entities['orders'])}")
    print(
        f"  Example Node (ID 0): Type={initial_entities['nodes'][0].type}, Coords={initial_entities['nodes'][0].coords}")
    if initial_entities['trucks']:
        print(
            f"  Example Truck (ID {list(initial_entities['trucks'].keys())[0]}): Start Node={list(initial_entities['trucks'].values())[0].current_node_id}")


def demonstrate_random_data_generation():
    print("\n--- Demonstrating Random Data Generation ---")

    # Configuration for DataLoader to use RandomDataGenerator
    random_data_loader_config = {
        "generator_type": "random",
        "generator_config": {
            "base_scale_factor": 5,  # Smaller scale for quick demo
            "num_nodes": 10,
            "num_customers": 5,
            "num_micro_hubs": 1,
            "num_trucks": 1,
            "num_drones": 1,
            "num_initial_orders": 3,
            "grid_size": [50.0, 50.0]
        }
    }

    data_loader = DataLoader(random_data_loader_config)
    raw_data = data_loader.load_initial_simulation_data()

    print("\nRaw data generated randomly (first 2 nodes and orders):")
    for i, node_data in enumerate(raw_data.get('nodes', [])):
        if i >= 2: break
        print(f"  Node: {node_data}")
    for i, order_data in enumerate(raw_data.get('orders', [])):
        if i >= 2: break
        print(f"  Order: {order_data}")

    # Now use ScenarioGenerator to build entities from this raw data
    scenario_generator = ScenarioGenerator(raw_data)
    initial_entities = scenario_generator.build_entities()

    print("\nEntities instantiated from random data:")
    print(f"  Number of Nodes: {len(initial_entities['nodes'])}")
    print(f"  Number of Edges: {len(initial_entities['edges'])}")
    print(f"  Number of Trucks: {len(initial_entities['trucks'])}")
    print(f"  Number of Drones: {len(initial_entities['drones'])}")
    print(f"  Number of MicroHubs: {len(initial_entities['micro_hubs'])}")
    print(f"  Number of Orders: {len(initial_entities['orders'])}")

    if initial_entities['nodes']:
        print(
            f"  Example Node (ID {list(initial_entities['nodes'].keys())[0]}): Type={list(initial_entities['nodes'].values())[0].type}, Coords={list(initial_entities['nodes'].values())[0].coords}")
    if initial_entities['trucks']:
        print(
            f"  Example Truck (ID {list(initial_entities['trucks'].keys())[0]}): Start Node={list(initial_entities['trucks'].values())[0].current_node_id}")


if __name__ == "__main__":
    demonstrate_json_data_loading()
    demonstrate_random_data_generation()
    print("\nAll Data Loading and Scenario Generation Examples Completed Successfully!")

