import sys
import os
import json

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = "C:\\Users\\SHAIK RIFSHU\\PycharmProjects\\Drone_Logistics\\ddls_src"
sys.path.insert(0, project_root)

# Import necessary classes
from ddls_src.core.global_state import GlobalState
from ddls_src.managers.resource_manager.fleet_manager import FleetManager
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator

# Import entity classes for type hinting and verification
from ddls_src.entities.node import Node
from ddls_src.entities.order import Order
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.micro_hub import MicroHub


def setup_initial_simulation_state_for_manager_demo():
    """
    Helper function to set up a basic initial simulation state for manager demonstrations.
    This mimics the initialization process in LogisticsSimulation.
    """
    print("\n--- Setting Up Initial Simulation State for FleetManager Demo ---")

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
            {"id": 101, "start_node_id": 0, "max_payload_capacity": 2, "max_speed": 60.0, "initial_fuel": 100.0,
             "fuel_consumption_rate": 0.1}
        ],
        "drones": [
            {"id": 201, "start_node_id": 0, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 0.5,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01}
        ],
        "micro_hubs": [],  # Micro-hubs are handled within the 'nodes' list now
        "orders": [
            {"id": 1001, "customer_node_id": 1, "time_received": 0.0, "SLA_deadline": 1800.0, "priority": 1},
            {"id": 1002, "customer_node_id": 1, "time_received": 0.0, "SLA_deadline": 1800.0, "priority": 1},
            {"id": 1003, "customer_node_id": 3, "time_received": 0.0, "SLA_deadline": 3000.0, "priority": 1}
            # Order for micro_hub
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

    global_state = GlobalState(initial_entities)

    print("Initial GlobalState setup complete for manager demo.")
    return global_state


def demonstrate_fleet_manager_functionality():
    """
    Demonstrates the functionalities of the FleetManager class.
    """
    print("--- Demonstrating FleetManager Class Functionality ---")

    global_state = setup_initial_simulation_state_for_manager_demo()
    fleet_manager = FleetManager(global_state)

    # Get initial entities for reference
    truck1 = global_state.get_entity("truck", 101)
    drone1 = global_state.get_entity("drone", 201)
    order1 = global_state.get_entity("order", 1001)
    order2 = global_state.get_entity("order", 1002)
    order3 = global_state.get_entity("order", 1003)
    depot_node = global_state.get_entity("node", 0)
    customer_node1 = global_state.get_entity("node", 1)
    micro_hub3 = global_state.get_entity("micro_hub", 3)

    print("\nInitial State:")
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}, Status: {truck1.status}")
    print(
        f"  Drone 201 Cargo: {drone1.get_cargo()}, Status: {drone1.status}, Battery: {drone1.battery_level * 100:.1f}%")
    print(f"  Order 1001 Status: {order1.status}")
    print(f"  Order 1002 Status: {order2.status}")
    print(f"  Order 1003 Status: {order3.status}")
    print(f"  Depot Node Packages: {depot_node.get_packages()}")
    print(f"  Customer Node 1 Packages: {customer_node1.get_packages()}")
    print(f"  MicroHub 3 Packages: {micro_hub3.get_packages()}")

    # Ensure orders are at the depot for loading demo
    depot_node.add_package(1001)
    depot_node.add_package(1002)
    print(f"\nOrders 1001, 1002 placed at Depot. Depot Packages: {depot_node.get_packages()}")

    # Ensure order 1003 is at MicroHub 3 for drone loading demo
    micro_hub3.add_package_to_holding(1003)
    print(f"Order 1003 placed at MicroHub 3. MicroHub 3 Packages: {micro_hub3.get_packages()}")

    # 1. Load Truck
    print("\n1. Loading Truck 101 with Order 1001:")
    truck1.current_node_id = 0  # Ensure truck is at depot
    truck1.set_status("idle")
    fleet_manager.load_truck(101, 1001)
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}, Status: {truck1.status}")
    print(f"  Order 1001 Status: {order1.status}")
    print(f"  Depot Node Packages: {depot_node.get_packages()}")

    print("\n   Loading Truck 101 with Order 1002 (should reach capacity):")
    fleet_manager.load_truck(101, 1002)
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}, Status: {truck1.status}")
    print(f"  Order 1002 Status: {order2.status}")
    print(f"  Depot Node Packages: {depot_node.get_packages()}")

    print("\n   Attempting to load more (should fail - truck full):")
    fleet_manager.load_truck(101, 9999)  # Non-existent order, but demonstrates capacity check
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}")

    # 2. Unload Truck
    print("\n2. Unloading Truck 101 (Order 1001 to Customer 1, Order 1002 to Customer 1):")
    truck1.current_node_id = 1  # Move truck to customer node 1
    truck1.set_status("idle")

    # Unload order 1001 (final delivery)
    fleet_manager.unload_truck(101, 1001)
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}, Status: {truck1.status}")
    print(f"  Order 1001 Status: {order1.status}, Delivery Time: {order1.delivery_time}")
    print(f"  Customer Node 1 Packages: {customer_node1.get_packages()}")

    # Unload order 1002 (final delivery)
    fleet_manager.unload_truck(101, 1002)
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}, Status: {truck1.status}")
    print(f"  Order 1002 Status: {order2.status}, Delivery Time: {order2.delivery_time}")
    print(f"  Customer Node 1 Packages: {customer_node1.get_packages()}")

    # 3. Drone Load
    print("\n3. Loading Drone 201 with Order 1003 (from MicroHub 3):")
    drone1.current_node_id = 3  # Ensure drone is at micro_hub 3
    drone1.set_status("idle")
    fleet_manager.drone_load(201, 1003)
    print(f"  Drone 201 Cargo: {drone1.get_cargo()}, Status: {drone1.status}")
    print(f"  Order 1003 Status: {order3.status}")
    print(f"  MicroHub 3 Packages: {micro_hub3.get_packages()}")

    # 4. Drone Unload
    print("\n4. Unloading Drone 201 (Order 1003 to MicroHub 3 - it's a transfer):")
    drone1.current_node_id = 3  # Ensure drone is at micro_hub 3 (for transfer demo)
    drone1.set_status("idle")

    fleet_manager.drone_unload(201, 1003)
    print(f"  Drone 201 Cargo: {drone1.get_cargo()}, Status: {drone1.status}")
    print(f"  Order 1003 Status: {order3.status}")  # Should be 'at_node' or 'at_micro_hub'
    print(f"  MicroHub 3 Packages: {micro_hub3.get_packages()}")

    # 5. Drone Charge
    print("\n5. Charging Drone 201 at Depot (Node 0):")
    drone1.current_node_id = 0  # Move drone to depot (which is charging station)
    drone1.battery_level = 0.2  # Set low battery for demo
    drone1.set_status("charging")  # Set status to charging
    print(f"  Drone 201 initial battery: {drone1.battery_level * 100:.1f}%, Status: {drone1.status}")

    charge_duration = 100.0  # seconds
    fleet_manager.drone_charge(201, charge_duration)
    print(f"  Drone 201 battery after {charge_duration}s charge: {drone1.battery_level * 100:.1f}%")

    # Attempt to charge when not in charging status
    drone1.set_status("idle")
    print(f"  Attempting to charge Drone 201 while idle:")
    fleet_manager.drone_charge(201, 50.0)
    print(f"  Drone 201 battery (should be same): {drone1.battery_level * 100:.1f}%")

    print("\n--- FleetManager Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_fleet_manager_functionality()

