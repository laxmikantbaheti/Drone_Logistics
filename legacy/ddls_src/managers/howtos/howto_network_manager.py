import sys
import os
import json

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = "C:\\Users\\SHAIK RIFSHU\\PycharmProjects\\Drone_Logistics\\ddls_src"
sys.path.insert(0, project_root)

# Import necessary classes
from ddls_src.core.global_state import GlobalState
from ddls_src.core.network import Network
from ddls_src.managers.network_manager import NetworkManager
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator

# Import entity classes for type hinting and verification
from ddls_src.entities.node import Node
from ddls_src.entities.edge import Edge
from ddls_src.entities.order import Order
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.micro_hub import MicroHub


def setup_initial_simulation_state_for_manager_demo():
    """
    Helper function to set up a basic initial simulation state for manager demonstrations.
    This mimics the initialization process in LogisticsSimulation.
    """
    print("\n--- Setting Up Initial Simulation State for NetworkManager Demo ---")

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
            {"id": 2, "coords": [15.0, 15.0], "type": "customer", "is_loadable": False, "is_unloadable": True,
             "is_charging_station": False},
            {"id": 3, "coords": [5.0, 20.0], "type": "micro_hub", "is_loadable": True, "is_unloadable": True,
             "is_charging_station": True, "num_charging_slots": 2},
            {"id": 4, "coords": [25.0, 10.0], "type": "customer", "is_loadable": False, "is_unloadable": True,
             "is_charging_station": False}
        ],
        "edges": [
            {"id": 0, "start_node_id": 0, "end_node_id": 1, "length": 11.18, "base_travel_time": 670.8},
            {"id": 1, "start_node_id": 1, "end_node_id": 0, "length": 11.18, "base_travel_time": 670.8},
            {"id": 2, "start_node_id": 0, "end_node_id": 3, "length": 20.62, "base_travel_time": 1237.2},
            {"id": 3, "start_node_id": 3, "end_node_id": 0, "length": 20.62, "base_travel_time": 1237.2},
            {"id": 4, "start_node_id": 1, "end_node_id": 2, "length": 11.18, "base_travel_time": 670.8},
            {"id": 5, "start_node_id": 2, "end_node_id": 1, "length": 11.18, "base_travel_time": 670.8},
            {"id": 6, "start_node_id": 2, "end_node_id": 3, "length": 11.18, "base_travel_time": 670.8},
            {"id": 7, "start_node_id": 3, "end_node_id": 2, "length": 11.18, "base_travel_time": 670.8},
            {"id": 8, "start_node_id": 3, "end_node_id": 4, "length": 20.0, "base_travel_time": 1200.0}
            # Added for drone pathing
        ],
        "trucks": [
            {"id": 101, "start_node_id": 0, "max_payload_capacity": 5, "max_speed": 60.0, "initial_fuel": 100.0,
             "fuel_consumption_rate": 0.1}
        ],
        "drones": [
            {"id": 201, "start_node_id": 0, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 1.0,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01},
            {"id": 202, "start_node_id": 3, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 0.8,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01}
        ],
        "micro_hubs": [],  # Micro-hubs are handled within the 'nodes' list now
        "orders": [
            {"id": 1001, "customer_node_id": 1, "time_received": 0.0, "SLA_deadline": 1800.0, "priority": 1},
            # Truck order
            {"id": 1002, "customer_node_id": 4, "time_received": 0.0, "SLA_deadline": 2400.0, "priority": 2}
            # Drone order
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
    network = Network(global_state)  # Network needs GlobalState
    global_state.network = network  # GlobalState also needs reference to Network

    print("Initial GlobalState and Network setup complete for manager demo.")
    return global_state, network


def demonstrate_network_manager_functionality():
    """
    Demonstrates the functionalities of the NetworkManager class.
    """
    print("--- Demonstrating NetworkManager Class Functionality ---")

    global_state, network = setup_initial_simulation_state_for_manager_demo()
    network_manager = NetworkManager(global_state, network)

    # Get initial entities for reference
    truck1 = global_state.get_entity("truck", 101)
    drone1 = global_state.get_entity("drone", 201)
    drone2 = global_state.get_entity("drone", 202)  # Drone at micro_hub 3
    order1 = global_state.get_entity("order", 1001)
    order2 = global_state.get_entity("order", 1002)
    micro_hub3 = global_state.get_entity("micro_hub", 3)
    depot_node = global_state.get_entity("node", 0)
    customer_node1 = global_state.get_entity("node", 1)
    customer_node4 = global_state.get_entity("node", 4)

    print("\nInitial State:")
    print(f"  Truck 101 Status: {truck1.status}, Node: {truck1.current_node_id}, Route: {truck1.current_route}")
    print(f"  Drone 201 Status: {drone1.status}, Node: {drone1.current_node_id}, Route: {drone1.current_route}")
    print(f"  Drone 202 Status: {drone2.status}, Node: {drone2.current_node_id}, Route: {drone2.current_route}")
    print(f"  Order 1001 Customer: {order1.customer_node_id}")
    print(f"  Order 1002 Customer: {order2.customer_node_id}")

    # 1. Truck to Node
    print("\n1. Commanding Truck 101 to Node 1 (customer_node1):")
    network_manager.truck_to_node(101, customer_node1.id)
    print(f"  Truck 101 Status: {truck1.status}, Route: {truck1.current_route}")

    print("\n   Simulating truck movement (usually done in main loop):")
    # To demonstrate movement, we need to call move_along_route repeatedly
    # For a single example, let's just make one step and check coords
    if truck1.status == "en_route":
        truck1.move_along_route(truck1.max_speed * 0.1, network_manager)  # Move for a short duration
        print(f"  Truck 101 moved. Coords: {truck1.current_location_coords}, Progress: {truck1.route_progress:.2f}")

    # 2. Launch Drone
    print("\n2. Launching Drone 201 with Order 1002 (to customer_node4):")
    # First, assign order 1002 to drone 201 and put it in cargo
    order2.assign_vehicle(201)
    drone1.add_cargo(1002)
    print(f"  Drone 201 cargo: {drone1.get_cargo()}")

    network_manager.launch_drone(201, order2.id)
    print(f"  Drone 201 Status: {drone1.status}, Route: {drone1.current_route}")

    print("\n   Simulating drone movement:")
    if drone1.status == "en_route":
        drone1.move_along_route(drone1.max_speed * 0.1, network_manager)  # Move for a short duration
        print(
            f"  Drone 201 moved. Coords: {drone1.current_location_coords}, Battery: {drone1.battery_level * 100:.1f}%")

    # 3. Drone to Charging Station
    print("\n3. Commanding Drone 201 to MicroHub 3 (charging station):")
    # Clear drone's current route/cargo for this demo
    drone1.current_route = []
    drone1.cargo_manifest = []
    drone1.set_status("idle")
    drone1.battery_level = 0.1  # Set low battery

    network_manager.drone_to_charging_station(201, micro_hub3.id)
    print(f"  Drone 201 Status: {drone1.status}, Route: {drone1.current_route}")

    print("\n   Simulating drone movement to charging station:")
    if drone1.status == "en_route":
        drone1.move_along_route(drone1.max_speed * 0.1, network_manager)
        print(
            f"  Drone 201 moved. Coords: {drone1.current_location_coords}, Battery: {drone1.battery_level * 100:.1f}%")

    # 4. Drone Landing (at customer node for delivery)
    print("\n4. Commanding Drone 202 to Land (at MicroHub 3, where it started):")
    # Simulate drone 202 having an order for Node 4 (customer)
    order_for_drone2 = Order(id=1004, customer_node_id=4, time_received=0.0, SLA_deadline=3600.0, priority=1)
    global_state.add_entity(order_for_drone2)
    order_for_drone2.assign_vehicle(202)
    drone2.add_cargo(1004)
    drone2.set_route([3, 4])  # Route from MicroHub 3 to Customer 4
    drone2.set_status("en_route")
    # Simulate drone arriving at customer node 4
    drone2.current_node_id = 4
    drone2.current_location_coords = global_state.get_entity("node", 4).coords
    drone2.route_progress = 1.0  # Arrived at node

    print(f"  Drone 202 Status: {drone2.status}, Node: {drone2.current_node_id}, Cargo: {drone2.get_cargo()}")
    print(f"  Order 1004 Status: {order_for_drone2.status}")
    print(f"  Customer Node 4 Packages: {customer_node4.get_packages()}")

    network_manager.drone_landing(202)
    print(f"  Drone 202 Status: {drone2.status}, Route: {drone2.current_route}, Cargo: {drone2.get_cargo()}")
    print(f"  Order 1004 Status: {order_for_drone2.status}, Delivery Time: {order_for_drone2.delivery_time}")
    print(f"  Customer Node 4 Packages: {customer_node4.get_packages()}")

    # 5. Re-route Truck
    print("\n5. Re-routing Truck 101 to Node 3 (MicroHub):")
    truck1.current_node_id = 1  # Assume truck is at Node 1
    truck1.set_status("en_route")  # Assume it was en-route
    network_manager.re_route_truck_to_node(101, micro_hub3.id)
    print(f"  Truck 101 Status: {truck1.status}, New Route: {truck1.current_route}")

    print("\n--- NetworkManager Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_network_manager_functionality()

