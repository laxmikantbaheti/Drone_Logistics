import sys
import os
import json

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = "C:\\Users\\SHAIK RIFSHU\\PycharmProjects\\Drone_Logistics\\ddls_src"
sys.path.insert(0, project_root)

# Import necessary classes
from ddls_src.core.global_state import GlobalState
from ddls_src.managers.supply_chain_manager import SupplyChainManager
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
    print("\n--- Setting Up Initial Simulation State for SupplyChainManager Demo ---")

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
             "is_charging_station": True, "num_charging_slots": 2}
        ],
        "edges": [
            {"id": 0, "start_node_id": 0, "end_node_id": 1, "length": 11.18, "base_travel_time": 670.8},
            {"id": 1, "start_node_id": 1, "end_node_id": 0, "length": 11.18, "base_travel_time": 670.8},
            {"id": 2, "start_node_id": 0, "end_node_id": 3, "length": 20.62, "base_travel_time": 1237.2}
        ],
        "trucks": [
            {"id": 101, "start_node_id": 0, "max_payload_capacity": 5, "max_speed": 60.0, "initial_fuel": 100.0,
             "fuel_consumption_rate": 0.1}
        ],
        "drones": [
            {"id": 201, "start_node_id": 0, "max_payload_capacity": 1, "max_speed": 30.0, "initial_battery": 1.0,
             "battery_drain_rate_flying": 0.005, "battery_drain_rate_idle": 0.001, "battery_charge_rate": 0.01}
        ],
        "micro_hubs": [],  # Micro-hubs are handled within the 'nodes' list now
        "orders": [
            {"id": 1001, "customer_node_id": 1, "time_received": 0.0, "SLA_deadline": 1800.0, "priority": 1},
            {"id": 1002, "customer_node_id": 2, "time_received": 0.0, "SLA_deadline": 2400.0, "priority": 2},
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


def demonstrate_supply_chain_manager_functionality():
    """
    Demonstrates the functionalities of the SupplyChainManager class.
    """
    print("--- Demonstrating SupplyChainManager Class Functionality ---")

    global_state = setup_initial_simulation_state_for_manager_demo()
    supply_chain_manager = SupplyChainManager(global_state)

    # Get initial entities for reference
    order1 = global_state.get_entity("order", 1001)
    order2 = global_state.get_entity("order", 1002)
    order3 = global_state.get_entity("order", 1003)  # Order for micro_hub
    truck1 = global_state.get_entity("truck", 101)
    drone1 = global_state.get_entity("drone", 201)
    # micro_hub3 should be accessed via global_state.micro_hubs, not global_state.get_entity("micro_hub", 3) if it's not guaranteed to be in nodes
    # No, it should be in global_state.micro_hubs because ScenarioGenerator adds it there.
    # And it should be in global_state.nodes because ScenarioGenerator.add_micro_hub adds it there.
    micro_hub3 = global_state.get_entity("micro_hub", 3)
    node0 = global_state.get_entity("node", 0)  # Depot

    print("\nInitial State:")
    print(f"  Order 1001 Status: {order1.status}, Assigned: {order1.assigned_vehicle_id}")
    print(f"  Order 1002 Status: {order2.status}, Assigned: {order2.assigned_vehicle_id}")
    print(f"  Order 1003 Status: {order3.status}, Assigned MicroHub: {order3.assigned_micro_hub_id}")
    print(f"  Truck 101 Status: {truck1.status}, Cargo: {truck1.get_cargo()}")
    print(f"  Drone 201 Status: {drone1.status}, Cargo: {drone1.get_cargo()}")
    print(f"  MicroHub 3 Packages: {micro_hub3.get_packages()}")
    print(f"  Node 0 (Depot) Packages: {node0.get_packages()}")

    # 1. Accept Order
    print("\n1. Accepting Order 1001:")
    supply_chain_manager.accept_order(1001)
    print(f"  Order 1001 Status: {order1.status}")

    # 2. Prioritize Order
    print("\n2. Prioritizing Order 1002:")
    supply_chain_manager.prioritize_order(1002, 3)
    print(f"  Order 1002 Priority: {order2.priority}")

    # 3. Assign Order to Truck
    print("\n3. Assigning Order 1001 to Truck 101:")
    # First, put order 1001 at the depot (node 0) for loading demonstration
    node0.add_package(1001)
    print(f"  Order 1001 now at Node 0. Node 0 packages: {node0.get_packages()}")

    supply_chain_manager.assign_order_to_truck(1001, 101)
    print(f"  Order 1001 Status: {order1.status}, Assigned: {order1.assigned_vehicle_id}")
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}")
    print(f"  Node 0 (Depot) Packages: {node0.get_packages()} (Order 1001 should be gone)")

    # 4. Assign Order to Drone (assuming drone is at node 0 and order 1002 is also there)
    print("\n4. Assigning Order 1002 to Drone 201:")
    node0.add_package(1002)
    print(f"  Order 1002 now at Node 0. Node 0 packages: {node0.get_packages()}")
    supply_chain_manager.assign_order_to_drone(1002, 201)
    print(f"  Order 1002 Status: {order2.status}, Assigned: {order2.assigned_vehicle_id}")
    print(f"  Drone 201 Cargo: {drone1.get_cargo()}")
    print(f"  Node 0 (Depot) Packages: {node0.get_packages()} (Order 1002 should be gone)")

    # 5. Assign Order to MicroHub
    print("\n5. Assigning Order 1003 to MicroHub 3:")
    # First, put order 1003 at the depot (node 0) to simulate transfer
    node0.add_package(1003)
    print(f"  Order 1003 now at Node 0. Node 0 packages: {node0.get_packages()}")
    micro_hub3.activate()  # MicroHub must be active for assignment
    supply_chain_manager.assign_order_to_micro_hub(1003, 3)
    print(f"  Order 1003 Status: {order3.status}, Assigned MicroHub: {order3.assigned_micro_hub_id}")
    print(f"  MicroHub 3 Packages: {micro_hub3.get_packages()}")
    print(f"  Node 0 (Depot) Packages: {node0.get_packages()} (Order 1003 should be gone)")

    # 6. Consolidate for Truck (assuming truck is at node 0, and new orders arrive)
    print("\n6. Consolidating for Truck 101:")
    # Add some new orders to depot
    new_order_id_1 = 1004
    new_order_id_2 = 1005
    new_order_1 = Order(id=new_order_id_1, customer_node_id=1, time_received=100.0, SLA_deadline=3000.0, priority=1)
    new_order_2 = Order(id=new_order_id_2, customer_node_id=2, time_received=100.0, SLA_deadline=3000.0, priority=1)
    global_state.add_entity(new_order_1)
    global_state.add_entity(new_order_2)
    node0.add_package(new_order_id_1)
    node0.add_package(new_order_id_2)
    print(f"  New orders {new_order_id_1}, {new_order_id_2} added to Node 0. Node 0 packages: {node0.get_packages()}")

    truck1.current_node_id = 0  # Ensure truck is at depot
    truck1.set_status("idle")  # Ensure truck is idle
    consolidated = supply_chain_manager.consolidate_for_truck(101)
    print(f"  Orders consolidated onto Truck 101: {consolidated}")
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}")
    print(f"  Node 0 (Depot) Packages: {node0.get_packages()}")

    # 7. Consolidate for Drone (assuming drone is at micro_hub 3, and new orders arrive there)
    print("\n7. Consolidating for Drone 201:")
    new_order_id_3 = 1006
    new_order_3 = Order(id=new_order_id_3, customer_node_id=1, time_received=100.0, SLA_deadline=3000.0, priority=1)
    global_state.add_entity(new_order_3)
    micro_hub3.add_package_to_holding(new_order_id_3)
    print(f"  New order {new_order_id_3} added to MicroHub 3. MicroHub 3 packages: {micro_hub3.get_packages()}")

    drone1.current_node_id = 3  # Ensure drone is at micro_hub 3
    drone1.set_status("idle")  # Ensure drone is idle
    consolidated_drone = supply_chain_manager.consolidate_for_drone(201)
    print(f"  Orders consolidated onto Drone 201: {consolidated_drone}")
    print(f"  Drone 201 Cargo: {drone1.get_cargo()}")
    print(f"  MicroHub 3 Packages: {micro_hub3.get_packages()}")

    # 8. Reassign Order
    print("\n8. Reassigning Order 1004 (from Truck 101) to Drone 201:")
    # Ensure drone has capacity and is available
    drone1.max_payload_capacity = 2  # Increase capacity for demo
    drone1.set_status("idle")

    reassigned = supply_chain_manager.reassign_order(1004, 201)
    print(f"  Order 1004 reassigned: {reassigned}")
    print(
        f"  Order 1004 Status: {global_state.get_entity('order', 1004).status}, Assigned: {global_state.get_entity('order', 1004).assigned_vehicle_id}")
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}")  # Should no longer have 1004
    print(f"  Drone 201 Cargo: {drone1.get_cargo()}")  # Should now have 1004

    # 9. Cancel Order
    print("\n9. Cancelling Order 1005 (was on Truck 101):")
    supply_chain_manager.cancel_order(1005)
    print(f"  Order 1005 Status: {global_state.get_entity('order', 1005).status}")
    print(f"  Truck 101 Cargo: {truck1.get_cargo()}")  # Should no longer have 1005

    # 10. Flag for Re-delivery
    print("\n10. Flagging Order 1001 for Re-delivery (was assigned to Truck 101):")
    # Simulate order 1001 being at node 1 (customer node) but not delivered
    # (e.g., delivery failed, so it's put back at the node)
    # For demo, we'll just change its status and unassign from truck
    order1.update_status("assigned")  # Reset status for demo
    order1.assign_vehicle(101)
    truck1.add_cargo(1001)
    print(f"  Order 1001 is now assigned to Truck 101. Truck Cargo: {truck1.get_cargo()}")

    supply_chain_manager.flag_for_re_delivery(1001)
    print(f"  Order 1001 Status: {order1.status}")
    print(f"  Order 1001 Assigned Vehicle: {order1.assigned_vehicle_id} (should be None)")
    print(f"  Truck 101 Cargo: {truck1.get_cargo()} (Order 1001 should be gone)")

    print("\n--- SupplyChainManager Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_supply_chain_manager_functionality()

