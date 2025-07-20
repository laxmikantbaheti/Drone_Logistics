import sys
import os

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import concrete Vehicle types (Truck and Drone) to demonstrate base class methods
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone

# Import Node for context, though not strictly needed for Vehicle methods themselves
from ddls_src.entities.node import Node


# Forward declaration for NetworkManager as it's a dependency for move_along_route
# For this example, we'll create a mock NetworkManager
class MockNetworkManager:
    def __init__(self, global_state):
        self.global_state = global_state

    def get_edge_between_nodes(self, start_node_id, end_node_id):
        # Create a dummy edge for demonstration
        class DummyEdge:
            is_blocked = False

            def get_current_travel_time(self): return 60.0  # 1 minute

            def get_drone_flight_time(self): return 30.0  # 0.5 minute

        return DummyEdge()

    def handle_vehicle_arrival(self, vehicle_id, arrived_node_id):
        print(f"  MockNetworkManager: Vehicle {vehicle_id} arrived at node {arrived_node_id}.")


def demonstrate_vehicle_base_functionality():
    """
    Demonstrates the common functionalities of the Vehicle base class
    using concrete Truck and Drone instances.
    """
    print("--- Demonstrating Vehicle Base Class Functionality ---")

    # Create dummy nodes for vehicle starting points and routes
    node0 = Node(id=0, coords=(0.0, 0.0), type="depot")
    node1 = Node(id=1, coords=(10.0, 0.0), type="customer")
    node2 = Node(id=2, coords=(20.0, 0.0), type="customer")

    # Create a minimal mock global state for NetworkManager
    class MockGlobalState:
        def __init__(self):
            self.nodes = {0: node0, 1: node1, 2: node2}
            self.network = None  # Will be set by a mock Network instance if needed
            self.current_time = 0.0

        def get_entity(self, entity_type, entity_id):
            if entity_type == "node": return self.nodes.get(entity_id)
            raise ValueError("Only node entity type supported in mock global state")

    mock_global_state = MockGlobalState()
    mock_network_manager = MockNetworkManager(mock_global_state)

    # 1. Initialize Truck and Drone (inheriting from Vehicle)
    print("\n1. Initializing Truck and Drone:")
    truck1 = Truck(id=101, start_node_id=0, max_payload_capacity=5, max_speed=60.0, initial_fuel=100.0,
                   fuel_consumption_rate=0.1)
    drone1 = Drone(id=201, start_node_id=0, max_payload_capacity=1, max_speed=30.0, initial_battery=1.0,
                   battery_drain_rate_flying=0.005, battery_drain_rate_idle=0.001, battery_charge_rate=0.01)

    print(
        f"  Truck 101: Type={truck1.type}, Status={truck1.status}, Current Node={truck1.current_node_id}, Coords={truck1.current_location_coords}, Cargo={truck1.get_cargo()}")
    print(
        f"  Drone 201: Type={drone1.type}, Status={drone1.status}, Current Node={drone1.current_node_id}, Coords={drone1.current_location_coords}, Cargo={drone1.get_cargo()}")

    # 2. Set Status
    print("\n2. Setting Vehicle Status:")
    truck1.set_status("maintenance")
    print(f"  Truck 101 status changed to: {truck1.status}")
    drone1.set_status("charging")
    print(f"  Drone 201 status changed to: {drone1.status}")
    truck1.set_status("idle")  # Reset for next demos

    # 3. Add and Remove Cargo
    print("\n3. Adding and Removing Cargo:")
    truck1.add_cargo(order_id=1001)
    truck1.add_cargo(order_id=1002)
    print(f"  Truck 101 cargo: {truck1.get_cargo()}")
    truck1.remove_cargo(order_id=1001)
    print(f"  Truck 101 cargo after removal: {truck1.get_cargo()}")

    drone1.add_cargo(order_id=2001)
    print(f"  Drone 201 cargo: {drone1.get_cargo()}")
    drone1.remove_cargo(order_id=2001)
    print(f"  Drone 201 cargo after removal: {drone1.get_cargo()}")

    # 4. Set Route and Basic Movement (using mock NetworkManager)
    print("\n4. Setting Route and Basic Movement:")
    truck1.set_route([0, 1, 2])  # Route: Node 0 -> Node 1 -> Node 2
    print(f"  Truck 101 route set to: {truck1.current_route}, Status: {truck1.status}")
    print(f"  Truck 101 current segment: {truck1.get_current_route_segment()}")

    # Simulate movement for a short duration
    delta_time_move = 30.0  # seconds
    print(f"  Simulating Truck 101 movement for {delta_time_move} seconds...")
    truck1.move_along_route(delta_time_move, mock_network_manager)
    print(f"  Truck 101 current coords: {truck1.current_location_coords}, Progress: {truck1.route_progress:.2f}")

    print(f"  Simulating Truck 101 movement for another {delta_time_move} seconds (should arrive at Node 1)...")
    truck1.move_along_route(delta_time_move, mock_network_manager)  # Should arrive at Node 1 (travel time 60s)
    print(
        f"  Truck 101 current node: {truck1.current_node_id}, Coords: {truck1.current_location_coords}, Status: {truck1.status}")
    print(f"  Truck 101 remaining route: {truck1.current_route}")
    print(f"  Truck 101 current segment: {truck1.get_current_route_segment()}")

    # Simulate drone movement
    drone1.set_route([0, 2])  # Route: Node 0 -> Node 2
    print(f"  Drone 201 route set to: {drone1.current_route}, Status: {drone1.status}")
    print(f"  Simulating Drone 201 movement for {delta_time_move} seconds...")
    drone1.move_along_route(delta_time_move, mock_network_manager)
    print(f"  Drone 201 current coords: {drone1.current_location_coords}, Progress: {drone1.route_progress:.2f}")

    print(f"  Simulating Drone 201 movement for another {delta_time_move} seconds (should arrive at Node 2)...")
    drone1.move_along_route(delta_time_move, mock_network_manager)  # Should arrive at Node 2 (travel time 60s)
    print(
        f"  Drone 201 current node: {drone1.current_node_id}, Coords: {drone1.current_location_coords}, Status: {drone1.status}")
    print(f"  Drone 201 remaining route: {drone1.current_route}")

    # 5. Check if at node
    print("\n5. Checking if Vehicle is at Node:")
    print(f"  Is Truck 101 at Node 1? {truck1.is_at_node(1)}")
    print(f"  Is Drone 201 at Node 0? {drone1.is_at_node(0)}")

    print("\n--- Vehicle Base Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_vehicle_base_functionality()

