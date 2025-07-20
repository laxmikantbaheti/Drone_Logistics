import sys
import os

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the Truck class
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.node import Node  # For context in movement demo


# Mock NetworkManager and GlobalState for move_along_route demonstration
class MockGlobalStateForTruck:
    def __init__(self):
        self.nodes = {
            0: Node(id=0, coords=(0.0, 0.0), type="depot"),
            1: Node(id=1, coords=(10.0, 0.0), type="customer"),
            2: Node(id=2, coords=(20.0, 0.0), type="customer")
        }
        self.network = MockNetworkForTruck(self.nodes)  # Mock Network instance
        self.current_time = 0.0

    def get_entity(self, entity_type, entity_id):
        if entity_type == "node": return self.nodes.get(entity_id)
        raise ValueError(f"Only node entity type supported in mock global state for {entity_type}")


class MockNetworkForTruck:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_edge_between_nodes(self, start_node_id, end_node_id):
        # Create a dummy edge for demonstration
        # FIX: Pass self.nodes to DummyEdge so it can access node coordinates
        class DummyEdge:
            def __init__(self, nodes_ref):  # Take nodes_ref as argument
                self.nodes = nodes_ref  # Store reference to nodes
                self.is_blocked = False

            # Assume 1 unit length = 1 second travel time for simplicity in this mock
            # So, 10 units length = 10 seconds travel time
            def get_current_travel_time(self):
                # Calculate length between mock nodes using the stored nodes reference
                start_coords = self.nodes[start_node_id].coords
                end_coords = self.nodes[end_node_id].coords
                length = ((start_coords[0] - end_coords[0]) ** 2 + (start_coords[1] - end_coords[1]) ** 2) ** 0.5
                return length * 1.0  # 1 second per unit length

            def get_drone_flight_time(self): return 1.0  # Not used for truck, but required by Edge interface

        return DummyEdge(self.nodes)  # FIX: Pass self.nodes when creating DummyEdge instance

    def calculate_shortest_path(self, start_node_id, end_node_id, vehicle_type):
        # Simple mock path: direct connection if exists
        if start_node_id in self.nodes and end_node_id in self.nodes:
            if self.get_edge_between_nodes(start_node_id, end_node_id):
                return [start_node_id, end_node_id]
        return []


class MockNetworkManagerForTruck:
    def __init__(self, global_state):
        self.global_state = global_state

    def handle_vehicle_arrival(self, vehicle_id, arrived_node_id):
        print(f"  MockNetworkManager: Vehicle {vehicle_id} arrived at node {arrived_node_id}.")


def demonstrate_truck_functionality():
    """
    Demonstrates the specific functionalities of the Truck class.
    """
    print("--- Demonstrating Truck Class Functionality ---")

    mock_global_state = MockGlobalStateForTruck()
    mock_network_manager = MockNetworkManagerForTruck(mock_global_state)

    # 1. Initialize a Truck
    print("\n1. Initializing a new Truck:")
    truck1 = Truck(id=101, start_node_id=0, max_payload_capacity=5, max_speed=60.0,
                   initial_fuel=100.0, fuel_consumption_rate=0.1)  # Fuel consumption per second
    print(f"  Truck ID: {truck1.id}")
    print(f"  Type: {truck1.type}")
    print(f"  Initial Fuel: {truck1.fuel_level:.2f}")
    print(f"  Fuel Consumption Rate: {truck1.fuel_consumption_rate}")
    print(f"  Max Fuel Capacity: {truck1.max_fuel_capacity:.2f}")
    print(f"  Status: {truck1.status}")

    # 2. Consume Fuel
    print("\n2. Consuming Fuel:")
    delta_time_consume = 50.0  # seconds
    truck1.consume_fuel(delta_time_consume)
    print(f"  Fuel after consuming for {delta_time_consume}s: {truck1.fuel_level:.2f}")

    # 3. Update Energy (Refueling)
    print("\n3. Updating Energy (Refueling):")
    refuel_amount = 20.0
    truck1.update_energy(refuel_amount)
    print(f"  Fuel after refueling by {refuel_amount}: {truck1.fuel_level:.2f}")

    # Attempt to over-refuel
    truck1.update_energy(200.0)
    print(f"  Fuel after over-refueling (should cap at max): {truck1.fuel_level:.2f}")

    # 4. Movement with Fuel Consumption
    print("\n4. Movement with Fuel Consumption:")
    truck1.fuel_level = 20.0  # Set fuel to a low amount for demonstration
    truck1.set_status("idle")  # Ensure status is idle before setting route
    truck1.set_route([0, 1])  # Route from Node 0 to Node 1 (length 10.0, travel time 10.0s)
    print(f"  Truck 101 fuel set to {truck1.fuel_level:.2f}. Route: {truck1.current_route}")
    print(f"  Simulating movement for 5.0 seconds (halfway to Node 1)...")
    truck1.move_along_route(5.0, mock_network_manager)
    print(f"  Truck 101 coords: {truck1.current_location_coords}, Fuel: {truck1.fuel_level:.2f}")

    print(f"  Simulating movement for another 6.0 seconds (should arrive and consume fuel)...")
    truck1.move_along_route(6.0, mock_network_manager)  # 1s more than needed to arrive
    print(
        f"  Truck 101 current node: {truck1.current_node_id}, Coords: {truck1.current_location_coords}, Status: {truck1.status}, Fuel: {truck1.fuel_level:.2f}")
    print(f"  Truck 101 route: {truck1.current_route}")  # Should be empty

    # 5. Running out of Fuel during Movement
    print("\n5. Running out of Fuel during Movement:")
    truck2 = Truck(id=102, start_node_id=0, max_payload_capacity=5, max_speed=60.0,
                   initial_fuel=0.5, fuel_consumption_rate=0.1)  # Very low fuel
    truck2.set_route([0, 1])
    print(f"  Truck 102 fuel set to {truck2.fuel_level:.2f}. Route: {truck2.current_route}")
    print(f"  Simulating movement for 10.0 seconds (should run out of fuel mid-route)...")
    truck2.move_along_route(10.0, mock_network_manager)
    print(f"  Truck 102 fuel: {truck2.fuel_level:.2f}, Status: {truck2.status}, Current Node: {truck2.current_node_id}")
    print(f"  Truck 102 route: {truck2.current_route}")  # Should be empty

    print("\n--- Truck Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_truck_functionality()

