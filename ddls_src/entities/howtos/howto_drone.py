import sys
import os

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the Drone class
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.node import Node  # For context in movement demo


# Mock NetworkManager and GlobalState for move_along_route demonstration
class MockGlobalStateForDrone:
    def __init__(self):
        self.nodes = {
            0: Node(id=0, coords=(0.0, 0.0), type="depot"),
            1: Node(id=1, coords=(10.0, 0.0), type="customer"),
            2: Node(id=2, coords=(20.0, 0.0), type="customer"),
            3: Node(id=3, coords=(5.0, 5.0), type="micro_hub", is_charging_station=True)  # A charging node
        }
        self.network = MockNetworkForDrone(self.nodes)  # Mock Network instance
        self.current_time = 0.0

    def get_entity(self, entity_type, entity_id):
        if entity_type == "node": return self.nodes.get(entity_id)
        raise ValueError(f"Only node entity type supported in mock global state for {entity_type}")


class MockNetworkForDrone:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_edge_between_nodes(self, start_node_id, end_node_id):
        # Create a dummy edge for demonstration
        class DummyEdge:
            def __init__(self, nodes_ref):
                self.nodes = nodes_ref
                self.is_blocked = False

            # Assume 1 unit length = 1 second travel time for simplicity in this mock
            # So, 10 units length = 10 seconds travel time
            def get_current_travel_time(self): return 1.0  # Not used for drone, but required by Edge interface

            def get_drone_flight_time(self):
                start_coords = self.nodes[start_node_id].coords
                end_coords = self.nodes[end_node_id].coords
                length = ((start_coords[0] - end_coords[0]) ** 2 + (start_coords[1] - end_coords[1]) ** 2) ** 0.5
                return length * 1.0  # 1 second per unit length

        return DummyEdge(self.nodes)

    def calculate_shortest_path(self, start_node_id, end_node_id, vehicle_type):
        # Simple mock path: direct connection if exists
        if start_node_id in self.nodes and end_node_id in self.nodes:
            if self.get_edge_between_nodes(start_node_id, end_node_id):
                return [start_node_id, end_node_id]
        return []


class MockNetworkManagerForDrone:
    def __init__(self, global_state):
        self.global_state = global_state

    def handle_vehicle_arrival(self, vehicle_id, arrived_node_id):
        print(f"  MockNetworkManager: Vehicle {vehicle_id} arrived at node {arrived_node_id}.")


def demonstrate_drone_functionality():
    """
    Demonstrates the specific functionalities of the Drone class.
    """
    print("--- Demonstrating Drone Class Functionality ---")

    mock_global_state = MockGlobalStateForDrone()
    mock_network_manager = MockNetworkManagerForDrone(mock_global_state)

    # 1. Initialize a Drone
    print("\n1. Initializing a new Drone:")
    drone1 = Drone(id=201, start_node_id=0, max_payload_capacity=1, max_speed=30.0,
                   initial_battery=1.0, battery_drain_rate_flying=0.005,
                   battery_drain_rate_idle=0.001, battery_charge_rate=0.01)  # Rates per second
    print(f"  Drone ID: {drone1.id}")
    print(f"  Type: {drone1.type}")
    print(f"  Initial Battery: {drone1.battery_level * 100:.1f}%")
    print(f"  Flying Drain Rate: {drone1.battery_drain_rate_flying}")
    print(f"  Idle Drain Rate: {drone1.battery_drain_rate_idle}")
    print(f"  Charge Rate: {drone1.battery_charge_rate}")
    print(f"  Max Battery Capacity: {drone1.max_battery_capacity * 100:.1f}%")
    print(f"  Status: {drone1.status}")

    # 2. Drain Battery (Idle and Flying)
    print("\n2. Draining Battery:")
    drone1.set_status("idle")
    delta_time_idle = 100.0  # seconds
    drone1.drain_battery(delta_time_idle)
    print(f"  Battery after idle drain for {delta_time_idle}s: {drone1.battery_level * 100:.1f}%")

    drone1.set_status("en_route")  # Simulate flying
    delta_time_flying = 50.0  # seconds
    drone1.drain_battery(delta_time_flying)
    print(f"  Battery after flying drain for {delta_time_flying}s: {drone1.battery_level * 100:.1f}%")
    drone1.set_status("idle")  # Reset status

    # 3. Charge Battery
    print("\n3. Charging Battery:")
    drone1.battery_level = 0.5  # Set battery to 50% for demo
    drone1.set_status("charging")
    charge_time = 20.0  # seconds
    drone1.charge_battery(charge_time)
    print(f"  Battery after charging for {charge_time}s: {drone1.battery_level * 100:.1f}%")

    # Attempt to over-charge
    drone1.charge_battery(200.0)
    print(f"  Battery after over-charging (should cap at max): {drone1.battery_level * 100:.1f}%")
    drone1.set_status("idle")  # Reset status

    # 4. Movement with Battery Consumption
    print("\n4. Movement with Battery Consumption:")
    drone1.battery_level = 0.05  # Set battery to 5% for demonstration
    drone1.set_status("idle")
    drone1.set_route([0, 1])  # Route from Node 0 to Node 1 (length 10.0, travel time 10.0s)
    print(f"  Drone 201 battery set to {drone1.battery_level * 100:.1f}%. Route: {drone1.current_route}")
    print(f"  Simulating movement for 5.0 seconds (halfway to Node 1)...")
    drone1.move_along_route(5.0, mock_network_manager)
    print(f"  Drone 201 coords: {drone1.current_location_coords}, Battery: {drone1.battery_level * 100:.1f}%")

    print(f"  Simulating movement for another 6.0 seconds (should arrive and consume battery)...")
    drone1.move_along_route(6.0, mock_network_manager)  # 1s more than needed to arrive
    print(
        f"  Drone 201 current node: {drone1.current_node_id}, Coords: {drone1.current_location_coords}, Status: {drone1.status}, Battery: {drone1.battery_level * 100:.1f}%")
    print(f"  Drone 201 route: {drone1.current_route}")  # Should be empty

    # 5. Running out of Battery during Movement
    print("\n5. Running out of Battery during Movement:")
    drone2 = Drone(id=202, start_node_id=0, max_payload_capacity=1, max_speed=30.0,
                   initial_battery=0.0001, battery_drain_rate_flying=0.005,  # Very low battery
                   battery_drain_rate_idle=0.001, battery_charge_rate=0.01)
    drone2.set_route([0, 1])
    print(f"  Drone 202 battery set to {drone2.battery_level * 100:.1f}%. Route: {drone2.current_route}")
    print(f"  Simulating movement for 10.0 seconds (should run out of battery mid-route)...")
    drone2.move_along_route(10.0, mock_network_manager)
    print(
        f"  Drone 202 battery: {drone2.battery_level * 100:.1f}%, Status: {drone2.status}, Current Node: {drone2.current_node_id}")
    print(f"  Drone 202 route: {drone2.current_route}")  # Should be empty

    print("\n--- Drone Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_drone_functionality()

