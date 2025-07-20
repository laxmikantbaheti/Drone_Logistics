import sys
import os

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the Edge class
from ddls_src.entities.edge import Edge

def demonstrate_edge_functionality():
    """
    Demonstrates the basic functionalities of the Edge class.
    """
    print("--- Demonstrating Edge Class Functionality ---")

    # 1. Initialize an Edge
    print("\n1. Initializing an Edge between Node 10 and Node 11:")
    edge1 = Edge(id=100, start_node_id=10, end_node_id=11, length=28.28, base_travel_time=1696.8)
    print(f"  Edge ID: {edge1.id}")
    print(f"  Start Node ID: {edge1.start_node_id}")
    print(f"  End Node ID: {edge1.end_node_id}")
    print(f"  Length: {edge1.length:.2f} units")
    print(f"  Base Travel Time: {edge1.base_travel_time:.2f} seconds")
    print(f"  Current Traffic Factor: {edge1.current_traffic_factor}")
    print(f"  Is Blocked: {edge1.is_blocked}")
    print(f"  Drone Flight Impact Factor: {edge1.drone_flight_impact_factor}")

    # 2. Calculate Travel Times
    print("\n2. Calculating Travel Times:")
    print(f"  Current Truck Travel Time (initial): {edge1.get_current_travel_time():.2f} seconds")
    print(f"  Current Drone Flight Time (initial): {edge1.get_drone_flight_time():.2f} seconds")

    # 3. Modify Traffic Factor
    print("\n3. Modifying Traffic Factor:")
    new_traffic_factor = 1.5 # 50% slower
    edge1.set_traffic_factor(new_traffic_factor)
    print(f"  Set Traffic Factor to: {new_traffic_factor}")
    print(f"  New Truck Travel Time: {edge1.get_current_travel_time():.2f} seconds")

    # 4. Modify Drone Flight Impact Factor
    print("\n4. Modifying Drone Flight Impact Factor:")
    new_drone_impact_factor = 2.0 # 100% slower for drones (e.g., bad weather)
    edge1.set_drone_flight_impact_factor(new_drone_impact_factor)
    print(f"  Set Drone Flight Impact Factor to: {new_drone_impact_factor}")
    print(f"  New Drone Flight Time: {edge1.get_drone_flight_time():.2f} seconds")

    # 5. Block and Unblock Edge
    print("\n5. Blocking and Unblocking Edge:")
    edge1.set_blocked(True)
    print(f"  Edge Blocked Status: {edge1.is_blocked}")
    print(f"  Truck Travel Time (when blocked): {edge1.get_current_travel_time()}") # Should be infinity
    print(f"  Drone Flight Time (when blocked): {edge1.get_drone_flight_time()}") # Should be infinity

    edge1.set_blocked(False)
    print(f"  Edge Blocked Status: {edge1.is_blocked}")
    print(f"  Truck Travel Time (after unblocking): {edge1.get_current_travel_time():.2f} seconds")

    print("\n--- Edge Class Functionality Demonstration Complete ---")

if __name__ == "__main__":
    demonstrate_edge_functionality()

