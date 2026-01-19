import sys
import os

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the Node class
from ddls_src.entities.node import Node

def demonstrate_node_functionality():
    """
    Demonstrates the basic functionalities of the Node class.
    """
    print("--- Demonstrating Node Class Functionality ---")

    # 1. Initialize a Node
    print("\n1. Initializing a Depot Node:")
    depot_node = Node(id=1, coords=(0.0, 0.0), type="depot",
                      is_loadable=True, is_unloadable=True, is_charging_station=True)
    print(f"  Depot Node ID: {depot_node.id}")
    print(f"  Type: {depot_node.type}")
    print(f"  Coordinates: {depot_node.coords}")
    print(f"  Is Loadable: {depot_node.is_loadable}")
    print(f"  Is Unloadable: {depot_node.is_unloadable}")
    print(f"  Is Charging Station: {depot_node.is_charging_station}")
    print(f"  Packages Held: {depot_node.get_packages()}")

    print("\n2. Initializing a Customer Node:")
    customer_node = Node(id=2, coords=(15.0, 10.0), type="customer",
                         is_loadable=False, is_unloadable=True, is_charging_station=False)
    print(f"  Customer Node ID: {customer_node.id}")
    print(f"  Type: {customer_node.type}")
    print(f"  Coordinates: {customer_node.coords}")
    print(f"  Is Loadable: {customer_node.is_loadable}")
    print(f"  Is Unloadable: {customer_node.is_unloadable}")
    print(f"  Is Charging Station: {customer_node.is_charging_station}")
    print(f"  Packages Held: {customer_node.get_packages()}")

    # 3. Add and Remove Packages
    print("\n3. Adding and Removing Packages from Depot Node:")
    depot_node.add_package(order_id=101)
    depot_node.add_package(order_id=102)
    print(f"  Depot Node packages after adding 101, 102: {depot_node.get_packages()}")
    depot_node.add_package(order_id=101) # Attempt to add duplicate
    print(f"  Depot Node packages after attempting duplicate add (should be same): {depot_node.get_packages()}")

    depot_node.remove_package(order_id=101)
    print(f"  Depot Node packages after removing 101: {depot_node.get_packages()}")
    depot_node.remove_package(order_id=999) # Attempt to remove non-existent
    print(f"  Depot Node packages after attempting to remove 999 (should be same): {depot_node.get_packages()}")

    # 4. Change Node Capabilities
    print("\n4. Changing Node Capabilities:")
    print(f"  Customer Node initial loadable status: {customer_node.is_loadable}")
    customer_node.set_loadable(True)
    print(f"  Customer Node new loadable status: {customer_node.is_loadable}")

    print(f"  Depot Node initial charging station status: {depot_node.is_charging_station}")
    depot_node.set_charging_station(False)
    print(f"  Depot Node new charging station status: {depot_node.is_charging_station}")

    print("\n--- Node Class Functionality Demonstration Complete ---")

if __name__ == "__main__":
    demonstrate_node_functionality()

