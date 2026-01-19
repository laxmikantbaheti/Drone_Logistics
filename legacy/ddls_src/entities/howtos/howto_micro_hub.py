import sys
import os

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the MicroHub and Drone classes
from ddls_src.entities.micro_hub import MicroHub
from ddls_src.entities.vehicles.drone import Drone  # To demonstrate charging slot assignment


def demonstrate_micro_hub_functionality():
    """
    Demonstrates the basic functionalities of the MicroHub class.
    """
    print("--- Demonstrating MicroHub Class Functionality ---")

    # 1. Initialize a MicroHub
    print("\n1. Initializing a new MicroHub:")
    micro_hub1 = MicroHub(id=301, coords=(50.0, 50.0), num_charging_slots=2)
    print(f"  MicroHub ID: {micro_hub1.id}")
    print(f"  Coordinates: {micro_hub1.coords}")
    print(f"  Type: {micro_hub1.type}")
    print(f"  Operational Status: {micro_hub1.operational_status}")
    print(f"  Charging Slots: {micro_hub1.charging_slots}")
    print(f"  Is Blocked for Launches: {micro_hub1.is_blocked_for_launches}")
    print(f"  Is Blocked for Recoveries: {micro_hub1.is_blocked_for_recoveries}")
    print(f"  Packages Held: {micro_hub1.get_packages()}")

    # 2. Activate and Deactivate MicroHub
    print("\n2. Activating and Deactivating MicroHub:")
    micro_hub1.activate()
    print(f"  Status after activation: {micro_hub1.operational_status}")
    micro_hub1.deactivate()
    print(f"  Status after deactivation: {micro_hub1.operational_status}")
    micro_hub1.activate()  # Activate again for further demos

    # 3. Manage Charging Slots
    print("\n3. Managing Charging Slots:")
    drone1 = Drone(id=201, start_node_id=301, max_payload_capacity=1, max_speed=30.0,
                   initial_battery=0.5, battery_drain_rate_flying=0.005, battery_drain_rate_idle=0.001,
                   battery_charge_rate=0.01)
    drone2 = Drone(id=202, start_node_id=301, max_payload_capacity=1, max_speed=30.0,
                   initial_battery=0.7, battery_drain_rate_flying=0.005, battery_drain_rate_idle=0.001,
                   battery_charge_rate=0.01)

    print(f"  Available slots initially: {micro_hub1.get_available_charging_slots()}")

    # Assign drone1 to a slot
    slot_assigned_drone1 = micro_hub1.assign_charging_slot(0, drone1.id)
    print(f"  Assigned drone {drone1.id} to slot 0: {slot_assigned_drone1}")
    print(f"  Current slots: {micro_hub1.charging_slots}")
    print(f"  Available slots: {micro_hub1.get_available_charging_slots()}")

    # Try to assign drone2 to the same slot (should fail)
    slot_assigned_drone2_fail = micro_hub1.assign_charging_slot(0, drone2.id)
    print(f"  Attempted to assign drone {drone2.id} to occupied slot 0: {slot_assigned_drone2_fail}")

    # Assign drone2 to another slot
    slot_assigned_drone2 = micro_hub1.assign_charging_slot(1, drone2.id)
    print(f"  Assigned drone {drone2.id} to slot 1: {slot_assigned_drone2}")
    print(f"  Current slots: {micro_hub1.charging_slots}")
    print(f"  Available slots: {micro_hub1.get_available_charging_slots()}")

    # Release a slot
    micro_hub1.release_charging_slot(0)
    print(f"  Released slot 0. Current slots: {micro_hub1.charging_slots}")
    print(f"  Available slots: {micro_hub1.get_available_charging_slots()}")

    # 4. Add and Remove Packages from Holding
    print("\n4. Adding and Removing Packages from Holding:")
    micro_hub1.add_package_to_holding(order_id=1001)
    micro_hub1.add_package_to_holding(order_id=1002)
    print(f"  Packages held after adding 1001, 1002: {micro_hub1.get_packages()}")
    micro_hub1.remove_package_from_holding(order_id=1001)
    print(f"  Packages held after removing 1001: {micro_hub1.get_packages()}")

    # 5. Flag and Release Services/Blocking
    print("\n5. Flagging and Releasing Services/Blocking:")
    micro_hub1.flag_service_unavailable("package_transfer")
    print(f"  Package transfer unavailable: {micro_hub1.is_package_transfer_unavailable}")
    micro_hub1.release_service_available("package_transfer")
    print(f"  Package transfer unavailable: {micro_hub1.is_package_transfer_unavailable}")

    micro_hub1.block_launches()
    print(f"  Blocked for launches: {micro_hub1.is_blocked_for_launches}")
    micro_hub1.unblock_launches()
    print(f"  Blocked for launches: {micro_hub1.is_blocked_for_launches}")

    print("\n--- MicroHub Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_micro_hub_functionality()

