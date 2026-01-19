import sys
import os

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the Order class
from ddls_src.entities.order import Order


def demonstrate_order_functionality():
    """
    Demonstrates the basic functionalities of the Order class.
    """
    print("--- Demonstrating Order Class Functionality ---")

    # 1. Initialize an Order
    print("\n1. Initializing a new Order:")
    order1 = Order(id=1001, customer_node_id=5, time_received=0.0, SLA_deadline=3600.0, priority=1)  # SLA 1 hour
    print(f"  Order ID: {order1.id}")
    print(f"  Customer Node ID: {order1.customer_node_id}")
    print(f"  Time Received: {order1.time_received:.2f}s")
    print(f"  SLA Deadline: {order1.SLA_deadline:.2f}s")
    print(f"  Priority: {order1.priority}")
    print(f"  Initial Status: {order1.status}")
    print(f"  Assigned Vehicle: {order1.assigned_vehicle_id}")
    print(f"  Delivery Time: {order1.delivery_time}")

    # 2. Update Order Status
    print("\n2. Updating Order Status:")
    order1.update_status("accepted")
    print(f"  Status after 'accepted': {order1.status}")
    order1.update_status("in_transit")
    print(f"  Status after 'in_transit': {order1.status}")

    # Attempt invalid status update
    try:
        order1.update_status("invalid_status")
    except ValueError as e:
        print(f"  Attempted invalid status update: {e}")
    print(f"  Status after invalid attempt (should be same): {order1.status}")

    # 3. Assign and Unassign Vehicle
    print("\n3. Assigning and Unassigning Vehicle:")
    order1.assign_vehicle(vehicle_id=101)
    print(f"  Assigned vehicle ID: {order1.assigned_vehicle_id}")
    print(f"  Status after vehicle assignment: {order1.status}")  # Should be 'assigned'

    order1.unassign_vehicle()
    print(f"  Assigned vehicle ID after unassignment: {order1.assigned_vehicle_id}")
    print(f"  Status after vehicle unassignment: {order1.status}")  # Should revert to 'pending' if it was 'assigned'

    # 4. Assign Micro-Hub
    print("\n4. Assigning Micro-Hub:")
    order1.assign_micro_hub(micro_hub_id=3)
    print(f"  Assigned Micro-Hub ID: {order1.assigned_micro_hub_id}")
    # Note: Status change for micro-hub assignment is handled by SupplyChainManager, not directly by Order.

    # 5. Get SLA Remaining Time
    print("\n5. Getting SLA Remaining Time:")
    current_time_1 = 600.0  # 10 minutes
    print(f"  At current time {current_time_1:.2f}s, SLA remaining: {order1.get_SLA_remaining(current_time_1):.2f}s")

    current_time_2 = 4000.0  # After SLA deadline
    print(
        f"  At current time {current_time_2:.2f}s, SLA remaining: {order1.get_SLA_remaining(current_time_2):.2f}s (negative means breached)")

    # 6. Final Status (e.g., delivered)
    print("\n6. Demonstrating Delivered Status:")
    order1.update_status("delivered")
    order1.delivery_time = 3500.0  # Manually set for demo, usually set by FleetManager
    print(f"  Status after 'delivered': {order1.status}")
    print(f"  Delivery Time: {order1.delivery_time:.2f}s")

    print("\n--- Order Class Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_order_functionality()

