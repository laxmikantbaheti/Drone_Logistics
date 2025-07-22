import numpy as np
from typing import Dict, Any, Tuple, Set, List
from pprint import pprint

# --- Mock Objects and Imports ---
# Import the REAL MLPro and entity classes to inherit from them
from mlpro.bf.systems import System
from ddls_src.entities.order import Order
from ddls_src.entities.vehicles.base import Vehicle
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone


# Create mock classes that inherit from the real classes
class MockOrder(Order):
    def __init__(self, id, status, global_state):
        # Initialize the parent with dummy data, then set the status for our test
        super().__init__(p_id=id, global_state=global_state, customer_node_id=99, time_received=0, SLA_deadline=999)
        self.status = status


class MockTruck(Truck):
    def __init__(self, id, status, global_state):
        # FIX: Set the problematic attribute BEFORE calling the parent constructor.
        self.fuel_level = 100.0

        # Initialize the parent with dummy data, then set the final status for our test
        super().__init__(p_id=id, global_state=global_state, start_node_id=0, max_payload_capacity=2, max_speed=60,
                         initial_fuel=100, fuel_consumption_rate=0.1)
        self.status = status


class MockGlobalState:
    def __init__(self):
        # FIX: Pass 'self' as the global_state reference to the mock objects
        self.orders = {
            0: MockOrder(id=0, status='pending', global_state=self),
            1: MockOrder(id=1, status='delivered', global_state=self)
        }
        self.trucks = {
            101: MockTruck(id=101, status='idle', global_state=self),
            102: MockTruck(id=102, status='en_route', global_state=self)
        }
        self.drones = {}


# Import the actual components we are demonstrating
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.actions.state_action_mapper import ActionIndex, StateActionMapper
from ddls_src.actions.constraints.base import Constraint, OrderAssignableConstraint, VehicleAvailableConstraint
from ddls_src.actions.action_masker import ActionMasker


# --- Helper function for clear output ---

def get_invalidation_reason(action_tuple: Tuple, global_state: MockGlobalState) -> str:
    """Checks the state and explains why a specific action is invalid."""
    action_type = action_tuple[0]
    reasons = []

    if "ASSIGN" in action_type.name:
        order_id, vehicle_id = action_tuple[1], action_tuple[2]
        order = global_state.orders.get(order_id)
        vehicle = global_state.trucks.get(vehicle_id)

        if order and order.status not in ['pending', 'accepted']:
            reasons.append(f"Order {order_id} is '{order.status}'")
        if vehicle and vehicle.status not in ['idle']:
            reasons.append(f"Truck {vehicle_id} is '{vehicle.status}'")

    elif "TRUCK_TO_NODE" in action_type.name:
        vehicle_id = action_tuple[1]
        vehicle = global_state.trucks.get(vehicle_id)
        if vehicle and vehicle.status not in ['idle']:
            reasons.append(f"Truck {vehicle_id} is '{vehicle.status}'")

    if not reasons:
        return "No constraints failed."
    return ", ".join(reasons)


# --- Demonstration Script ---

def demonstrate_action_masking():
    """
    A self-contained script to demonstrate the entire action masking pipeline.
    """
    print("=============================================")
    print("=== Demonstrating Action Masking Strategy ===")
    print("=============================================")

    print("\n[Step 1] Setting up mock GlobalState and action_map...")
    global_state = MockGlobalState()

    action_map = {
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 101): 0,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 102): 1,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, 101): 2,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, 102): 3,
        (SimulationAction.TRUCK_TO_NODE, 101, 5): 4,
        (SimulationAction.TRUCK_TO_NODE, 102, 5): 5,
    }
    reverse_action_map = {v: k for k, v in action_map.items()}

    print("\n*** Initial State of the World ***")
    print(f"  - Order 0: status='{global_state.orders[0].status}'")
    print(f"  - Order 1: status='{global_state.orders[1].status}'")
    print(f"  - Truck 101: status='{global_state.trucks[101].status}'")
    print(f"  - Truck 102: status='{global_state.trucks[102].status}'")
    print("**********************************")

    print("\n[Step 2] Defining the set of pluggable constraints...")
    constraints_to_use = [
        OrderAssignableConstraint(),
        VehicleAvailableConstraint()
    ]
    print(f"  - Using constraints: {[c.__class__.__name__ for c in constraints_to_use]}")

    print("\n[Step 3] Initializing the ActionMasker...")
    print("  - This will create the ActionIndex and StateActionMapper.")
    agent_config = {
        'num_orders': 2,
        'num_vehicles': 2,
        'vehicle_map': {0: 101, 1: 102}
    }
    action_masker = ActionMasker(global_state, action_map, constraints_to_use, agent_config)
    print("  - Initialization complete.")

    print("\n\n--- VALIDATION: Inspecting Internal Databases ---")
    print("\n[A] ActionIndex Database:")
    pprint(dict(action_masker.mapper.action_index.actions_by_type), indent=4)
    print("\n  - Actions grouped by entity involved:")
    pprint(dict(action_masker.mapper.action_index.actions_involving_entity), indent=4)

    print("\n[B] StateActionMapper's Final Invalidation Map:")
    pprint(action_masker.mapper._invalidation_map, indent=4)
    print("-------------------------------------------------")

    print("\n[Step 4] Generating the low-level System Mask...")
    system_mask = action_masker.generate_system_mask()

    print("\n  --- System Mask Results ---")
    for i, is_valid in enumerate(system_mask):
        action_tuple = reverse_action_map[i]
        status = "VALID" if is_valid else "INVALID"
        reason = get_invalidation_reason(action_tuple, global_state) if not is_valid else ""
        print(f"    - Action {i} {str(action_tuple):<50} -> {status:<8} | Reason: {reason}")

    print("\n[Step 5] Generating the high-level Agent Mask...")
    agent_mask = action_masker.generate_agent_mask(system_mask)

    print("\n  --- Agent Mask Results (Assignment-Only) ---")
    agent_action_meanings = {
        0: "(Assign Order 0 to Truck 101)",
        1: "(Assign Order 0 to Truck 102)",
        2: "(Assign Order 1 to Truck 101)",
        3: "(Assign Order 1 to Truck 102)"
    }

    for i, is_valid in enumerate(agent_mask):
        meaning = agent_action_meanings[i]
        status = "VALID" if is_valid else "INVALID"

        system_action_idx = action_masker._agent_to_system_map.get(i)
        reason = ""
        if not is_valid and system_action_idx is not None:
            system_tuple = reverse_action_map[system_action_idx]
            reason = get_invalidation_reason(system_tuple, global_state)

        print(f"    - Agent Action {i} {meaning:<35} -> {status:<8} | Reason: {reason}")

    print("\n=============================================")
    print("========= Demonstration Complete ==========")
    print("=============================================")


if __name__ == "__main__":
    demonstrate_action_masking()
