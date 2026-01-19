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
            1: MockOrder(id=1, status='delivered', global_state=self),
            2: MockOrder(id=2, status='pending', global_state=self)  # Added for a longer scenario
        }
        self.trucks = {
            101: MockTruck(id=101, status='idle', global_state=self),
            102: MockTruck(id=102, status='en_route', global_state=self),
            103: MockTruck(id=103, status='idle', global_state=self)  # Added for a longer scenario
        }
        self.drones = {}

    def apply_action(self, action_tuple: Tuple):
        """A simple method to update the state based on an action."""
        action_type = action_tuple[0]
        if "ASSIGN_ORDER_TO_TRUCK" in action_type.name:
            order_id, truck_id = action_tuple[1], action_tuple[2]
            if order_id in self.orders and truck_id in self.trucks:
                self.orders[order_id].status = 'assigned'
                self.trucks[truck_id].status = 'en_route'  # Assume it starts moving right away
                print(
                    f"\n  >>> STATE CHANGE: Order {order_id} status -> 'assigned', Truck {truck_id} status -> 'en_route'")


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
    A self-contained script to demonstrate the entire action masking pipeline over several steps.
    """
    print("=============================================")
    print("=== Demonstrating Action Masking Strategy ===")
    print("=============================================")

    # 1. Initial Setup
    global_state = MockGlobalState()
    # Expanded action map for the larger scenario
    action_map = {
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 101): 0,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 102): 1,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 103): 2,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, 101): 3,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, 102): 4,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, 103): 5,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 2, 101): 6,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 2, 102): 7,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 2, 103): 8,
        (SimulationAction.TRUCK_TO_NODE, 101, 5): 9,
        (SimulationAction.TRUCK_TO_NODE, 102, 5): 10,
        (SimulationAction.TRUCK_TO_NODE, 103, 5): 11,
    }
    reverse_action_map = {v: k for k, v in action_map.items()}
    constraints_to_use = [OrderAssignableConstraint(), VehicleAvailableConstraint()]
    # Expanded agent config
    agent_config = {'num_orders': 3, 'num_vehicles': 3, 'vehicle_map': {0: 101, 1: 102, 2: 103}}

    action_masker = ActionMasker(global_state, action_map, constraints_to_use, agent_config)

    # 2. Simulation Loop
    for i in range(15):  # Increased number of steps to 15
        print(f"\n\n------------------- SIMULATION CYCLE {i + 1} -------------------")

        print("\n*** Current State of the World ***")
        for order_id, order in sorted(global_state.orders.items()):
            print(f"  - Order {order_id}: status='{order.status}'")
        for truck_id, truck in sorted(global_state.trucks.items()):
            print(f"  - Truck {truck_id}: status='{truck.status}'")
        print("**********************************")

        # Generate masks for the current state
        print("\n[Step A] Generating Masks for Current State...")
        system_mask = action_masker.generate_system_mask()
        agent_mask = action_masker.generate_agent_mask(system_mask)
        print("  - Masks generated.")

        print("\n  --- Agent Mask Results ---")
        agent_action_meanings = {
            0: "(Assign Order 0 to Truck 101)", 1: "(Assign Order 0 to Truck 102)", 2: "(Assign Order 0 to Truck 103)",
            3: "(Assign Order 1 to Truck 101)", 4: "(Assign Order 1 to Truck 102)", 5: "(Assign Order 1 to Truck 103)",
            6: "(Assign Order 2 to Truck 101)", 7: "(Assign Order 2 to Truck 102)", 8: "(Assign Order 2 to Truck 103)",
        }
        for j, is_valid in enumerate(agent_mask):
            meaning = agent_action_meanings[j]
            status = "VALID" if is_valid else "INVALID"
            print(f"    - Agent Action {j} {meaning:<35} -> {status}")

        # Simulate agent taking the first valid action
        print("\n[Step B] Simulating Agent Action...")
        valid_agent_actions = np.where(agent_mask)[0]
        if len(valid_agent_actions) > 0:
            action_to_take = valid_agent_actions[0]
            print(f"  - Agent finds a valid action: Index {action_to_take} {agent_action_meanings[action_to_take]}")

            # Translate agent action back to system action to update the state
            system_action_idx = action_masker._agent_to_system_map.get(action_to_take)
            system_action_tuple = reverse_action_map[system_action_idx]

            # Apply the action's effect to the world state
            global_state.apply_action(system_action_tuple)

        else:
            print("  - No valid actions for the agent to take. Continuing simulation step.")

    print("\n=============================================")
    print("========= Demonstration Complete ==========")
    print("=============================================")


if __name__ == "__main__":
    demonstrate_action_masking()
