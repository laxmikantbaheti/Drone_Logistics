import numpy as np
from typing import Dict, Any, Tuple, Set, List
from pprint import pprint
import random

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
        self.trip_timer = 0


class MockGlobalState:
    def __init__(self):
        # FIX: Pass 'self' as the global_state reference to the mock objects
        self.orders = {
            0: MockOrder(id=0, status='pending', global_state=self),
            1: MockOrder(id=1, status='delivered', global_state=self),
            2: MockOrder(id=2, status='pending', global_state=self),
            3: MockOrder(id=3, status='non_existent', global_state=self),  # Dynamic order
            4: MockOrder(id=4, status='non_existent', global_state=self)  # Dynamic order
        }
        self.trucks = {
            101: MockTruck(id=101, status='idle', global_state=self),
            102: MockTruck(id=102, status='en_route', global_state=self),
            103: MockTruck(id=103, status='idle', global_state=self)
        }
        self.drones = {}
        self.current_cycle = 0

    def apply_action(self, action_tuple: Tuple):
        """A simple method to update the state based on an action."""
        action_type = action_tuple[0]
        if "ASSIGN_ORDER_TO_TRUCK" in action_type.name:
            order_id, truck_id = action_tuple[1], action_tuple[2]
            if order_id in self.orders and truck_id in self.trucks:
                self.orders[order_id].status = 'assigned'
                self.trucks[truck_id].status = 'en_route'
                self.trucks[truck_id].trip_timer = random.randint(3, 6)  # Truck will be busy for 3-6 cycles
                print(f"\n  >>> STATE CHANGE: Order {order_id} assigned to Truck {truck_id}. Truck is now 'en_route'.")

    def advance_cycle(self):
        """Simulates the passage of time, with dynamic events."""
        self.current_cycle += 1
        print(f"\n  >>> TIME PASSES: Advancing to Cycle {self.current_cycle}...")

        # Check for new order arrivals
        if self.current_cycle == 5 and self.orders[3].status == 'non_existent':
            self.orders[3].status = 'pending'
            print("  >>> DYNAMIC EVENT: Order 3 has arrived and is now 'pending'!")
        if self.current_cycle == 10 and self.orders[4].status == 'non_existent':
            self.orders[4].status = 'pending'
            print("  >>> DYNAMIC EVENT: Order 4 has arrived and is now 'pending'!")

        # Check for trucks finishing their trips
        for truck in self.trucks.values():
            if truck.status == 'en_route':
                truck.trip_timer -= 1
                if truck.trip_timer <= 0:
                    truck.status = 'idle'
                    print(f"  >>> DYNAMIC EVENT: Truck {truck.id} has completed its trip and is now 'idle'!")


# Import the actual components we are demonstrating
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.actions.state_action_mapper import ActionIndex, StateActionMapper
from ddls_src.actions.constraints.base import Constraint, OrderAssignableConstraint, VehicleAvailableConstraint
from ddls_src.actions.action_masker import ActionMasker


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

    # Action map and agent config must account for ALL potential orders/vehicles
    MAX_ORDERS = 5
    MAX_TRUCKS = 3

    action_map = {}
    idx = 0
    for o in range(MAX_ORDERS):
        for t_idx, t_id in enumerate([101, 102, 103]):
            action_map[(SimulationAction.ASSIGN_ORDER_TO_TRUCK, o, t_id)] = idx
            idx += 1

    reverse_action_map = {v: k for k, v in action_map.items()}
    constraints_to_use = [OrderAssignableConstraint(), VehicleAvailableConstraint()]
    agent_config = {'num_orders': MAX_ORDERS, 'num_vehicles': MAX_TRUCKS, 'vehicle_map': {0: 101, 1: 102, 2: 103}}

    action_masker = ActionMasker(global_state, action_map, constraints_to_use, agent_config)

    # 2. Simulation Loop
    agent_action_meanings = {
        0: "(Assign Order 0 to Truck 101)", 1: "(Assign Order 0 to Truck 102)", 2: "(Assign Order 0 to Truck 103)",
        3: "(Assign Order 1 to Truck 101)", 4: "(Assign Order 1 to Truck 102)", 5: "(Assign Order 1 to Truck 103)",
        6: "(Assign Order 2 to Truck 101)", 7: "(Assign Order 2 to Truck 102)", 8: "(Assign Order 2 to Truck 103)",
        9: "(Assign Order 3 to Truck 101)", 10: "(Assign Order 3 to Truck 102)", 11: "(Assign Order 3 to Truck 103)",
        12: "(Assign Order 4 to Truck 101)", 13: "(Assign Order 4 to Truck 102)", 14: "(Assign Order 4 to Truck 103)",
    }

    # NEW: Print the key for interpreting the agent action mask at the beginning
    print("\n--- Agent Action Space Key ---")
    for idx, meaning in agent_action_meanings.items():
        print(f"  - Index {idx}: {meaning}")
    print("----------------------------")

    for i in range(100):
        print(f"\n\n------------------- SIMULATION CYCLE {i + 1} -------------------")

        print("\n*** Current State of the World ***")
        for order_id, order in sorted(global_state.orders.items()):
            if order.status != 'non_existent':
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
        print(f"  - Raw Agent Mask (as 0s and 1s): {agent_mask.astype(int)}")

        # Simulate agent taking the first valid action
        print("\n[Step B] Simulating Agent Action...")
        valid_agent_actions = np.where(agent_mask)[0]
        if len(valid_agent_actions) > 0:
            action_to_take = valid_agent_actions[0]

            system_action_idx = action_masker._agent_to_system_map.get(action_to_take)
            system_action_tuple = reverse_action_map[system_action_idx]

            print(f"\n  - Agent chose to act: {agent_action_meanings[action_to_take]}")
            global_state.apply_action(system_action_tuple)
        else:
            print("\n  - Agent has no valid actions to take in this cycle.")

        # Advance the simulation time
        global_state.advance_cycle()

    print("\n=============================================")
    print("========= Demonstration Complete ==========")
    print("=============================================")


if __name__ == "__main__":
    demonstrate_action_masking()
