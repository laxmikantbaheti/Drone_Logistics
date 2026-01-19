import numpy as np
from typing import Dict, Any, Tuple, Set, List
from pprint import pprint
import random
import os

# --- Mock Objects and Imports ---
from mlpro.bf.systems import System
from mlpro.bf.events import Event
from ddls_src.entities.order import Order
from ddls_src.entities.vehicles.base import Vehicle
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.scenarios.generators.order_generator import OrderGenerator


# Create mock classes that inherit from the real classes
class MockOrder(Order):
    def __init__(self, id, status, global_state):
        super().__init__(p_id=id, global_state=global_state, customer_node_id=99, time_received=0, SLA_deadline=999)
        self.status = status


class MockTruck(Truck):
    def __init__(self, id, status, global_state):
        self.fuel_level = 100.0
        super().__init__(p_id=id, global_state=global_state, start_node_id=0, max_payload_capacity=2, max_speed=60,
                         initial_fuel=100, fuel_consumption_rate=0.1)
        self.status = status
        self.trip_timer = 0


class MockGlobalState:
    def __init__(self):
        self.orders = {
            0: MockOrder(id=0, status='pending', global_state=self),
            1: MockOrder(id=1, status='delivered', global_state=self),
        }
        self.trucks = {
            101: MockTruck(id=101, status='idle', global_state=self),
            102: MockTruck(id=102, status='en_route', global_state=self),
        }
        self.drones = {}
        self.nodes = {99: type('MockNode', (), {'id': 99, 'type': 'customer'})()}
        self.current_time = 0.0
        self.current_cycle = 0

    def add_entity(self, entity):
        # FIX: Accept any instance of the base Order class, not just MockOrder
        if isinstance(entity, Order):
            self.orders[entity.id] = entity

    def apply_action(self, action_tuple: Tuple):
        """A simple method to update the state based on an assignment action."""
        action_type = action_tuple[0]
        if "ASSIGN_ORDER_TO_TRUCK" in action_type.name:
            order_id, truck_id = action_tuple[1], action_tuple[2]
            if order_id in self.orders and truck_id in self.trucks:
                self.orders[order_id].status = 'assigned'
                self.trucks[truck_id].status = 'en_route'
                self.trucks[truck_id].trip_timer = random.randint(3, 6)
                print(f"\n  >>> STATE CHANGE: Order {order_id} assigned to Truck {truck_id}. Truck is now 'en_route'.")

    def advance_time(self, duration: float):
        """Simulates the passage of time for dynamic events."""
        self.current_time += duration
        self.current_cycle += 1
        print(f"\n  >>> TIME PASSES: Advancing to Cycle {self.current_cycle} (Time: {self.current_time}s)...")

        for truck in self.trucks.values():
            if truck.status == 'en_route':
                truck.trip_timer -= 1
                if truck.trip_timer <= 0:
                    truck.status = 'idle'
                    print(f"  >>> DYNAMIC EVENT: Truck {truck.id} has completed its trip and is now 'idle'!")


class MockEventManager:
    C_EVENT_NEW_ORDER = 'NEW_ORDER_CREATED'

    def __init__(self):
        self._handlers = {}

    def register_event_handler(self, p_event_id, p_handler):
        self._handlers[p_event_id] = p_handler

    def _raise_event(self, p_event_id, p_event_object):
        print(f"  >>> MOCK EVENT RAISED: ID='{p_event_id}', Order ID='{p_event_object.get_data()['order'].id}'")
        if p_event_id in self._handlers:
            self._handlers[p_event_id](p_event_id, p_event_object)


# Import the actual components we are demonstrating
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.actions.state_action_mapper import ActionIndex, StateActionMapper
from ddls_src.actions.constraints.base import Constraint, OrderAssignableConstraint, VehicleAvailableConstraint
from ddls_src.actions.action_masker import ActionMasker


# --- Demonstration Script ---

def demonstrate_action_masking():
    print("=============================================")
    print("=== Demonstrating Action Masking Strategy ===")
    print("=============================================")

    # 1. Initial Setup
    global_state = MockGlobalState()

    action_map = {
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 101): 0,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 102): 1,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, 101): 2,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, 102): 3,
    }
    reverse_action_map = {v: k for k, v in action_map.items()}
    constraints_to_use = [OrderAssignableConstraint(), VehicleAvailableConstraint()]

    MAX_ORDERS = 5
    MAX_TRUCKS = 2
    agent_config = {'num_orders': MAX_ORDERS, 'num_vehicles': MAX_TRUCKS, 'vehicle_map': {0: 101, 1: 102}}

    action_masker = ActionMasker(global_state, action_map, constraints_to_use, agent_config)

    # Setup the OrderGenerator and a mock event handler
    mock_event_manager = MockEventManager()
    order_generator_config = {"arrival_schedule": {"900.0": 1, "1800.0": 1}}
    order_generator = OrderGenerator(global_state, mock_event_manager, order_generator_config)

    # This handler simulates the job of the StateActionMapper
    def new_order_handler(p_event_id, p_event_object: Event):
        new_order = p_event_object.get_data()['order']
        global_state.add_entity(new_order)
        action_masker.mapper.update_for_new_order(new_order.id)
        print(f"  >>> MOCK HANDLER: Processed new Order {new_order.id} and updated action maps.")

    mock_event_manager.register_event_handler(MockEventManager.C_EVENT_NEW_ORDER, new_order_handler)

    # 2. Simulation Loop
    for i in range(15):
        print(
            f"\n\n------------------- SIMULATION CYCLE {i + 1} (Time: {global_state.current_time}s) -------------------")

        print("\n*** Current State of the World ***")
        for order_id, order in sorted(global_state.orders.items()):
            print(f"  - Order {order_id}: status='{order.status}'")
        for truck_id, truck in sorted(global_state.trucks.items()):
            print(f"  - Truck {truck_id}: status='{truck.status}'")
        print("**********************************")

        system_mask = action_masker.generate_system_mask()

        valid_system_actions = np.where(system_mask)[0]
        if len(valid_system_actions) > 0:
            action_to_take_idx = valid_system_actions[0]
            system_action_tuple = reverse_action_map.get(action_to_take_idx)
            if system_action_tuple:
                print(f"\n  - Agent chose to act: {system_action_tuple}")
                global_state.apply_action(system_action_tuple)
        else:
            print("\n  - Agent has no valid actions to take in this cycle.")

        # Advance time and trigger the generator
        global_state.advance_time(300.0)
        order_generator.generate(global_state.current_time)

    print("\n=============================================")
    print("========= Demonstration Complete ==========")
    print("=============================================")


if __name__ == "__main__":
    demonstrate_action_masking()
