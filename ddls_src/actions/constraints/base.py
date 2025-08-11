from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Set
from collections import defaultdict
from pprint import pprint

# Local Imports
from ddls_src.actions.base import SimulationAction
from mlpro.bf.systems import System  # Import System for mock object inheritance


# Forward declarations
class GlobalState: pass


class ActionIndex: pass


class Truck: pass


class Drone: pass


# -------------------------------------------------------------------------
# -- Pluggable Constraint Architecture (Self-Configuring)
# -------------------------------------------------------------------------

class Constraint(ABC):
    """
    Abstract base class for a pluggable constraint rule.
    """

    def __init__(self):
        # Find all actions that use this constraint from the blueprint
        self.actions_to_check = set()
        for action in SimulationAction:
            # Check if the function's class is in the list of constraint classes
            if self.__class__ in [c.__class__ for c in action.constraints]:
                self.actions_to_check.add(action)

    @abstractmethod
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        """
        Generates the invalidation map for this specific rule.
        """
        pass


# --- Default Constraint Implementations ---

class OrderAssignableConstraint(Constraint):
    """Invalidates actions if an order is not in an assignable state."""

    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        invalidation_map = {}

        action_indices_to_check = set()
        for action in self.actions_to_check:
            action_indices_to_check.update(action_index.actions_by_type[action])

        for order in global_state.orders.values():
            # FIX: Use get_id() for MLPro compatibility
            order_id = order.get_id()
            order_actions = action_index.actions_involving_entity[('Order', order_id)]
            relevant_actions = action_indices_to_check.intersection(order_actions)

            if not relevant_actions: continue

            for status in ["assigned", "in_transit", "delivered", "cancelled", "non_existent"]:
                state_tuple = ('Order', order_id, 'status', status)
                invalidation_map[state_tuple] = relevant_actions

        return invalidation_map


class VehicleAvailableConstraint(Constraint):
    """Invalidates actions if a vehicle is not available."""

    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        invalidation_map = {}

        action_indices_to_check = set()
        for action in self.actions_to_check:
            action_indices_to_check.update(action_index.actions_by_type[action])

        from ddls_src.entities.vehicles.truck import Truck
        from ddls_src.entities.vehicles.drone import Drone

        all_vehicles = list(global_state.trucks.values()) + list(global_state.drones.values())
        for vehicle in all_vehicles:
            entity_type = "Truck" if isinstance(vehicle, Truck) else "Drone"
            # FIX: Use get_id() for MLPro compatibility
            vehicle_id = vehicle.get_id()

            vehicle_actions = action_index.actions_involving_entity[(entity_type, vehicle_id)]
            relevant_actions = action_indices_to_check.intersection(vehicle_actions)

            if not relevant_actions: continue

            for status in ["en_route", "maintenance", "broken_down"]:
                state_tuple = (entity_type, vehicle_id, 'status', status)
                invalidation_map[state_tuple] = relevant_actions

        return invalidation_map


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Create Mock Objects for the test
    class MockEntity(System):  # Inherit from System to get get_id()
        def __init__(self, p_id):
            super().__init__(p_id=p_id)
            self.status = 'idle'

        @staticmethod
        def setup_spaces():  # Required by System
            return None, None


    class MockGlobalState:
        def __init__(self):
            self.orders = {0: MockEntity(p_id=0)}
            self.trucks = {101: MockEntity(p_id=101)}
            self.drones = {}


    class MockActionIndex:
        def __init__(self):
            self.actions_by_type = defaultdict(set, {
                SimulationAction.ASSIGN_ORDER_TO_TRUCK: {0, 1},
                SimulationAction.TRUCK_TO_NODE: {2}
            })
            self.actions_involving_entity = defaultdict(set, {
                ('Order', 0): {0},
                ('Truck', 101): {0, 2}
            })


    mock_gs = MockGlobalState()
    mock_ai = MockActionIndex()

    print("--- Validating Self-Configuring Constraints ---")

    # 2. Instantiate and test OrderAssignableConstraint
    print("\n[A] Testing OrderAssignableConstraint...")
    order_constraint = OrderAssignableConstraint()
    order_invalidations = order_constraint.get_invalidations(mock_gs, mock_ai)
    print("  - Generated Invalidation Map:")
    pprint(order_invalidations)

    # 3. Instantiate and test VehicleAvailableConstraint
    print("\n[B] Testing VehicleAvailableConstraint...")
    # We need to mock the Truck class for isinstance to work
    from ddls_src.entities.vehicles.truck import Truck as RealTruck

    mock_gs.trucks[101].__class__ = RealTruck

    vehicle_constraint = VehicleAvailableConstraint()
    vehicle_invalidations = vehicle_constraint.get_invalidations(mock_gs, mock_ai)
    print("  - Generated Invalidation Map:")
    pprint(vehicle_invalidations)

    print("\n--- Validation Complete ---")
