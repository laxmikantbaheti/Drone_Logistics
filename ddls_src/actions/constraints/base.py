from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Set

# Local Imports
from ..action_enums import SimulationAction


# Forward declarations
class GlobalState: pass


class ActionIndex: pass


class Truck: pass  # Needed for isinstance check


class Drone: pass  # Needed for isinstance check


# -------------------------------------------------------------------------
# -- Pluggable Constraint Architecture
# -------------------------------------------------------------------------

class Constraint(ABC):
    """
    Abstract base class for a pluggable constraint rule. Each constraint is responsible
    for generating its own state-action invalidation map using the ActionIndex.
    """

    @abstractmethod
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        """
        Generates a dictionary mapping state tuples to the set of action indices they invalidate.

        Returns:
            Dict[Tuple, Set[int]]: The invalidation map for this specific rule.
        """
        pass


# --- Default Constraint Implementations ---

class OrderAssignableConstraint(Constraint):
    """Invalidates assignment actions if an order is not in an assignable state."""

    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        invalidation_map = {}
        # Get all assignment actions from the index
        assign_actions = action_index.actions_by_type[SimulationAction.ASSIGN_ORDER_TO_TRUCK].union(
            action_index.actions_by_type[SimulationAction.ASSIGN_ORDER_TO_DRONE])

        for order in global_state.orders.values():
            # Find all assignment actions that involve this specific order
            order_actions = action_index.actions_involving_entity[('Order', order.id)]
            relevant_actions = assign_actions.intersection(order_actions)

            if not relevant_actions: continue

            # Define which states of this order invalidate these actions
            for status in ["assigned", "in_transit", "delivered", "cancelled"]:
                state_tuple = ('Order', order.id, 'status', status)
                invalidation_map[state_tuple] = relevant_actions

        return invalidation_map


class VehicleAvailableConstraint(Constraint):
    """Invalidates actions if a vehicle is not available."""

    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        invalidation_map = {}
        # Define actions that require a vehicle to be available
        actions_requiring_availability = action_index.actions_by_type[SimulationAction.ASSIGN_ORDER_TO_TRUCK].union(
            action_index.actions_by_type[SimulationAction.ASSIGN_ORDER_TO_DRONE],
            action_index.actions_by_type[SimulationAction.TRUCK_TO_NODE])

        # We need to import the actual classes for isinstance to work
        from ...entities.vehicles.truck import Truck
        from ...entities.vehicles.drone import Drone

        all_vehicles = list(global_state.trucks.items()) + list(global_state.drones.items())
        for vehicle_id, vehicle in all_vehicles:
            entity_type = "Truck" if isinstance(vehicle, Truck) else "Drone"

            vehicle_actions = action_index.actions_involving_entity[(entity_type, vehicle_id)]
            relevant_actions = actions_requiring_availability.intersection(vehicle_actions)

            if not relevant_actions: continue

            for status in ["en_route", "maintenance", "broken_down"]:
                state_tuple = (entity_type, vehicle_id, 'status', status)
                invalidation_map[state_tuple] = relevant_actions

        return invalidation_map
