import numpy as np
from typing import Dict, Any, Tuple, Set, List
from abc import ABC, abstractmethod
from collections import defaultdict

# MLPro Imports
from mlpro.bf.events import Event

# Local Imports
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.actions.constraints.base import Constraint


# Forward declarations
class GlobalState: pass


class Order: pass


# -------------------------------------------------------------------------
# -- ActionIndex (Dynamic)
# -------------------------------------------------------------------------

class ActionIndex:
    """
    Pre-processes the global action_map into a structured database of groups
    and subgroups for hyper-efficient lookups. Now supports dynamic updates.
    """

    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        self.actions_by_type: Dict[SimulationAction, Set[int]] = defaultdict(set)
        self.actions_involving_entity: Dict[Tuple, Set[int]] = defaultdict(set)
        self._next_action_index = 0
        self.global_state = global_state  # Store reference for dynamic updates

        print("ActionIndex: Building initial action database...")
        self._build_indexes(global_state, action_map)
        print("ActionIndex: Database built successfully.")

    def _build_indexes(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        """Parses the initial action_map to create structured groups of action indices."""
        for action_tuple, action_index in action_map.items():
            self._index_single_action(action_tuple, action_index)
            self._next_action_index = max(self._next_action_index, action_index + 1)

    def _index_single_action(self, action_tuple: Tuple, action_index: int):
        """Adds a single action tuple and its index to the database."""
        action_type = action_tuple[0]
        self.actions_by_type[action_type].add(action_index)

        if "ASSIGN" in action_type.name:
            order_id, vehicle_id = action_tuple[1], action_tuple[2]
            entity_type = "Truck" if vehicle_id in self.global_state.trucks else "Drone"
            self.actions_involving_entity[('Order', order_id)].add(action_index)
            self.actions_involving_entity[(entity_type, vehicle_id)].add(action_index)

        elif action_type in [SimulationAction.TRUCK_TO_NODE, SimulationAction.RE_ROUTE_TRUCK_TO_NODE]:
            truck_id = action_tuple[1]
            self.actions_involving_entity[('Truck', truck_id)].add(action_index)

    def add_actions_for_new_order(self, order_id: int, action_map_ref: Dict[Tuple, int]):
        """Dynamically adds all assignment actions for a new order."""
        print(f"ActionIndex: Adding new actions for Order ID {order_id}...")
        all_vehicle_ids = list(self.global_state.trucks.keys()) + list(self.global_state.drones.keys())

        for vehicle_id in all_vehicle_ids:
            action_enum = SimulationAction.ASSIGN_ORDER_TO_TRUCK if vehicle_id in self.global_state.trucks else SimulationAction.ASSIGN_ORDER_TO_DRONE
            new_action_tuple = (action_enum, order_id, vehicle_id)

            if new_action_tuple not in action_map_ref:
                new_action_index = self._next_action_index
                action_map_ref[new_action_tuple] = new_action_index
                self._index_single_action(new_action_tuple, new_action_index)
                self._next_action_index += 1


# -------------------------------------------------------------------------
# -- StateActionMapper (Now a Subscriber)
# -------------------------------------------------------------------------

class StateActionMapper:
    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int], constraints: List[Constraint]):
        self.global_state = global_state
        self.action_map = action_map
        self.action_index = ActionIndex(global_state, action_map)
        self.constraints = constraints
        self._invalidation_map: Dict[Tuple, Set[int]] = {}

        self._build_map()

    def _build_map(self):
        """Builds the initial map and can be called to refresh it."""
        self._invalidation_map.clear()
        for constraint in self.constraints:
            rule_map = constraint.get_invalidations(self.global_state, self.action_index)
            for state_tuple, action_indices in rule_map.items():
                if state_tuple not in self._invalidation_map:
                    self._invalidation_map[state_tuple] = set()
                self._invalidation_map[state_tuple].update(action_indices)

    def _update_for_new_order(self, order_id: int):
        """Dynamically updates the maps and indexes for a newly arrived order."""
        self.action_index.add_actions_for_new_order(order_id, self.action_map)
        print("StateActionMapper: Re-building invalidation map to include new order...")
        self._build_map()
        print("StateActionMapper: Invalidation map updated.")

    def handle_new_order_event(self, p_event_id, p_event_object: Event):
        """
        Event handler method that subscribes to 'NEW_ORDER_CREATED' events.
        """
        print(f"StateActionMapper: Received event '{p_event_id}'.")
        new_order: 'Order' = p_event_object.get_data().get('order')
        if new_order:
            self._update_for_new_order(new_order.id)

    def generate_mask(self) -> np.ndarray:
        mask = np.ones(len(self.action_map), dtype=bool)
        invalid_indices = set()

        for entity_type in ["orders", "trucks", "drones"]:
            # Check if the entity type exists in global_state before iterating
            if hasattr(self.global_state, entity_type):
                for entity in getattr(self.global_state, entity_type).values():
                    # Construct state tuple using a simple type name
                    simple_type_name = entity.__class__.__name__.replace("Mock", "")
                    state_tuple = (simple_type_name, entity.id, 'status', entity.status)
                    invalid_indices.update(self._invalidation_map.get(state_tuple, set()))

        if invalid_indices:
            valid_indices_to_update = [idx for idx in invalid_indices if idx < len(mask)]
            if valid_indices_to_update:
                mask[valid_indices_to_update] = False

        return mask
