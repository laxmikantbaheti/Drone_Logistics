from abc import ABC, abstractmethod
from collections import defaultdict
from ddls_src.actions.base import SimulationActions, ActionIndex
from ddls_src.entities import *
from ddls_src.entities.base import LogisticEntity
from ddls_src.entities.micro_hub import MicroHub
from ddls_src.entities.node import Node
from ddls_src.entities.order import PseudoOrder, Order, NodePair
from ddls_src.entities.vehicles.base import Vehicle
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.vehicles.truck import Truck
from mlpro.bf.events import Event, EventManager
from mlpro.bf.various import Log
from typing import Dict, Tuple, Set, List, Iterable


# -------------------------------------------------------------------------------------------------
# -- Part 1: Pluggable Constraint Architecture (Unified)
# -------------------------------------------------------------------------------------------------

class Constraint(ABC, EventManager):
    """
    Abstract base class for a pluggable constraint rule.
    """
    C_ACTIVE = True
    C_ASSOCIATED_ENTITIES = []
    C_ACTIONS_AFFECTED = []
    C_DEFAULT_EFFECT = True
    C_NAME = None
    C_EVENT_CONSTRAINT_UPDATE = "ConstraintUpdate"

    def __init__(self, p_reverse_action_map, p_action_index, custom_log=False):
        EventManager.__init__(self, p_logging=False)
        self.reverse_action_map = p_reverse_action_map

        # State tracking
        self._entity_invalidation_map = defaultdict(set)
        self.action_index = p_action_index
        self.associated_action_index = None
        self.find_associated_actions()
        self.initiated = False
        self.custom_log = custom_log
        self.evaluation_history = []

    def find_associated_actions(self):
        if self.C_ACTIONS_AFFECTED:
            self.associated_action_index = set(self.action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED))
        else:
            self.associated_action_index = set()

    def raise_constraint_change_event(self, p_entities, p_effect):
        p_event_data = {"entities": p_entities, "effect": p_effect}
        self._raise_event(p_event_id=Constraint.C_EVENT_CONSTRAINT_UPDATE,
                          p_event_object=Event(p_raising_object=self,
                                               p_event_data=p_event_data))

    # def get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
    #     if not self.initiated:
    #         self.initiated = self.initiate_masks()
    #     return self._get_restricted_actions(p_entity, p_action_index, **p_kwargs)

    # @abstractmethod
    # def initiate_masks(self):
    #     """Initiate the first default masks of the system"""
    #
    #     raise NotImplementedError

    @abstractmethod
    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        """
        Pure logic method: Determines which actions should be RESTRICTED (masked) based on CURRENT state.
        Returns: (List of indices to MASK, List of indices to UNMASK)
        """
        raise NotImplementedError

    def evaluate_impact(self, p_entity, p_action_index: ActionIndex, deck) -> Tuple[List[int], List[int]]:
        """
        Calculates the Delta (Impact) of this constraint.
        """
        # 1. Get current "Desired Blocks"
        current_actions_to_block, current_actions_to_unblock = self._get_restricted_actions(p_entity, p_action_index)
        current_block_set = set(current_actions_to_block) if current_actions_to_block else set()

        # 2. Get "Previous Blocks"
        entity_id = p_entity.get_id()
        previous_block_set = self._entity_invalidation_map[entity_id]

        # 3. Calculate Deltas
        to_block = list(current_block_set.difference(previous_block_set))
        to_unblock = list(previous_block_set.difference(current_block_set))
        for action in to_block:
            deck[action].add(f"{self.C_NAME} - {p_entity.C_NAME} {p_entity.get_id()}")
        for action in to_unblock:
            deck[action].remove(f"{self.C_NAME} - {p_entity.C_NAME} {p_entity.get_id()}")
        # 4. Update Memory
        self._entity_invalidation_map[entity_id] = current_block_set

        self.evaluation_history.append(
            f"{p_entity.C_NAME} - {p_entity.get_id()} --> to block: {current_actions_to_block}, to unblock: {current_actions_to_unblock}")

        return to_block, to_unblock

    def clear_cache(self):
        self._entity_invalidation_map.clear()

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        """Legacy method for full invalidation calculation."""
        return [], []

    def update_operability(self, p_entity: LogisticEntity, **p_kwargs):
        pass


# -------------------------------------------------------------------------------------------------
# -- Part 2: Concrete Constraints
# -------------------------------------------------------------------------------------------------

class VehicleAvailableConstraint(Constraint):
    C_NAME = "VehicleAvailabilityConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE,
                          SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]

    def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
        if not isinstance(p_entity, Vehicle):
            raise TypeError(f"The {self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} types for associated entities")

        related_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
        if p_entity.get_state_value_by_dim_name(Vehicle.C_DIM_AVAILABLE[0]):
            return [], list(related_actions)
        return list(related_actions), []

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        if not isinstance(p_entity, Vehicle):
            raise TypeError("Vehicle Availability Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        if vehicle.get_state_value_by_dim_name(Vehicle.C_DIM_AVAILABLE[0]):
            ids_to_unblock = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            return [], list(ids_to_unblock)
        else:
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_type.intersection(actions_by_entity))
            return invalidation_idx, []

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, Vehicle):
            return
        is_available = p_entity.get_state_value_by_dim_name(Vehicle.C_DIM_AVAILABLE[0])
        is_operable = bool(is_available)
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_operable


#
# class VehicleAtDeliveryNodeConstraint(Constraint):
#     C_NAME = "VehicleAtDeliveryNodeConstraint"
#     C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
#     C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
#                           SimulationActions.UNLOAD_DRONE_ACTION,
#                           SimulationActions.DRONE_LAND,
#                           SimulationActions.DRONE_LAUNCH]
#
#     def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
#         if not isinstance(p_entity, Vehicle):
#             raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
#
#         relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
#         if p_entity.current_node_id is not None:
#             relevant_orders = [o for o in p_entity.get_current_cargo() if
#                                o.get_delivery_node_id() == p_entity.current_node_id]
#             if len(relevant_orders) > 0:
#                 actions_related_to_valid_orders = set()
#                 for o in relevant_orders:
#                     actions_related_to_valid_orders.update(
#                         o.associated_actions.intersection(p_entity.associated_action_indexes).intersection(
#                             self.associated_action_index)
#                     )
#                 actions_to_block = relevant_actions.difference(actions_related_to_valid_orders)
#                 return list(actions_to_block), list(actions_related_to_valid_orders)
#             else:
#                 return list(relevant_actions), []
#         return list(relevant_actions), []
#
#     # --- [LEGACY METHODS] ---
#     def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
#         if not isinstance(p_entity, (Drone, Truck)):
#             raise TypeError("Vehicle At Delivery Node Constraint can only be applied to a vehicle entity.")
#
#         vehicle = p_entity
#         node_vehicle = vehicle.get_current_node()
#         delivery_orders = vehicle.get_delivery_orders()
#
#         is_at_delivery_node = False
#         if delivery_orders:
#             for order_obj in delivery_orders:
#                 try:
#                     if order_obj.get_delivery_node_id() == node_vehicle:
#                         is_at_delivery_node = True
#                         break
#                 except KeyError:
#                     continue
#
#         if is_at_delivery_node:
#             actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
#             actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
#
#             relevant_orders = [o.get_id() for o in vehicle.delivery_orders if o.get_delivery_node_id() == node_vehicle]
#             relevant_actions_by_order = set()
#             for o in relevant_orders:
#                 actions_by_order = p_action_index.actions_involving_entity[("Order", o)]
#                 relevant_actions_by_order.update(actions_by_order.intersection(actions_by_type))
#
#             actions_to_mask = actions_by_entity.intersection(actions_by_type)
#             actions_to_mask = list(actions_to_mask.difference(relevant_actions_by_order))
#             return actions_to_mask, []
#
#         actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
#         actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
#         actions_to_mask = actions_by_entity.intersection(actions_by_type)
#         return list(actions_to_mask), []
#
#     def update_operability(self, p_entity, **p_kwargs):
#         if not isinstance(p_entity, (Truck, Drone)):
#             return
#         vehicle = p_entity
#         node_vehicle = vehicle.get_current_node()
#         delivery_orders = vehicle.get_delivery_orders()
#         valid_orders = set()
#         if delivery_orders:
#             for order_obj in delivery_orders:
#                 try:
#                     if order_obj.get_delivery_node_id() == node_vehicle:
#                         valid_orders.add(order_obj.get_id())
#                 except (KeyError, AttributeError):
#                     continue
#         is_operable = valid_orders if valid_orders else False
#         for action_type in self.C_ACTIONS_AFFECTED:
#             if action_type in p_entity.action_operability:
#                 p_entity.action_operability[action_type] = is_operable
#
#
# class VehicleAtPickUpNodeConstraint(Constraint):
#     C_NAME = "VehicleAtPickUpNodeConstraint"
#     C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
#     C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
#                           SimulationActions.LOAD_DRONE_ACTION,
#                           SimulationActions.DRONE_LAUNCH,
#                           SimulationActions.DRONE_LAND]
#
#     def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
#         if not isinstance(p_entity, Vehicle):
#             raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
#
#         relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
#         if p_entity.current_node_id is not None:
#             relevant_orders = [o for o in p_entity.get_pickup_orders() if
#                                o.get_pickup_node_id() == p_entity.current_node_id]
#             if len(relevant_orders) > 0:
#                 actions_related_to_valid_orders = set()
#                 for o in relevant_orders:
#                     actions_related_to_valid_orders.update(
#                         o.associated_actions.intersection(p_entity.associated_action_indexes).intersection(
#                             self.associated_action_index)
#                     )
#                 actions_to_block = relevant_actions.difference(actions_related_to_valid_orders)
#                 return list(actions_to_block), list(actions_related_to_valid_orders)
#             else:
#                 return list(relevant_actions), []
#         return list(relevant_actions), []
#
#     # --- [LEGACY METHODS] ---
#     def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
#         if not isinstance(p_entity, (Drone, Truck)):
#             raise TypeError("Vehicle At PickUp Node Constraint can only be applied to a vehicle entity.")
#
#         vehicle = p_entity
#         node_vehicle = vehicle.get_current_node()
#         pickup_orders = vehicle.get_pickup_orders()
#
#         is_at_pickup_node = False
#         if pickup_orders:
#             for order_obj in pickup_orders:
#                 try:
#                     if order_obj.get_pickup_node_id() == node_vehicle:
#                         is_at_pickup_node = True
#                         break
#                 except KeyError:
#                     continue
#
#         if is_at_pickup_node:
#             actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
#             actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
#
#             relevant_orders = [o.get_id() for o in vehicle.pickup_orders if o.get_pickup_node_id() == node_vehicle]
#             relevant_actions_by_order = set()
#             for o in relevant_orders:
#                 actions_by_order = p_action_index.actions_involving_entity[("Order", o)]
#                 relevant_actions_by_order.update(actions_by_order.intersection(actions_by_type))
#
#             actions_to_mask = actions_by_entity.intersection(actions_by_type)
#             actions_to_mask = list(actions_to_mask.difference(relevant_actions_by_order))
#             return actions_to_mask, []
#
#         actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
#         actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
#         actions_to_mask = actions_by_entity.intersection(actions_by_type)
#         return list(actions_to_mask), []
#
#     def update_operability(self, p_entity, **p_kwargs):
#         if not isinstance(p_entity, (Truck, Drone)):
#             return
#         vehicle = p_entity
#         node_vehicle = vehicle.get_current_node()
#         if node_vehicle is None:
#             for action_type in self.C_ACTIONS_AFFECTED:
#                 if action_type in p_entity.action_operability:
#                     p_entity.action_operability[action_type] = False
#             return
#         pickup_orders = vehicle.get_pickup_orders()
#         valid_orders = set()
#         if pickup_orders:
#             for order_obj in pickup_orders:
#                 try:
#                     if order_obj.get_pickup_node_id() == node_vehicle:
#                         valid_orders.add(order_obj.get_id())
#                 except (KeyError, AttributeError):
#                     continue
#         is_operable = valid_orders if valid_orders else False
#         for action_type in self.C_ACTIONS_AFFECTED:
#             if action_type in p_entity.action_operability:
#                 p_entity.action_operability[action_type] = is_operable


class OrderRequestAssignabilityConstraint(Constraint):
    C_NAME = "OrderRequestAssignabilityConstraint"
    C_ACTIVE = True
    C_ASSOCIATED_ENTITIES = ["Node Pair"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    #
    # def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
    #     if not isinstance(p_entity, Order):
    #         raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entity.")
    #
    #     relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
    #     current_status = p_entity.get_state_value_by_dim_name(Order.C_DIM_DELIVERY_STATUS[0])
    #
    #     if current_status == Order.C_STATUS_PLACED:
    #         return [], list(relevant_actions)
    #     else:
    #         return list(relevant_actions), []

    #
    # def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
    #     if not isinstance(p_entity, NodePair):
    #         raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entity.")
    #     if p_entity.global_state is None:
    #         return [],[]
    #     associated_actions = set()
    #     for node_pair in p_entity.global_state.get_order_requests().keys():
    #         np_actions = p_action_index.actions_involving_entity["Node Pair", node_pair]
    #         associated_actions.update(np_actions)
    #     actions_to_mask = self.associated_action_index.difference(associated_actions)
    #     actions_to_unmask = self.associated_action_index.intersection(associated_actions)
    #
    #     return actions_to_mask, actions_to_unmask

    def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
        if not isinstance(p_entity, NodePair):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entity.")
        if p_entity.global_state is None:
            return [], []
        associated_actions = set()
        relevant_actions = p_entity.associated_action_indexes.intersection(self.associated_action_index)
        if p_entity.get_id() not in p_entity.global_state.get_order_requests():
            return list(relevant_actions), []
        else:
            return [], list(relevant_actions)

        return actions_to_mask, actions_to_unmask

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        vehicles = list(p_entity.global_state.trucks.values()) + list(p_entity.global_state.drones.values())
        unassignabile_vehicles = [veh for veh in vehicles if not veh.check_assignability()]
        vehicle_related_actions = p_action_index.get_actions_involving_entities("Vehicle", unassignabile_vehicles)
        inv_veh_asgn_actions = vehicle_related_actions.intersection(
            p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED))

        order_request_node_pairs = list(p_entity.global_state.get_order_requests().keys())
        invalid_order_requests = [pair for pair in p_entity.global_state.node_pairs if
                                  pair not in order_request_node_pairs]
        pair_related_actions = p_action_index.get_actions_involving_entities("Node Pair", invalid_order_requests)
        inv_pair_asgn_actions = pair_related_actions.intersection(
            p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED))

        invalidation_idx = list(inv_pair_asgn_actions.union(inv_veh_asgn_actions))
        return invalidation_idx, []

    def update_operability(self, p_entity, **p_kwargs):
        if isinstance(p_entity, (Truck, Drone)):
            is_assignable = p_entity.check_assignability()
            for action_type in self.C_ACTIONS_AFFECTED:
                if action_type in p_entity.action_operability:
                    p_entity.action_operability[action_type] = is_assignable

        elif isinstance(p_entity, Order):
            current_status = p_entity.get_state_value_by_dim_name(Order.C_DIM_DELIVERY_STATUS[0])
            is_active_request = (current_status == Order.C_STATUS_PLACED)
            for action_type in self.C_ACTIONS_AFFECTED:
                if action_type in p_entity.action_operability:
                    p_entity.action_operability[action_type] = is_active_request


# class VehicleAssignabilityConstraint(Constraint):
#     C_NAME = "Vehicle Assignability Constraint"
#     C_ACTIVE = True
#     C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
#     C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
#                           SimulationActions.ASSIGN_ORDER_TO_DRONE]

# def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
#     if not isinstance(p_entity, Vehicle):
#         raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
#
#     relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
#     state = p_entity.get_state_value_by_dim_name(p_entity.C_DIM_TRIP_STATE[0])
#     is_en_route = (state == p_entity.C_TRIP_STATE_EN_ROUTE)
#     is_available = p_entity.get_state_value_by_dim_name(p_entity.C_DIM_AVAILABLE[0])
#
#     if (not is_en_route) and is_available:
#         return [], list(relevant_actions)
#     else:
#         return list(relevant_actions), []
#
# # --- [LEGACY METHODS] ---
# def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
#     return [], []

class VehicleAssignabilityConstraint(Constraint):
    C_NAME = "Vehicle Assignability Constraint"
    C_ACTIVE = True
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
        if not isinstance(p_entity, Vehicle):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)

        # Fetch our new state boundaries
        current_status = p_entity.get_state_value_by_dim_name(p_entity.C_DIM_TRIP_STATE[0])
        is_locked = getattr(p_entity, 'consolidation_confirmed', False)
        is_available = p_entity.get_state_value_by_dim_name(p_entity.C_DIM_AVAILABLE[0])

        # The agent can ONLY assign orders if the vehicle is strictly IDLE, unlocked, and generally available
        if current_status == p_entity.C_TRIP_STATE_IDLE and not is_locked and is_available:
            return [], list(relevant_actions)  # Unblock
        else:
            return list(relevant_actions), []  # Block

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        return [], []


# class VehicleCapacityConstraint(Constraint):
#     C_NAME = "VehicleCapacityConstraint"
#     C_ACTIVE = True
#     C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
#     C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
#                           SimulationActions.ASSIGN_ORDER_TO_DRONE]
#
#     def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
#         if not isinstance(p_entity, Vehicle):
#             raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
#
#         relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
#         capacity = p_entity.get_cargo_capacity()
#         committed_load = len(p_entity.get_pickup_orders()) + len(p_entity.get_delivery_orders())
#
#         if committed_load >= capacity:
#             return list(relevant_actions), []
#         else:
#             return [], list(relevant_actions)
#
#     # --- [LEGACY METHODS] ---
#     def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
#         invalidation_idx = []
#         vehicle = p_entity
#
#         vehicle_capacity = vehicle.get_cargo_capacity()
#         committed_load = len(vehicle.get_pickup_orders()) + len(vehicle.get_delivery_orders())
#
#         if committed_load >= vehicle_capacity:
#             actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
#             actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
#             invalidation_idx = list(actions_by_entity.intersection(actions_by_type))
#             return invalidation_idx, []
#         else:
#             return [], []
#
#     def update_operability(self, p_entity, **p_kwargs):
#         if not isinstance(p_entity, (Truck, Drone)):
#             return
#         vehicle = p_entity
#         vehicle_capacity = vehicle.get_cargo_capacity()
#         current_cargo_size = vehicle.get_current_cargo_size()
#         has_capacity = (vehicle_capacity - current_cargo_size >= 1)
#         for action_type in self.C_ACTIONS_AFFECTED:
#             if action_type in p_entity.action_operability:
#                 p_entity.action_operability[action_type] = has_capacity

class VehicleCapacityConstraint(Constraint):
    C_NAME = "VehicleCapacityConstraint"
    C_ACTIVE = True
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
        if not isinstance(p_entity, Vehicle):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
        capacity = p_entity.get_cargo_capacity()

        # --- THE FIX: Use a set to prevent double-counting ---
        # This grabs everything on the clipboard and everything physically in the back of the truck
        # and ensures each unique order is only counted exactly once.
        active_orders = set(p_entity.get_pickup_orders() +
                            p_entity.get_delivery_orders() +
                            p_entity.get_current_cargo())

        committed_load = len(active_orders)

        if committed_load >= capacity:
            return list(relevant_actions), []
        else:
            return [], list(relevant_actions)

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        invalidation_idx = []
        vehicle = p_entity

        vehicle_capacity = vehicle.get_cargo_capacity()

        # Apply the exact same fix to the legacy method just to be safe
        active_orders = set(vehicle.get_pickup_orders() +
                            vehicle.get_delivery_orders() +
                            vehicle.get_current_cargo())
        committed_load = len(active_orders)

        if committed_load >= vehicle_capacity:
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_entity.intersection(actions_by_type))
            return invalidation_idx, []
        else:
            return [], []

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, (Truck, Drone)):
            return
        vehicle = p_entity
        vehicle_capacity = vehicle.get_cargo_capacity()

        # Update operability to also reflect the true committed load, not just physical cargo
        active_orders = set(vehicle.get_pickup_orders() +
                            vehicle.get_delivery_orders() +
                            vehicle.get_current_cargo())
        committed_load = len(active_orders)

        has_capacity = (vehicle_capacity - committed_load >= 1)

        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = has_capacity


class TripWithinRangeConstraint(Constraint):
    C_ACTIVE = False
    C_NAME = "TripWithinRangeConstraint"
    C_ASSOCIATED_ENTITIES = ["Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
        if not isinstance(p_entity, Drone):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
        available_range = p_entity.get_remaining_range()
        orders = p_entity.global_state.orders
        orders_out_of_range = []

        for order in orders.values():
            try:
                loc_pick_up = order.get_pickup_node_id()
                loc_delivery = order.get_delivery_node_id()
                distance = p_entity.global_state.network.calculate_distance(loc_pick_up, loc_delivery)
                if distance >= available_range:
                    orders_out_of_range.append(order)
            except:
                continue

        if not orders_out_of_range:
            return [], list(relevant_actions)

        actions_to_block = set()
        for order in orders_out_of_range:
            actions_by_order = p_action_index.actions_involving_entity.get(("Order", order.get_id()), set())
            actions_to_block.update(relevant_actions.intersection(actions_by_order))

        return list(actions_to_block), []

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        drone = p_entity
        if not isinstance(drone, Drone):
            raise ValueError("The 'Trip Within Range Constraint' is only applicable to Drones.")

        available_range = drone.get_remaining_range()
        orders = drone.global_state.orders
        orders_not_in_range = []

        for order in orders.values():
            loc_pick_up = order.get_pickup_node_id()
            loc_delivery = order.get_delivery_node_id()
            distance = drone.global_state.network.calculate_distance(loc_pick_up, loc_delivery)
            if distance >= available_range:
                orders_not_in_range.append(order)

        if not orders_not_in_range:
            return [], []

        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        actions_by_drone = p_action_index.actions_involving_entity[(drone.C_NAME, drone.get_id())]
        actions_by_orders = set()
        for order in orders_not_in_range:
            actions_by_orders.update(p_action_index.actions_involving_entity[(order.C_NAME, order.get_id())])

        invalidation_set = actions_by_drone.intersection(actions_by_type, actions_by_orders)
        return list(invalidation_set), []

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, Drone):
            return
        drone = p_entity
        available_range = drone.get_remaining_range()
        orders = drone.global_state.orders
        reachable_orders = set()
        for order in orders.values():
            try:
                loc_pick_up = order.get_pickup_node_id()
                loc_delivery = order.get_delivery_node_id()
                distance = drone.global_state.network.calculate_distance(loc_pick_up, loc_delivery)
                if distance < available_range:
                    reachable_orders.add(order.get_id())
            except (KeyError, AttributeError):
                continue
        is_operable = reachable_orders if reachable_orders else False
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_operable


class VehicleRoutingConstraint(Constraint):
    C_NAME = "Vehicle Routing Constraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]

    def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
        if not isinstance(p_entity, Vehicle):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        pickup_nodes = [o.get_pickup_node_id() for o in p_entity.get_pickup_orders()]
        delivery_nodes = [o.get_delivery_node_id() for o in p_entity.get_delivery_orders()]
        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)

        idx_to_unmask = set()
        for node_id in pickup_nodes + delivery_nodes:
            idx_to_unmask.update(p_action_index.actions_involving_entity.get(("Node", node_id), set()))

        ids_to_mask = relevant_actions.difference(idx_to_unmask)
        return list(ids_to_mask), list(idx_to_unmask)

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        if isinstance(p_entity, (Truck, Drone)):
            vehicle = p_entity
            pickup_nodes = [order.get_pickup_node_id() for order in vehicle.get_pickup_orders()]
            delivery_nodes = [order.get_delivery_node_id() for order in vehicle.get_delivery_orders()]

            all_possible_move_actions = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)

            idx_to_unmask = set()
            for node_id in pickup_nodes + delivery_nodes:
                idx_to_unmask.update(p_action_index.actions_involving_entity[("Node", node_id)])

            invalidation_idx = list(all_possible_move_actions.difference(idx_to_unmask))
            return invalidation_idx, []
        return [], []

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, (Truck, Drone)):
            return
        vehicle = p_entity
        valid_destinations = set()
        for order in vehicle.get_pickup_orders():
            try:
                valid_destinations.add(order.get_pickup_node_id())
            except (KeyError, AttributeError):
                continue
        for order in vehicle.get_delivery_orders():
            try:
                valid_destinations.add(order.get_delivery_node_id())
            except (KeyError, AttributeError):
                continue
        is_operable = valid_destinations if valid_destinations else False
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_operable


class ConsolidationConstraint(Constraint):
    C_NAME = "ConsolidationConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE_FOR_TRUCK,
                          SimulationActions.CONSOLIDATE_FOR_DRONE]

    # def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
    #     if not isinstance(p_entity, Vehicle):
    #         raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
    #
    #     consolidation_action_ids = self.associated_action_index.intersection(p_entity.associated_action_indexes)
    #     if p_entity.get_state_value_by_dim_name(p_entity.C_DIM_TRIP_STATE[0]) in [p_entity.C_TRIP_STATE_EN_ROUTE]:
    #         return list(consolidation_action_ids), []
    #
    #     if not (len(p_entity.get_pickup_orders()) or len(p_entity.get_delivery_orders())):
    #         return list(consolidation_action_ids), []
    #
    #     current_node_id = p_entity.get_current_node()
    #
    #     has_pending_loads = False
    #     for order in p_entity.get_pickup_orders():
    #         if order.get_pickup_node_id() == current_node_id:
    #             has_pending_loads = True
    #             break
    #
    #     has_pending_unloads = False
    #     if not has_pending_loads:
    #         for order in p_entity.get_delivery_orders():
    #             if order.get_delivery_node_id() == current_node_id:
    #                 has_pending_unloads = True
    #                 break
    #
    #     if has_pending_loads or has_pending_unloads:
    #         return list(consolidation_action_ids), []
    #     else:
    #         return [], list(consolidation_action_ids)

    def _get_restricted_actions(self, p_entity, p_action_index, **p_kwargs):
        if not isinstance(p_entity, Vehicle):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        consolidation_action_ids = self.associated_action_index.intersection(p_entity.associated_action_indexes)

        current_status = p_entity.get_state_value_by_dim_name(p_entity.C_DIM_TRIP_STATE[0])
        is_locked = getattr(p_entity, 'consolidation_confirmed', False)

        # 1. Guard: If it's already locked or not IDLE, block it immediately
        if current_status != p_entity.C_TRIP_STATE_IDLE and is_locked:
            return list(consolidation_action_ids), []

        # 2. Guard: Does it actually have orders to consolidate?
        # current_node = p_entity.current_node_id
        # nodes = p_entity.pickup_node_ids+p_entity.delivery_node_ids
        # if current_node in nodes:
        #     has_pending_tasks = True
        # else:
        #     has_pending_tasks = True
        #
        # if not has_pending_tasks:
        #     return list(consolidation_action_ids), []

        # If it is IDLE, unlocked, and has tasks waiting, allow consolidation!

        if (len(p_entity.staged_pickup_orders) or len(p_entity.staged_delivery_orders) or len(
                p_entity.staged_pickup_leg2_orders) or len(
                p_entity.staged_delivery_leg2_orders)) and not p_entity.consolidation_confirmed:
            return [], list(consolidation_action_ids)

        return list(consolidation_action_ids), []

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        invalidation_idx = []
        if not isinstance(p_entity, (Truck, Drone)):
            return invalidation_idx, []

        vehicle = p_entity
        is_ready_for_consolidation = (vehicle.get_state_value_by_dim_name(vehicle.C_DIM_TRIP_STATE[0])
                                      in [vehicle.C_TRIP_STATE_IDLE, vehicle.C_TRIP_STATE_HALT]
                                      and (len(vehicle.pickup_orders) > 0 or p_entity.get_current_cargo_size() > 0))

        vehicle_node_id = vehicle.current_node_id
        assigned_orders_at_node = (
                [ordr for ordr in vehicle.get_pickup_orders() if ordr.get_pickup_node_id() == vehicle_node_id]
                + [ordr for ordr in vehicle.get_delivery_orders() if
                   ordr.get_delivery_node_id() == vehicle_node_id])

        if len(assigned_orders_at_node):
            valid_relay_orders = True
            for ordr in assigned_orders_at_node:
                valid_relay_orders = ordr.check_order_precedence() and True
            is_ready_for_consolidation = not valid_relay_orders

        if not is_ready_for_consolidation:
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_type.intersection(actions_by_entity))

        return invalidation_idx, []

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, (Truck, Drone)):
            return
        vehicle = p_entity
        trip_state = vehicle.get_state_value_by_dim_name(Vehicle.C_DIM_TRIP_STATE[0])
        is_stable_state = (trip_state in [Vehicle.C_TRIP_STATE_IDLE, Vehicle.C_TRIP_STATE_HALT])
        has_orders = (len(vehicle.pickup_orders) > 0 or vehicle.get_current_cargo_size() > 0)
        is_ready = is_stable_state and has_orders
        if is_ready:
            current_node = vehicle.current_node_id
            orders_at_node = []
            if vehicle.pickup_orders:
                orders_at_node.extend([o for o in vehicle.pickup_orders if o.get_pickup_node_id() == current_node])
            if vehicle.delivery_orders:
                orders_at_node.extend([o for o in vehicle.delivery_orders if o.get_delivery_node_id() == current_node])
            if orders_at_node:
                are_orders_actionable = True
                for ordr in orders_at_node:
                    are_orders_actionable = ordr.check_order_precedence() and True
                is_ready = not are_orders_actionable
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_ready


class MicroHubAssignabilityConstraint(Constraint):
    C_NAME = "MicroHubAssignabilityConstraint"
    C_ASSOCIATED_ENTITIES = ["Node Pair"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB]

    # def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
    #     if not isinstance(p_entity, Order):
    #         raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
    #
    #     relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
    #     forbidden_hub_ids = set()
    #
    #     if isinstance(p_entity, PseudoOrder):
    #         if p_entity.parent_order.assigned_micro_hub_id is not None:
    #             forbidden_hub_ids.add(p_entity.parent_order.assigned_micro_hub_id)
    #         current_ancestor = p_entity.parent_order
    #         while isinstance(current_ancestor, PseudoOrder):
    #             if current_ancestor.assigned_micro_hub_id is not None:
    #                 forbidden_hub_ids.add(current_ancestor.assigned_micro_hub_id)
    #             current_ancestor = current_ancestor.parent_order
    #         if hasattr(current_ancestor, "assigned_micro_hub_id") and current_ancestor.assigned_micro_hub_id:
    #             forbidden_hub_ids.add(current_ancestor.assigned_micro_hub_id)
    #
    #     actions_to_block = set()
    #     if forbidden_hub_ids:
    #         for hub_id in forbidden_hub_ids:
    #             hub_actions = p_action_index.actions_involving_entity.get(("MicroHub", hub_id), set())
    #             actions_to_block.update(relevant_actions.intersection(hub_actions))
    #
    #     return list(actions_to_block), []

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
        if not isinstance(p_entity, NodePair):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = p_entity.associated_action_indexes.intersection(self.associated_action_index)
        order_requests = p_entity.global_state.get_order_requests()
        if p_entity.get_id() not in order_requests.keys():
            return [], list(relevant_actions)
        current_order = order_requests[p_entity.get_id()][0]
        if not isinstance(current_order, PseudoOrder):
            return [], list(relevant_actions)
        if current_order.get_state_value_by_dim_name(current_order.C_DIM_DELIVERY_STATUS[0]) not in [
            current_order.C_STATUS_PLACED]:
            return [], list(relevant_actions)
        mh_history = current_order.mh_assignment_history
        actions_to_block = set()
        for mh in mh_history:
            actions_to_block.update(mh.associated_action_indexes.intersection(self.associated_action_index))
        return list(actions_to_block), list(relevant_actions.difference(actions_to_block))

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        invalidation_idx = []
        if not isinstance(p_entity, Order):
            raise TypeError("The \"Micro-Hub assignability constraint\" is only applicable to an Order entity.")

        if isinstance(p_entity, PseudoOrder):
            mh_node_id = [p_entity.parent_order.assigned_micro_hub_id]
        else:
            mh_node_id = []

        ps_order = p_entity
        mh_node_ids = mh_node_id + ps_order.mh_assignment_history_ids

        delivery_node_id = ps_order.get_delivery_node_id()
        pickup_node_id = ps_order.get_pickup_node_id()
        node_pair = (pickup_node_id, delivery_node_id)

        actions_by_type = p_action_index.get_actions_of_type([SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB])
        actions_by_node_pair = p_action_index.actions_involving_entity["Node Pair", node_pair]
        actions_by_mh = []
        for mh_id in set(mh_node_ids):
            actions_by_mh.extend(p_action_index.actions_involving_entity["MicroHub", mh_id])

        invalidation_idx.extend(
            list(actions_by_type.intersection(actions_by_node_pair).intersection(set(actions_by_mh))))
        return invalidation_idx, []

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, Order):
            return
        forbidden_ids = set()
        if isinstance(p_entity, PseudoOrder):
            if p_entity.parent_order.assigned_micro_hub_id is not None:
                forbidden_ids.add(p_entity.parent_order.assigned_micro_hub_id)
            if hasattr(p_entity, 'mh_assignment_history'):
                forbidden_ids.update(p_entity.mh_assignment_history_ids)
        all_hub_ids = set(p_entity.global_state.micro_hubs.keys())
        allowed_ids = all_hub_ids.difference(forbidden_ids)
        is_operable = allowed_ids if allowed_ids else False
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_operable


#
# class CoordinatedDeliveryAssignmentConstraint(Constraint):
#     C_NAME = "Co-ordinated Delivery Assignment Constraint"
#     C_ASSOCIATED_ENTITIES = ["Order"]
#     C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
#                           SimulationActions.ASSIGN_ORDER_TO_DRONE]
#
#     def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
#         if not isinstance(p_entity, Order):
#             return [], []
#
#         if not isinstance(p_entity, PseudoOrder):
#             return [], []
#
#         relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
#         is_predecessor_unassigned = False
#         if p_entity.predecessor_orders:
#             for pre_order in p_entity.predecessor_orders:
#                 status = pre_order.get_state_value_by_dim_name(Order.C_DIM_DELIVERY_STATUS[0])
#                 if status == Order.C_STATUS_PLACED:
#                     is_predecessor_unassigned = True
#                     break
#
#         if is_predecessor_unassigned:
#             return list(relevant_actions), []
#         else:
#             return [], list(relevant_actions)
#
#     # --- [LEGACY METHODS] ---
#     def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
#         def check_ass_precedence(p_order):
#             ass_precedence_satisfied = True
#             for pre_order in p_order.predecessor_orders:
#                 if pre_order.get_state_value_by_dim_name(pre_order.C_DIM_DELIVERY_STATUS[0]) not in [
#                     pre_order.C_STATUS_PLACED,
#                     pre_order.C_STATUS_ACCEPTED,
#                     pre_order.C_STATUS_FAILED]:
#                     ass_precedence_satisfied = True and ass_precedence_satisfied
#                 else:
#                     ass_precedence_satisfied = False
#             return ass_precedence_satisfied
#
#         invalidation_idx = []
#         validation_idx = []
#         if not (isinstance(p_entity, Order) or isinstance(p_entity, (Truck, Drone))):
#             raise TypeError("Wrong entity type for the constraint")
#
#         actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
#         pseudo_orders = [order for order in p_entity.global_state.get_all_entities_by_type("order").values()
#                          if (isinstance(order, PseudoOrder) and len(
#                 order.predecessor_orders) and not check_ass_precedence(order))]
#
#         for ps_ordr in pseudo_orders:
#             actions_by_entity = p_action_index.actions_involving_entity[
#                 "Node Pair", (ps_ordr.get_pickup_node_id(), ps_ordr.get_delivery_node_id())]
#             invalidation_idx.extend(actions_by_entity.intersection(actions_by_type))
#
#         return invalidation_idx, validation_idx
#
#     def update_operability(self, p_entity, **p_kwargs):
#         if not isinstance(p_entity, Order):
#             return
#         is_ready = True
#         if isinstance(p_entity, PseudoOrder) and p_entity.predecessor_orders:
#             for pre_order in p_entity.predecessor_orders:
#                 status = pre_order.get_state_value_by_dim_name(Order.C_DIM_DELIVERY_STATUS[0])
#                 if status in [Order.C_STATUS_PLACED, Order.C_STATUS_ACCEPTED, Order.C_STATUS_FAILED]:
#                     is_ready = False
#                     break
#         for action_type in self.C_ACTIONS_AFFECTED:
#             if action_type in p_entity.action_operability:
#                 p_entity.action_operability[action_type] = is_ready

class CoordinatedDeliveryAssignmentConstraint(Constraint):
    """
    Enforces sequential assignment of coordinated orders.
    Strictly localized to Node Pair. Relies on the Environment to raise an event
    for this Node Pair when its predecessor order changes state.
    """
    C_NAME = "Co-ordinated Delivery Assignment Constraint"
    C_ASSOCIATED_ENTITIES = ["Node Pair"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        # 1. Strict Entity Check
        if not isinstance(p_entity, NodePair):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = p_entity.associated_action_indexes.intersection(self.associated_action_index)
        if not relevant_actions:
            return [], []

        global_state = p_entity.global_state
        if global_state is None:
            return [], list(relevant_actions)

        # 2. O(1) Lookup: What order is sitting at this Node Pair?
        order_requests = global_state.get_order_requests()
        node_pair_id = p_entity.get_id()

        if node_pair_id not in order_requests:
            return [], list(relevant_actions)

        target_order = order_requests[node_pair_id][0]

        # 3. Fast bypass: If it has no predecessor, it's not Leg 2, so it's safe.
        if not isinstance(target_order, PseudoOrder) or not target_order.predecessor_orders:
            return [], list(relevant_actions)

        # 4. The Sequence Check
        for pre_order in target_order.predecessor_orders:
            status = pre_order.get_state_value_by_dim_name(Order.C_DIM_DELIVERY_STATUS[0])
            # If Leg 1 is NOT actively moving or delivered, block Leg 2!
            if status in [Order.C_STATUS_PLACED, Order.C_STATUS_ACCEPTED, Order.C_STATUS_FAILED]:
                return list(relevant_actions), []

        # If we passed the check, Leg 1 is assigned/moving. Unblock Leg 2!
        return [], list(relevant_actions)


class OrderLoadConstraint(Constraint):
    C_NAME = "OrderLoadConstraint"
    C_ASSOCIATED_ENTITIES = ["Order"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION]

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
        if not isinstance(p_entity, Order):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entity.")

        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
        assigned_veh = p_entity.assigned_vehicle
        if not assigned_veh:
            return relevant_actions, []
        veh_actions = self.associated_action_index.intersection(assigned_veh.associated_action_indexes)
        actions_to_unblock = relevant_actions.intersection(veh_actions)
        actions_to_mask = relevant_actions.difference(veh_actions)
        return actions_to_mask, actions_to_unblock

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        invalidation_idx = []

        if isinstance(p_entity, Vehicle):
            actions_by_order = []
            for ordr in p_entity.get_pickup_orders():
                actions_by_order.extend(p_action_index.actions_involving_entity["Order", ordr.get_id()])
            actions_by_order = list(set(actions_by_order))

            if isinstance(p_entity, Truck):
                actions_by_vehicle = p_action_index.actions_involving_entity["Truck", p_entity.get_id()]
            elif isinstance(p_entity, Drone):
                actions_by_vehicle = p_action_index.actions_involving_entity["Drone", p_entity.get_id()]

            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            relevant_actions = actions_by_vehicle.intersection(actions_by_type)
            invalidation_idx = list(relevant_actions.difference(actions_by_order))
            return invalidation_idx, []

        elif isinstance(p_entity, Order):
            assigned_vehicle_id = p_entity.assigned_vehicle_id
            actions_by_order = p_action_index.actions_involving_entity["Order", p_entity.get_id()]
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            relevant_actions = actions_by_type.intersection(actions_by_order)

            if ((assigned_vehicle_id is not None) and
                    (p_entity.get_state_value_by_dim_name(p_entity.C_DIM_DELIVERY_STATUS[0]) in [
                        p_entity.C_STATUS_ASSIGNED])):
                try:
                    vehicle = p_entity.global_state.get_entity("truck", assigned_vehicle_id)
                    actions_by_vehicle = p_action_index.actions_involving_entity["Truck", assigned_vehicle_id]
                except KeyError:
                    vehicle = p_entity.global_state.get_entity("drone", assigned_vehicle_id)
                    actions_by_vehicle = p_action_index.actions_involving_entity["Drone", assigned_vehicle_id]

                if vehicle.current_node_id == p_entity.get_pickup_node_id():
                    invalidation_idx = list(relevant_actions.difference(actions_by_vehicle))
                else:
                    return list(relevant_actions), []
                return invalidation_idx, []

            return list(relevant_actions), []

    def update_operability(self, p_entity, **p_kwargs):
        if isinstance(p_entity, (Truck, Drone)):
            vehicle = p_entity
            valid_orders = set()
            for order in vehicle.get_pickup_orders():
                valid_orders.add(order.get_id())
            new_val = valid_orders if valid_orders else False
            for action in self.C_ACTIONS_AFFECTED:
                if action in p_entity.action_operability:
                    current_val = p_entity.action_operability[action]
                    if current_val is False or new_val is False:
                        p_entity.action_operability[action] = False
                    elif current_val is True:
                        p_entity.action_operability[action] = new_val
                    else:
                        intersection = current_val.intersection(new_val)
                        p_entity.action_operability[action] = intersection if intersection else False

        elif isinstance(p_entity, Order):
            order = p_entity
            valid_vehicles = set()
            assigned_id = order.assigned_vehicle_id
            status = order.get_state_value_by_dim_name(Order.C_DIM_DELIVERY_STATUS[0])
            if assigned_id is not None and status == Order.C_STATUS_ASSIGNED:
                vehicle = None
                if assigned_id in order.global_state.trucks:
                    vehicle = order.global_state.trucks[assigned_id]
                elif assigned_id in order.global_state.drones:
                    vehicle = order.global_state.drones[assigned_id]
                if vehicle and vehicle.current_node_id == order.get_pickup_node_id():
                    valid_vehicles.add(assigned_id)
            new_val = valid_vehicles if valid_vehicles else False
            for action in self.C_ACTIONS_AFFECTED:
                if action in p_entity.action_operability:
                    current_val = p_entity.action_operability[action]
                    if current_val is False or new_val is False:
                        p_entity.action_operability[action] = False
                    elif current_val is True:
                        p_entity.action_operability[action] = new_val
                    else:
                        intersection = current_val.intersection(new_val)
                        p_entity.action_operability[action] = intersection if intersection else False


class CoordinatedOrderLoadConstraint(Constraint):
    C_NAME = "CoordinatedOrderLoadConstraint"
    C_ASSOCIATED_ENTITIES = ["Order"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION, SimulationActions.LOAD_DRONE_ACTION]

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
        if not isinstance(p_entity, Order):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
        if not isinstance(p_entity, PseudoOrder):
            return [], []
        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
        loadable = True
        for order in p_entity.predecessor_orders:
            if order.get_state_value_by_dim_name(p_entity.C_DIM_DELIVERY_STATUS[0]) in [p_entity.C_STATUS_DELIVERED]:
                loadable = loadable and True
            else:
                loadable = loadable and False
        if loadable:
            return [], list(relevant_actions)
        else:
            return list(relevant_actions), []


class VehicleLoadConstraint(Constraint):
    C_NAME = "VehicleLoadConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION]

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
        if not isinstance(p_entity, Vehicle):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)

        if p_entity.get_state_value_by_dim_name(p_entity.C_DIM_TRIP_STATE[0]) not in [p_entity.C_TRIP_STATE_HALT]:
            return list(relevant_actions), []

        current_node = p_entity.get_current_node()
        valid_orders = []

        for order in p_entity.get_pickup_orders():
            if order.get_pickup_node_id() == current_node:
                valid_orders.append(order)

        valid_action_ids = set()
        for order in valid_orders:
            actions_for_order = p_action_index.actions_involving_entity.get(("Order", order.get_id()), set())
            valid_action_ids.update(relevant_actions.intersection(actions_for_order))

        to_mask = relevant_actions.difference(valid_action_ids)
        return list(to_mask), list(valid_action_ids)

    # NO LEGACY METHODS (New Constraint)


# class VehicleLoadConstraint(Constraint):
#     C_NAME = "VehicleLoadConstraint"
#     C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
#     C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
#                           SimulationActions.LOAD_DRONE_ACTION]
#
#     def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
#         if not isinstance(p_entity, Vehicle):
#             raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
#
#         relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
#
#         # 1. Strict State Guard: Must be explicitly parked (HALT) and locked into a mission
#         current_status = p_entity.get_state_value_by_dim_name(p_entity.C_DIM_TRIP_STATE[0])
#         if current_status != p_entity.C_TRIP_STATE_HALT or not getattr(p_entity, 'consolidation_confirmed', False):
#             return list(relevant_actions), []  # Block everything
#
#         # 2. Strict Sequence Guard: Only allow orders at the CURRENT sequence index
#         valid_action_ids = set()
#
#         # Safely fetch the list of orders assigned to the current step
#         current_step_orders = getattr(p_entity, 'planned_order_sequence', {}).get(getattr(p_entity, 'current_sequence_index', -1), [])
#
#         for order in current_step_orders:
#             # Verify it is actually meant to be picked up
#             if order in p_entity.get_pickup_orders():
#                 actions_for_order = p_action_index.actions_involving_entity.get(("Order", order.get_id()), set())
#                 valid_action_ids.update(relevant_actions.intersection(actions_for_order))
#
#         to_mask = relevant_actions.difference(valid_action_ids)
#         return list(to_mask), list(valid_action_ids)

# NO LEGACY METHODS


class OrderUnloadConstraint(Constraint):
    C_NAME = "OrderUnloadConstraint"
    C_ASSOCIATED_ENTITIES = ["Order"]
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
                          SimulationActions.UNLOAD_DRONE_ACTION]

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
        if not isinstance(p_entity, Order):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
        carrying_vehicle = p_entity.carrying_vehicle
        if carrying_vehicle is None:
            return relevant_actions, []
        current_location = carrying_vehicle.get_current_node()
        if not current_location == p_entity.get_delivery_node_id():
            return relevant_actions, []
        if p_entity not in carrying_vehicle.get_current_cargo():
            return relevant_actions, []
        veh_actions = carrying_vehicle.associated_action_indexes
        actions_to_unblock = relevant_actions.intersection(veh_actions)
        actions_to_block = relevant_actions.difference(actions_to_unblock)
        return actions_to_block, actions_to_unblock

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        invalidation_idx = []

        if isinstance(p_entity, Vehicle):
            actions_by_order = []
            for ordr in p_entity.get_delivery_orders():
                if ordr in p_entity.get_current_cargo():
                    actions_by_order.extend(p_action_index.actions_involving_entity["Order", ordr.get_id()])
            actions_by_order = list(set(actions_by_order))

            if isinstance(p_entity, Truck):
                actions_by_vehicle = p_action_index.actions_involving_entity["Truck", p_entity.get_id()]
            elif isinstance(p_entity, Drone):
                actions_by_vehicle = p_action_index.actions_involving_entity["Drone", p_entity.get_id()]

            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            relevant_actions = actions_by_vehicle.intersection(actions_by_type)
            invalidation_idx = list(relevant_actions.difference(actions_by_order))
            return invalidation_idx, []

        elif isinstance(p_entity, Order):
            assigned_vehicle_id = p_entity.assigned_vehicle_id
            actions_by_order = p_action_index.actions_involving_entity["Order", p_entity.get_id()]
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            relevant_actions = actions_by_type.intersection(actions_by_order)

            if ((assigned_vehicle_id is not None)
                    and (p_entity.get_state_value_by_dim_name(p_entity.C_DIM_DELIVERY_STATUS[0]) in [
                        p_entity.C_STATUS_EN_ROUTE])):
                try:
                    vehicle = p_entity.global_state.get_entity("truck", assigned_vehicle_id)
                    actions_by_vehicle = p_action_index.actions_involving_entity["Truck", assigned_vehicle_id]
                except KeyError:
                    vehicle = p_entity.global_state.get_entity("drone", assigned_vehicle_id)
                    actions_by_vehicle = p_action_index.actions_involving_entity["Drone", assigned_vehicle_id]

                if vehicle.current_node_id == p_entity.get_delivery_node_id():
                    invalidation_idx = list(relevant_actions.difference(actions_by_vehicle))
                    return invalidation_idx, []

            return list(relevant_actions), []

    def update_operability(self, p_entity, **p_kwargs):
        if isinstance(p_entity, (Truck, Drone)):
            vehicle = p_entity
            current_cargo_ids = set(vehicle.get_current_cargo())
            delivery_manifest_ids = set([o.get_id() for o in vehicle.get_delivery_orders()])
            valid_orders = current_cargo_ids.intersection(delivery_manifest_ids)
            new_val = valid_orders if valid_orders else False
            for action in self.C_ACTIONS_AFFECTED:
                if action in p_entity.action_operability:
                    current_val = p_entity.action_operability[action]
                    if current_val is False or new_val is False:
                        p_entity.action_operability[action] = False
                    elif current_val is True:
                        p_entity.action_operability[action] = new_val
                    else:
                        intersection = current_val.intersection(new_val)
                        p_entity.action_operability[action] = intersection if intersection else False

        elif isinstance(p_entity, Order):
            order = p_entity
            valid_vehicles = set()
            assigned_id = order.assigned_vehicle_id
            status = order.get_state_value_by_dim_name(Order.C_DIM_DELIVERY_STATUS[0])
            if assigned_id is not None and status == Order.C_STATUS_EN_ROUTE:
                vehicle = None
                if assigned_id in order.global_state.trucks:
                    vehicle = order.global_state.trucks[assigned_id]
                elif assigned_id in order.global_state.drones:
                    vehicle = order.global_state.drones[assigned_id]
                if vehicle and vehicle.current_node_id == order.get_delivery_node_id():
                    valid_vehicles.add(assigned_id)
            new_val = valid_vehicles if valid_vehicles else False
            for action in self.C_ACTIONS_AFFECTED:
                if action in p_entity.action_operability:
                    current_val = p_entity.action_operability[action]
                    if current_val is False or new_val is False:
                        p_entity.action_operability[action] = False
                    elif current_val is True:
                        p_entity.action_operability[action] = new_val
                    else:
                        intersection = current_val.intersection(new_val)
                        p_entity.action_operability[action] = intersection if intersection else False


class VehicleUnloadConstraint(Constraint):
    C_NAME = "VehicleUnloadConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
                          SimulationActions.UNLOAD_DRONE_ACTION]

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
        if not isinstance(p_entity, Vehicle):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)

        if p_entity.get_state_value_by_dim_name(p_entity.C_DIM_TRIP_STATE[0]) not in [p_entity.C_TRIP_STATE_HALT]:
            return list(relevant_actions), []

        current_node = p_entity.get_current_node()
        current_cargo = p_entity.get_current_cargo()

        valid_orders = []
        for order in current_cargo:
            if order.get_delivery_node_id() == current_node:
                valid_orders.append(order)

        valid_action_ids = set()
        for order in valid_orders:
            actions_for_order = p_action_index.actions_involving_entity.get(("Order", order.get_id()), set())
            valid_action_ids.update(relevant_actions.intersection(actions_for_order))

        to_mask = relevant_actions.difference(valid_action_ids)
        return list(to_mask), list(valid_action_ids)

    # NO LEGACY METHODS (New Constraint)


# class VehicleUnloadConstraint(Constraint):
#     C_NAME = "VehicleUnloadConstraint"
#     C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
#     C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
#                           SimulationActions.UNLOAD_DRONE_ACTION]
#
#     def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs):
#         if not isinstance(p_entity, Vehicle):
#             raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
#
#         relevant_actions = self.associated_action_index.intersection(p_entity.associated_action_indexes)
#
#         # 1. Strict State Guard: Must be explicitly parked (HALT) and locked into a mission
#         current_status = p_entity.get_state_value_by_dim_name(p_entity.C_DIM_TRIP_STATE[0])
#         if current_status != p_entity.C_TRIP_STATE_HALT or not getattr(p_entity, 'consolidation_confirmed', False):
#             return list(relevant_actions), []  # Block everything
#
#         # 2. Strict Sequence Guard: Only allow orders at the CURRENT sequence index
#         valid_action_ids = set()
#
#         # Safely fetch the list of orders assigned to the current step
#         current_step_orders = getattr(p_entity, 'planned_order_sequence', {}).get(getattr(p_entity, 'current_sequence_index', -1), [])
#
#         for order in current_step_orders:
#             # Verify it is actually meant to be delivered AND is physically in the cargo
#             if order in p_entity.get_delivery_orders() and order in p_entity.get_current_cargo():
#                 actions_for_order = p_action_index.actions_involving_entity.get(("Order", order.get_id()), set())
#                 valid_action_ids.update(relevant_actions.intersection(actions_for_order))
#
#         to_mask = relevant_actions.difference(valid_action_ids)
#         return list(to_mask), list(valid_action_ids)

# NO LEGACY METHODS

class OrderAtDeliveryNode(Constraint):
    C_NAME = "OrderAtDeliveryNode"
    C_ASSOCIATED_ENTITIES = ["Order"]
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION, SimulationActions.UNLOAD_DRONE_ACTION]
    C_ACTIVE = False

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        return [], []


#
# class DeadlockPreventionConstraint(Constraint):
#     """
#     Prevents distributed circular waits by analyzing the dependency graph.
#     Strictly associated with the Node Pair entity, it evaluates if assigning
#     the active order request of this node pair to a specific vehicle creates a cycle.
#     """
#     C_NAME = "DeadlockPreventionConstraint"
#     C_ACTIVE = True
#     C_ASSOCIATED_ENTITIES = ["Node Pair"]
#     C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
#                           SimulationActions.ASSIGN_ORDER_TO_DRONE]
#
#     def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
#         # 1. Strict Entity Check
#         if not isinstance(p_entity, NodePair):
#             raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")
#
#         # Fast-fail: Get assignment actions strictly related to this specific Node Pair
#         relevant_actions = p_entity.associated_action_indexes.intersection(self.associated_action_index)
#         if not relevant_actions:
#             return [], []
#
#         global_state = p_entity.global_state
#         if global_state is None:
#             return [], list(relevant_actions)
#
#         # 2. Resolve the Node Pair to its active Order Request
#         order_requests = global_state.get_order_requests()
#         node_pair_id = p_entity.get_id()
#
#         # If there's no active request, we don't block anything here
#         # (OrderRequestAssignabilityConstraint handles masking empty requests)
#         if node_pair_id not in order_requests:
#             return [], list(relevant_actions)
#
#         target_order = order_requests[node_pair_id][0]
#
#         # If the order doesn't have predecessors, it physically cannot cause a wait loop
#         if not hasattr(target_order, 'predecessor_orders') or not target_order.predecessor_orders:
#             return [], list(relevant_actions)
#
#         # 3. Build the Dependency Graph of the Network
#         dependencies = defaultdict(set)
#         all_vehicles = list(global_state.trucks.values()) + list(global_state.drones.values())
#
#         for v in all_vehicles:
#             for order in v.get_pickup_orders() + v.get_current_cargo():
#                 if hasattr(order, 'predecessor_orders') and order.predecessor_orders:
#                     for pred in order.predecessor_orders:
#                         if pred.get_state_value_by_dim_name(pred.C_DIM_DELIVERY_STATUS[0]) != pred.C_STATUS_DELIVERED:
#                             assigned_veh_id = getattr(pred, 'assigned_vehicle_id', None)
#                             if assigned_veh_id is not None and assigned_veh_id != v.get_id():
#                                 dependencies[v.get_id()].add(assigned_veh_id)
#
#         # Helper: Cycle Detection
#         def has_path(start_veh_id, target_veh_id, visited):
#             if start_veh_id == target_veh_id:
#                 return True
#             visited.add(start_veh_id)
#             for neighbor in dependencies.get(start_veh_id, set()):
#                 if neighbor not in visited:
#                     if has_path(neighbor, target_veh_id, visited):
#                         return True
#             return False
#
#         actions_to_block = set()
#
#         # 4. Evaluate each specific vehicle assignment for this Node Pair
#         for action_idx in relevant_actions:
#             action_tuple = self.reverse_action_map.get(action_idx)
#             if not action_tuple:
#                 continue
#
#             # Action Tuple Format: (ActionType, NodePair_ID, Vehicle_ID)
#             vehicle_id = action_tuple[2]
#
#             # Check if assigning this order to THIS vehicle creates a cycle
#             for pred in target_order.predecessor_orders:
#                 if pred.get_state_value_by_dim_name(pred.C_DIM_DELIVERY_STATUS[0]) != pred.C_STATUS_DELIVERED:
#                     pred_veh_id = getattr(pred, 'assigned_vehicle_id', None)
#
#                     if pred_veh_id is not None and pred_veh_id != vehicle_id:
#                         # CYCLE CHECK: Does the vehicle carrying the predecessor depend on the current vehicle?
#                         if has_path(pred_veh_id, vehicle_id, set()):
#                             actions_to_block.add(action_idx)
#                             break
#
#         actions_to_unblock = relevant_actions.difference(actions_to_block)
#         return list(actions_to_block), list(actions_to_unblock)

class DeadlockPreventionConstraint(Constraint):
    """
    Prevents distributed circular waits by building a hypothetical global
    dependency graph for every proposed assignment.
    """
    C_NAME = "DeadlockPreventionConstraint"
    C_ACTIVE = True
    C_ASSOCIATED_ENTITIES = ["Node Pair"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        if not isinstance(p_entity, NodePair):
            raise TypeError(f"{self.C_NAME} needs {self.C_ASSOCIATED_ENTITIES} as type for associated entities.")

        relevant_actions = p_entity.associated_action_indexes.intersection(self.associated_action_index)
        if not relevant_actions:
            return [], []

        global_state = p_entity.global_state
        if global_state is None:
            return [], list(relevant_actions)

        order_requests = global_state.get_order_requests()
        node_pair_id = p_entity.get_id()

        if node_pair_id not in order_requests:
            return [], list(relevant_actions)

        target_order = order_requests[node_pair_id][0]

        # --- THE FIX: Recursive Drill-Down across ALL 3 Phases ---
        def get_responsible_vehicles(order, proposed_order_id=None, proposed_veh_id=None):
            veh_ids = set()

            # Phase 1: Hypothetical (The order we are testing right now)
            if str(order.get_id()) == str(proposed_order_id):
                veh_ids.add(proposed_veh_id)
            # Phase 2: Waiting at Node (Assigned but not loaded)
            elif getattr(order, 'assigned_vehicle_id', None) is not None:
                veh_ids.add(order.assigned_vehicle_id)
            # Phase 3: THE MISSING LINK -> Actively inside Cargo
            elif getattr(order, 'carrying_vehicle', None) is not None:
                veh_ids.add(order.carrying_vehicle.get_id())

            # Nested Assignment
            if hasattr(order, 'pseudo_orders') and order.pseudo_orders:
                for sub_order in order.pseudo_orders:
                    if sub_order.get_state_value_by_dim_name(
                            sub_order.C_DIM_DELIVERY_STATUS[0]) != sub_order.C_STATUS_DELIVERED:
                        veh_ids.update(get_responsible_vehicles(sub_order, proposed_order_id, proposed_veh_id))
            return veh_ids

        # ---------------------------------------------------------------

        all_vehicles = list(global_state.trucks.values()) + list(global_state.drones.values())
        actions_to_block = set()

        for action_idx in relevant_actions:
            action_tuple = self.reverse_action_map.get(action_idx)
            if not action_tuple:
                continue

            vehicle_id = action_tuple[2]
            dependencies = defaultdict(set)

            # Build the Dependency Graph
            for v in all_vehicles:
                # Add our hypothetical target order to the vehicle we are testing
                manifest = v.get_pickup_orders() + v.get_current_cargo()
                if v.get_id() == vehicle_id:
                    manifest = manifest + [target_order]

                for order in manifest:
                    if hasattr(order, 'predecessor_orders') and order.predecessor_orders:
                        for pred in order.predecessor_orders:
                            if pred.get_state_value_by_dim_name(
                                    pred.C_DIM_DELIVERY_STATUS[0]) != pred.C_STATUS_DELIVERED:
                                pred_veh_ids = get_responsible_vehicles(pred, target_order.get_id(), vehicle_id)
                                dependencies[v.get_id()].update(pred_veh_ids)

            # ---------------------------------------------------------------
            # Cycle Detection (Depth First Search)
            def has_cycle(current_veh, visited, rec_stack):
                visited.add(current_veh)
                rec_stack.add(current_veh)
                for neighbor in dependencies.get(current_veh, set()):
                    if neighbor not in visited:
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True
                rec_stack.remove(current_veh)
                return False

            visited = set()
            rec_stack = set()

            # If giving this order to this vehicle puts the vehicle in a loop, block it.
            if has_cycle(vehicle_id, visited, rec_stack):
                actions_to_block.add(action_idx)

        actions_to_unblock = relevant_actions.difference(actions_to_block)
        return list(actions_to_block), list(actions_to_unblock)

        # ---------------------------------------------------------------

        # Build the Dependency Graph
        dependencies = defaultdict(set)
        all_vehicles = list(global_state.trucks.values()) + list(global_state.drones.values())

        for v in all_vehicles:
            for order in v.get_pickup_orders() + v.get_current_cargo():
                if hasattr(order, 'predecessor_orders') and order.predecessor_orders:
                    for pred in order.predecessor_orders:
                        if pred.get_state_value_by_dim_name(pred.C_DIM_DELIVERY_STATUS[0]) != pred.C_STATUS_DELIVERED:
                            # Use the helper to extract ALL nested vehicles
                            pred_veh_ids = get_responsible_vehicles(pred)
                            dependencies[v.get_id()].update(pred_veh_ids)

        def has_path(start_veh_id, target_veh_id, visited):
            if start_veh_id == target_veh_id:
                return True
            visited.add(start_veh_id)
            for neighbor in dependencies.get(start_veh_id, set()):
                if neighbor not in visited:
                    if has_path(neighbor, target_veh_id, visited):
                        return True
            return False

        actions_to_block = set()

        for action_idx in relevant_actions:
            action_tuple = self.reverse_action_map.get(action_idx)
            if not action_tuple:
                continue

            vehicle_id = action_tuple[2]

            for pred in target_order.predecessor_orders:
                if pred.get_state_value_by_dim_name(pred.C_DIM_DELIVERY_STATUS[0]) != pred.C_STATUS_DELIVERED:

                    # Extract ALL vehicles carrying the predecessor or its sub-legs
                    pred_veh_ids = get_responsible_vehicles(pred)

                    # Cycle Check against every nested vehicle
                    for p_veh_id in pred_veh_ids:
                        if has_path(p_veh_id, vehicle_id, set()):
                            actions_to_block.add(action_idx)
                            break

        actions_to_unblock = relevant_actions.difference(actions_to_block)
        return list(actions_to_block), list(actions_to_unblock)


# -------------------------------------------------------------------------------------------------
# -- Part 3: Managers
# -------------------------------------------------------------------------------------------------


class ConstraintManager(EventManager):
    """
    Manages all constraints in the simulation.
    """
    C_NAME = "Constraint Manager"
    C_EVENT_MASK_UPDATED = "New Masks Necessary"

    def __init__(self, action_index: ActionIndex, reverse_action_map, custom_log=False):
        EventManager.__init__(self, p_logging=False)
        self.masks = []
        self.action_map = None
        self._update_counter = 0
        self.constraints = set()
        self.entity_constraints = {}
        self.reverse_action_map = reverse_action_map
        self.action_index = action_index
        self.setup_constraint_entity_map()
        self.custom_log = custom_log
        self.constraint_deck = {key: set() for key in self.reverse_action_map}
        self.masks = [0 for i in range(len(self.reverse_action_map))]
        if self.custom_log:
            print("Constraints Setup")

    def setup_constraint_entity_map(self):
        self.entity_constraints = {}
        for con in Constraint.__subclasses__():
            # Skip abstract or base classes if they somehow get in
            if con.C_ACTIVE and con is not Constraint:
                constr = con(p_reverse_action_map=self.reverse_action_map, p_action_index=self.action_index)
                self.constraints.add(constr)
                for entity_name in con.C_ASSOCIATED_ENTITIES:
                    if entity_name in self.entity_constraints:
                        self.entity_constraints[entity_name].append(constr)
                    else:
                        self.entity_constraints[entity_name] = [constr]

        print("Constraint dict updated")

    def get_constraints_by_entity(self, p_entity):
        if p_entity.C_NAME in self.entity_constraints:
            return self.entity_constraints[p_entity.C_NAME]
        return []

    def handle_entity_state_change(self, p_event_id, p_event_object):
        # DEBUG 1: Did we even get called?
        if self.custom_log:
            print(
                f"[ConstraintManager] Event received: {p_event_id} from {p_event_object.get_raising_object().get_id()}")

        self._update_counter += 1
        entity = p_event_object.get_raising_object()

        total_to_block = []
        total_to_unblock = []

        constraints_to_check = self.get_constraints_by_entity(entity)

        # DEBUG 2: Did we find constraints?
        if self.custom_log:
            print(f"[ConstraintManager] Found {len(constraints_to_check)} constraints for entity {entity.C_NAME}")

        for constraint in constraints_to_check:
            to_block, to_unblock = constraint.evaluate_impact(p_entity=entity, p_action_index=self.action_index,
                                                              deck=self.constraint_deck)

            # DEBUG 3: specific constraint output
            if to_block or to_unblock:
                if self.custom_log:
                    print(f"   -> {constraint.C_NAME}: Block={len(to_block)}, Unblock={len(to_unblock)}")

            total_to_block.extend(to_block)
            total_to_unblock.extend(to_unblock)

        if len(total_to_block) > 0 or len(total_to_unblock) > 0:
            event_data = {
                "to_block": total_to_block,
                "to_unblock": total_to_unblock
            }
            if self.custom_log:
                print(f"[ConstraintManager] Raising update event! (+{len(total_to_block)} / -{len(total_to_unblock)})")
            self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                              p_event_object=Event(p_raising_object=self,
                                                   to_block=total_to_block,
                                                   to_unblock=total_to_unblock))
        else:
            if self.custom_log:
                print("[ConstraintManager] No net change in masks. Event skipped.")

    def update_constraints(self, global_state, reverse_action_map):
        """
        Full initialization/Reset.
        """
        if self.custom_log:
            print("Update constraints method is called")
        self.reverse_action_map = reverse_action_map
        self.constraint_deck = {key: set() for key in self.reverse_action_map.keys()}
        self.masks = [0 for i in range(len(self.reverse_action_map))]
        for constraint in self.constraints:
            constraint.clear_cache()
            constraint.reverse_action_map = self.reverse_action_map

        total_to_block = []

        for entity_dict in global_state.get_all_entities():
            for entity in entity_dict.values():
                constraints_to_check = self.get_constraints_by_entity(entity)
                for constraint in constraints_to_check:
                    to_block, _ = constraint.evaluate_impact(p_entity=entity, p_action_index=self.action_index,
                                                             deck=self.constraint_deck)
                    total_to_block.extend(to_block)

        if total_to_block:
            event_data = {
                "to_block": total_to_block,
                "to_unblock": []
            }
            self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                              p_event_object=Event(p_raising_object=self,
                                                   p_event_data=event_data))

    def update_action_index(self, action_map, action_map_old, reverse_action_map_old):
        for constraint in self.constraints:
            as_action_index_old = list(constraint.associated_action_index)
            constraint.associated_action_index = self.action_index.get_actions_of_type(
                constraint.C_ACTIONS_AFFECTED).copy()
            # for i in as_action_index_old:
            #     constraint.associated_action_index.add(action_map[reverse_action_map_old[i]])
            # print("action_indexes_updated")
        self.update_entity_invalidation_maps(action_map, reverse_action_map_old)
        constraint_deck_old = self.constraint_deck.copy()
        self.constraint_deck = {key: set() for key in action_map.values()}
        for action in reverse_action_map_old.values():
            self.constraint_deck[action_map[action]] = constraint_deck_old[action_map_old[action]]
        self.action_map = action_map
        self.update_masks()
        return

    def update_entity_invalidation_maps(self, action_map, reverse_action_map_old):
        for constraint in self.constraints:
            for entity, action_set in constraint._entity_invalidation_map.items():
                new_action_set = set()
                for old_action in action_set:
                    if not reverse_action_map_old[old_action] in action_map:
                        print("Something is wrong. I am tired.")
                        raise TypeError
                    new_action_set.add(action_map[reverse_action_map_old[old_action]])
                constraint._entity_invalidation_map[entity] = new_action_set

    def update_masks(self):
        self.masks = [False for i in range(len(self.constraint_deck.keys()))]
        for key, value in self.constraint_deck.items():
            if len(value):
                self.masks[key] = False
            else:
                self.masks[key] = True

    def get_masks(self):
        for key, value in self.constraint_deck.items():
            if len(value):
                self.masks[key] = 0
            else:
                self.masks[key] = 1

        return self.masks


class StateActionMapper:
    """
    Maps the system state to a valid action mask using reference counting.
    """

    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int], reverse_action_map, custom_log=False):
        self.old_counters = None
        self.global_state = global_state
        self.action_map = action_map
        self.action_index = ActionIndex(global_state, action_map)

        self.mask_counters = [0] * len(action_map)
        self.masks = [True] * len(action_map)
        self.reverse_action_map = reverse_action_map

        self.permanent_valid_actions = set(self.action_index.get_actions_of_type([SimulationActions.NO_OPERATION]))
        self.custom_log = custom_log

    def update_counters_and_masks(self, indices_to_block: Iterable[int], indices_to_unblock: Iterable[int]):
        """
        Updates counters and flips boolean masks on 0 <-> 1 transitions.
        """
        masked = 0
        unmasked = 0
        # --- BLOCK LOGIC ---
        if self.custom_log:
            print("Masks updated")
        for idx in indices_to_block:
            if idx not in self.permanent_valid_actions:
                self.mask_counters[idx] += 1
                if self.mask_counters[idx] >= 1:
                    self.masks[idx] = False
                    masked += 1

        # --- UNBLOCK LOGIC ---
        for idx in indices_to_unblock:
            if idx not in self.permanent_valid_actions:
                self.mask_counters[idx] -= 1
                if self.mask_counters[idx] == 0:
                    self.masks[idx] = True
                    unmasked += 1

                if self.mask_counters[idx] < 0:
                    if self.custom_log:
                        print(f"[StateActionMapper] Warning: Counter negative for index {idx}. Resetting to 0.")
                    self.mask_counters[idx] = 0
                    self.masks[idx] = True
        print(f"Masks updated: Masked --> {masked} , Unmaked --> {unmasked}")
        print(
            f"Changes requested: to block --> {len(list(indices_to_block))}, to unblock --> {len(list(indices_to_unblock))}")
        return 0

    def handle_new_masks_event(self, p_event_id, p_event_object):
        """
        Handles the event from ConstraintManager.
        """
        raising_object = p_event_object.get_raising_object()
        if isinstance(raising_object, ConstraintManager):
            # [FIXED] Extract data from dictionary
            data = p_event_object.get_data()
            if data:
                to_block = data.get('to_block', [])
                to_unblock = data.get('to_unblock', [])
                self.update_counters_and_masks(to_block, to_unblock)
        else:
            return

    def generate_masks(self) -> List[bool]:
        return self.masks

    def reset_masks(self):
        self.mask_counters = [0] * len(self.mask_counters)
        self.masks = [True] * len(self.masks)

    def update_action_space(self, action_map, old_action_map):
        # TODO: migrate the handling of masks to Numpy
        self.old_counters = self.mask_counters.copy()
        self.mask_counters = [0] * len(action_map)
        for a, idx in old_action_map.items():
            self.mask_counters[action_map[a]] = self.old_counters[old_action_map[a]]
        self.masks = [True] * len(action_map)
        self.action_map = action_map
        self.update_masks()

    def update_masks(self):
        for i, counter in enumerate(self.mask_counters):
            if counter:
                self.masks[i] = False
            else:
                self.masks[i] = True
        if self.custom_log:
            print("Masks updated after micro-hub assignement")


if __name__ == '__main__':
    # Debugging: Print discovered constraints
    print([c.C_ASSOCIATED_ENTITIES for c in Constraint.__subclasses__() if c is not Constraint])