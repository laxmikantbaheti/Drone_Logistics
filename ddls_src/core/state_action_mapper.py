import numpy as np
from typing import Dict, Any, Tuple, Set, List
from collections import defaultdict
from pprint import pprint
from abc import ABC, abstractmethod
# Local Imports
from ddls_src.actions.base import SimulationActions
from mlpro.bf.systems import System  # Import System for mock object inheritance
from mlpro.bf.events import Event, EventManager  # Import Event for type hinting
from ddls_src.actions.base import ActionIndex
from ddls_src.entities.base import LogisticEntity
from ddls_src.entities import *
from mlpro.bf.various import Log

# # Forward declarations
# class GlobalState: pass
# class Vehicle: pass
# class Order: pass
# class Truck: pass
# class Drone: pass
# class MicroHub: pass
# class Node: pass
# class Edge: pass
# class Network: pass

# -------------------------------------------------------------------------------------------------
# -- Part 1: Pluggable Constraint Architecture (Unified)
# -------------------------------------------------------------------------------------------------


class Constraint(ABC, EventManager):
    """
    Abstract base class for a pluggable constraint rule. Each implementation
    represents a single, specific rule in the simulation.
    """
    C_ASSOCIATED_ENTITIES = []
    C_ACTIONS_AFFECTED = []
    C_DEFAULT_EFFECT = True
    C_NAME = None
    C_EVENT_CONSTRAINT_UPDATE = "ConstraintUpdate"

    def raise_constraint_change_event(self, p_entities, p_effect):
        p_event_data = [p_entities, p_effect]
        self._raise_event(p_event_id=Constraint.C_EVENT_CONSTRAINT_UPDATE,
                          p_event_object=Event(p_raising_object=self,
                                               p_event_data=p_event_data))

    @abstractmethod
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:

        raise NotImplementedError


class VehicleAvailableConstraint(Constraint):

    C_NAME = "VehicleAvailabilityConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]
    C_DEFAULT_EFFECT = True

    # def _handle_vehicle_availabiliy(self, p_event_id, p_event_object):
    #     constraint_satisfied = True
    #     vehicle = p_event_object.get_data()["Vehicle"]
    #     if vehicle.get_availability():
    #         constraint_satisfied = True
    #     else:
    #         constraint_satisfied = False
    #     self.raise_constraint_change_event(p_entities=[vehicle],
    #                                        p_effect=constraint_satisfied)

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) ->List:
        invalidation_idx = []
        # We don't need to check for validation idx, since all the corresponding actions related to the entity are
        # first set to unmasked. And with combinations of constraints the corresponding actions are masked. Rest are
        # set to be True (Except for the permanently masked ones).
        # validation_idx = []
        if not isinstance(p_entity, Vehicle):
            raise TypeError("Vehicle Availability Constraint can only be applied to vehicle entity. Please check the "
                            "corresponding constraint configurations.")

        vehicle = p_entity
        if vehicle.get_state_value_by_dim_name(Vehicle.C_DIM_AVAILABLE[0]):
            constraint_satisfied = True
            # We only return the actions to be masked. If none, we return an empty list
            return invalidation_idx
        else:
            constraint_satisfied = False

            actions_by_type = p_action_index.get_actions_of_type([SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                                                                  SimulationActions.TRUCK_TO_NODE,
                                                                  SimulationActions.DRONE_TO_NODE])

            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_type.intersection(actions_by_entity))
            return invalidation_idx


class VehicleAtDeliveryNodeConstraint(Constraint):
    C_NAME = "VehicleAtDeliveryNodeConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
                          SimulationActions.UNLOAD_DRONE_ACTION,
                          SimulationActions.DRONE_LAND,
                          SimulationActions.DRONE_LAUNCH]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        invalidation_idx = []
        if not (isinstance(p_entity, Drone) or isinstance(p_entity, Truck)):
            raise TypeError("Vehicle At Delivery Node Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        node_vehicle = vehicle.get_current_node()
        delivery_orders = vehicle.get_delivery_orders()

        # Check if the vehicle is at the delivery node for any of its delivery orders
        is_at_delivery_node = False
        if delivery_orders:
            for order_obj in delivery_orders:
                try:
                    # order_obj = vehicle.global_state.get_entity("order", order_id)
                    node_next_delivery_order = order_obj.get_delivery_node_id()

                    if node_next_delivery_order == node_vehicle:
                        is_at_delivery_node = True
                        break  # Found a match, no need to check other orders
                except KeyError:
                    continue  # Order not found, skip it

        if is_at_delivery_node:
            # The constraint is satisfied, so we don't return any invalidations.
            return []

        # If no orders or the vehicle is not at a delivery node for any of them, mask the actions.
        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
        return list(actions_by_entity.intersection(actions_by_type))


class VehicleAtPickUpNodeConstraint(Constraint):
    C_NAME = "VehicleAtPickUpNodeConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION,
                          SimulationActions.DRONE_LAUNCH,
                          SimulationActions.DRONE_LAND]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        invalidation_idx = []
        if not (isinstance(p_entity, Drone) or isinstance(p_entity, Truck)):
            raise TypeError("Vehicle At Delivery Node Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        node_vehicle = vehicle.get_current_node()
        pickup_orders = vehicle.get_pickup_orders()

        # Check if the vehicle is at the delivery node for any of its delivery orders
        is_at_pickup_node = False
        if pickup_orders:
            for order_obj in pickup_orders:
                try:
                    # order_obj = vehicle.global_state.get_entity("order", order_id)
                    node_next_pickup_order = order_obj.get_pickup_node_id()

                    if node_next_pickup_order == node_vehicle:
                        is_at_pickup_node = True
                        break  # Found a match, no need to check other orders
                except KeyError:
                    continue  # Order not found, skip it

        if is_at_pickup_node:
            # The constraint is satisfied, so we don't return any invalidations.
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            relevant_orders = [o.get_id() for o in vehicle.pickup_orders if o.get_pickup_node_id()==node_vehicle]
            actions_to_mask = actions_by_entity.intersection(actions_by_type)
            actions_to_mask = list(actions_to_mask.difference(relevant_orders))
            return actions_to_mask

        # If no orders or the vehicle is not at a delivery node for any of them, mask the actions.
        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
        return list(actions_by_entity.intersection(actions_by_type))

class OrderRequestAssignabilityConstraint(Constraint):
    C_NAME = "OrderTripAssignabilityConstraint"
    C_ASSOCIATED_ENTITIES = ["Order","Truck", "Drone", "Micro-Hub"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        invalidation_idx = []
        if not (isinstance(p_entity, Vehicle)
                or isinstance(p_entity, MicroHub)
                or isinstance(p_entity, Node)
                or isinstance(p_entity, Order)):
            raise TypeError("The \"Order Request Assignability Constraint\" is only applicable to a vehicle "
                            "or a Micro-Hub.")
        if isinstance(p_entity, Truck) or isinstance(p_entity, Drone):
            vehicle = p_entity
            if ((vehicle.get_state_value_by_dim_name(vehicle.C_DIM_TRIP_STATE[0]) in ["En Route", "Charging"]) or
                    (not vehicle.get_state_value_by_dim_name(vehicle.C_DIM_AVAILABLE[0]))):
                constraint_satisfied = False
                actions_by_type = p_action_index.get_actions_of_type([SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                                                                      SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB,
                                                                      SimulationActions.ASSIGN_ORDER_TO_DRONE])
                actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
                invalidation_idx = list(actions_by_entity.intersection(actions_by_type))
                return invalidation_idx
        if isinstance(p_entity, MicroHub):
            micro_hub = p_entity
            if micro_hub.get_state_value_by_dim_name(MicroHub.C_DIM_AVAILABILITY[0]) not in [0]:
                invalidation_idx = list(p_action_index.actions_involving_entity[(type(micro_hub), micro_hub.get_id())])
                return invalidation_idx
        if isinstance(p_entity, Order):
            order = p_entity
            global_state = order.get_global_state()
            # order_requests = global_state.get_order_requests()
            pick_up_node = order.get_pickup_node_id()
            delivery_node = order.get_delivery_node_id()
            order_requests = global_state.get_order_requests()[(pick_up_node, delivery_node)]
            if len(order_requests) == 0:
                constraint_satisfied = False
                invalidation_idx = list(p_action_index.actions_involving_entity[("Node Pair",(pick_up_node, delivery_node))])
                return invalidation_idx
        return invalidation_idx


class OrderAssignableConstraint(Constraint):
    C_NAME = "OrderAssignableConstraint"
    C_ASSOCIATED_ENTITIES = ["Order"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    # def _handle_order_state_change(self, p_event_id, p_event_object):
    #     constraint_satisfied = True
    #     order = p_event_object.get_data()['Order']
    #     order_state = order.get_state()
    #     if order_state not in ["Assigned", "Delivered", "Cancelled"]:
    #         constraint_satisfied = True
    #         return constraint_satisfied
    #     else:
    #         constraint_satisfied = False
    #         return constraint_satisfied

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        invalidation_idx = []
        order = p_entity
        order_state = order.get_state()
        if order_state not in ["Assigned", "Delivered", "Cancelled"]:
            constraint_satisfied = True
            return invalidation_idx
        else:
            constraint_satisfied = False
            actions_by_type = p_action_index.get_actions_of_type([SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                                                                  SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB,
                                                                  SimulationActions.ASSIGN_ORDER_TO_DRONE])
            actions_by_entity = p_action_index.actions_involving_entity[(order.C_NAME, order.get_id())]
            invalidation_idx = list(actions_by_entity.intersection(actions_by_type))
            return invalidation_idx


class VehicleCapacityConstraint(Constraint):
    C_NAME = "VehicleAtDeliveryNodeConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    # def _handle_vehicle_cargo_change(self, p_event_id, p_event_object):
    #     constraint_satisfied = True
    #     vehicle = p_event_object.get_data()['Vehicle']
    #     vehicle_capacity = vehicle.get_config()['Capacity']
    #     current_cargo_size = vehicle.get_current_cargo_size()
    #     if vehicle_capacity - current_cargo_size >= 1:
    #         constraint_satisfied = True
    #         return constraint_satisfied
    #     else:
    #         constraint_satisfied = False
    #         return constraint_satisfied

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        invalidation_idx = []
        vehicle = p_entity
        vehicle_capacity = vehicle.get_cargo_capacity()
        current_cargo_size = vehicle.get_current_cargo_size()
        if vehicle_capacity - current_cargo_size >= 1:
            constraint_satisfied = True
            return invalidation_idx
        else:
            constraint_satisfied = False
            actions_by_type = p_action_index.get_actions_of_type([SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                                                                  SimulationActions.ASSIGN_ORDER_TO_DRONE])
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_entity.intersection(actions_by_type))
            return invalidation_idx


class TripWithinRangeConstraint(Constraint):
    C_NAME = "TripWithinRangeConstraint"
    C_ASSOCIATED_ENTITIES = ["Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_DRONE]
    # def _handle_trip_assingment_request(self, p_event_id, p_event_object):
    #     constraint_satisfied = True
    #     vehicle = p_event_object.get_data()['Vehicle']
    #     available_range = vehicle.get_remaining_range()
    #     vehicle.get_distance(vehicle.get_current_location(), vehicle.get_delivery_orders())

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        invalidation_idx = []
        orders_not_in_range = []
        drone = p_entity
        if not isinstance(drone, Drone):
            raise ValueError("The \'Trip Within Range Constraint\' is currently only applicable to Drones. Trucks can "
                             "carry trips without range limits.")
        available_range = drone.get_remaining_range()

        orders = drone.global_state.orders
        orders_not_in_range = []
        for order in orders.values():
            loc_pick_up = order.get_pickup_node_id()
            loc_delivery = order.get_delivery_node_id()
            distance = drone.global_state.network.calculate_distance(loc_pick_up, loc_delivery)
            if distance < available_range:
                constraint_satisfied = True
            else:
                constraint_satisfied = False
                orders_not_in_range.append(order)
        actions_by_type = p_action_index.get_actions_of_type([SimulationActions.ASSIGN_ORDER_TO_DRONE])
        actions_by_drone = p_action_index.actions_involving_entity[(drone.C_NAME, drone.get_id())]
        actions_by_orders = []
        for order in orders_not_in_range:
            actions_by_orders.append(p_action_index.actions_involving_entity[(order.C_NAME, order.get_id())])
        invalidation_idx = list(actions_by_drone.intersection(actions_by_type, actions_by_orders))
        return invalidation_idx


class VehicleRoutingConstraint(Constraint):
    C_NAME = "Vehicle Routing Constraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "Node"]
    C_ACTIONS_AFFECTED = [SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        if isinstance(p_entity, Truck) or isinstance(p_entity, Drone):
            invalidation_idx = []
            vehicle = p_entity
            pickup_nodes = [orders.get_id() for orders in vehicle.get_pickup_orders()]
            delivery_requests = [orders.get_id() for orders in vehicle.get_delivery_orders()]
            invalidation_idx = p_action_index.get_actions_of_type([SimulationActions.TRUCK_TO_NODE,
                                                                   SimulationActions.DRONE_TO_NODE])
            idx_to_unmask = set()
            for nd in pickup_nodes+delivery_requests:
                idx_to_unmask.update(p_action_index.actions_involving_entity[("Node", nd)])

            invalidation_idx = list(invalidation_idx.difference(idx_to_unmask))
            return invalidation_idx
        elif isinstance(p_entity, Node):
            return []


class ConsolidationConstraint(Constraint):
    C_NAME = "ConsolidationConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE_FOR_TRUCK,
                          SimulationActions.CONSOLIDATE_FOR_DRONE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        invalidation_idx = []
        if not (isinstance(p_entity, Truck) or isinstance(p_entity, Drone)):
            return invalidation_idx

        vehicle = p_entity

        # Consolidation is only valid if the vehicle is not en-route and has assigned delivery orders.
        is_ready_for_consolidation = (vehicle.get_state_value_by_dim_name(vehicle.C_DIM_TRIP_STATE[0]) == vehicle.C_TRIP_STATE_IDLE
                                      and len(vehicle.pickup_orders) > 0)

        if not is_ready_for_consolidation:
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_type.intersection(actions_by_entity))

        return invalidation_idx


# class LoadUnloadConstraint(Constraint):
#     C_NAME = "LoadUnloadConstraint"
#     C_ASSOCIATED_ENTITIES = ["Order"]
#     C_ACTIONS_AFFECTED = [SimulationActions.LOAD_DRONE_ACTION,
#                           SimulationActions.LOAD_TRUCK_ACTION,
#                           SimulationActions.UNLOAD_DRONE_ACTION,
#                           SimulationActions.UNLOAD_TRUCK_ACTION]
#
#
#     def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
#
#         invalidation_idx = []
#         constraint_satisfied = False
#
#         if not isinstance(p_entity, Order):
#             raise TypeError("The load unload constraint is only to be monitored for the state of the Order entitties.")
#
#         order = p_entity
#         vehicle_id = p_entity.get_assigned_vehicle_id()
#         if order.get_state_value_by_dim_name(order.C_DIM_DELIVERY_STATUS[0]) in [order.C_STATUS_EN_ROUTE,
#                                                                                  order.C_STATUS_FAILED,
#                                                                                  order.C_STATUS_DELIVERED]:
#
#             actions_by_entity = p_action_index.actions_involving_entity["Order", order.get_id()]
#             actions_by_type = p_action_index.get_actions_of_type([SimulationActions.LOAD_DRONE_ACTION,
#                                                                   SimulationActions.UNLOAD_DRONE_ACTION,
#                                                                   SimulationActions.LOAD_TRUCK_ACTION,
#                                                                   SimulationActions.UNLOAD_TRUCK_ACTION])
#
#             invalidation_idx = list(actions_by_entity.intersection(actions_by_type))
#
#             return invalidation_idx
#
#         return invalidation_idx


class ConstraintManager(EventManager):

    C_NAME = "Constraint Manager"
    C_EVENT_MASK_UPDATED = "New Masks Necessary"

    def __init__(self,
                 action_index: ActionIndex,
                 action_map):

        EventManager.__init__(self)
        # self.constraint_config = p_entity_constraint_config.copy()
        # self.entity_constraints = {}
        # constraints = set()
        # for entity, constraints in self.constraint_config.items():
        #     constraints.update(constraints)
        # for constraint_class in Constraint.__subclasses__():
        #
        # self.constraints = []
        # self.constraint_names = []
        # for constraint in constraints:
        #     self.constraints.append(constraint.__call__())
        #     self.constraint_names.append(constraint.C_NAME)
        self.entity_constraints = {}
        self.setup_constraint_entity_map()
        self.action_index = action_index
        self.action_map = action_map

    def setup_constraint_entity_map(self):
        """

        :return:
        """
        self.entity_constraints = {}
        for con in Constraint.__subclasses__():
            for entity in con.C_ASSOCIATED_ENTITIES:
                if entity in self.entity_constraints:
                    self.entity_constraints[entity].append(con.__call__())
                else:
                    self.entity_constraints[entity] = [con.__call__()]

    def get_constraints_by_entity(self, p_entity):
        """

        :param p_entity:
        :return:
        """
        if p_entity.C_NAME in self.entity_constraints.keys():
            return self.entity_constraints[p_entity.C_NAME]
        return []

    def handle_entity_state_change(self, p_event_id, p_event_object):
        entity = p_event_object.get_raising_object()
        idx_to_mask = set()
        # We are not checking for unmasking, since we are now checking constraints on entity state. So we always start
        # with completely unmasked version of the actions related to that entity.
        # ids_to_unmask = set()
        # Unmask all actions related to the entity
        related_actions = set()
        related_actions_by_entity = self.action_index.actions_involving_entity[(entity.C_NAME, entity.get_id())]

        if isinstance(entity, Order):
            pickup_node = entity.get_pickup_node_id()
            delivery_node = entity.get_delivery_node_id()
            related_actions_by_entity.update(self.action_index.actions_involving_entity[("Node Pair", (pickup_node, delivery_node))])
        constraints_to_check = self.get_constraints_by_entity(entity)
        related_actions_by_constraint = set()
        for constraint in constraints_to_check:
            self.log(Log.C_LOG_TYPE_I, f"Checking constraint: {constraint.C_NAME}")
            idx = constraint.get_invalidations(p_entity=entity,
                                               p_action_index=self.action_index)
            if idx is None:
                print(constraint)
            idx_to_mask.update(idx)
            related_actions_by_constraint.update(self.action_index.get_actions_of_type(constraint.C_ACTIONS_AFFECTED))
        related_actions = related_actions_by_constraint.intersection(related_actions_by_entity)
        idx_to_unmask = related_actions.difference(idx_to_mask)
        if len(idx_to_mask) or len(idx_to_unmask):
            self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                              p_event_object=Event(p_raising_object=self,
                                                   idx_to_mask=idx_to_mask,
                                                   idx_to_unmask=idx_to_unmask))


# -------------------------------------------------------------------------
# -- StateActionMapper (Now Fully Self-Configuring)
# -------------------------------------------------------------------------

class StateActionMapper:
    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        self.global_state = global_state
        self.action_map = action_map
        self.action_index = ActionIndex(global_state, action_map)
        self._invalidation_map: Dict[Tuple, Set[int]] = {}
        self.permanent_masks = set()
        self.masks = [False for i in range(len(action_map))]
        self.permanent_valid_actions = list(self.action_index.get_actions_of_type([SimulationActions.NO_OPERATION]))
        self.masks[self.permanent_valid_actions[0]] = True

    def update_masks(self, idx_to_mask, idx_to_unmask):
        if not len(self.masks):
            raise ValueError("Error in instantiation of process masks.")
        for idx in idx_to_mask:
            self.masks[idx] = False
        idx_to_unmask = idx_to_unmask.difference(self.permanent_masks)
        for idx in idx_to_unmask:
            self.masks[idx] = True

    def handle_new_masks_event(self, p_event_id, p_event_object):
        raising_object = p_event_object.get_raising_object()
        if isinstance(raising_object, ConstraintManager):
            idx_to_mask = p_event_object.get_data()['idx_to_mask']
            idx_to_unmask = p_event_object.get_data()['idx_to_unmask']
            self.update_masks(idx_to_mask, idx_to_unmask)
        else:
            return

    def generate_masks(self):
        return self.masks


# -------------------------------------------------------------------------
# -- Validation Block (Expanded for Larger Instances)
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # # 1. Create a more comprehensive set of Mock Objects for the test
    # from ddls_src.entities.order import Order
    # from ddls_src.entities.vehicles.truck import Truck
    #
    #
    # class MockOrder(Order):
    #     def __init__(self, p_id, status):
    #         super().__init__(p_id=p_id, customer_node_id=0, time_received=0, SLA_deadline=0)
    #         self.status = status
    #
    #
    # class MockTruck(Truck):
    #     def __init__(self, p_id, status, cargo_count=0, capacity=1):
    #         super().__init__(p_id=p_id, start_node_id=0)
    #         self.status = status
    #         self.cargo_manifest = [0] * cargo_count
    #         self.max_payload_capacity = capacity
    #
    #
    # class MockGlobalState:
    #     def __init__(self):
    #         self.orders = {
    #             0: MockOrder(p_id=0, status='pending'),
    #             1: MockOrder(p_id=1, status='delivered'),
    #             2: MockOrder(p_id=2, status='pending'),
    #             3: MockOrder(p_id=3, status='cancelled')
    #         }
    #         self.trucks = {
    #             101: MockTruck(p_id=101, status='idle', cargo_count=0, capacity=2),
    #             102: MockTruck(p_id=102, status='en_route'),
    #             103: MockTruck(p_id=103, status='idle', cargo_count=2, capacity=2),  # This truck is full
    #             104: MockTruck(p_id=104, status='maintenance')
    #         }
    #         self.drones = {}
    #         self.micro_hubs = {}
    #
    #
    # mock_gs = MockGlobalState()
    #
    # # Create a more comprehensive mock action map for the test
    # mock_action_map = {
    #     (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 101): 0,
    #     (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 2, 101): 1,
    #     (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 1, 101): 2,
    #     (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 3, 101): 3,
    #     (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 102): 4,
    #     (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 104): 5,
    #     (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 103): 6,
    # }
    #
    # print("--- Validating Self-Configuring StateActionMapper (Large Instance) ---")
    #
    # # 2. Instantiate the StateActionMapper
    # mapper = StateActionMapper(mock_gs, mock_action_map)
    #
    # print("\n[A] Generated Invalidation Map (Rulebook):")
    # pprint(mapper._invalidation_map)
    #
    # # 3. Generate the mask and perform assertions
    # print("\n[B] Generating Mask and Running Assertions...")
    # final_mask = mapper.generate_mask()
    #
    # print(f"  - Final Mask: {final_mask.astype(int)}")
    #
    # # Assertions for Valid Actions
    # assert final_mask[0] == True, "Test Case 1 FAILED: Assigning pending order 0 to idle truck 101 should be valid"
    # assert final_mask[1] == True, "Test Case 2 FAILED: Assigning pending order 2 to idle truck 101 should be valid"
    # print("  - PASSED: All expected valid actions are correctly marked as valid.")
    #
    # # Assertions for Invalid Actions
    # assert final_mask[2] == False, "Test Case 3 FAILED: Assigning delivered order 1 should be invalid"
    # assert final_mask[3] == False, "Test Case 4 FAILED: Assigning cancelled order 3 should be invalid"
    # assert final_mask[4] == False, "Test Case 5 FAILED: Assigning to busy truck 102 should be invalid"
    # assert final_mask[5] == False, "Test Case 6 FAILED: Assigning to maintenance truck 104 should be invalid"
    # assert final_mask[6] == False, "Test Case 7 FAILED: Assigning to full truck 103 should be invalid"
    # print("  - PASSED: All expected invalid actions are correctly marked as invalid.")
    #
    # print("\n--- Validation Complete ---")
    print([c.C_ASSOCIATED_ENTITIES for c in Constraint.__subclasses__()])
    # print([c for c in LogisticEntity.__subclasses__()])

