from abc import ABC, abstractmethod
from typing import Dict, Tuple, Set, List, Iterable
from collections import defaultdict

from mlpro.bf.events import Event, EventManager
from mlpro.bf.various import Log

from ddls_src.actions.base import SimulationActions, ActionIndex
from ddls_src.entities.base import LogisticEntity
from ddls_src.entities import *
from ddls_src.entities.order import PseudoOrder, Order
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.vehicles.base import Vehicle
from ddls_src.entities.node import Node
from ddls_src.entities.micro_hub import MicroHub


# -------------------------------------------------------------------------------------------------
# -- Part 1: Pluggable Constraint Architecture (Unified)
# -------------------------------------------------------------------------------------------------

class Constraint(ABC, EventManager):
    """
    Abstract base class for a pluggable constraint rule.

    [MODIFIED] Now Responsible for its own blocking/unblocking logic:
    It maintains a map (`_entity_invalidation_map`) to remember which actions
    it blocked previously for each entity, allowing it to calculate the delta (diff)
    between state updates.
    """
    C_ACTIVE = True
    C_ASSOCIATED_ENTITIES = []
    C_ACTIONS_AFFECTED = []
    C_DEFAULT_EFFECT = True
    C_NAME = None
    C_EVENT_CONSTRAINT_UPDATE = "ConstraintUpdate"

    def __init__(self, p_reverse_action_map, p_action_index):
        EventManager.__init__(self, p_logging=False)
        self.reverse_action_map = p_reverse_action_map

        # [NEW] State tracking: Key=EntityID, Value=Set of Action Indices currently blocked by THIS constraint
        self._entity_invalidation_map = defaultdict(set)
        self.action_index = p_action_index
        self.associated_action_index = None
        self.find_associated_actions()

    def find_associated_actions(self):
        self.associated_action_index = self.action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)


    def raise_constraint_change_event(self, p_entities, p_effect):
        p_event_data = [p_entities, p_effect]
        self._raise_event(p_event_id=Constraint.C_EVENT_CONSTRAINT_UPDATE,
                          p_event_object=Event(p_raising_object=self,
                                               p_event_data=p_event_data))

    @abstractmethod
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        """
        Pure logic method: Determines which actions should be invalidated based on CURRENT state.
        Returns: (List of invalidated indices, List of validated indices)
        """
        raise NotImplementedError

    def evaluate_impact(self, p_entity, p_action_index: ActionIndex) -> Tuple[List[int], List[int]]:
        """
        [NEW] Calculates the Delta (Impact) of this constraint.
        Compares current invalidations against previous invalidations to determine
        what needs to be newly blocked and what needs to be unblocked.
        """
        # 1. Get current status
        current_invalid_indices, _ = self.get_invalidations(p_entity, p_action_index)
        current_invalid_set = set(current_invalid_indices) if current_invalid_indices else set()

        # 2. Get previous status
        entity_id = p_entity.get_id()
        previous_invalid_set = self._entity_invalidation_map[entity_id]

        # 3. Calculate Deltas
        # Actions that are in current but not previous -> Must increase block counter
        to_block = list(current_invalid_set.difference(previous_invalid_set))

        # Actions that were in previous but not current -> Must decrease block counter
        to_unblock = list(previous_invalid_set.difference(current_invalid_set))

        # 4. Update State
        self._entity_invalidation_map[entity_id] = current_invalid_set

        return to_block, to_unblock

    def clear_cache(self):
        """Resets the internal memory of the constraint."""
        self._entity_invalidation_map.clear()

    def update_operability(self, p_entity: LogisticEntity, **p_kwargs):
        pass


# -------------------------------------------------------------------------------------------------

class VehicleAvailableConstraint(Constraint):
    C_NAME = "VehicleAvailabilityConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]
    C_DEFAULT_EFFECT = True
    C_DIMS = [Vehicle.C_DIM_AVAILABLE[0]]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        if not isinstance(p_entity, Vehicle):
            raise TypeError("Vehicle Availability Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        if vehicle.get_state_value_by_dim_name(Vehicle.C_DIM_AVAILABLE[0]):
            ids_to_unblock = list(p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())])
            return [], ids_to_unblock
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


class VehicleAtDeliveryNodeConstraint(Constraint):
    C_NAME = "VehicleAtDeliveryNodeConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
                          SimulationActions.UNLOAD_DRONE_ACTION,
                          SimulationActions.DRONE_LAND,
                          SimulationActions.DRONE_LAUNCH]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        if not isinstance(p_entity, (Drone, Truck)):
            raise TypeError("Vehicle At Delivery Node Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        node_vehicle = vehicle.get_current_node()
        delivery_orders = vehicle.get_delivery_orders()

        is_at_delivery_node = False
        if delivery_orders:
            for order_obj in delivery_orders:
                try:
                    if order_obj.get_delivery_node_id() == node_vehicle:
                        is_at_delivery_node = True
                        break
                except KeyError:
                    continue

        if is_at_delivery_node:
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]

            relevant_orders = [o.get_id() for o in vehicle.delivery_orders if o.get_delivery_node_id() == node_vehicle]
            relevant_actions_by_order = set()
            for o in relevant_orders:
                actions_by_order = p_action_index.actions_involving_entity[("Order", o)]
                relevant_actions_by_order.update(actions_by_order.intersection(actions_by_type))

            actions_to_mask = actions_by_entity.intersection(actions_by_type)
            actions_to_mask = list(actions_to_mask.difference(relevant_actions_by_order))
            return actions_to_mask, []

        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
        actions_to_mask = actions_by_entity.intersection(actions_by_type)
        return list(actions_to_mask), []

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, (Truck, Drone)):
            return
        vehicle = p_entity
        node_vehicle = vehicle.get_current_node()
        delivery_orders = vehicle.get_delivery_orders()
        valid_orders = set()
        if delivery_orders:
            for order_obj in delivery_orders:
                try:
                    if order_obj.get_delivery_node_id() == node_vehicle:
                        valid_orders.add(order_obj.get_id())
                except (KeyError, AttributeError):
                    continue
        is_operable = valid_orders if valid_orders else False
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_operable


class VehicleAtPickUpNodeConstraint(Constraint):
    C_NAME = "VehicleAtPickUpNodeConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION,
                          SimulationActions.DRONE_LAUNCH,
                          SimulationActions.DRONE_LAND]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        if not isinstance(p_entity, (Drone, Truck)):
            raise TypeError("Vehicle At PickUp Node Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        node_vehicle = vehicle.get_current_node()
        pickup_orders = vehicle.get_pickup_orders()

        is_at_pickup_node = False
        if pickup_orders:
            for order_obj in pickup_orders:
                try:
                    if order_obj.get_pickup_node_id() == node_vehicle:
                        is_at_pickup_node = True
                        break
                except KeyError:
                    continue

        if is_at_pickup_node:
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]

            relevant_orders = [o.get_id() for o in vehicle.pickup_orders if o.get_pickup_node_id() == node_vehicle]
            relevant_actions_by_order = set()
            for o in relevant_orders:
                actions_by_order = p_action_index.actions_involving_entity[("Order", o)]
                relevant_actions_by_order.update(actions_by_order.intersection(actions_by_type))

            actions_to_mask = actions_by_entity.intersection(actions_by_type)
            actions_to_mask = list(actions_to_mask.difference(relevant_actions_by_order))
            return actions_to_mask, []

        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
        actions_to_mask = actions_by_entity.intersection(actions_by_type)
        return list(actions_to_mask), []

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, (Truck, Drone)):
            return
        vehicle = p_entity
        node_vehicle = vehicle.get_current_node()
        if node_vehicle is None:
            for action_type in self.C_ACTIONS_AFFECTED:
                if action_type in p_entity.action_operability:
                    p_entity.action_operability[action_type] = False
            return
        pickup_orders = vehicle.get_pickup_orders()
        valid_orders = set()
        if pickup_orders:
            for order_obj in pickup_orders:
                try:
                    if order_obj.get_pickup_node_id() == node_vehicle:
                        valid_orders.add(order_obj.get_id())
                except (KeyError, AttributeError):
                    continue
        is_operable = valid_orders if valid_orders else False
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_operable


class OrderRequestAssignabilityConstraint(Constraint):
    C_NAME = "OrderTripAssignabilityConstraint"
    C_ACTIVE = True
    C_ASSOCIATED_ENTITIES = ["Order", "Truck", "Drone", "Micro-Hub"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

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


class VehicleCapacityConstraint(Constraint):
    C_NAME = "VehicleCapacityConstraint"
    C_ACTIVE = True
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        invalidation_idx = []
        vehicle = p_entity

        vehicle_capacity = vehicle.get_cargo_capacity()
        committed_load = len(vehicle.get_pickup_orders()) + len(vehicle.get_delivery_orders())

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
        current_cargo_size = vehicle.get_current_cargo_size()
        has_capacity = (vehicle_capacity - current_cargo_size >= 1)
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = has_capacity


class TripWithinRangeConstraint(Constraint):
    C_NAME = "TripWithinRangeConstraint"
    C_ASSOCIATED_ENTITIES = ["Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_DRONE]

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
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "Node"]
    C_ACTIONS_AFFECTED = [SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]

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
    C_ASSOCIATED_ENTITIES = ["Order"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        invalidation_idx = []
        if not isinstance(p_entity, Order):
            raise TypeError("The \"Micro-Hub assignability constraint\" is only applicable to an Order entity.")

        if isinstance(p_entity, PseudoOrder):
            mh_node_id = [p_entity.parent_order.assigned_micro_hub_id]
        else:
            mh_node_id = []

        ps_order = p_entity
        mh_node_ids = mh_node_id + ps_order.mh_assignment_history

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
                forbidden_ids.update(p_entity.mh_assignment_history)
        all_hub_ids = set(p_entity.global_state.micro_hubs.keys())
        allowed_ids = all_hub_ids.difference(forbidden_ids)
        is_operable = allowed_ids if allowed_ids else False
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_operable


class CoordinatedDeliveryAssignmentConstraint(Constraint):
    C_NAME = "Co-ordinated Delivery Assignment Constraint"
    C_ASSOCIATED_ENTITIES = ["Order", "Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        def check_ass_precedence(p_order):
            ass_precedence_satisfied = True
            for pre_order in p_order.predecessor_orders:
                if pre_order.get_state_value_by_dim_name(pre_order.C_DIM_DELIVERY_STATUS[0]) not in [
                    pre_order.C_STATUS_PLACED,
                    pre_order.C_STATUS_ACCEPTED,
                    pre_order.C_STATUS_FAILED]:
                    ass_precedence_satisfied = True and ass_precedence_satisfied
                else:
                    ass_precedence_satisfied = False
            return ass_precedence_satisfied

        invalidation_idx = []
        validation_idx = []
        if not (isinstance(p_entity, Order) or isinstance(p_entity, (Truck, Drone))):
            raise TypeError("Wrong entity type for the constraint")

        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        pseudo_orders = [order for order in p_entity.global_state.get_all_entities_by_type("order").values()
                         if (isinstance(order, PseudoOrder) and len(
                order.predecessor_orders) and not check_ass_precedence(order))]

        for ps_ordr in pseudo_orders:
            actions_by_entity = p_action_index.actions_involving_entity[
                "Node Pair", (ps_ordr.get_pickup_node_id(), ps_ordr.get_delivery_node_id())]
            invalidation_idx.extend(actions_by_entity.intersection(actions_by_type))

        return invalidation_idx, validation_idx

    def update_operability(self, p_entity, **p_kwargs):
        if not isinstance(p_entity, Order):
            return
        is_ready = True
        if isinstance(p_entity, PseudoOrder) and p_entity.predecessor_orders:
            for pre_order in p_entity.predecessor_orders:
                status = pre_order.get_state_value_by_dim_name(Order.C_DIM_DELIVERY_STATUS[0])
                if status in [Order.C_STATUS_PLACED, Order.C_STATUS_ACCEPTED, Order.C_STATUS_FAILED]:
                    is_ready = False
                    break
        for action_type in self.C_ACTIONS_AFFECTED:
            if action_type in p_entity.action_operability:
                p_entity.action_operability[action_type] = is_ready


class OrderLoadConstraint(Constraint):
    C_NAME = "OrderLoadConstraint"
    C_ASSOCIATED_ENTITIES = ["Order", "Vehicle"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION]

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


class OrderUnloadConstraint(Constraint):
    C_NAME = "OrderUnloadConstraint"
    C_ASSOCIATED_ENTITIES = ["Order", "Vehicle"]
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
                          SimulationActions.UNLOAD_DRONE_ACTION]

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


# -------------------------------------------------------------------------------------------------

class ConstraintManager(EventManager):
    """
    Manages all constraints in the simulation.

    [MODIFIED] Simplified Responsibility:
    Does NOT maintain a cache of invalidations.
    Delegates the 'impact' calculation (block/unblock deltas) to the Constraint objects.
    Aggregates these deltas and notifies the StateActionMapper.
    """
    C_NAME = "Constraint Manager"
    C_EVENT_MASK_UPDATED = "New Masks Necessary"

    def __init__(self, action_index: ActionIndex, reverse_action_map):
        EventManager.__init__(self, p_logging=False)
        self._update_counter = 0
        self.constraints = set()
        self.entity_constraints = {}
        self.reverse_action_map = reverse_action_map
        self.action_index = action_index
        self.setup_constraint_entity_map()
        print("Constraints Setup")

    def setup_constraint_entity_map(self):
        self.entity_constraints = {}
        for con in Constraint.__subclasses__():
            if con.C_ACTIVE:
                constr = con(p_reverse_action_map=self.reverse_action_map, p_action_index=self.action_index)
                self.constraints.add(constr)
                for entity_name in con.C_ASSOCIATED_ENTITIES:
                    if entity_name in self.entity_constraints:
                        self.entity_constraints[entity_name].append(constr)
                    else:
                        self.entity_constraints[entity_name] = [constr]

    def get_constraints_by_entity(self, p_entity):
        if p_entity.C_NAME in self.entity_constraints:
            return self.entity_constraints[p_entity.C_NAME]
        return []

    def handle_entity_state_change(self, p_event_id, p_event_object):
        """
        Triggered when an entity state changes.
        Polls relevant constraints for their calculated impact (deltas).
        """
        self._update_counter += 1
        entity = p_event_object.get_raising_object()

        total_to_block = []
        total_to_unblock = []

        constraints_to_check = self.get_constraints_by_entity(entity)

        for constraint in constraints_to_check:
            self.log(Log.C_LOG_TYPE_I, f"Checking constraint: {constraint.C_NAME}")

            # [MODIFIED] Ask the constraint to evaluate its own impact (Delta)
            to_block, to_unblock = constraint.evaluate_impact(p_entity=entity, p_action_index=self.action_index)

            total_to_block.extend(to_block)
            total_to_unblock.extend(to_unblock)

        if len(total_to_block) > 0 or len(total_to_unblock) > 0:
            self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                              p_event_object=Event(p_raising_object=self,
                                                   to_block=total_to_block,
                                                   to_unblock=total_to_unblock))

    def update_constraints(self, global_state, reverse_action_map):
        """
        Full initialization/Reset.
        Clears all constraint caches and calculates initial blocks.
        """
        self.reverse_action_map = reverse_action_map

        # Helper to clear caches inside constraints
        for constraint in self.constraints:
            constraint.clear_cache()
            constraint.reverse_action_map = self.reverse_action_map

        total_to_block = []

        for entity_dict in global_state.get_all_entities():
            for entity in entity_dict.values():
                constraints_to_check = self.get_constraints_by_entity(entity)

                for constraint in constraints_to_check:
                    # Initial evaluation: impact against empty cache = full current invalidation
                    to_block, _ = constraint.evaluate_impact(p_entity=entity, p_action_index=self.action_index)
                    total_to_block.extend(to_block)

        if total_to_block:
            self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                              p_event_object=Event(p_raising_object=self,
                                                   to_block=total_to_block,
                                                   to_unblock=[]))


# -------------------------------------------------------------------------------------------------
# -- StateActionMapper (Counter & Mask Logic)
# -------------------------------------------------------------------------------------------------

class StateActionMapper:
    """
    Maps the system state to a valid action mask.

    [MODIFIED]
    1. Maintains 'mask_counters': Integer count of how many constraints block an action.
    2. Maintains 'masks': Boolean array (True=Valid, False=Invalid).
    3. Updates 'masks' ONLY when counters transition between 0 and 1.
    """

    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        self.global_state = global_state
        self.action_map = action_map
        self.action_index = ActionIndex(global_state, action_map)

        # Initialize counters (0) and masks (True/Valid)
        self.mask_counters = [0] * len(action_map)
        self.masks = [True] * len(action_map)

        # Actions that are permanently valid (e.g. No-Op) are protected from modification
        self.permanent_valid_actions = set(self.action_index.get_actions_of_type([SimulationActions.NO_OPERATION]))

    def update_counters_and_masks(self, indices_to_block: Iterable[int], indices_to_unblock: Iterable[int]):
        """
        Updates counters and flips boolean masks on 0 <-> 1 transitions.
        """
        # --- BLOCK LOGIC ---
        for idx in indices_to_block:
            if idx not in self.permanent_valid_actions:
                self.mask_counters[idx] += 1

                # TRANSITION 0 -> 1: Action becomes blocked
                if self.mask_counters[idx] == 1:
                    self.masks[idx] = False

        # --- UNBLOCK LOGIC ---
        for idx in indices_to_unblock:
            if idx not in self.permanent_valid_actions:
                self.mask_counters[idx] -= 1

                # TRANSITION 1 -> 0: Action becomes unblocked
                if self.mask_counters[idx] == 0:
                    self.masks[idx] = True

                # Safety check
                if self.mask_counters[idx] < 0:
                    print(f"[StateActionMapper] Counter negative for index {idx}. Resetting.")
                    self.mask_counters[idx] = 0
                    self.masks[idx] = True

    def handle_new_masks_event(self, p_event_id, p_event_object):
        """
        Handles the event from ConstraintManager containing the lists of indices.
        """
        raising_object = p_event_object.get_raising_object()
        if isinstance(raising_object, ConstraintManager):
            to_block = p_event_object.get_data().get('to_block', [])
            to_unblock = p_event_object.get_data().get('to_unblock', [])
            self.update_counters_and_masks(to_block, to_unblock)
        else:
            return

    def generate_masks(self) -> List[bool]:
        """
        Returns the maintained boolean mask.
        """
        return self.masks

    def reset_masks(self):
        """Resets all counters to zero and masks to True."""
        self.mask_counters = [0] * len(self.mask_counters)
        self.masks = [True] * len(self.masks)

    def update_action_space(self, action_map, old_action_map):
        # Reset on resize
        self.mask_counters = [0] * len(action_map)
        self.masks = [True] * len(action_map)
        self.action_map = action_map


if __name__ == '__main__':
    # Debugging: Print discovered constraints
    print([c.C_ASSOCIATED_ENTITIES for c in Constraint.__subclasses__()])