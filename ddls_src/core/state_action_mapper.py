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
    represents a single, specific rule in the simulation that can invalidate
    certain actions based on the system's state.

    Attributes
    ----------
    C_ASSOCIATED_ENTITIES : List[str]
        A list of entity type names this constraint applies to.
    C_ACTIONS_AFFECTED : List[SimulationActions]
        A list of simulation actions that this constraint can invalidate.
    C_DEFAULT_EFFECT : bool
        The default boolean effect of the constraint.
    C_NAME : str
        The unique name of the constraint.
    C_EVENT_CONSTRAINT_UPDATE : str
        The event ID for constraint updates.
    """
    C_ASSOCIATED_ENTITIES = []
    C_ACTIONS_AFFECTED = []
    C_DEFAULT_EFFECT = True
    C_NAME = None
    C_EVENT_CONSTRAINT_UPDATE = "ConstraintUpdate"

    def __init__(self, p_reverse_action_map):
        """
        Initializes the Constraint.

        Parameters
        ----------
        p_reverse_action_map : Dict
            A mapping from action index to action details.
        """
        EventManager.__init__(self, p_logging=False)
        self.reverse_action_map = p_reverse_action_map

    def raise_constraint_change_event(self, p_entities, p_effect):
        """
        Raises an event to signal a change in the constraint's satisfaction.

        Parameters
        ----------
        p_entities : List
            The list of entities affected by the constraint change.
        p_effect : bool
            The new effect or satisfaction status of the constraint.
        """
        p_event_data = [p_entities, p_effect]
        self._raise_event(p_event_id=Constraint.C_EVENT_CONSTRAINT_UPDATE,
                          p_event_object=Event(p_raising_object=self,
                                               p_event_data=p_event_data))

    @abstractmethod
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Determines which actions should be invalidated for a given entity.

        Parameters
        ----------
        p_entity : object
            The entity instance being checked.
        p_action_index : ActionIndex
            The action indexer to find relevant action IDs.
        **p_kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        List
            A list of action indices that are invalidated by this constraint.
        """
        raise NotImplementedError


# -------------------------------------------------------------------------------------------------


class VehicleAvailableConstraint(Constraint):
    """
    Constraint that invalidates actions if a vehicle is not available.
    This applies to actions like assigning orders or moving the vehicle.
    """
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

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Invalidates vehicle actions if the vehicle is not available.
        """
        invalidation_idx = []
        if not isinstance(p_entity, Vehicle):
            raise TypeError("Vehicle Availability Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        if vehicle.get_state_value_by_dim_name(Vehicle.C_DIM_AVAILABLE[0]):
            # Vehicle is available, so no actions are invalidated.
            return invalidation_idx
        else:
            # Vehicle is not available, find and return all related actions to be masked.
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_type.intersection(actions_by_entity))
            return invalidation_idx


# -------------------------------------------------------------------------------------------------


class VehicleAtDeliveryNodeConstraint(Constraint):
    """
    Constraint that invalidates unloading actions if a vehicle is not at the
    correct delivery node for any of its assigned orders.
    """
    C_NAME = "VehicleAtDeliveryNodeConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
                          SimulationActions.UNLOAD_DRONE_ACTION,
                          SimulationActions.DRONE_LAND,
                          SimulationActions.DRONE_LAUNCH]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Invalidates unloading actions if the vehicle is not at a delivery node.
        """
        invalidation_idx = []
        if not (isinstance(p_entity, Drone) or isinstance(p_entity, Truck)):
            raise TypeError("Vehicle At Delivery Node Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        node_vehicle = vehicle.get_current_node()
        delivery_orders = vehicle.get_delivery_orders()

        is_at_delivery_node = False
        if delivery_orders:
            for order_obj in delivery_orders:
                try:
                    node_next_delivery_order = order_obj.get_delivery_node_id()
                    if node_next_delivery_order == node_vehicle:
                        is_at_delivery_node = True
                        break
                except KeyError:
                    continue

        if is_at_delivery_node:
            # Vehicle is at a correct delivery node. Unmask actions for relevant orders.
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            relevant_orders = [o.get_id() for o in vehicle.delivery_orders if o.get_delivery_node_id() == node_vehicle]

            relevant_actions_by_order = set()
            for o in relevant_orders:
                actions_by_order = p_action_index.actions_involving_entity[("Order", o)]
                relevant_actions_by_order.update(actions_by_order.intersection(actions_by_type))

            actions_to_mask = actions_by_entity.intersection(actions_by_type)
            actions_to_mask = list(actions_to_mask.difference(relevant_actions_by_order))
            return actions_to_mask

        # Vehicle is not at any delivery node, so mask all related unload actions.
        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
        actions_to_mask = actions_by_entity.intersection(actions_by_type)
        return list(actions_to_mask)


# -------------------------------------------------------------------------------------------------


class VehicleAtPickUpNodeConstraint(Constraint):
    """
    Constraint that invalidates loading actions if a vehicle is not at the
    correct pickup node for any of its assigned orders.
    """
    C_NAME = "VehicleAtPickUpNodeConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION,
                          SimulationActions.DRONE_LAUNCH,
                          SimulationActions.DRONE_LAND]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Invalidates loading actions if the vehicle is not at a pickup node.
        """
        if not (isinstance(p_entity, Drone) or isinstance(p_entity, Truck)):
            raise TypeError("Vehicle At PickUp Node Constraint can only be applied to a vehicle entity.")

        vehicle = p_entity
        node_vehicle = vehicle.get_current_node()
        pickup_orders = vehicle.get_pickup_orders()

        is_at_pickup_node = False
        if pickup_orders:
            for order_obj in pickup_orders:
                try:
                    node_next_pickup_order = order_obj.get_pickup_node_id()
                    if node_next_pickup_order == node_vehicle:
                        is_at_pickup_node = True
                        break
                except KeyError:
                    continue

        if is_at_pickup_node:
            # Vehicle is at a correct pickup node. Unmask actions for relevant orders.
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            relevant_orders = [o.get_id() for o in vehicle.pickup_orders if o.get_pickup_node_id() == node_vehicle]

            relevant_actions_by_order = set()
            for o in relevant_orders:
                actions_by_order = p_action_index.actions_involving_entity[("Order", o)]
                relevant_actions_by_order.update(actions_by_order.intersection(actions_by_type))

            actions_to_mask = actions_by_entity.intersection(actions_by_type)
            actions_to_mask = list(actions_to_mask.difference(relevant_actions_by_order))
            return actions_to_mask

        # Vehicle is not at any pickup node, so mask all related load actions.
        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
        actions_to_mask = actions_by_entity.intersection(actions_by_type)
        return list(actions_to_mask)


# -------------------------------------------------------------------------------------------------


class OrderRequestAssignabilityConstraint(Constraint):
    """
    Constraint that invalidates order assignment actions based on vehicle/hub status
    and the existence of active order requests.
    """
    C_NAME = "OrderTripAssignabilityConstraint"
    C_ASSOCIATED_ENTITIES = ["Order", "Truck", "Drone", "Micro-Hub"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Invalidates order assignments if there's no valid request or the target is unavailable.
        """
        invalidation_idx = []
        if not (isinstance(p_entity, Vehicle)
                or isinstance(p_entity, MicroHub)
                or isinstance(p_entity, Node)
                or isinstance(p_entity, Order)):
            raise TypeError("This constraint applies to Vehicle, MicroHub, Node, or Order entities.")

        # Invalidate actions for node pairs that have no active order requests.
        invalid_order_requests = [pair for pair in p_entity.global_state.node_pairs
                                  if pair not in p_entity.global_state.get_order_requests()]
        for inv_order in invalid_order_requests:
            actions_to_invalidate = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_for_node_pair = p_action_index.actions_involving_entity[("Node Pair", inv_order)]
            invalidation_idx.extend(list(actions_to_invalidate.intersection(actions_for_node_pair)))

        if isinstance(p_entity, (Truck, Drone)):
            vehicle = p_entity
            if (not vehicle.get_state_value_by_dim_name(vehicle.C_DIM_TRIP_STATE[0]) == vehicle.C_TRIP_STATE_IDLE or
                    not vehicle.get_state_value_by_dim_name(vehicle.C_DIM_AVAILABLE[0])):
                actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
                actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
                invalidation_idx.extend(actions_by_entity.intersection(actions_by_type))

        if isinstance(p_entity, MicroHub):
            micro_hub = p_entity
            if micro_hub.get_state_value_by_dim_name(MicroHub.C_DIM_AVAILABILITY[0]) != 0:
                invalidation_idx.extend(
                    list(p_action_index.actions_involving_entity[(type(micro_hub), micro_hub.get_id())]))

        return invalidation_idx


# -------------------------------------------------------------------------------------------------


class VehicleCapacityConstraint(Constraint):
    """
    Constraint that invalidates assigning orders to a vehicle if it has no
    remaining cargo capacity.
    """
    C_NAME = "VehicleCapacityConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Invalidates order assignment if the vehicle is at full capacity.
        """
        invalidation_idx = []
        vehicle = p_entity
        vehicle_capacity = vehicle.get_cargo_capacity()
        current_cargo_size = vehicle.get_current_cargo_size()

        if vehicle_capacity - current_cargo_size >= 1:
            # Vehicle has capacity, no invalidations.
            return invalidation_idx
        else:
            # Vehicle is full, invalidate assignment actions.
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_entity.intersection(actions_by_type))
            return invalidation_idx


# -------------------------------------------------------------------------------------------------


class TripWithinRangeConstraint(Constraint):
    """
    Constraint that ensures a trip is within a drone's available flight range
    before assigning an order.
    """
    C_NAME = "TripWithinRangeConstraint"
    C_ASSOCIATED_ENTITIES = ["Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_DRONE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Invalidates drone order assignments for trips that exceed the drone's range.
        """
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
            return []

        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        actions_by_drone = p_action_index.actions_involving_entity[(drone.C_NAME, drone.get_id())]

        actions_by_orders = set()
        for order in orders_not_in_range:
            actions_by_orders.update(p_action_index.actions_involving_entity[(order.C_NAME, order.get_id())])

        invalidation_set = actions_by_drone.intersection(actions_by_type, actions_by_orders)
        return list(invalidation_set)


# -------------------------------------------------------------------------------------------------


class VehicleRoutingConstraint(Constraint):
    """
    Constraint that restricts vehicle movement actions to only valid pickup or
    delivery nodes based on its current orders.
    """
    C_NAME = "Vehicle Routing Constraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "Node"]
    C_ACTIONS_AFFECTED = [SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Invalidates movement to nodes that are not part of the current route plan.
        """
        if isinstance(p_entity, (Truck, Drone)):
            vehicle = p_entity
            pickup_nodes = [order.get_pickup_node_id() for order in vehicle.get_pickup_orders()]
            delivery_nodes = [order.get_delivery_node_id() for order in vehicle.get_delivery_orders()]

            all_possible_move_actions = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)

            idx_to_unmask = set()
            for node_id in pickup_nodes + delivery_nodes:
                idx_to_unmask.update(p_action_index.actions_involving_entity[("Node", node_id)])

            invalidation_idx = list(all_possible_move_actions.difference(idx_to_unmask))
            return invalidation_idx
        elif isinstance(p_entity, Node):
            # The constraint logic is driven by vehicle state, not node state.
            return []
        return []


# -------------------------------------------------------------------------------------------------


class ConsolidationConstraint(Constraint):
    """
    Constraint that ensures consolidation actions are only valid when a vehicle
    is idle or halted and has orders to process.
    """
    C_NAME = "ConsolidationConstraint"
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE_FOR_TRUCK,
                          SimulationActions.CONSOLIDATE_FOR_DRONE]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> List:
        """
        Invalidates consolidation if the vehicle is en-route or has no orders.
        """
        invalidation_idx = []
        if not isinstance(p_entity, (Truck, Drone)):
            return invalidation_idx

        vehicle = p_entity

        is_ready_for_consolidation = (vehicle.get_state_value_by_dim_name(vehicle.C_DIM_TRIP_STATE[0])
                                      in [vehicle.C_TRIP_STATE_IDLE, vehicle.C_TRIP_STATE_HALT]
                                      and (len(vehicle.pickup_orders) > 0 or p_entity.get_current_cargo_size() > 0))

        if not is_ready_for_consolidation:
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_type.intersection(actions_by_entity))

        return invalidation_idx


# -------------------------------------------------------------------------------------------------


class ConstraintManager(EventManager):
    """
    Manages all constraints in the simulation. It maps constraints to entities,
    listens for entity state changes, and raises events to update action masks.
    """
    C_NAME = "Constraint Manager"
    C_EVENT_MASK_UPDATED = "New Masks Necessary"

    def __init__(self, action_index: ActionIndex, reverse_action_map):
        """
        Initializes the ConstraintManager.

        Parameters
        ----------
        action_index : ActionIndex
            The indexer for all possible actions in the simulation.
        reverse_action_map : Dict
            A mapping from action index to action details.
        """
        EventManager.__init__(self, p_logging=False)
        self.constraints = set()
        self.entity_constraints = {}
        self.reverse_action_map = reverse_action_map
        self.action_index = action_index
        self.setup_constraint_entity_map()

    def setup_constraint_entity_map(self):
        """
        Automatically discovers all Constraint subclasses and maps them to the
        entities they are associated with.
        """
        self.entity_constraints = {}
        for con in Constraint.__subclasses__():
            constr = con(p_reverse_action_map=self.reverse_action_map)
            self.constraints.add(constr)
            for entity_name in con.C_ASSOCIATED_ENTITIES:
                if entity_name in self.entity_constraints:
                    self.entity_constraints[entity_name].append(constr)
                else:
                    self.entity_constraints[entity_name] = [constr]

    def get_constraints_by_entity(self, p_entity):
        """
        Retrieves all constraints associated with a given entity type.

        Parameters
        ----------
        p_entity : object
            The entity instance.

        Returns
        -------
        List[Constraint]
            A list of constraint objects applicable to the entity.
        """
        if p_entity.C_NAME in self.entity_constraints:
            return self.entity_constraints[p_entity.C_NAME]
        return []

    def handle_entity_state_change(self, p_event_id, p_event_object):
        """
        Event handler for entity state changes. It re-evaluates all relevant
        constraints and raises an event with updated masks.

        Parameters
        ----------
        p_event_id : str
            The ID of the event.
        p_event_object : Event
            The event object containing details about the state change.
        """
        entity = p_event_object.get_raising_object()
        idx_to_mask = set()

        related_actions_by_entity = self.action_index.actions_involving_entity.get((entity.C_NAME, entity.get_id()),
                                                                                   set())

        if isinstance(entity, Order):
            pickup_node = entity.get_pickup_node_id()
            delivery_node = entity.get_delivery_node_id()
            node_pair_actions = self.action_index.actions_involving_entity.get(
                ("Node Pair", (pickup_node, delivery_node)), set())
            related_actions_by_entity.update(node_pair_actions)

        constraints_to_check = self.get_constraints_by_entity(entity)
        related_actions_by_constraint = set()

        for constraint in constraints_to_check:
            self.log(Log.C_LOG_TYPE_I, f"Checking constraint: {constraint.C_NAME}")
            invalidations = constraint.get_invalidations(p_entity=entity, p_action_index=self.action_index)
            if invalidations is not None:
                idx_to_mask.update(invalidations)
            related_actions_by_constraint.update(self.action_index.get_actions_of_type(constraint.C_ACTIONS_AFFECTED))

        related_actions = related_actions_by_constraint.intersection(related_actions_by_entity)
        idx_to_unmask = related_actions.difference(idx_to_mask)

        if len(idx_to_mask) > 0 or len(idx_to_unmask) > 0:
            self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                              p_event_object=Event(p_raising_object=self,
                                                   idx_to_mask=idx_to_mask,
                                                   idx_to_unmask=idx_to_unmask))

    def update_constraints(self, global_state, reverse_action_map):
        """
        Updates constraints for all entities in the global state. Used for full
        re-synchronization.

        Parameters
        ----------
        global_state : GlobalState
            The global state of the simulation.
        reverse_action_map : Dict
            The updated reverse action map.
        """
        self.reverse_action_map = reverse_action_map
        for entity_dict in global_state.get_all_entities():
            for entity in entity_dict.values():
                idx_to_mask = set()

                related_actions_by_entity = self.action_index.actions_involving_entity.get(
                    (entity.C_NAME, entity.get_id()), set())

                if isinstance(entity, Order):
                    pickup_node = entity.get_pickup_node_id()
                    delivery_node = entity.get_delivery_node_id()
                    node_pair_actions = self.action_index.actions_involving_entity.get(
                        ("Node Pair", (pickup_node, delivery_node)), set())
                    related_actions_by_entity.update(node_pair_actions)

                constraints_to_check = self.get_constraints_by_entity(entity)
                related_actions_by_constraint = set()

                for constraint in constraints_to_check:
                    constraint.reverse_action_map = self.reverse_action_map
                    self.log(Log.C_LOG_TYPE_I, f"Checking constraint: {constraint.C_NAME}")
                    invalidations = constraint.get_invalidations(p_entity=entity, p_action_index=self.action_index)
                    if invalidations:
                        idx_to_mask.update(invalidations)
                    related_actions_by_constraint.update(
                        self.action_index.get_actions_of_type(constraint.C_ACTIONS_AFFECTED))

                related_actions = related_actions_by_constraint.intersection(related_actions_by_entity)
                idx_to_unmask = related_actions.difference(idx_to_mask)

                if len(idx_to_mask) > 0 or len(idx_to_unmask) > 0:
                    self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                                      p_event_object=Event(p_raising_object=self,
                                                           idx_to_mask=idx_to_mask,
                                                           idx_to_unmask=idx_to_unmask))


# -------------------------------------------------------------------------------------------------
# -- StateActionMapper (Now Fully Self-Configuring)
# -------------------------------------------------------------------------------------------------


class StateActionMapper:
    """
    Maps the system state to a valid action mask by applying constraints.
    It listens to the ConstraintManager for updates and modifies the mask accordingly.
    """

    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        """
        Initializes the StateActionMapper.

        Parameters
        ----------
        global_state : GlobalState
            The global state of the simulation.
        action_map : Dict[Tuple, int]
            A mapping from action tuple to action index.
        """
        self.global_state = global_state
        self.action_map = action_map
        self.action_index = ActionIndex(global_state, action_map)
        self._invalidation_map: Dict[Tuple, Set[int]] = {}
        self.permanent_masks = set()
        self.masks = [False for _ in range(len(action_map))]
        self.permanent_valid_actions = list(self.action_index.get_actions_of_type([SimulationActions.NO_OPERATION]))
        if self.permanent_valid_actions:
            self.masks[self.permanent_valid_actions[0]] = True

    def update_masks(self, idx_to_mask, idx_to_unmask):
        """
        Updates the action mask based on indices to mask and unmask.

        Parameters
        ----------
        idx_to_mask : Set[int]
            A set of action indices to be masked (set to False).
        idx_to_unmask : Set[int]
            A set of action indices to be unmasked (set to True).
        """
        if not self.masks:
            raise ValueError("Error in instantiation of process masks.")
        for idx in idx_to_mask:
            self.masks[idx] = False

        idx_to_unmask = idx_to_unmask.difference(self.permanent_masks)
        for idx in idx_to_unmask:
            self.masks[idx] = True

    def handle_new_masks_event(self, p_event_id, p_event_object):
        """
        Event handler for receiving new mask information from the ConstraintManager.

        Parameters
        ----------
        p_event_id : str
            The ID of the event.
        p_event_object : Event
            The event object containing indices to mask and unmask.
        """
        raising_object = p_event_object.get_raising_object()
        if isinstance(raising_object, ConstraintManager):
            idx_to_mask = p_event_object.get_data().get('idx_to_mask', set())
            idx_to_unmask = p_event_object.get_data().get('idx_to_unmask', set())
            self.update_masks(idx_to_mask, idx_to_unmask)
        else:
            return

    def generate_masks(self) -> List[bool]:
        """
        Returns the current action mask.

        Returns
        -------
        List[bool]
            The boolean mask where True indicates a valid action.
        """
        return self.masks

    def update_action_space(self, action_map):
        self.masks = [False for _ in range(len(action_map))]


# -------------------------------------------------------------------------------------------------
# -- Validation Block (Expanded for Larger Instances)
# -------------------------------------------------------------------------------------------------


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