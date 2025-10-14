# Import numpy for numerical operations, although not directly used in this snippet, it's a common dependency.
import numpy as np
# Import various typing hints for better code readability and static analysis.
from typing import Dict, Any, Tuple, Set, List
# Import defaultdict for creating dictionaries with default values for missing keys.
from collections import defaultdict
# Import pprint for pretty-printing complex data structures.
from pprint import pprint
# Import ABC (Abstract Base Class) and abstractmethod decorator for creating abstract classes.
from abc import ABC, abstractmethod

from networkx import predecessor
# Import shiboken6 invalidate, potentially for UI integration or memory management in a Qt environment.
# from shiboken6 import invalidate

# Local Imports from the project's source code.
# Import the base class for simulation actions.
from ddls_src.actions.base import SimulationActions
# Import the base System class from MLPro's framework, likely for creating mock objects or type hinting.
from mlpro.bf.systems import System  # Import System for mock object inheritance
# Import Event and EventManager for handling an event-driven architecture.
from mlpro.bf.events import Event, EventManager  # Import Event for type hinting
# Import the ActionIndex class for efficient action lookups.
from ddls_src.actions.base import ActionIndex
# Import the base class for all logistic entities in the simulation.
from ddls_src.entities.base import LogisticEntity
# Import all specific entity classes from the entities module.
from ddls_src.entities import *
# Import the Log class for structured logging.
from mlpro.bf.various import Log
from ddls_src.entities.order import PseudoOrder
from ddls_src.core.global_state import GlobalState

# # Forward declarations for type hinting to avoid circular import issues.
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
    # Class constant defining which entity types this constraint is associated with.
    C_ACTIVE = True

    C_ASSOCIATED_ENTITIES = []
    # Class constant listing the simulation actions this constraint can affect.
    C_ACTIONS_AFFECTED = []
    # Class constant for the default effect of the constraint (e.g., True might mean 'valid').
    C_DEFAULT_EFFECT = True
    # Class constant for the unique name of the constraint.
    C_NAME = None
    # Class constant for the event ID triggered when the constraint's status changes.
    C_EVENT_CONSTRAINT_UPDATE = "ConstraintUpdate"


    def __init__(self, p_reverse_action_map):
        """
        Initializes the Constraint.

        Parameters
        ----------
        p_reverse_action_map : Dict
            A mapping from action index to action details.
        """
        # Initialize the EventManager parent class, disabling its logging by default.
        EventManager.__init__(self, p_logging=False)
        # Store the reverse action map, which maps integer indices back to action tuples.
        self.reverse_action_map = p_reverse_action_map

#----------------------------------------------------------------------------------------------------

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
        # Prepare the data payload for the event, containing affected entities and the new effect.
        p_event_data = [p_entities, p_effect]
        # Raise the event using the predefined event ID.
        self._raise_event(p_event_id=Constraint.C_EVENT_CONSTRAINT_UPDATE,
                          p_event_object=Event(p_raising_object=self,
                                               p_event_data=p_event_data))

#----------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
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
        # This is an abstract method that must be implemented by all subclasses.
        raise NotImplementedError

#----------------------------------------------------------------------------------------------------





class VehicleAvailableConstraint(Constraint):
    """
    Constraint that invalidates actions if a vehicle is not available.
    This applies to actions like assigning orders or moving the vehicle.
    """
    # Define the unique name for this specific constraint.
    C_NAME = "VehicleAvailabilityConstraint"
    # Specify that this constraint applies to "Truck" and "Drone" entities.
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    # List the actions that are affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]
    # Set the default effect of the constraint.
    C_DEFAULT_EFFECT = True

    # # This is a commented-out event handler method, potentially for future use.
    # def _handle_vehicle_availabiliy(self, p_event_id, p_event_object):
    #     constraint_satisfied = True
    #     vehicle = p_event_object.get_data()["Vehicle"]
    #     if vehicle.get_availability():
    #         constraint_satisfied = True
    #     else:
    #         constraint_satisfied = False
    #     self.raise_constraint_change_event(p_entities=[vehicle],
    #                                        p_effect=constraint_satisfied)


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        """
        Invalidates vehicle actions if the vehicle is not available.
        """
        # Initialize an empty list to store the indices of actions to be invalidated.
        invalidation_idx = []
        # Check if the provided entity is an instance of Vehicle.
        if not isinstance(p_entity, Vehicle):
            # If not, raise a TypeError as this constraint is only for vehicles.
            raise TypeError("Vehicle Availability Constraint can only be applied to a vehicle entity.")

        # Assign the entity to a more specific variable name.
        vehicle = p_entity
        # Check the vehicle's availability status from its state.
        if vehicle.get_state_value_by_dim_name(Vehicle.C_DIM_AVAILABLE[0]):
            # If the vehicle is available, return the empty list (no invalidations).
            return invalidation_idx, []
        else:
            # If the vehicle is not available, find all actions related to it that should be masked.
            # Get all action indices of the types affected by this constraint.
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            # Get all action indices involving this specific vehicle instance.
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            # Find the intersection of the two sets to get the actions to invalidate.
            invalidation_idx = list(actions_by_type.intersection(actions_by_entity))
            # Return the list of invalidated action indices.
            return invalidation_idx, []

#----------------------------------------------------------------------------------------------------





class VehicleAtDeliveryNodeConstraint(Constraint):
    """
    Constraint that invalidates unloading actions if a vehicle is not at the
    correct delivery node for any of its assigned orders.
    """
    # Define the unique name for this constraint.
    C_NAME = "VehicleAtDeliveryNodeConstraint"
    # Specify that this constraint applies to "Truck" and "Drone" entities.
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    # List the unload and related actions affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
                          SimulationActions.UNLOAD_DRONE_ACTION,
                          SimulationActions.DRONE_LAND,
                          SimulationActions.DRONE_LAUNCH]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        """
        Invalidates unloading actions if the vehicle is not at a delivery node.
        """
        # Initialize an empty list for invalidated action indices.
        invalidation_idx = []
        # Check if the entity is a Drone or a Truck.
        if not (isinstance(p_entity, Drone) or isinstance(p_entity, Truck)):
            # Raise an error if the entity type is incorrect.
            raise TypeError("Vehicle At Delivery Node Constraint can only be applied to a vehicle entity.")

        # Assign the entity to a more specific variable name.
        vehicle = p_entity
        # Get the current node (location) of the vehicle.
        node_vehicle = vehicle.get_current_node()
        # Get the list of orders the vehicle is supposed to deliver.
        delivery_orders = vehicle.get_delivery_orders()

        # Flag to check if the vehicle is at any of the required delivery nodes.
        is_at_delivery_node = False
        # If the vehicle has delivery orders, iterate through them.
        if delivery_orders:
            for order_obj in delivery_orders:
                try:
                    # Get the delivery node ID for the current order.
                    node_next_delivery_order = order_obj.get_delivery_node_id()
                    # If the vehicle's current node matches the order's delivery node...
                    if node_next_delivery_order == node_vehicle:
                        # ...set the flag to True and break the loop.
                        is_at_delivery_node = True
                        break
                except KeyError:
                    # If there's an issue getting the node ID, skip to the next order.
                    continue

        # If the vehicle is at a correct delivery node...
        if is_at_delivery_node:
            # Unmask actions only for the specific orders that can be delivered at this node.
            # Get all actions of the affected types (e.g., unload actions).
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            # Get all actions involving this specific vehicle.
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            # Find which orders are deliverable at the current node.
            relevant_orders = [o.get_id() for o in vehicle.delivery_orders if o.get_delivery_node_id() == node_vehicle]

            # Collect all actions related to these relevant orders.
            relevant_actions_by_order = set()
            for o in relevant_orders:
                actions_by_order = p_action_index.actions_involving_entity[("Order", o)]
                # Add the intersection of order-specific actions and affected action types to the set.
                relevant_actions_by_order.update(actions_by_order.intersection(actions_by_type))

            # Start with all potentially masked actions for this vehicle.
            actions_to_mask = actions_by_entity.intersection(actions_by_type)
            # Unmask the relevant actions by taking the set difference.
            actions_to_mask = list(actions_to_mask.difference(relevant_actions_by_order))
            # Return the final list of actions to be masked.
            return actions_to_mask, []

        # If the vehicle is not at any delivery node, mask all related unload actions.
        # Get all actions of the affected types.
        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        # Get all actions involving this vehicle.
        actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
        # The actions to mask are the intersection of the two sets.
        actions_to_mask = actions_by_entity.intersection(actions_by_type)
        # Return the list of actions to mask.
        return list(actions_to_mask), []

#----------------------------------------------------------------------------------------------------





class VehicleAtPickUpNodeConstraint(Constraint):
    """
    Constraint that invalidates loading actions if a vehicle is not at the
    correct pickup node for any of its assigned orders.
    """
    # Define the unique name for this constraint.
    C_NAME = "VehicleAtPickUpNodeConstraint"
    # Specify that this constraint applies to "Truck" and "Drone" entities.
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    # List the load and related actions affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION,
                          SimulationActions.DRONE_LAUNCH,
                          SimulationActions.DRONE_LAND]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        """
        Invalidates loading actions if the vehicle is not at a pickup node.
        """
        # Check if the entity is a Drone or a Truck.
        if not (isinstance(p_entity, Drone) or isinstance(p_entity, Truck)):
            # Raise an error if the entity type is incorrect.
            raise TypeError("Vehicle At PickUp Node Constraint can only be applied to a vehicle entity.")

        # Assign the entity to a more specific variable name.
        vehicle = p_entity
        # Get the current node (location) of the vehicle.
        node_vehicle = vehicle.get_current_node()
        # Get the list of orders the vehicle is supposed to pick up.
        pickup_orders = vehicle.get_pickup_orders()

        # Flag to check if the vehicle is at any of the required pickup nodes.
        is_at_pickup_node = False
        # If the vehicle has pickup orders, iterate through them.
        if pickup_orders:
            for order_obj in pickup_orders:
                try:
                    # Get the pickup node ID for the current order.
                    node_next_pickup_order = order_obj.get_pickup_node_id()
                    # If the vehicle's current node matches the order's pickup node...
                    if node_next_pickup_order == node_vehicle:
                        # ...set the flag to True and break the loop.
                        is_at_pickup_node = True
                        break
                except KeyError:
                    # If there's an issue getting the node ID, skip to the next order.
                    continue

        # If the vehicle is at a correct pickup node...
        if is_at_pickup_node:
            # Unmask actions only for the specific orders that can be picked up at this node.
            # Get all actions of the affected types (e.g., load actions).
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            # Get all actions involving this specific vehicle.
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            # Find which orders are pickup-able at the current node.
            relevant_orders = [o.get_id() for o in vehicle.pickup_orders if o.get_pickup_node_id() == node_vehicle]

            # Collect all actions related to these relevant orders.
            relevant_actions_by_order = set()
            for o in relevant_orders:
                actions_by_order = p_action_index.actions_involving_entity[("Order", o)]
                # Add the intersection of order-specific actions and affected action types to the set.
                relevant_actions_by_order.update(actions_by_order.intersection(actions_by_type))

            # Start with all potentially masked actions for this vehicle.
            actions_to_mask = actions_by_entity.intersection(actions_by_type)
            # Unmask the relevant actions by taking the set difference.
            actions_to_mask = list(actions_to_mask.difference(relevant_actions_by_order))
            # Return the final list of actions to be masked.
            return actions_to_mask, []

        # If the vehicle is not at any pickup node, mask all related load actions.
        # Get all actions of the affected types.
        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        # Get all actions involving this vehicle.
        actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
        # The actions to mask are the intersection of the two sets.
        actions_to_mask = actions_by_entity.intersection(actions_by_type)
        # Return the list of actions to mask.
        return list(actions_to_mask), []

#----------------------------------------------------------------------------------------------------





class OrderRequestAssignabilityConstraint(Constraint):
    """
    Constraint that invalidates order assignment actions based on vehicle/hub status
    and the existence of active order requests.
    """
    # Define the unique name for this constraint.
    C_NAME = "OrderTripAssignabilityConstraint"
    # Specify the entity types this constraint is associated with.
    C_ASSOCIATED_ENTITIES = ["Order", "Truck", "Drone", "Micro-Hub"]
    # List the assignment actions affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        """
        Invalidates order assignments if there's no valid request or the target is unavailable.
        """
        # Initialize an empty list for invalidated action indices.
        invalidation_idx = []
        # Check if the entity is of an applicable type.
        if not (isinstance(p_entity, Vehicle)
                or isinstance(p_entity, MicroHub)
                or isinstance(p_entity, Node)
                or isinstance(p_entity, Order)):
            # Raise an error if the entity type is incorrect.
            raise TypeError("This constraint applies to Vehicle, MicroHub, Node, or Order entities.")

        # Invalidate actions for node pairs (trips) that have no active order requests.
        # Get a list of node pairs that do NOT have active orders.
        invalid_order_requests = [pair for pair in p_entity.global_state.node_pairs
                                  if pair not in p_entity.global_state.get_order_requests()]
        # Iterate through these invalid node pairs.
        for inv_order in invalid_order_requests:
            # Get all actions of the affected assignment types.
            actions_to_invalidate = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            # Get all actions associated with this specific node pair.
            actions_for_node_pair = p_action_index.actions_involving_entity[("Node Pair", inv_order)]
            # Add the intersection to the list of invalidations.
            invalidation_idx.extend(list(actions_to_invalidate.intersection(actions_for_node_pair)))

        # If the entity is a Truck or Drone, check its status.
        if isinstance(p_entity, (Truck, Drone)):
            vehicle = p_entity
            # If the vehicle is not idle OR not available...
            if (not vehicle.get_state_value_by_dim_name(vehicle.C_DIM_TRIP_STATE[0]) == vehicle.C_TRIP_STATE_IDLE or
                    not vehicle.get_state_value_by_dim_name(vehicle.C_DIM_AVAILABLE[0])):
                # ...invalidate all assignment actions for this vehicle.
                actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
                actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
                invalidation_idx.extend(actions_by_entity.intersection(actions_by_type))

        # If the entity is a MicroHub, check its availability.
        if isinstance(p_entity, MicroHub):
            micro_hub = p_entity
            # If the micro-hub is not available...
            if micro_hub.get_state_value_by_dim_name(MicroHub.C_DIM_AVAILABILITY[0]) != 0:
                # ...invalidate all actions associated with it.
                invalidation_idx.extend(
                    list(p_action_index.actions_involving_entity[(type(micro_hub), micro_hub.get_id())]))

        # Return the final list of invalidated action indices.
        return invalidation_idx, []

#----------------------------------------------------------------------------------------------------





class VehicleCapacityConstraint(Constraint):
    """
    Constraint that invalidates assigning orders to a vehicle if it has no
    remaining cargo capacity.
    """
    # Define the unique name for this constraint.
    C_NAME = "VehicleCapacityConstraint"
    # Specify that this constraint applies to "Truck" and "Drone" entities.
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    # List the order assignment actions affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        """
        Invalidates order assignment if the vehicle is at full capacity.
        """
        # Initialize an empty list for invalidated action indices.
        invalidation_idx = []
        # Assign the entity to a more specific variable name.
        vehicle = p_entity
        # Get the vehicle's total cargo capacity.
        vehicle_capacity = vehicle.get_cargo_capacity()
        # Get the current size of the cargo being carried.
        current_cargo_size = vehicle.get_current_cargo_size()

        # Check if there is space for at least one more item.
        if vehicle_capacity - current_cargo_size >= 1:
            # If there is capacity, return the empty list (no invalidations).
            return invalidation_idx, []
        else:
            # If the vehicle is full, invalidate assignment actions.
            # Get all actions of the affected assignment types.
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            # Get all actions involving this specific vehicle.
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            # The actions to invalidate are the intersection of the two sets.
            invalidation_idx = list(actions_by_entity.intersection(actions_by_type))
            # Return the list of invalidations.
            return invalidation_idx, []

#----------------------------------------------------------------------------------------------------





class TripWithinRangeConstraint(Constraint):
    """
    Constraint that ensures a trip is within a drone's available flight range
    before assigning an order.
    """
    # Define the unique name for this constraint.
    C_NAME = "TripWithinRangeConstraint"
    # Specify that this constraint only applies to "Drone" entities.
    C_ASSOCIATED_ENTITIES = ["Drone"]
    # List the drone order assignment action as the one affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_DRONE]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        """
        Invalidates drone order assignments for trips that exceed the drone's range.
        """
        # Assign the entity to a more specific variable name.
        drone = p_entity
        # Check if the entity is actually a Drone.
        if not isinstance(drone, Drone):
            # Raise an error if it's not, as this constraint is drone-specific.
            raise ValueError("The 'Trip Within Range Constraint' is only applicable to Drones.")

        # Get the remaining flight range of the drone.
        available_range = drone.get_remaining_range()
        # Get all orders from the global state.
        orders = drone.global_state.orders
        # Initialize a list to store orders that are out of range.
        orders_not_in_range = []

        # Iterate through all available orders.
        for order in orders.values():
            # Get the pickup and delivery locations for the order.
            loc_pick_up = order.get_pickup_node_id()
            loc_delivery = order.get_delivery_node_id()
            # Calculate the distance for this trip using the network map.
            distance = drone.global_state.network.calculate_distance(loc_pick_up, loc_delivery)
            # If the trip distance is greater than or equal to the available range...
            if distance >= available_range:
                # ...add the order to the out-of-range list.
                orders_not_in_range.append(order)

        # If no orders are out of range, there are no invalidations.
        if not orders_not_in_range:
            return [], []

        # Get all actions of the affected assignment type.
        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        # Get all actions involving this specific drone.
        actions_by_drone = p_action_index.actions_involving_entity[(drone.C_NAME, drone.get_id())]

        # Collect all actions related to the out-of-range orders.
        actions_by_orders = set()
        for order in orders_not_in_range:
            actions_by_orders.update(p_action_index.actions_involving_entity[(order.C_NAME, order.get_id())])

        # The invalidation set is the intersection of actions involving the drone, the correct action type, and the out-of-range orders.
        invalidation_set = actions_by_drone.intersection(actions_by_type, actions_by_orders)
        # Return the resulting list of invalidations.
        return list(invalidation_set), []

#----------------------------------------------------------------------------------------------------





class VehicleRoutingConstraint(Constraint):
    """
    Constraint that restricts vehicle movement actions to only valid pickup or
    delivery nodes based on its current orders.
    """
    # Define the unique name for this constraint.
    C_NAME = "Vehicle Routing Constraint"
    # Specify the entity types this constraint is associated with.
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "Node"]
    # List the vehicle movement actions affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.TRUCK_TO_NODE,
                          SimulationActions.DRONE_TO_NODE]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        """
        Invalidates movement to nodes that are not part of the current route plan.
        """
        # Check if the entity is a Truck or Drone.
        if isinstance(p_entity, (Truck, Drone)):
            # Assign the entity to a more specific variable name.
            vehicle = p_entity
            # Get a list of all pickup node IDs from the vehicle's assigned orders.
            pickup_nodes = [order.get_pickup_node_id() for order in vehicle.get_pickup_orders()]
            # Get a list of all delivery node IDs from the vehicle's assigned orders.
            delivery_nodes = [order.get_delivery_node_id() for order in vehicle.get_delivery_orders()]

            # Get all possible movement actions from the action index.
            all_possible_move_actions = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)

            # Create a set to store the indices of actions that should be valid (unmasked).
            idx_to_unmask = set()
            # For every valid pickup and delivery node...
            for node_id in pickup_nodes + delivery_nodes:
                # ...find all actions involving that node and add them to the unmask set.
                idx_to_unmask.update(p_action_index.actions_involving_entity[("Node", node_id)])

            # The invalid actions are all possible move actions minus the ones we want to unmask.
            invalidation_idx = list(all_possible_move_actions.difference(idx_to_unmask))
            # Return the list of invalidations.
            return invalidation_idx, []
        # If the entity is a Node, the logic is driven by the vehicle, so we do nothing here.
        elif isinstance(p_entity, Node):
            return [], []
        # For any other entity type, return an empty list.
        return [], []

#----------------------------------------------------------------------------------------------------





class ConsolidationConstraint(Constraint):
    """
    Constraint that ensures consolidation actions are only valid when a vehicle
    is idle or halted and has orders to process.
    """
    # Define the unique name for this constraint.
    C_NAME = "ConsolidationConstraint"
    # Specify that this constraint applies to "Truck" and "Drone" entities.
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    # List the consolidation actions affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE_FOR_TRUCK,
                          SimulationActions.CONSOLIDATE_FOR_DRONE]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        """
        Invalidates consolidation if the vehicle is en-route or has no orders.
        """
        # Initialize an empty list for invalidated action indices.
        invalidation_idx = []
        # Check if the entity is a Truck or Drone.
        if not isinstance(p_entity, (Truck, Drone)):
            # If not, return immediately as this constraint does not apply.
            return invalidation_idx, []

        # Assign the entity to a more specific variable name.
        vehicle = p_entity

        # Determine if the vehicle is in a state where consolidation is allowed.
        # It must be idle or halted, and have either pickup orders or cargo on board.
        is_ready_for_consolidation = (vehicle.get_state_value_by_dim_name(vehicle.C_DIM_TRIP_STATE[0])
                                      in [vehicle.C_TRIP_STATE_IDLE, vehicle.C_TRIP_STATE_HALT]
                                      and (len(vehicle.pickup_orders) > 0 or p_entity.get_current_cargo_size() > 0))

        # Get the vehicle's current node ID.
        vehicle_node_id = vehicle.current_node_id
        # Check if there are any assigned orders (pickup or delivery) at the vehicle's current location.
        assigned_orders_at_node = ([ordr for ordr in vehicle.get_pickup_orders() if ordr.get_pickup_node_id() == vehicle_node_id]
                                   + [ordr for ordr in vehicle.get_delivery_orders() if ordr.get_delivery_node_id() == vehicle_node_id])
        # If there are orders to be processed at the current node, consolidation should not happen.
        if len(assigned_orders_at_node):
            valid_relay_orders = True
            for ordr in assigned_orders_at_node:
                valid_relay_orders = ordr.check_order_precedence() and True
            is_ready_for_consolidation = not valid_relay_orders
        #     is_ready_for_consolidation = True

        # If the vehicle is not ready for consolidation...
        if not is_ready_for_consolidation:
            # ...find and invalidate the relevant consolidation actions.
            actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            actions_by_entity = p_action_index.actions_involving_entity[(vehicle.C_NAME, vehicle.get_id())]
            invalidation_idx = list(actions_by_type.intersection(actions_by_entity))

        # Return the final list of invalidations.
        return invalidation_idx, []

#----------------------------------------------------------------------------------------------------



# # This entire class is commented out, representing a potential or deprecated constraint.
class CollaborationPrecedenceConstraint(Constraint):
    C_NAME = "Collaboration Precedence Constraint"
    C_ACTIVE = False
    C_ASSOCIATED_ENTITIES = ["Order", "Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION, SimulationActions.LOAD_DRONE_ACTION]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        invalidation_idx = []
        # if not isinstance(p_entity, Order):
        #     raise TypeError("The \'Collaboration Precedence Constraint\' is only applicable to an Order entity.")

        # def check_order_precedence(p_order):
        #     predecessor_orders: [Order] = p_order.predecessor_orders
        #     if not len(predecessor_orders):
        #         return True
        #
        #     else:
        #         precedence_satisfied = True
        #         for ordr in predecessor_orders:
        #             if isinstance(ordr, Order):
        #                 if ordr.get_state_value_by_dim_name(ordr.C_DIM_DELIVERY_STATUS[0]) == ordr.C_STATUS_DELIVERED:
        #                     precedence_satisfied = True and precedence_satisfied
        #                 else:
        #                     precedence_satisfied = False
        #         return precedence_satisfied

        if isinstance(p_entity, Order):
            order = p_entity
            # predecessor_orders:[Order] = order.predecessor_orders
            # if not len(predecessor_orders):
            #     return invalidation_idx, []
            #
            # else:
            #     precedence_satisfied = True
            #     for ordr in predecessor_orders:
            #         if isinstance(ordr, Order):
            #             if ordr.get_state_value_by_dim_name(ordr.C_DIM_DELIVERY_STATUS[0]) == ordr.C_STATUS_DELIVERED:
            #                 precedence_satisfied = True and precedence_satisfied
            #             else:
            #                 precedence_satisfied = False
            # precedence_satisfied = check_order_precedence(order)
            precedence_satisfied = order.check_order_precedence()
            if not precedence_satisfied:
                actions_by_entity = p_action_index.actions_involving_entity["Order", p_entity.get_id()]
                actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
                invalidation_idx = list(actions_by_entity.intersection(actions_by_type))

        if isinstance(p_entity, Vehicle):
            if p_entity.current_node_id is not None:
                current_node_id = p_entity.current_node_id
                pickup_orders_at_nodes = [ordr for ordr in p_entity.get_pickup_orders() if ordr.get_pickup_node_id() == current_node_id]
                relevant_orders = [ordr for ordr in pickup_orders_at_nodes if not ordr.check_order_precedence()]
                actions_by_order = set()
                for rel_order in relevant_orders:
                    actions_by_order.update(p_action_index.actions_involving_entity["Order", rel_order.get_id()])
                if isinstance(p_entity, Truck):
                    actions_by_vehicle = p_action_index.actions_involving_entity["Truck", p_entity.get_id()]
                elif isinstance(p_entity, Drone):
                    actions_by_vehicle = p_action_index.actions_involving_entity["Drone", p_entity.get_id()]
                actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
                invalidation_idx = actions_by_type.intersection(actions_by_vehicle).intersection(actions_by_order)
#



        return invalidation_idx, []





class MicroHubAssignabilityConstraint(Constraint):
    # Define the unique name for this constraint.
    C_NAME = "MicroHubAssignabilityConstraint"
    # Specify that this constraint applies only to "Order" entities.
    C_ASSOCIATED_ENTITIES = ["Order"]
    # List the Micro-Hub assignment action as the one affected by this constraint.
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
        # Initialize an empty list for invalidated action indices.
        invalidation_idx = []
        # Check if the entity is an Order.
        if not isinstance(p_entity, Order):
            # Raise an error if the entity type is incorrect.
            raise TypeError("The \"Micro-Hub assignability constraint\" is only applicable to an Order entity.")
        try:
            # Get the ID of the Micro-Hub this order is assigned to.
            mh_node_id = p_entity.assigned_micro_hub_id
            # # This is a commented-out alternative way to get the ID.
            # mh_node_id = assignment.get_id()
            # Get the pseudo-orders associated with the main order (legs of the trip).
            pseudo_orders = p_entity.pseudo_orders
            # Iterate through each leg of the trip.
            for ps_order in pseudo_orders:
                # Get the delivery and pickup nodes for this leg.
                delivery_node_id = ps_order.get_delivery_node_id()
                pickup_node_id = ps_order.get_pickup_node_id()
                # Create a tuple representing the node pair (the trip).
                node_pair = (pickup_node_id, delivery_node_id)
                # Get all actions of the Micro-Hub assignment type.
                actions_by_type = p_action_index.get_actions_of_type([SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB])
                # Get all actions associated with this specific node pair.
                actions_by_node_pair = p_action_index.actions_involving_entity["Node Pair", node_pair]
                # Get all actions associated with the assigned Micro-Hub.
                actions_by_mh = p_action_index.actions_involving_entity["MicroHub", mh_node_id]
                # Find actions that match all three criteria and add them to the invalidation list.
                invalidation_idx.extend(list(actions_by_type.intersection(actions_by_node_pair).intersection(actions_by_mh)))

            # Return the final list of invalidations.
            return invalidation_idx, []
        except:
            # If any error occurs (e.g., the order is not assigned to a Micro-Hub), return the empty list.
            return invalidation_idx, []

#----------------------------------------------------------------------------------------------------




class CoOrdinatedDeliveryAssignmentConstraint(Constraint):
    C_NAME = "Co-ordinated Delivery Assignment Constraint"
    C_ASSOCIATED_ENTITIES = ["Order", "Truck", "Drone"]
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_TRUCK,
                          SimulationActions.ASSIGN_ORDER_TO_DRONE]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):

        def check_ass_precedence(p_order):
            ass_precedence_satisfied = True
            for pre_order in p_order.predecessor_orders:
                if pre_order.get_state_value_by_dim_name(pre_order.C_DIM_DELIVERY_STATUS[0]) not in [pre_order.C_STATUS_PLACED,
                                                                                                     pre_order.C_STATUS_ACCEPTED,
                                                                                                     pre_order.C_STATUS_FAILED]:
                    ass_precedence_satisfied = True and ass_precedence_satisfied
                else:
                    ass_precedence_satisfied = False

            return ass_precedence_satisfied

        invalidation_idx = []
        validation_idx = []
        if not (isinstance(p_entity, Order) or isinstance(p_entity, Truck), isinstance(p_entity, Drone)):
            raise TypeError("Wrong entity type for the constraint")
        actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
        pseudo_orders = [order for order in p_entity.global_state.get_all_entities_by_type("order").values()
                         if (isinstance(order, PseudoOrder) and len(order.predecessor_orders) and not check_ass_precedence(order))]
        # for ps_ordr in pseudo_orders:
        #     ass_precedence_satisfied = True
        #     for pre_order in ps_ordr.predecessor_orders:
        #         if pre_order.get_state_value_by_dim_name(pre_order.C_DIM_DELIVERY_STATUS[0]) not in [pre_order.C_STATUS_PLACED,
        #                                                                                              pre_order.C_STATUS_ACCEPTED,
        #                                                                                              pre_order.C_STATUS_FALIED]:
        #             ass_precedence_satisfied = True and ass_precedence_satisfied
        #         else:
        #             ass_precedence_satisfied = False

            # if ps_ordr.get_state_value_by_dim_name(ps_ordr.C_DIM_DELIVERY_STATUS[0]) == ps_ordr.C_STATUS_PLACED:
        for ps_ordr in pseudo_orders:
            actions_by_entity = p_action_index.actions_involving_entity["Node Pair", (ps_ordr.get_pickup_node_id(), ps_ordr.get_delivery_node_id())]
            invalidation_idx.extend(actions_by_entity.intersection(actions_by_type))

        return invalidation_idx, validation_idx


# This is another commented-out, likely deprecated or incomplete, constraint class.


# class CoOrdinationPrecedenceConstraint(Constraint):
#     C_NAME = "Co-ordination Precedence Constraint"
#     C_ASSOCIATED_ENTITIES = ["Order"]
#     C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
#                           SimulationActions.LOAD_DRONE_ACTION]
#
#
#     def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
#         invalidation_idx = []
#         if not isinstance(p_entity, Order):
#             raise TypeError("The \"Co-ordination Precedence Constraint\" is only applicable to Order entity.")
#
#         actions_by_type = p_action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
#         actions_by_entity = p_action_index.actions_involving_entity["Order", p_entity.get_id()]
#
# 
#
#
#         # invalidation_idx.extend(list(actions_by_type.intersection(actions_by_entity)))
#         return invalidation_idx, []



class OrderLoadConstraint(Constraint):
    C_NAME = "OrderLoadConstraint"
    C_ASSOCIATED_ENTITIES = ["Order", "Vehicle"]
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION]


    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
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
                    (p_entity.get_state_value_by_dim_name(p_entity.C_DIM_DELIVERY_STATUS[0]) in [p_entity.C_STATUS_ASSIGNED])):
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


class OrderUnloadConstraint(Constraint):
    C_NAME = "OrderUnloadConstraint"
    C_ASSOCIATED_ENTITIES = ["Order", "Vehicle"]
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_TRUCK_ACTION,
                          SimulationActions.UNLOAD_DRONE_ACTION]

    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> (List, List):
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
                    and (p_entity.get_state_value_by_dim_name(p_entity.C_DIM_DELIVERY_STATUS[0]) in [p_entity.C_STATUS_EN_ROUTE])):
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



# -------------------------------------------------------------------------------------------------





class ConstraintManager(EventManager):
    """
    Manages all constraints in the simulation. It maps constraints to entities,
    listens for entity state changes, and raises events to update action masks.
    """
    # Define the unique name for the manager.
    C_NAME = "Constraint Manager"
    # Define the event ID for when action masks need to be updated.
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
        # Initialize the EventManager parent class disabling its logging.
        EventManager.__init__(self, p_logging=False)
        # A set to hold all instantiated constraint objects.
        self.constraints = set()
        # A dictionary to map entity type names to a list of applicable constraints.
        self.entity_constraints = {}
        # Store the reverse action map.
        self.reverse_action_map = reverse_action_map
        # Store the action indexer.
        self.action_index = action_index
        # Call the method to build the constraint-entity mapping.
        self.setup_constraint_entity_map()

#----------------------------------------------------------------------------------------------------

    def setup_constraint_entity_map(self):
        """
        Automatically discovers all Constraint subclasses and maps them to the
        entities they are associated with.
        """
        # Reset the mapping dictionary.
        self.entity_constraints = {}
        # Iterate through all direct subclasses of the base Constraint class.
        for con in Constraint.__subclasses__():
            if con.C_ACTIVE:
                # Instantiate the constraint class.
                constr = con(p_reverse_action_map=self.reverse_action_map)
                # Add the instance to the set of all constraints.
                self.constraints.add(constr)
                # Iterate through the entity names associated with this constraint.
                for entity_name in con.C_ASSOCIATED_ENTITIES:
                    # If the entity name is already a key in the map...
                    if entity_name in self.entity_constraints:
                        # ...append the new constraint to the existing list.
                        self.entity_constraints[entity_name].append(constr)
                    else:
                        # ...otherwise, create a new list with this constraint.
                        self.entity_constraints[entity_name] = [constr]

#----------------------------------------------------------------------------------------------------

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
        # Check if the entity's name exists as a key in the mapping.
        if p_entity.C_NAME in self.entity_constraints:
            # If yes, return the list of associated constraints.
            return self.entity_constraints[p_entity.C_NAME]
        # Otherwise, return an empty list.
        return []

#----------------------------------------------------------------------------------------------------

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
        # Get the entity that raised the state change event.
        entity = p_event_object.get_raising_object()
        # Initialize a set to store indices of actions to be masked.
        idx_to_mask = set()
        validations = set()

        # Get all actions involving this specific entity instance.
        related_actions_by_entity = self.action_index.actions_involving_entity.get((entity.C_NAME, entity.get_id()),
                                                                                   set())

        # If the entity is an Order, also consider actions related to its node pair (trip).
        if isinstance(entity, Order):
            pickup_node = entity.get_pickup_node_id()
            delivery_node = entity.get_delivery_node_id()
            node_pair_actions = self.action_index.actions_involving_entity.get(
                ("Node Pair", (pickup_node, delivery_node)), set())
            related_actions_by_entity.update(node_pair_actions)

        # Get all constraints that apply to this entity type.
        constraints_to_check = self.get_constraints_by_entity(entity)
        # Initialize a set to store all actions affected by these constraints.
        related_actions_by_constraint = set()

        # Iterate through the applicable constraints.
        for constraint in constraints_to_check:
            # Log which constraint is being checked.
            self.log(Log.C_LOG_TYPE_I, f"Checking constraint: {constraint.C_NAME}")
            # Get the list of invalidated action indices from the constraint.
            invalidations, validations = constraint.get_invalidations(p_entity=entity, p_action_index=self.action_index)
            # If the constraint returned any invalidations, add them to the mask set.
            if invalidations is not None:
                idx_to_mask.update(set(invalidations))
            # Add the action types affected by this constraint to the related actions set.
            related_actions_by_constraint.update(self.action_index.get_actions_of_type(constraint.C_ACTIONS_AFFECTED))

        # Determine the total set of actions that are relevant to this entity and its constraints.
        related_actions = related_actions_by_constraint.intersection(related_actions_by_entity)
        # Determine which of these related actions should be unmasked (i.e., are not in the mask list).
        idx_to_unmask = related_actions.difference(idx_to_mask)
        idx_to_unmask.update(validations)

        # If there are any changes (actions to mask or unmask), raise an event.
        if len(idx_to_mask) > 0 or len(idx_to_unmask) > 0:
            self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                              p_event_object=Event(p_raising_object=self,
                                                   idx_to_mask=idx_to_mask,
                                                   idx_to_unmask=idx_to_unmask))

#----------------------------------------------------------------------------------------------------

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
        # Update the reverse action map for this manager and all its constraints.
        self.reverse_action_map = reverse_action_map
        # Iterate through all entity types in the global state.
        for entity_dict in global_state.get_all_entities():
            # Iterate through all instances of that entity type.
            for entity in entity_dict.values():
                # Initialize a set for actions to mask for this entity.
                idx_to_mask = set()
                validations = set()

                # Get all actions involving this specific entity instance.
                related_actions_by_entity = self.action_index.actions_involving_entity.get(
                    (entity.C_NAME, entity.get_id()), set())

                # Special handling for Order entities to include node pair actions.
                if isinstance(entity, Order):
                    pickup_node = entity.get_pickup_node_id()
                    delivery_node = entity.get_delivery_node_id()
                    node_pair_actions = self.action_index.actions_involving_entity.get(
                        ("Node Pair", (pickup_node, delivery_node)), set())
                    related_actions_by_entity.update(node_pair_actions)

                # Get all constraints applicable to this entity.
                constraints_to_check = self.get_constraints_by_entity(entity)
                # Initialize a set for actions affected by these constraints.
                related_actions_by_constraint = set()

                # Iterate through the constraints.
                for constraint in constraints_to_check:
                    # Ensure the constraint has the latest reverse action map.
                    constraint.reverse_action_map = self.reverse_action_map
                    # Log the check.
                    self.log(Log.C_LOG_TYPE_I, f"Checking constraint: {constraint.C_NAME}")
                    # Get invalidations from the constraint.
                    invalidations, validations = constraint.get_invalidations(p_entity=entity, p_action_index=self.action_index)
                    # If there are invalidations, add them to the mask set.
                    if invalidations:
                        idx_to_mask.update(set(invalidations))
                    # Update the set of related actions based on the constraint's affected types.
                    related_actions_by_constraint.update(
                        self.action_index.get_actions_of_type(constraint.C_ACTIONS_AFFECTED))

                # Determine the total set of relevant actions.
                related_actions = related_actions_by_constraint.intersection(related_actions_by_entity)
                # Determine which actions should be unmasked.
                idx_to_unmask = related_actions.difference(idx_to_mask)
                idx_to_unmask.update(validations)

                # If any changes are needed, raise the update event.
                if len(idx_to_mask) > 0 or len(idx_to_unmask) > 0:
                    self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                                      p_event_object=Event(p_raising_object=self,
                                                           idx_to_mask=idx_to_mask,
                                                           idx_to_unmask=idx_to_unmask))


# -------------------------------------------------------------------------------------------------
# -- StateActionMapper (Now Fully Self-Configuring)
# -------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------





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
        # Store the global state object.
        self.global_state = global_state
        # Store the mapping from action tuples to integer indices.
        self.action_map = action_map
        # Initialize the ActionIndex helper class with the global state and action map.
        self.action_index = ActionIndex(global_state, action_map)
        # Initialize a dictionary to store invalidation mappings (commented as "Rulebook").
        self._invalidation_map: Dict[Tuple, Set[int]] = {}
        # Initialize a set for actions that are permanently masked.
        self.permanent_masks = set()
        # Initialize the main action mask as a list of False values, one for each action.
        self.masks = [False for _ in range(len(action_map))]
        # Get the index for the NO_OPERATION action and mark it as permanently valid.
        self.permanent_valid_actions = list(self.action_index.get_actions_of_type([SimulationActions.NO_OPERATION]))
        # If a NO_OPERATION action exists, set its corresponding mask value to True.
        if self.permanent_valid_actions:
            self.masks[self.permanent_valid_actions[0]] = True

#----------------------------------------------------------------------------------------------------

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
        # Check if the masks list has been properly initialized.
        if not self.masks:
            raise ValueError("Error in instantiation of process masks.")
        # Iterate through the indices that need to be masked.
        for idx in idx_to_mask:
            # Set the corresponding mask value to False (invalid action).
            self.masks[idx] = False

        # Remove any permanently masked actions from the set of actions to unmask.
        idx_to_unmask = idx_to_unmask.difference(self.permanent_masks)
        # Iterate through the indices that need to be unmasked.
        for idx in idx_to_unmask:
            # Set the corresponding mask value to True (valid action).
            self.masks[idx] = True

#----------------------------------------------------------------------------------------------------

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
        # Get the object that raised the event.
        raising_object = p_event_object.get_raising_object()
        # Check if the event came from a ConstraintManager instance.
        if isinstance(raising_object, ConstraintManager):
            # Extract the sets of indices to mask and unmask from the event data.
            idx_to_mask = p_event_object.get_data().get('idx_to_mask', set())
            idx_to_unmask = p_event_object.get_data().get('idx_to_unmask', set())
            # Call the update_masks method with the received data.
            self.update_masks(idx_to_mask, idx_to_unmask)
        else:
            # If the event is from an unexpected source, do nothing.
            return

#----------------------------------------------------------------------------------------------------

    def generate_masks(self) -> List[bool]:
        """
        Returns the current action mask.

        Returns
        -------
        List[bool]
            The boolean mask where True indicates a valid action.
        """
        # Return the current state of the masks list.
        return self.masks

#----------------------------------------------------------------------------------------------------

    def update_action_space(self, action_map):
        # When the action space changes, re-initialize the masks list to the new size.
        self.masks = [False for _ in range(len(action_map))]


# -------------------------------------------------------------------------------------------------
# -- Validation Block (Expanded for Larger Instances)
# -------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # This block of code runs only when the script is executed directly.
    # It contains a validation test suite, which is currently commented out.
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
    # This line prints the associated entities for each discovered Constraint subclass. Useful for debugging the auto-configuration.
    print([c.C_ASSOCIATED_ENTITIES for c in Constraint.__subclasses__()])
    # # This line prints all subclasses of LogisticEntity, also for debugging purposes.
    # print([c for c in LogisticEntity.__subclasses__()])