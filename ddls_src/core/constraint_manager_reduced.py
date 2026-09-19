from abc import ABC, abstractmethod
from collections import defaultdict
from ddls_src.actions.base import SimulationActions, ActionIndex
from ddls_src.actions.conditional_actions import SimulationActions
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
    C_GLOBAL_CONSTRAINT = False
    C_NAME = None
    C_EVENT_CONSTRAINT_UPDATE = "ConstraintUpdate"

    def __init__(self, p_reverse_action_map, p_action_index, custom_log=False):
        EventManager.__init__(self, p_logging=False)
        self.reverse_action_map = p_reverse_action_map

        # State tracking
        if not self.C_GLOBAL_CONSTRAINT:
            self._entity_invalidation_map = defaultdict(set)
        else:
            self._entity_invalidation_map = set()
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

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        """
        Pure logic method: Determines which actions should be RESTRICTED (masked) based on CURRENT state.
        Returns: (List of indices to MASK, List of indices to UNMASK)
        """
        raise NotImplementedError

    def evaluate_impact(self, p_entity, p_action_index: ActionIndex, deck) -> Tuple[Set[int], Set[int]]:
        """
        Calculates the Delta (Impact) of this constraint.
        """
        # 1. Get current "Desired Blocks"
        if not self.C_GLOBAL_CONSTRAINT:
            current_actions_to_unblock, current_actions_to_block = self._get_restricted_actions(p_entity, p_action_index)
            current_block_set = set(current_actions_to_block) if current_actions_to_block else set()

            # 2. Get "Previous Blocks"
            entity_id = p_entity.get_id()
            previous_block_set = self._entity_invalidation_map[entity_id]

            # 3. Calculate Deltas
            to_block = (current_block_set.difference(previous_block_set))
            to_unblock = (previous_block_set.difference(current_block_set))
            self.update_constraint_deck(to_block, to_unblock, deck, p_entity)
            # 4. Update Memory
            self._entity_invalidation_map[entity_id] = current_block_set

            self.evaluation_history.append(
                f"{p_entity.C_NAME} - {p_entity.get_id()} --> to block: {current_actions_to_block}, to unblock: {current_actions_to_unblock}")

        else:
            masks = self.evaluate_impact_global(p_entity=p_entity, p_action_index=p_action_index,
                                                               deck=deck)
            to_unblock, to_block = masks
            # current_block_set = set(current_actions_to_block) if current_actions_to_block else set()
            # to_block, to_unblock = self.examine_global_cache(current_block_set)
            # self.update_constraint_deck(to_block, to_unblock, deck, p_entity)
            self.update_constraint_deck_global(to_block, to_unblock, deck)

        return to_block, to_unblock


    def evaluate_impact_global(self, p_entity, p_action_index, deck):

        raise NotImplementedError

    def update_constraint_deck(self, to_block, to_unblock, deck, p_entity):
        for action in to_block:
            deck[action].add(f"{self.C_NAME} - {p_entity.C_NAME} {p_entity.get_id()}")
        for action in to_unblock:
            deck[action].remove(f"{self.C_NAME} - {p_entity.C_NAME} {p_entity.get_id()}")

    def update_constraint_deck_global(self, to_block, to_unblock, deck):
        for action in to_block:
            deck[action].add(f"{self.C_NAME}")
        for action in to_unblock:
            if f"{self.C_NAME}" in deck[action]:
                deck[action].remove(f"{self.C_NAME}")

    def clear_cache(self):
        self._entity_invalidation_map.clear()

    # --- [LEGACY METHODS] ---
    def get_invalidations(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        """Legacy method for full invalidation calculation."""
        return [], []

    def update_operability(self, p_entity: LogisticEntity, **p_kwargs):
        pass

    def examine_global_cache(self, current_block_set):

        raise NotImplementedError


# -------------------------------------------------------------------------------------------------
# -- Part 2: Concrete Constraints
# -------------------------------------------------------------------------------------------------

class VehicleAvailableConstraint(Constraint):
    C_NAME = "VehicleAvailableConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK,
                          SimulationActions.SELECT_DRONE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity : LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:

        actions_to_block = set()
        actions_to_unblock = set()
        for vehicle in p_entity.global_state.get_vehicles():
            if vehicle.get_state_value_by_dim_name(Vehicle.C_DIM_TRIP_STATE[0]) in [Vehicle.C_TRIP_STATE_EN_ROUTE,
                                                                                    Vehicle.C_TRIP_STATE_HALT]:
                actions_to_block.update(vehicle.associated_action_indexes.intersection(self.associated_action_index))
            else:
                actions_to_unblock.update(vehicle.associated_action_indexes.intersection(self.associated_action_index))

        return list(actions_to_unblock), list(actions_to_block)


class VehicleCapacityConstraint(Constraint):
    C_NAME = "VehicleCapacityConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: Vehicle, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:

        active_resource = p_entity.global_state.active_resource
        if active_resource is None:
            return list(self.associated_action_index), []
        elif not active_resource == p_entity:
            return [], []
        actions_to_block = set()
        actions_to_unblock = set()
        current_capacity = active_resource.get_remaining_capacity()
        dems = p_entity.global_state.get_pending_demands()
        for n_pair, demand in dems.items():
            if demand[0] <= current_capacity:
                actions_to_unblock.update(p_entity.global_state.node_pairs[n_pair].associated_action_indexes.intersection(self.associated_action_index))

            else:
                actions_to_block.update(p_entity.global_state.node_pairs[n_pair].associated_action_indexes.intersection(self.associated_action_index))

        return list(actions_to_unblock), list(actions_to_block)


class ResourceAssignabilityConstraint(Constraint):
    C_NAME = "VehicleAssignabilityConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        relevant_actions = self.associated_action_index
        active_resource = p_entity.global_state.active_resource
        masks = set()
        if active_resource is None:
            return [], list(relevant_actions)
        elif isinstance(active_resource, Truck):
            for order in active_resource.global_state.pseudo_orders.values():
                if order.leg == 2:
                    masks.update(order.node_pair.associated_action_indexes.intersection(self.associated_action_index))
            return list(relevant_actions.difference(masks)), list(masks)


        return list(relevant_actions), []


class ActiveResourceConstraint(Constraint):
    C_NAME = "ActiveResourceConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK,
                          SimulationActions.SELECT_DRONE,
                          SimulationActions.SELECT_MICROHUB]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity:LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        active_resource = p_entity.global_state.active_resource
        relevant_actions = self.associated_action_index
        mask = set()
        if not len(p_entity.global_state.get_next_demands(False)):
            return [], list(relevant_actions)
        if (isinstance(active_resource, Truck)
                or isinstance(active_resource, Drone)
                or isinstance(active_resource, MicroHub)):

            return list(), list(relevant_actions)

        elif active_resource is None:

            for mh in p_entity.global_state.micro_hubs.values():
                if not len(p_entity.global_state.get_next_demands()):
                    mask.update(mh.associated_action_indexes.intersection(relevant_actions))
                elif mh.get_remaining_capacity() < min(p_entity.global_state.get_next_demands()):
                    mask.update(mh.associated_action_indexes.intersection(relevant_actions))

            unmask = relevant_actions.difference(mask)

            return list(unmask), list(mask)

        else:
            raise TypeError("Invalid resource type for the selected/active resource for the decision epoch.")


class OrderRequestAssignabilityConstraint(Constraint):
    C_NAME = "OrderRequestAssignability"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:

        relevant_actions = self.associated_action_index
        actions_to_unblock = set()
        actions_to_block = set()

        orders = p_entity.global_state.orders_by_nodes

        for key, value in orders.items():
            if len(value):
                n_pair = p_entity.global_state.node_pairs[key]
                actions_to_unblock.update(relevant_actions.intersection(n_pair.associated_action_indexes))
            else:
                n_pair = p_entity.global_state.node_pairs[key]
                actions_to_block.update(relevant_actions.intersection(n_pair.associated_action_indexes))

        return list(actions_to_unblock), list(actions_to_block)


# Redundant with VehicleAssignabilityConstraint, can be renamed there to ResourceAssignabilityConstraint
# TO BE WORKED XXX

# class MicroHubAssignabilityConstraint(Constraint):
#     C_NAME = "MicroHubAssignabilityConstraint"
#     C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
#     C_ASSOCIATED_ENTITIES = ["MicroHub"]
#     C_GLOBAL_CONSTRAINT = True
#
#     def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
#         return [],[]


class VehicleLoadConstraint(Constraint):
    C_NAME = "VehicleLoadConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION,
                          SimulationActions.LOAD_DRONE_ACTION]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        actions_to_block = set()
        actions_to_unblock = set()
        relevant_actions = self.associated_action_index
        for vehicle in p_entity.global_state.get_vehicles():
            current_node = vehicle.current_node_id
            if current_node is not None and (vehicle.get_state_value_by_dim_name(vehicle.C_DIM_TRIP_STATE[0]) in vehicle.C_TRIP_STATE_HALT):
                pickup_orders = [o for o in vehicle.get_pickup_orders() if o.get_pickup_node_id() == current_node]
                if len(pickup_orders):
                    precedence = [o.check_order_precedence() for o in pickup_orders]
                    if False in precedence:
                        actions_to_block.update(relevant_actions.intersection(vehicle.associated_action_indexes))
                    else:
                        actions_to_unblock.update(relevant_actions.intersection(vehicle.associated_action_indexes))
                else:
                    actions_to_block.update(relevant_actions.intersection(vehicle.associated_action_indexes))
            else:
                actions_to_block.update(relevant_actions.intersection(vehicle.associated_action_indexes))
        return list(actions_to_unblock), list(actions_to_block)


class VehicleUnloadConstraint(Constraint):
    C_NAME = "VehicleUnloadConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_DRONE_ACTION,
                          SimulationActions.UNLOAD_TRUCK_ACTION]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity,
                               p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:

        relevant_actions = self.associated_action_index
        actions_to_block = set()
        actions_to_unblock = set()
        for v in p_entity.global_state.get_vehicles():
            current_node = v.current_node_id
            if current_node is not None:
                cargo_nodes = [o.get_delivery_node_id() for o in v.get_current_cargo()]
                if current_node in cargo_nodes:
                    actions_to_unblock.update(relevant_actions.intersection(v.associated_action_indexes))

                else:
                    actions_to_block.update(relevant_actions.intersection(v.associated_action_indexes))
            else:
                actions_to_block.update(relevant_actions.intersection(v.associated_action_indexes))

        return list(actions_to_unblock), list(actions_to_block)


class ConsolidationConstraint(Constraint):
    C_NAME = "ConsolidationConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_ACTIVE = False
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:

        active_resource = p_entity.global_state.active_resource

        if active_resource is None:
            return [], list(self.associated_action_index)
        if isinstance(active_resource, Vehicle):
            if len(active_resource.get_current_cargo()) or len(active_resource.get_pickup_orders()):
                return list(self.associated_action_index), []
            else:
                return [], list(self.associated_action_index)
        elif isinstance(active_resource, MicroHub):
                return list(self.associated_action_index), []

class FullCapacityConsolidationConstraint(Constraint):
    C_NAME = "ConsolidationConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_ACTIVE = True
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:

        active_resource = p_entity.global_state.active_resource

        if active_resource is None:
            return [], list(self.associated_action_index)
        if isinstance(active_resource, Vehicle):
            if len(active_resource.get_current_cargo()) or len(active_resource.get_pickup_orders()):
                av_cap = active_resource.get_remaining_capacity()
                dems = active_resource.global_state.get_next_demands()
                if not len(dems):
                    return list(self.associated_action_index), []
                if av_cap>=min(active_resource.global_state.get_next_demands()):
                    return [], list(self.associated_action_index)
                else:
                    return list(self.associated_action_index), []
            else:
                return [], list(self.associated_action_index)
        elif isinstance(active_resource, MicroHub):
            if not len(p_entity.global_state.get_next_demands()):
                return list(self.associated_action_index), []
            if active_resource.get_remaining_capacity() >= min(p_entity.global_state.get_next_demands()):
                return [], list(self.associated_action_index)
            else:
                return list(self.associated_action_index), []




class PseudoOrderAssignmentConstraint(Constraint):

    C_NAME = "PrecedenceConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Node Pair"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:

        relevant_actions = self.associated_action_index
        mh = p_entity.global_state.micro_hubs
        mh_deliveries, mh_pickups = p_entity.global_state.get_microhub_orders()
        actions_to_block = set()
        actions_to_unblock = set()
        for n_pair, o in mh_pickups.items():
            assignment_precedence = o[0].check_assignment_precedence()
            if not assignment_precedence:
                actions_to_block.update(relevant_actions.intersection(mh[n_pair[0]].associated_action_indexes))
            else:
                actions_to_unblock.update(relevant_actions.intersection(mh[n_pair[0]].associated_action_indexes))
        return list(actions_to_unblock), list(actions_to_block)


class MicroHubFirstConstraint(Constraint):

    C_NAME = "MicroHubFirstConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK,
                          SimulationActions.SELECT_DRONE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub"]
    C_ACTIVE = True
    C_GLOBAL_CONSTRAINT = True

    # def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
    #
    #     relevant_actions = list(self.associated_action_index)
    #     mh_phase = p_entity.global_state.micro_hub_phase
    #     if mh_phase:
    #         return [], relevant_actions
    #     else:
    #         return relevant_actions, []
    #     pass

    def evaluate_impact_global(self, p_entity, p_action_index, deck):
        relevant_actions = list(self.associated_action_index)
        mh_phase = p_entity.global_state.micro_hub_phase
        if mh_phase:
            return [], relevant_actions
        else:
            return relevant_actions, []
        pass


class MicroHubConsolidation(Constraint):
    C_NAME = "MicroHubConsolidation"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Drone", "MicroHub"]
    C_ACTIVE = True

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        if isinstance(p_entity.global_state.active_resource, MicroHub):
            relevant_actions = self.associated_action_index
            available_cap = p_entity.get_remaining_capacity()
            if not len(p_entity.global_state.get_next_demands()):
                return list(relevant_actions), []
            if available_cap >= min(p_entity.global_state.get_next_demands()):
                return [], list(relevant_actions)
            else:
                return list(relevant_actions), []
        return [], []


class DeadlockConstraint(Constraint):

    C_NAME = "DeadLockConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIVE = False

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        pass


class TwoEchelonConstraint(Constraint):
    C_NAME = "TwoEchelonConstraint"
    C_GLOBAL_CONSTRAINT = True
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Drone", "MicroHub", "Truck", "Order"]

    def evaluate_impact_global(self, p_entity, p_action_index, deck):
        mask = set()
        mh_routes = p_entity.global_state.get_microhub_routes()
        relevant_actions = [list(route.associated_action_indexes)[0] for route in mh_routes.values()]
        active_resource = p_entity.global_state.active_resource
        if isinstance(active_resource, MicroHub):
            return [], relevant_actions
        else:
            return relevant_actions, []


class ConstraintManager(EventManager):
    """
    Manages all constraints in the simulation.
    """
    C_NAME = "Constraint Manager"
    C_EVENT_MASK_UPDATED = "New Masks Necessary"

    def __init__(self, action_index: ActionIndex, reverse_action_map, custom_log=False):
        EventManager.__init__(self, p_logging=False)
        self.custom_log = custom_log
        self.masks = []
        self.action_map = None
        self._update_counter = 0
        self.constraints = set()
        self.entity_constraints = {}
        self.reverse_action_map = reverse_action_map
        self.action_index = action_index
        self.global_constraints = []
        self.setup_constraint_entity_map()
        self.constraint_deck = {key: set() for key in self.reverse_action_map}
        self.masks = [0 for i in range(len(self.reverse_action_map))]
        if self.custom_log:
            print("Constraints Setup")

    def setup_constraint_entity_map(self):
        self.entity_constraints = {}
        for con in Constraint.__subclasses__():
            if con.C_GLOBAL_CONSTRAINT:
                self.global_constraints.append(
                    con(p_reverse_action_map=self.reverse_action_map, p_action_index=self.action_index))
            # Skip abstract or base classes if they somehow get in
            if con.C_ACTIVE and con is not Constraint:
                constr = con(p_reverse_action_map=self.reverse_action_map, p_action_index=self.action_index)
                self.constraints.add(constr)
                for entity_name in con.C_ASSOCIATED_ENTITIES:
                    if entity_name in self.entity_constraints:
                        self.entity_constraints[entity_name].append(constr)
                    else:
                        self.entity_constraints[entity_name] = [constr]
        if self.custom_log:
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

        # for gl_constraint in self.global_constraints:
        #     to_block, to_unblock = gl_constraint.evaluate_impact(p_entity=entity, p_action_index=self.action_index, deck=self.constraint_deck)
        #
        #     # DEBUG 3: specific constraint output
        #     if to_block or to_unblock:
        #         if self.custom_log:
        #             print(f"   -> {gl_constraint.C_NAME}: Block={len(to_block)}, Unblock={len(to_unblock)}")
        #
        #     total_to_block.extend(to_block)
        #     total_to_unblock.extend(to_unblock)

        # if len(total_to_block) > 0 or len(total_to_unblock) > 0:
        #     event_data = {
        #         "to_block": total_to_block,
        #         "to_unblock": total_to_unblock
        #     }
        #     # if self.custom_log:
        #     print(f"[ConstraintManager] Raising update event! (+{len(total_to_block)} / -{len(total_to_unblock)})")
        # self._raise_event(p_event_id = ConstraintManager.C_EVENT_MASK_UPDATED,
        #                   p_event_object = Event(p_raising_object=self,
        #                                          to_block = total_to_block,
        #                                          to_unblock = total_to_unblock))
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
            if not constraint.C_GLOBAL_CONSTRAINT:
                for entity, action_set in constraint._entity_invalidation_map.items():
                    new_action_set = set()
                    for old_action in action_set:
                        if not reverse_action_map_old[old_action] in action_map:
                            print("Something is wrong. I am tired.")
                            raise TypeError
                        new_action_set.add(action_map[reverse_action_map_old[old_action]])
                    constraint._entity_invalidation_map[entity] = new_action_set
            else:
                new_action_set = set()
                for idx, old_action in enumerate(constraint._entity_invalidation_map):
                    # new_action_set = set()
                    # for old_action in action_set:
                    #     if not reverse_action_map_old[old_action] in action_map:
                    #         print("Something is wrong. I am tired.")
                    #         raise TypeError
                    #     new_action_set.add(action_map[reverse_action_map_old[old_action]])
                    new_action_set.add(action_map[reverse_action_map_old[old_action]])
                constraint._entity_invalidation_map = new_action_set

    def update_masks(self):
        self.masks = [0 for i in range(len(self.constraint_deck.keys()))]
        for key, value in self.constraint_deck.items():
            if len(value):
                self.masks[key] = 0
            else:
                self.masks[key] = 1

    # def get_masks(self):
    #     for key, value in self.constraint_deck.items():
    #         if len(value):
    #             self.masks[key] = 0
    #         else:
    #             self.masks[key] = 1
    #
    #     return self.masks

    # def get_masks(self):
    #     for key, value in self.constraint_deck.items():
    #         self.masks[key] = 0 if value else 1
    #
    #     return self.masks

    def get_masks(self):
        self.masks = [0 if value else 1 for key, value in self.constraint_deck.items()]
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
        if self.custom_log:
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