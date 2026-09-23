from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple, Set, List, Iterable
import numpy as np

from ddls_src.actions.base import ActionIndex
from ddls_src.actions.conditional_actions import SimulationActions
from ddls_src.entities.base import LogisticEntity
from ddls_src.entities.micro_hub import MicroHub
from ddls_src.entities.node import Node
from ddls_src.entities.order import PseudoOrder, Order, NodePair
from ddls_src.entities.vehicles.base import Vehicle
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.vehicles.truck import Truck
from mlpro.bf.events import Event, EventManager


# -------------------------------------------------------------------------------------------------
# -- Part 1: Pluggable Constraint Architecture (Optimized)
# -------------------------------------------------------------------------------------------------

class Constraint(ABC, EventManager):
    """
    Abstract base class for pluggable constraints.
    Tracks state transitions with integer reference counting in mask_counters.
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
        self.action_index = p_action_index
        self.custom_log = custom_log

        if not self.C_GLOBAL_CONSTRAINT:
            self._entity_invalidation_map = defaultdict(set)
        else:
            self._entity_invalidation_map = set()

        self.associated_action_index = set()
        self.find_associated_actions()
        self.initiated = False

    def find_associated_actions(self):
        if self.C_ACTIONS_AFFECTED:
            self.associated_action_index = set(self.action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED))
        else:
            self.associated_action_index = set()

    def evaluate_impact(self, p_entity, p_action_index: ActionIndex, mask_counters: np.ndarray) -> Tuple[Set[int], Set[int]]:
        """
        Calculates delta masks and applies them directly into mask_counters using vectorized operations.
        """
        if not self.C_GLOBAL_CONSTRAINT:
            current_actions_to_unblock, current_actions_to_block = self._get_restricted_actions(p_entity, p_action_index)
            current_block_set = set(current_actions_to_block) if current_actions_to_block else set()

            entity_id = p_entity.get_id()
            previous_block_set = self._entity_invalidation_map[entity_id]

            to_block = current_block_set.difference(previous_block_set)
            to_unblock = previous_block_set.difference(current_block_set)

            if to_block:
                idx_block = np.fromiter(to_block, dtype=np.int32, count=len(to_block))
                mask_counters[idx_block] += 1
            if to_unblock:
                idx_unblock = np.fromiter(to_unblock, dtype=np.int32, count=len(to_unblock))
                mask_counters[idx_unblock] = np.maximum(mask_counters[idx_unblock] - 1, 0)

            self._entity_invalidation_map[entity_id] = current_block_set
        else:
            to_unblock_set, to_block_set = self.evaluate_impact_global(
                p_entity=p_entity, p_action_index=p_action_index, deck=None
            )
            current_block_set = to_block_set if isinstance(to_block_set, set) else set(to_block_set)
            previous_block_set = self._entity_invalidation_map

            to_block = current_block_set.difference(previous_block_set)
            to_unblock = previous_block_set.difference(current_block_set)

            if to_block:
                idx_block = np.fromiter(to_block, dtype=np.int32, count=len(to_block))
                mask_counters[idx_block] += 1
            if to_unblock:
                idx_unblock = np.fromiter(to_unblock, dtype=np.int32, count=len(to_unblock))
                mask_counters[idx_unblock] = np.maximum(mask_counters[idx_unblock] - 1, 0)

            self._entity_invalidation_map = current_block_set

        return to_block, to_unblock

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        raise NotImplementedError

    def evaluate_impact_global(self, p_entity, p_action_index, deck=None) -> Tuple[Iterable, Iterable]:
        raise NotImplementedError

    def clear_cache(self):
        if not self.C_GLOBAL_CONSTRAINT:
            self._entity_invalidation_map.clear()
        else:
            self._entity_invalidation_map = set()


# -------------------------------------------------------------------------------------------------
# -- Part 2: Concrete Constraints
# -------------------------------------------------------------------------------------------------

class VehicleAvailableConstraint(Constraint):
    C_NAME = "VehicleAvailableConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK, SimulationActions.SELECT_DRONE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        raw_block = set()
        raw_unblock = set()

        dim_trip_state = Vehicle.C_DIM_TRIP_STATE[0]
        halt_states = (Vehicle.C_TRIP_STATE_EN_ROUTE, Vehicle.C_TRIP_STATE_HALT)

        for vehicle in p_entity.global_state.get_vehicles():
            state = vehicle.get_state_value_by_dim_name(dim_trip_state)
            if state in halt_states:
                raw_block.update(vehicle.associated_action_indexes)
            else:
                raw_unblock.update(vehicle.associated_action_indexes)

        actions_to_block = raw_block.intersection(self.associated_action_index)
        actions_to_unblock = raw_unblock.intersection(self.associated_action_index)
        return actions_to_unblock, actions_to_block


class VehicleCapacityConstraint(Constraint):
    C_NAME = "VehicleCapacityConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: Vehicle, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        active_resource = p_entity.global_state.active_resource
        if active_resource is None:
            return self.associated_action_index, set()
        elif active_resource != p_entity:
            return set(), set()

        raw_block = set()
        raw_unblock = set()
        current_capacity = active_resource.get_remaining_capacity()
        dems = p_entity.global_state.get_pending_demands()
        node_pairs = p_entity.global_state.node_pairs

        for n_pair, demand in dems.items():
            if demand[0] <= current_capacity:
                raw_unblock.update(node_pairs[n_pair].associated_action_indexes)
            else:
                raw_block.update(node_pairs[n_pair].associated_action_indexes)

        actions_to_unblock = raw_unblock.intersection(self.associated_action_index)
        actions_to_block = raw_block.intersection(self.associated_action_index)
        return actions_to_unblock, actions_to_block


class ResourceAssignabilityConstraint(Constraint):
    C_NAME = "VehicleAssignabilityConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        relevant_actions = self.associated_action_index
        active_resource = p_entity.global_state.active_resource

        if active_resource is None:
            return set(), relevant_actions
        elif isinstance(active_resource, Truck):
            raw_masks = set()
            for order in active_resource.global_state.pseudo_orders.values():
                if order.leg == 2:
                    raw_masks.update(order.node_pair.associated_action_indexes)
            masks = raw_masks.intersection(relevant_actions)
            return relevant_actions.difference(masks), masks

        return relevant_actions, set()


class ActiveResourceConstraint(Constraint):
    C_NAME = "ActiveResourceConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK,
                          SimulationActions.SELECT_DRONE,
                          SimulationActions.SELECT_MICROHUB]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        active_resource = p_entity.global_state.active_resource
        relevant_actions = self.associated_action_index
        next_demands = p_entity.global_state.get_next_demands(False)

        if not next_demands:
            return set(), relevant_actions

        if isinstance(active_resource, (Truck, Drone, MicroHub)):
            return set(), relevant_actions
        elif active_resource is None:
            raw_mask = set()
            active_next_demands = p_entity.global_state.get_next_demands()
            min_demand = min(active_next_demands) if active_next_demands else None

            for mh in p_entity.global_state.micro_hubs.values():
                if min_demand is None or mh.get_remaining_capacity() < min_demand:
                    raw_mask.update(mh.associated_action_indexes)

            mask = raw_mask.intersection(relevant_actions)
            unmask = relevant_actions.difference(mask)
            return unmask, mask
        else:
            raise TypeError("Invalid resource type for the selected/active resource for the decision epoch.")


class OrderRequestAssignabilityConstraint(Constraint):
    C_NAME = "OrderRequestAssignability"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        relevant_actions = self.associated_action_index
        raw_unblock = set()
        raw_block = set()

        orders = p_entity.global_state.orders_by_nodes
        node_pairs = p_entity.global_state.node_pairs

        for key, value in orders.items():
            pair_acts = node_pairs[key].associated_action_indexes
            if value:
                raw_unblock.update(pair_acts)
            else:
                raw_block.update(pair_acts)

        actions_to_unblock = raw_unblock.intersection(relevant_actions)
        actions_to_block = raw_block.intersection(relevant_actions)
        return actions_to_unblock, actions_to_block


class VehicleLoadConstraint(Constraint):
    C_NAME = "VehicleLoadConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION, SimulationActions.LOAD_DRONE_ACTION]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        raw_block = set()
        raw_unblock = set()

        dim_trip_state = Vehicle.C_DIM_TRIP_STATE[0]
        trip_halt = Vehicle.C_TRIP_STATE_HALT

        for vehicle in p_entity.global_state.get_vehicles():
            current_node = vehicle.current_node_id
            if current_node is not None and (vehicle.get_state_value_by_dim_name(dim_trip_state) in trip_halt):
                pickup_orders = [o for o in vehicle.get_pickup_orders() if o.get_pickup_node_id() == current_node]
                if pickup_orders and all(o.check_order_precedence() for o in pickup_orders):
                    raw_unblock.update(vehicle.associated_action_indexes)
                else:
                    raw_block.update(vehicle.associated_action_indexes)
            else:
                raw_block.update(vehicle.associated_action_indexes)

        actions_to_unblock = raw_unblock.intersection(self.associated_action_index)
        actions_to_block = raw_block.intersection(self.associated_action_index)
        return actions_to_unblock, actions_to_block


class VehicleUnloadConstraint(Constraint):
    C_NAME = "VehicleUnloadConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_DRONE_ACTION, SimulationActions.UNLOAD_TRUCK_ACTION]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        raw_block = set()
        raw_unblock = set()

        for v in p_entity.global_state.get_vehicles():
            current_node = v.current_node_id
            if current_node is not None:
                has_cargo = any(o.get_delivery_node_id() == current_node for o in v.get_current_cargo())
                if has_cargo:
                    raw_unblock.update(v.associated_action_indexes)
                else:
                    raw_block.update(v.associated_action_indexes)
            else:
                raw_block.update(v.associated_action_indexes)

        actions_to_unblock = raw_unblock.intersection(self.associated_action_index)
        actions_to_block = raw_block.intersection(self.associated_action_index)
        return actions_to_unblock, actions_to_block


class ConsolidationConstraint(Constraint):
    C_NAME = "ConsolidationConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_ACTIVE = False
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        active_resource = p_entity.global_state.active_resource
        if active_resource is None:
            return set(), self.associated_action_index
        if isinstance(active_resource, Vehicle):
            if active_resource.get_current_cargo() or active_resource.get_pickup_orders():
                return self.associated_action_index, set()
            return set(), self.associated_action_index
        elif isinstance(active_resource, MicroHub):
            return self.associated_action_index, set()
        return set(), self.associated_action_index


class FullCapacityConsolidationConstraint(Constraint):
    C_NAME = "ConsolidationConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_ACTIVE = True
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity: LogisticEntity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        active_resource = p_entity.global_state.active_resource
        if active_resource is None:
            return set(), self.associated_action_index

        dems = p_entity.global_state.get_next_demands()
        min_demand = min(dems) if dems else None

        if isinstance(active_resource, Vehicle):
            if active_resource.get_current_cargo() or active_resource.get_pickup_orders():
                if min_demand is None or active_resource.get_remaining_capacity() < min_demand:
                    return self.associated_action_index, set()
                return set(), self.associated_action_index
            return set(), self.associated_action_index

        elif isinstance(active_resource, MicroHub):
            if min_demand is not None and active_resource.get_remaining_capacity() >= min_demand:
                return set(), self.associated_action_index
            return self.associated_action_index, set()

        return set(), self.associated_action_index


class PseudoOrderAssignmentConstraint(Constraint):
    C_NAME = "PrecedenceConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Node Pair"]
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[Set[int], Set[int]]:
        mh = p_entity.global_state.micro_hubs
        _, mh_pickups = p_entity.global_state.get_microhub_orders()
        raw_block = set()
        raw_unblock = set()

        for n_pair, o in mh_pickups.items():
            if not o[0].check_assignment_precedence():
                raw_block.update(mh[n_pair[0]].associated_action_indexes)
            else:
                raw_unblock.update(mh[n_pair[0]].associated_action_indexes)

        actions_to_block = raw_block.intersection(self.associated_action_index)
        actions_to_unblock = raw_unblock.intersection(self.associated_action_index)
        return actions_to_unblock, actions_to_block


class MicroHubFirstConstraint(Constraint):
    C_NAME = "MicroHubFirstConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK, SimulationActions.SELECT_DRONE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub"]
    C_ACTIVE = True
    C_GLOBAL_CONSTRAINT = True

    def evaluate_impact_global(self, p_entity, p_action_index, deck=None) -> Tuple[Set[int], Set[int]]:
        if p_entity.global_state.micro_hub_phase:
            return set(), self.associated_action_index
        return self.associated_action_index, set()


class MicroHubConsolidation(Constraint):
    C_NAME = "MicroHubConsolidation"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Drone", "MicroHub"]
    C_ACTIVE = True

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        if isinstance(p_entity.global_state.active_resource, MicroHub):
            dems = p_entity.global_state.get_next_demands()
            if dems and p_entity.get_remaining_capacity() >= min(dems):
                return [], list(self.associated_action_index)
            return list(self.associated_action_index), []
        return [], []


class DeadlockConstraint(Constraint):
    C_NAME = "DeadLockConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIVE = False

    def _get_restricted_actions(self, p_entity, p_action_index: ActionIndex, **p_kwargs) -> Tuple[List, List]:
        return [], []


class TwoEchelonConstraint(Constraint):
    C_NAME = "TwoEchelonConstraint"
    C_GLOBAL_CONSTRAINT = True
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Drone", "MicroHub", "Truck", "Order"]

    def evaluate_impact_global(self, p_entity, p_action_index, deck=None) -> Tuple[Set[int], Set[int]]:
        mh_routes = p_entity.global_state.get_microhub_routes()
        relevant_actions = {next(iter(route.associated_action_indexes)) for route in mh_routes.values() if route.associated_action_indexes}
        if isinstance(p_entity.global_state.active_resource, MicroHub):
            return set(), relevant_actions
        return relevant_actions, set()


# -------------------------------------------------------------------------------------------------
# -- Part 3: Constraint Manager (Vectorized Deck & Fast Mask Lookups)
# -------------------------------------------------------------------------------------------------

class ConstraintManager(EventManager):
    C_NAME = "Constraint Manager"
    C_EVENT_MASK_UPDATED = "New Masks Necessary"

    def __init__(self, action_index: ActionIndex, reverse_action_map, custom_log=False):
        EventManager.__init__(self, p_logging=False)
        self.custom_log = custom_log
        self._update_counter = 0
        self.constraints = set()
        self.entity_constraints = {}
        self.reverse_action_map = reverse_action_map
        self.action_index = action_index
        self.global_constraints = []
        self.setup_constraint_entity_map()

        self.num_actions = len(self.reverse_action_map)
        self.mask_counters = np.zeros(self.num_actions, dtype=np.int32)
        self.masks = np.ones(self.num_actions, dtype=np.int8)

    def setup_constraint_entity_map(self):
        self.entity_constraints = {}
        for con in Constraint.__subclasses__():
            if con.C_GLOBAL_CONSTRAINT:
                self.global_constraints.append(con(p_reverse_action_map=self.reverse_action_map, p_action_index=self.action_index))
            if con.C_ACTIVE and con is not Constraint:
                constr = con(p_reverse_action_map=self.reverse_action_map, p_action_index=self.action_index)
                self.constraints.add(constr)
                for entity_name in con.C_ASSOCIATED_ENTITIES:
                    self.entity_constraints.setdefault(entity_name, []).append(constr)

    def get_constraints_by_entity(self, p_entity):
        return self.entity_constraints.get(p_entity.C_NAME, [])

    def handle_entity_state_change(self, p_event_id, p_event_object):
        self._update_counter += 1
        entity = p_event_object.get_raising_object()
        constraints_to_check = self.get_constraints_by_entity(entity)

        total_to_block = []
        total_to_unblock = []

        for constraint in constraints_to_check:
            to_block, to_unblock = constraint.evaluate_impact(
                p_entity=entity, p_action_index=self.action_index, mask_counters=self.mask_counters
            )
            total_to_block.extend(to_block)
            total_to_unblock.extend(to_unblock)

    def update_constraints(self, global_state, reverse_action_map):
        self.reverse_action_map = reverse_action_map
        self.num_actions = len(self.reverse_action_map)
        self.mask_counters = np.zeros(self.num_actions, dtype=np.int32)

        for constraint in self.constraints:
            constraint.clear_cache()
            constraint.reverse_action_map = self.reverse_action_map

        total_to_block = []
        for entity_dict in global_state.get_all_entities():
            for entity in entity_dict.values():
                for constraint in self.get_constraints_by_entity(entity):
                    to_block, _ = constraint.evaluate_impact(
                        p_entity=entity, p_action_index=self.action_index, mask_counters=self.mask_counters
                    )
                    total_to_block.extend(to_block)

        if total_to_block:
            event_data = {"to_block": total_to_block, "to_unblock": []}
            self._raise_event(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                              p_event_object=Event(p_raising_object=self, p_event_data=event_data))

    def update_action_index(self, action_map, action_map_old, reverse_action_map_old):
        for constraint in self.constraints:
            constraint.associated_action_index = set(
                self.action_index.get_actions_of_type(constraint.C_ACTIONS_AFFECTED)
            )

        self.update_entity_invalidation_maps(action_map, reverse_action_map_old)
        old_counters = self.mask_counters.copy()
        self.num_actions = len(action_map)
        self.mask_counters = np.zeros(self.num_actions, dtype=np.int32)

        for action, old_idx in action_map_old.items():
            if action in action_map:
                self.mask_counters[action_map[action]] = old_counters[old_idx]

        self.update_masks()

    def update_entity_invalidation_maps(self, action_map, reverse_action_map_old):
        for constraint in self.constraints:
            if not constraint.C_GLOBAL_CONSTRAINT:
                for entity, action_set in constraint._entity_invalidation_map.items():
                    constraint._entity_invalidation_map[entity] = {
                        action_map[reverse_action_map_old[act]] for act in action_set if reverse_action_map_old[act] in action_map
                    }
            else:
                constraint._entity_invalidation_map = {
                    action_map[reverse_action_map_old[act]] for act in constraint._entity_invalidation_map if reverse_action_map_old[act] in action_map
                }

    def update_masks(self):
        self.masks = (self.mask_counters == 0).astype(np.int8)

    def get_masks(self) -> List[int]:
        return (self.mask_counters == 0).astype(np.int8).tolist()