from abc import ABC, abstractmethod
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
# -- Part 1: Direct Boolean Mask Architecture
# -------------------------------------------------------------------------------------------------

class Constraint(ABC, EventManager):
    """
    Abstract base class for direct boolean slice constraints.
    Directly mutates a boolean mask array in-place.
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

        self.associated_action_index = set()
        self.associated_action_indices = np.array([], dtype=np.int32)
        self.find_associated_actions()
        self.initiated = False

    def find_associated_actions(self):
        if self.C_ACTIONS_AFFECTED:
            acts = self.action_index.get_actions_of_type(self.C_ACTIONS_AFFECTED)
            self.associated_action_index = set(acts)
            self.associated_action_indices = np.array(list(acts), dtype=np.int32)
        else:
            self.associated_action_index = set()
            self.associated_action_indices = np.array([], dtype=np.int32)

    @abstractmethod
    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        """
        Directly modifies mask in-place: mask[restricted_indices] = False
        """
        raise NotImplementedError

    def clear_cache(self):
        pass


# -------------------------------------------------------------------------------------------------
# -- Part 2: Concrete Constraints (Direct In-Place Masking)
# -------------------------------------------------------------------------------------------------

class VehicleAvailableConstraint(Constraint):
    C_NAME = "VehicleAvailableConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK, SimulationActions.SELECT_DRONE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        dim_trip_state = Vehicle.C_DIM_TRIP_STATE[0]
        halt_states = (Vehicle.C_TRIP_STATE_EN_ROUTE, Vehicle.C_TRIP_STATE_HALT)

        for vehicle in p_entity.global_state.get_vehicles():
            state = vehicle.get_state_value_by_dim_name(dim_trip_state)
            if state in halt_states:
                acts = getattr(vehicle, 'associated_action_indices_arr', None)
                if acts is None:
                    acts = np.array(list(vehicle.associated_action_indexes), dtype=np.int32)
                    vehicle.associated_action_indices_arr = acts
                mask[acts] = False


class VehicleCapacityConstraint(Constraint):
    C_NAME = "VehicleCapacityConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub"]
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        active_resource = p_entity.global_state.active_resource
        if active_resource is None:
            return

        current_capacity = active_resource.get_remaining_capacity()
        dems = p_entity.global_state.get_pending_demands()
        node_pairs = p_entity.global_state.node_pairs

        for n_pair, demand in dems.items():
            if demand[0] > current_capacity:
                np_obj = node_pairs[n_pair]
                acts = getattr(np_obj, 'associated_action_indices_arr', None)
                if acts is None:
                    acts = np.array(list(np_obj.associated_action_indexes), dtype=np.int32)
                    np_obj.associated_action_indices_arr = acts
                mask[acts] = False


class ResourceAssignabilityConstraint(Constraint):
    C_NAME = "VehicleAssignabilityConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        active_resource = p_entity.global_state.active_resource
        if active_resource is None:
            mask[self.associated_action_indices] = False
        elif isinstance(active_resource, Truck):
            for order in active_resource.global_state.pseudo_orders.values():
                if order.leg == 2:
                    np_obj = order.node_pair
                    acts = getattr(np_obj, 'associated_action_indices_arr', None)
                    if acts is None:
                        acts = np.array(list(np_obj.associated_action_indexes), dtype=np.int32)
                        np_obj.associated_action_indices_arr = acts
                    mask[acts] = False


class ActiveResourceConstraint(Constraint):
    C_NAME = "ActiveResourceConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK,
                          SimulationActions.SELECT_DRONE,
                          SimulationActions.SELECT_MICROHUB]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        active_resource = p_entity.global_state.active_resource
        next_demands = p_entity.global_state.get_next_demands(False)

        if not next_demands:
            mask[self.associated_action_indices] = False
            return

        if isinstance(active_resource, (Truck, Drone, MicroHub)):
            mask[self.associated_action_indices] = False
        elif active_resource is None:
            active_next_demands = p_entity.global_state.get_next_demands()
            min_demand = min(active_next_demands) if active_next_demands else None

            for mh in p_entity.global_state.micro_hubs.values():
                if min_demand is None or mh.get_remaining_capacity() < min_demand:
                    acts = getattr(mh, 'associated_action_indices_arr', None)
                    if acts is None:
                        acts = np.array(list(mh.associated_action_indexes), dtype=np.int32)
                        mh.associated_action_indices_arr = acts
                    mask[acts] = False
        else:
            raise TypeError("Invalid resource type for the selected/active resource for the decision epoch.")


class OrderRequestAssignabilityConstraint(Constraint):
    C_NAME = "OrderRequestAssignability"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        orders = p_entity.global_state.orders_by_nodes
        node_pairs = p_entity.global_state.node_pairs

        for key, value in orders.items():
            if not value:
                np_obj = node_pairs[key]
                acts = getattr(np_obj, 'associated_action_indices_arr', None)
                if acts is None:
                    acts = np.array(list(np_obj.associated_action_indexes), dtype=np.int32)
                    np_obj.associated_action_indices_arr = acts
                mask[acts] = False


class VehicleLoadConstraint(Constraint):
    C_NAME = "VehicleLoadConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.LOAD_TRUCK_ACTION, SimulationActions.LOAD_DRONE_ACTION]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "Order"]
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        dim_trip_state = Vehicle.C_DIM_TRIP_STATE[0]
        trip_halt = Vehicle.C_TRIP_STATE_HALT

        for vehicle in p_entity.global_state.get_vehicles():
            current_node = vehicle.current_node_id
            acts = getattr(vehicle, 'associated_action_indices_arr', None)
            if acts is None:
                acts = np.array(list(vehicle.associated_action_indexes), dtype=np.int32)
                vehicle.associated_action_indices_arr = acts

            if current_node is not None and (vehicle.get_state_value_by_dim_name(dim_trip_state) in trip_halt):
                pickup_orders = [o for o in vehicle.get_pickup_orders() if o.get_pickup_node_id() == current_node]
                if not pickup_orders or not all(o.check_order_precedence() for o in pickup_orders):
                    mask[acts] = False
            else:
                mask[acts] = False


class VehicleUnloadConstraint(Constraint):
    C_NAME = "VehicleUnloadConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.UNLOAD_DRONE_ACTION, SimulationActions.UNLOAD_TRUCK_ACTION]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        for v in p_entity.global_state.get_vehicles():
            acts = getattr(v, 'associated_action_indices_arr', None)
            if acts is None:
                acts = np.array(list(v.associated_action_indexes), dtype=np.int32)
                v.associated_action_indices_arr = acts

            current_node = v.current_node_id
            if current_node is not None:
                has_cargo = any(o.get_delivery_node_id() == current_node for o in v.get_current_cargo())
                if not has_cargo:
                    mask[acts] = False
            else:
                mask[acts] = False


class ConsolidationConstraint(Constraint):
    C_NAME = "ConsolidationConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_ACTIVE = False
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        active_resource = p_entity.global_state.active_resource
        if active_resource is None:
            mask[self.associated_action_indices] = False
        elif isinstance(active_resource, Vehicle):
            if not active_resource.get_current_cargo() and not active_resource.get_pickup_orders():
                mask[self.associated_action_indices] = False


class FullCapacityConsolidationConstraint(Constraint):
    C_NAME = "ConsolidationConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub", "Node Pair"]
    C_ACTIVE = True
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        active_resource = p_entity.global_state.active_resource
        if active_resource is None:
            mask[self.associated_action_indices] = False
            return

        dems = p_entity.global_state.get_next_demands()
        min_demand = min(dems) if dems else None

        if isinstance(active_resource, Vehicle):
            if active_resource.get_current_cargo() or active_resource.get_pickup_orders():
                if min_demand is not None and active_resource.get_remaining_capacity() >= min_demand:
                    mask[self.associated_action_indices] = False
            else:
                mask[self.associated_action_indices] = False
        elif isinstance(active_resource, MicroHub):
            if min_demand is not None and active_resource.get_remaining_capacity() >= min_demand:
                mask[self.associated_action_indices] = False


class PseudoOrderAssignmentConstraint(Constraint):
    C_NAME = "PrecedenceConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Node Pair"]
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        mh = p_entity.global_state.micro_hubs
        _, mh_pickups = p_entity.global_state.get_microhub_orders()

        for n_pair, o in mh_pickups.items():
            if not o[0].check_assignment_precedence():
                mh_obj = mh[n_pair[0]]
                acts = getattr(mh_obj, 'associated_action_indices_arr', None)
                if acts is None:
                    acts = np.array(list(mh_obj.associated_action_indexes), dtype=np.int32)
                    mh_obj.associated_action_indices_arr = acts
                mask[acts] = False


class MicroHubFirstConstraint(Constraint):
    C_NAME = "MicroHubFirstConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.SELECT_TRUCK, SimulationActions.SELECT_DRONE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone", "MicroHub"]
    C_ACTIVE = True
    C_GLOBAL_CONSTRAINT = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        if p_entity.global_state.micro_hub_phase:
            mask[self.associated_action_indices] = False


class MicroHubConsolidation(Constraint):
    C_NAME = "MicroHubConsolidation"
    C_ACTIONS_AFFECTED = [SimulationActions.CONSOLIDATE]
    C_ASSOCIATED_ENTITIES = ["Drone", "MicroHub"]
    C_ACTIVE = True

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        if isinstance(p_entity.global_state.active_resource, MicroHub):
            dems = p_entity.global_state.get_next_demands()
            if dems and p_entity.get_remaining_capacity() >= min(dems):
                mask[self.associated_action_indices] = False


class DeadlockConstraint(Constraint):
    C_NAME = "DeadLockConstraint"
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Truck", "Drone"]
    C_ACTIVE = False

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        pass


class TwoEchelonConstraint(Constraint):
    C_NAME = "TwoEchelonConstraint"
    C_GLOBAL_CONSTRAINT = True
    C_ACTIONS_AFFECTED = [SimulationActions.ASSIGN_ORDER_TO_RESOURCE]
    C_ASSOCIATED_ENTITIES = ["Drone", "MicroHub", "Truck", "Order"]

    def apply_constraint(self, p_entity: LogisticEntity, mask: np.ndarray):
        if isinstance(p_entity.global_state.active_resource, MicroHub):
            mh_routes = p_entity.global_state.get_microhub_routes()
            route_acts = [next(iter(r.associated_action_indexes)) for r in mh_routes.values() if r.associated_action_indexes]
            if route_acts:
                mask[route_acts] = False


# -------------------------------------------------------------------------------------------------
# -- Part 3: Constraint Manager (Direct Mask Dispatch)
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
        self.masks = np.ones(self.num_actions, dtype=bool)
        self.global_state = None

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

    def handle_entity_state_change(self, p_event_id, p_event_object):
        self._update_counter += 1
        entity = p_event_object.get_raising_object()
        if self.global_state is None and hasattr(entity, 'global_state'):
            self.global_state = entity.global_state

    def update_constraints(self, global_state, reverse_action_map):
        self.global_state = global_state
        self.reverse_action_map = reverse_action_map
        self.num_actions = len(self.reverse_action_map)

        for constraint in self.constraints:
            constraint.clear_cache()
            constraint.reverse_action_map = self.reverse_action_map
            constraint.find_associated_actions()

        self.recompute_masks()

    def update_action_index(self, action_map, action_map_old, reverse_action_map_old):
        self.num_actions = len(action_map)
        for constraint in self.constraints:
            constraint.find_associated_actions()
        self.recompute_masks()

    def recompute_masks(self):
        """
        Direct in-place mask reconstruction.
        """
        self.masks = np.ones(self.num_actions, dtype=bool)

        if self.global_state is None:
            return

        entity_context = LogisticEntity(p_id="context", p_name="Context")
        entity_context.global_state = self.global_state

        for constraint in self.constraints:
            constraint.apply_constraint(entity_context, self.masks)

    def get_masks(self) -> List[int]:
        self.recompute_masks()
        return self.masks.astype(np.int8).tolist()


# -------------------------------------------------------------------------------------------------
# -- Part 4: State Action Mapper (Vectorized Direct Masking)
# -------------------------------------------------------------------------------------------------

class StateActionMapper:
    """
    Maps the system state to a valid action mask using vectorized NumPy arrays.
    """

    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int], reverse_action_map, custom_log=False):
        self.global_state = global_state
        self.action_map = action_map
        self.reverse_action_map = reverse_action_map
        self.action_index = ActionIndex(global_state, action_map)
        self.custom_log = custom_log

        num_actions = len(action_map)
        # Vectorized reference counters and boolean masks
        self.mask_counters = np.zeros(num_actions, dtype=np.int32)
        self.masks = np.ones(num_actions, dtype=bool)

        # Permanent valid actions stored as a fast boolean lookup mask
        perm_acts = self.action_index.get_actions_of_type([SimulationActions.NO_OPERATION])
        self.permanent_valid_actions = set(perm_acts)
        self.is_permanent_mask = np.zeros(num_actions, dtype=bool)
        if perm_acts:
            self.is_permanent_mask[list(perm_acts)] = True

    def update_counters_and_masks(self, indices_to_block: Iterable[int], indices_to_unblock: Iterable[int]):
        """
        Updates counters and flips boolean masks using vectorized NumPy array slicing.
        """
        masked = 0
        unmasked = 0

        # --- BLOCK LOGIC ---
        if indices_to_block:
            block_arr = np.fromiter(indices_to_block, dtype=np.int32, count=len(list(indices_to_block)) if not isinstance(indices_to_block, (list, set, np.ndarray)) else len(indices_to_block))
            if len(block_arr) > 0:
                valid_block = block_arr[~self.is_permanent_mask[block_arr]]
                if len(valid_block) > 0:
                    self.mask_counters[valid_block] += 1
                    blocked_mask = (self.mask_counters[valid_block] >= 1)
                    target_indices = valid_block[blocked_mask]
                    self.masks[target_indices] = False
                    masked = len(target_indices)

        # --- UNBLOCK LOGIC ---
        if indices_to_unblock:
            unblock_arr = np.fromiter(indices_to_unblock, dtype=np.int32, count=len(list(indices_to_unblock)) if not isinstance(indices_to_unblock, (list, set, np.ndarray)) else len(indices_to_unblock))
            if len(unblock_arr) > 0:
                valid_unblock = unblock_arr[~self.is_permanent_mask[unblock_arr]]
                if len(valid_unblock) > 0:
                    self.mask_counters[valid_unblock] = np.maximum(self.mask_counters[valid_unblock] - 1, 0)
                    unblocked_mask = (self.mask_counters[valid_unblock] == 0)
                    target_indices = valid_unblock[unblocked_mask]
                    self.masks[target_indices] = True
                    unmasked = len(target_indices)

        if self.custom_log:
            print(f"Masks updated: Masked --> {masked} , Unmasked --> {unmasked}")
        return 0

    def handle_new_masks_event(self, p_event_id, p_event_object):
        """
        Handles mask update events from ConstraintManager.
        """
        raising_object = p_event_object.get_raising_object()
        if isinstance(raising_object, ConstraintManager):
            data = p_event_object.get_data()
            if data:
                to_block = data.get('to_block', [])
                to_unblock = data.get('to_unblock', [])
                self.update_counters_and_masks(to_block, to_unblock)

    def generate_masks(self) -> List[bool]:
        return self.masks.tolist()

    def reset_masks(self):
        self.mask_counters.fill(0)
        self.masks.fill(True)

    def update_action_space(self, action_map, old_action_map):
        old_counters = self.mask_counters.copy()
        num_actions = len(action_map)

        self.mask_counters = np.zeros(num_actions, dtype=np.int32)
        for a, idx in old_action_map.items():
            if a in action_map:
                self.mask_counters[action_map[a]] = old_counters[idx]

        self.action_map = action_map
        self.masks = (self.mask_counters == 0)

        perm_acts = self.action_index.get_actions_of_type([SimulationActions.NO_OPERATION])
        self.permanent_valid_actions = set(perm_acts)
        self.is_permanent_mask = np.zeros(num_actions, dtype=bool)
        if perm_acts:
            self.is_permanent_mask[list(perm_acts)] = True

        if self.custom_log:
            print("Masks updated after micro-hub assignment")

    def update_masks(self):
        self.masks = (self.mask_counters == 0)