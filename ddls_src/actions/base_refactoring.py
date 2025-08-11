from enum import Enum, auto
from typing import List, Dict, Any, Callable, Tuple, Type, Set
from abc import ABC, abstractmethod
from collections import defaultdict
from pprint import pprint

# MLPro Imports (for validation block)
from mlpro.bf.systems import System


# -------------------------------------------------------------------------------------------------
# -- Part 1: Pluggable Constraint Architecture
# -------------------------------------------------------------------------------------------------

class Constraint(ABC):
    """
    Abstract base class for a pluggable constraint rule.
    """

    def __init__(self):
        self.actions_to_check = set()
        for action in SimulationAction:
            if self.__class__ in action.constraints:
                self.actions_to_check.add(action)

    @abstractmethod
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        pass


# --- Concrete Constraint Implementations ---

class OrderAssignableConstraint(Constraint):
    """Invalidates actions if an order is not in an assignable state."""

    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        invalidation_map = defaultdict(set)
        action_indices_to_check = action_index.get_actions_of_type(list(self.actions_to_check))

        for order in global_state.orders.values():
            order_id = order.get_id()
            order_actions = action_index.actions_involving_entity[('Order', order_id)]
            relevant_actions = action_indices_to_check.intersection(order_actions)
            if not relevant_actions: continue

            for status in ["assigned", "in_transit", "delivered", "cancelled"]:
                action_tuple = ('Order', order_id, 'status', status)
                invalidation_map[action_tuple].update(relevant_actions)
        return invalidation_map


class VehicleAvailableConstraint(Constraint):
    """Invalidates actions if a vehicle is not available."""

    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        invalidation_map = defaultdict(set)
        action_indices_to_check = action_index.get_actions_of_type(list(self.actions_to_check))

        from ddls_src.entities.vehicles.truck import Truck
        from ddls_src.entities.vehicles.drone import Drone
        all_vehicles = list(global_state.trucks.values()) + list(global_state.drones.values())

        for vehicle in all_vehicles:
            entity_type = "Truck" if isinstance(vehicle, Truck) else "Drone"
            vehicle_id = vehicle.get_id()
            vehicle_actions = action_index.actions_involving_entity[(entity_type, vehicle_id)]
            relevant_actions = action_indices_to_check.intersection(vehicle_actions)
            if not relevant_actions: continue

            for status in ["en_route", "maintenance", "broken_down"]:
                state_tuple = (entity_type, vehicle_id, 'status', status)
                invalidation_map[state_tuple].update(relevant_actions)
        return invalidation_map


class VehicleCapacityConstraint(Constraint):
    """Invalidates actions if a vehicle is at full capacity."""

    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        invalidation_map = defaultdict(set)
        action_indices_to_check = action_index.get_actions_of_type(list(self.actions_to_check))

        from ddls_src.entities.vehicles.truck import Truck
        from ddls_src.entities.vehicles.drone import Drone
        all_vehicles = list(global_state.trucks.values()) + list(global_state.drones.values())

        for vehicle in all_vehicles:
            if len(vehicle.cargo_manifest) >= vehicle.max_payload_capacity:
                entity_type = "Truck" if isinstance(vehicle, Truck) else "Drone"
                vehicle_id = vehicle.get_id()
                vehicle_actions = action_index.actions_involving_entity[(entity_type, vehicle_id)]
                relevant_actions = action_indices_to_check.intersection(vehicle_actions)
                if not relevant_actions: continue

                state_tuple = (entity_type, vehicle_id, 'capacity', 'full')
                invalidation_map[state_tuple].update(relevant_actions)
        return invalidation_map


class HubIsActiveConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        invalidation_map = defaultdict(set)
        action_indices_to_check = action_index.get_actions_of_type(list(self.actions_to_check))

        for hub in global_state.micro_hubs.values():
            hub_id = hub.get_id()
            hub_actions = action_index.actions_involving_entity[('MicroHub', hub_id)]
            relevant_actions = action_indices_to_check.intersection(hub_actions)
            if not relevant_actions: continue

            state_tuple = ('MicroHub', hub_id, 'status', 'inactive')
            invalidation_map[state_tuple].update(relevant_actions)
        return invalidation_map


class VehicleAtNodeConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        return {}


class OrderAtNodeConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        return {}


class OrderInCargoConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        return {}


class DroneBatteryConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        return {}


# -------------------------------------------------------------------------------------------------
# -- Part 2: ActionIndex (The "Database")
# -------------------------------------------------------------------------------------------------

class ActionIndex:
    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        self.actions_by_type: Dict[SimulationAction, Set[int]] = defaultdict(set)
        self.actions_involving_entity: Dict[Tuple, Set[int]] = defaultdict(set)
        self._build_indexes(global_state, action_map)

    def _build_indexes(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        for action_tuple, action_index in action_map.items():
            action_type = action_tuple[0]
            self.actions_by_type[action_type].add(action_index)
            if not action_type.params: continue
            for i, param_def in enumerate(action_type.params):
                entity_type = param_def['type']
                entity_id = action_tuple[i + 1]
                self.actions_involving_entity[(entity_type, entity_id)].add(action_index)

    def get_actions_of_type(self, action_types: List['SimulationAction']) -> Set[int]:
        ids = set()
        for action_type in action_types:
            ids.update(self.actions_by_type[action_type])
        return ids


# -------------------------------------------------------------------------------------------------
# -- Part 3: SimulationAction Enum (The Central Source of Truth)
# -------------------------------------------------------------------------------------------------

class SimulationAction(Enum):

    def __init__(self,
                 id_val: int,
                 params: List[Dict[str, Any]],
                 constraints: List[Type[Constraint]],
                 is_automatic: bool,
                 handler: str):
        self._id_val = id_val
        self.params = params
        self.constraints = constraints
        self.is_automatic = is_automatic
        self.handler = handler

    @property
    def id(self):
        return self.value[0]

    ACCEPT_ORDER = (
    auto(), [{'name': 'order_id', 'type': 'Order'}], [OrderAssignableConstraint], False, "SupplyChainManager")
    PRIORITIZE_ORDER = (auto(), [{'name': 'order_id', 'type': 'Order'}, {'name': 'priority', 'type': 'int'}], [], False,
                        "SupplyChainManager")
    CANCEL_ORDER = (auto(), [{'name': 'order_id', 'type': 'Order'}], [], False, "SupplyChainManager")
    FLAG_FOR_RE_DELIVERY = (auto(), [{'name': 'order_id', 'type': 'Order'}], [], False, "SupplyChainManager")
    ASSIGN_ORDER_TO_TRUCK = (auto(), [{'name': 'order_id', 'type': 'Order'}, {'name': 'truck_id', 'type': 'Truck'}],
                             [OrderAssignableConstraint, VehicleAvailableConstraint, VehicleCapacityConstraint], False,
                             "SupplyChainManager")
    ASSIGN_ORDER_TO_DRONE = (auto(), [{'name': 'order_id', 'type': 'Order'}, {'name': 'drone_id', 'type': 'Drone'}],
                             [OrderAssignableConstraint, VehicleAvailableConstraint, VehicleCapacityConstraint], False,
                             "SupplyChainManager")
    ASSIGN_ORDER_TO_MICRO_HUB = (
    auto(), [{'name': 'order_id', 'type': 'Order'}, {'name': 'micro_hub_id', 'type': 'MicroHub'}],
    [OrderAssignableConstraint, HubIsActiveConstraint], False, "SupplyChainManager")
    REASSIGN_ORDER = (auto(), [{'name': 'order_id', 'type': 'Order'}, {'name': 'vehicle_id', 'type': 'Vehicle'}],
                      [OrderAssignableConstraint, VehicleAvailableConstraint, VehicleCapacityConstraint], False,
                      "SupplyChainManager")
    CONSOLIDATE_FOR_TRUCK = (
    auto(), [{'name': 'truck_id', 'type': 'Truck'}], [VehicleAvailableConstraint, VehicleAtNodeConstraint], False,
    "SupplyChainManager")
    CONSOLIDATE_FOR_DRONE = (
    auto(), [{'name': 'drone_id', 'type': 'Drone'}], [VehicleAvailableConstraint, VehicleAtNodeConstraint], False,
    "SupplyChainManager")
    LOAD_TRUCK_ACTION = (auto(), [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'order_id', 'type': 'Order'}],
                         [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderAtNodeConstraint,
                          VehicleCapacityConstraint], True, "ResourceManager")
    UNLOAD_TRUCK_ACTION = (auto(), [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'order_id', 'type': 'Order'}],
                           [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderInCargoConstraint], True,
                           "ResourceManager")
    DRONE_LOAD_ACTION = (auto(), [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'order_id', 'type': 'Order'}],
                         [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderAtNodeConstraint,
                          VehicleCapacityConstraint], True, "ResourceManager")
    DRONE_UNLOAD_ACTION = (auto(), [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'order_id', 'type': 'Order'}],
                           [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderInCargoConstraint], True,
                           "ResourceManager")
    DRONE_CHARGE_ACTION = (
    auto(), [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'duration', 'type': 'int'}], [VehicleAtNodeConstraint],
    True, "ResourceManager")
    ACTIVATE_MICRO_HUB = (auto(), [{'name': 'micro_hub_id', 'type': 'MicroHub'}], [], False, "ResourceManager")
    DEACTIVATE_MICRO_HUB = (auto(), [{'name': 'micro_hub_id', 'type': 'MicroHub'}], [], False, "ResourceManager")
    ADD_TO_CHARGING_QUEUE = (
    auto(), [{'name': 'micro_hub_id', 'type': 'MicroHub'}, {'name': 'drone_id', 'type': 'Drone'}],
    [HubIsActiveConstraint, VehicleAtNodeConstraint], True, "ResourceManager")
    FLAG_VEHICLE_FOR_MAINTENANCE = (auto(), [{'name': 'vehicle_id', 'type': 'Vehicle'}], [], False, "ResourceManager")
    FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB = (
    auto(), [{'name': 'micro_hub_id', 'type': 'MicroHub'}, {'name': 'service_type', 'type': 'str'}], [], False,
    "ResourceManager")
    TRUCK_TO_NODE = (auto(), [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'destination_node_id', 'type': 'Node'}],
                     [VehicleAvailableConstraint], True, "NetworkManager")
    RE_ROUTE_TRUCK_TO_NODE = (
    auto(), [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'destination_node_id', 'type': 'Node'}], [], False,
    "NetworkManager")
    LAUNCH_DRONE = (auto(), [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'order_id', 'type': 'Order'}],
                    [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderInCargoConstraint,
                     DroneBatteryConstraint], True, "NetworkManager")
    DRONE_TO_NODE = (auto(), [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'destination_node_id', 'type': 'Node'}],
                     [VehicleAvailableConstraint], True, "NetworkManager")
    RE_ROUTE_DRONE_TO_NODE = (
    auto(), [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'destination_node_id', 'type': 'Node'}], [], False,
    "NetworkManager")
    DRONE_LANDING_ACTION = (auto(), [{'name': 'drone_id', 'type': 'Drone'}], [], True, "NetworkManager")
    DRONE_TO_CHARGING_STATION = (
    auto(), [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'station_id', 'type': 'Node'}],
    [VehicleAvailableConstraint], True, "NetworkManager")
    NO_OPERATION = (auto(), [], [], False, None)


# -------------------------------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    class MockEntity(System):
        def __init__(self, p_id, status, cargo_count=0, capacity=1):
            super().__init__(p_id=p_id)
            self.status = status
            self.cargo_manifest = [0] * cargo_count
            self.max_payload_capacity = capacity

        @staticmethod
        def setup_spaces(): return None, None


    class MockGlobalState:
        def __init__(self):
            self.orders = {
                0: MockEntity(p_id=0, status='pending'),
                1: MockEntity(p_id=1, status='delivered')
            }
            self.trucks = {
                101: MockEntity(p_id=101, status='idle', cargo_count=0, capacity=2),
                102: MockEntity(p_id=102, status='en_route'),
                103: MockEntity(p_id=103, status='idle', cargo_count=2, capacity=2)
            }
            self.drones = {}
            self.micro_hubs = {}


    mock_gs = MockGlobalState()
    mock_action_map = {
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 101): 0,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, 101): 1,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 102): 2,
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 103): 3,
    }
    mock_ai = ActionIndex(mock_gs, mock_action_map)

    print("--- Validating Self-Configuring Constraints ---")

    # Test OrderAssignableConstraint
    print("\n[A] Testing OrderAssignableConstraint...")
    order_constraint = OrderAssignableConstraint()
    order_invalidations = order_constraint.get_invalidations(mock_gs, mock_ai)
    print("  - Generated Invalidation Map:")
    pprint(order_invalidations)
    # FIX: Corrected assertion to check for the expected invalidation
    assert 1 in order_invalidations[('Order', 1, 'status', 'delivered')]
    print("  - PASSED: Correctly invalidates actions for delivered orders.")

    # Test VehicleAvailableConstraint
    print("\n[B] Testing VehicleAvailableConstraint...")
    from ddls_src.entities.vehicles.truck import Truck as RealTruck

    mock_gs.trucks[101].__class__ = RealTruck
    mock_gs.trucks[102].__class__ = RealTruck
    mock_gs.trucks[103].__class__ = RealTruck
    vehicle_constraint = VehicleAvailableConstraint()
    vehicle_invalidations = vehicle_constraint.get_invalidations(mock_gs, mock_ai)
    print("  - Generated Invalidation Map:")
    pprint(vehicle_invalidations)
    assert 2 in vehicle_invalidations[('Truck', 102, 'status', 'en_route')]
    print("  - PASSED: Correctly invalidates actions for unavailable vehicles.")

    # Test VehicleCapacityConstraint
    print("\n[C] Testing VehicleCapacityConstraint...")
    capacity_constraint = VehicleCapacityConstraint()
    capacity_invalidations = capacity_constraint.get_invalidations(mock_gs, mock_ai)
    print("  - Generated Invalidation Map:")
    pprint(capacity_invalidations)
    assert 3 in capacity_invalidations[('Truck', 103, 'capacity', 'full')]
    print("  - PASSED: Correctly invalidates actions for full vehicles.")

    print("\n--- Validation Complete ---")
