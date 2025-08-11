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

    @abstractmethod
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        pass


# --- Concrete Constraint Implementations ---

class OrderAssignableConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        # Logic to be implemented
        return {}


class VehicleAvailableConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        # Logic to be implemented
        return {}


class VehicleCapacityConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        # Logic to be implemented
        return {}


class HubIsActiveConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex') -> Dict[Tuple, Set[int]]:
        return {}


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
        self.actions_by_type: Dict['ActionType', Set[int]] = defaultdict(set)
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

    def get_actions_of_type(self, action_types: List['ActionType']) -> Set[int]:
        ids = set()
        for action_type in action_types:
            ids.update(self.actions_by_type[action_type])
        return ids


# -------------------------------------------------------------------------------------------------
# -- Part 3: Class-based Action Blueprint (The Central Source of Truth)
# -------------------------------------------------------------------------------------------------

class ActionType:
    """
    A simple data class to hold the blueprint for a single action type.
    """
    _id_counter = 0
    _id_map = {}

    def __init__(self, name: str, params: List, constraints: List, is_automatic: bool, handler: str):
        self.id = ActionType._id_counter
        self.name = name
        self.params = params
        self.constraints = constraints
        self.is_automatic = is_automatic
        self.handler = handler

        ActionType._id_map[self.id] = self
        ActionType._id_counter += 1

    @classmethod
    def get_by_id(cls, action_id: int):
        return cls._id_map.get(action_id)


class SimulationActions:
    """
    A namespace class that holds all action blueprints as class attributes.
    This serves as our single source of truth for all action definitions.
    """
    ACCEPT_ORDER = ActionType("ACCEPT_ORDER", [{'name': 'order_id', 'type': 'Order'}], [OrderAssignableConstraint],
                              False, "SupplyChainManager")
    PRIORITIZE_ORDER = ActionType("PRIORITIZE_ORDER",
                                  [{'name': 'order_id', 'type': 'Order'}, {'name': 'priority', 'type': 'int'}], [],
                                  False, "SupplyChainManager")
    CANCEL_ORDER = ActionType("CANCEL_ORDER", [{'name': 'order_id', 'type': 'Order'}], [], False, "SupplyChainManager")
    FLAG_FOR_RE_DELIVERY = ActionType("FLAG_FOR_RE_DELIVERY", [{'name': 'order_id', 'type': 'Order'}], [], False,
                                      "SupplyChainManager")
    ASSIGN_ORDER_TO_TRUCK = ActionType("ASSIGN_ORDER_TO_TRUCK",
                                       [{'name': 'order_id', 'type': 'Order'}, {'name': 'truck_id', 'type': 'Truck'}],
                                       [OrderAssignableConstraint, VehicleAvailableConstraint,
                                        VehicleCapacityConstraint], False, "SupplyChainManager")
    ASSIGN_ORDER_TO_DRONE = ActionType("ASSIGN_ORDER_TO_DRONE",
                                       [{'name': 'order_id', 'type': 'Order'}, {'name': 'drone_id', 'type': 'Drone'}],
                                       [OrderAssignableConstraint, VehicleAvailableConstraint,
                                        VehicleCapacityConstraint], False, "SupplyChainManager")
    ASSIGN_ORDER_TO_MICRO_HUB = ActionType("ASSIGN_ORDER_TO_MICRO_HUB", [{'name': 'order_id', 'type': 'Order'},
                                                                         {'name': 'micro_hub_id', 'type': 'MicroHub'}],
                                           [OrderAssignableConstraint, HubIsActiveConstraint], False,
                                           "SupplyChainManager")
    REASSIGN_ORDER = ActionType("REASSIGN_ORDER",
                                [{'name': 'order_id', 'type': 'Order'}, {'name': 'vehicle_id', 'type': 'Vehicle'}],
                                [OrderAssignableConstraint, VehicleAvailableConstraint, VehicleCapacityConstraint],
                                False, "SupplyChainManager")
    CONSOLIDATE_FOR_TRUCK = ActionType("CONSOLIDATE_FOR_TRUCK", [{'name': 'truck_id', 'type': 'Truck'}],
                                       [VehicleAvailableConstraint, VehicleAtNodeConstraint], False,
                                       "SupplyChainManager")
    CONSOLIDATE_FOR_DRONE = ActionType("CONSOLIDATE_FOR_DRONE", [{'name': 'drone_id', 'type': 'Drone'}],
                                       [VehicleAvailableConstraint, VehicleAtNodeConstraint], False,
                                       "SupplyChainManager")
    LOAD_TRUCK_ACTION = ActionType("LOAD_TRUCK_ACTION",
                                   [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'order_id', 'type': 'Order'}],
                                   [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderAtNodeConstraint,
                                    VehicleCapacityConstraint], True, "ResourceManager")
    UNLOAD_TRUCK_ACTION = ActionType("UNLOAD_TRUCK_ACTION",
                                     [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'order_id', 'type': 'Order'}],
                                     [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderInCargoConstraint],
                                     True, "ResourceManager")
    DRONE_LOAD_ACTION = ActionType("DRONE_LOAD_ACTION",
                                   [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'order_id', 'type': 'Order'}],
                                   [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderAtNodeConstraint,
                                    VehicleCapacityConstraint], True, "ResourceManager")
    DRONE_UNLOAD_ACTION = ActionType("DRONE_UNLOAD_ACTION",
                                     [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'order_id', 'type': 'Order'}],
                                     [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderInCargoConstraint],
                                     True, "ResourceManager")
    DRONE_CHARGE_ACTION = ActionType("DRONE_CHARGE_ACTION",
                                     [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'duration', 'type': 'int'}],
                                     [VehicleAtNodeConstraint], True, "ResourceManager")
    ACTIVATE_MICRO_HUB = ActionType("ACTIVATE_MICRO_HUB", [{'name': 'micro_hub_id', 'type': 'MicroHub'}], [], False,
                                    "ResourceManager")
    DEACTIVATE_MICRO_HUB = ActionType("DEACTIVATE_MICRO_HUB", [{'name': 'micro_hub_id', 'type': 'MicroHub'}], [], False,
                                      "ResourceManager")
    ADD_TO_CHARGING_QUEUE = ActionType("ADD_TO_CHARGING_QUEUE", [{'name': 'micro_hub_id', 'type': 'MicroHub'},
                                                                 {'name': 'drone_id', 'type': 'Drone'}],
                                       [HubIsActiveConstraint, VehicleAtNodeConstraint], True, "ResourceManager")
    FLAG_VEHICLE_FOR_MAINTENANCE = ActionType("FLAG_VEHICLE_FOR_MAINTENANCE",
                                              [{'name': 'vehicle_id', 'type': 'Vehicle'}], [], False, "ResourceManager")
    FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB = ActionType("FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB",
                                                             [{'name': 'micro_hub_id', 'type': 'MicroHub'},
                                                              {'name': 'service_type', 'type': 'str'}], [], False,
                                                             "ResourceManager")
    TRUCK_TO_NODE = ActionType("TRUCK_TO_NODE",
                               [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'destination_node_id', 'type': 'Node'}],
                               [VehicleAvailableConstraint], True, "NetworkManager")
    RE_ROUTE_TRUCK_TO_NODE = ActionType("RE_ROUTE_TRUCK_TO_NODE", [{'name': 'truck_id', 'type': 'Truck'},
                                                                   {'name': 'destination_node_id', 'type': 'Node'}], [],
                                        False, "NetworkManager")
    LAUNCH_DRONE = ActionType("LAUNCH_DRONE",
                              [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'order_id', 'type': 'Order'}],
                              [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderInCargoConstraint,
                               DroneBatteryConstraint], True, "NetworkManager")
    DRONE_TO_NODE = ActionType("DRONE_TO_NODE",
                               [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'destination_node_id', 'type': 'Node'}],
                               [VehicleAvailableConstraint], True, "NetworkManager")
    RE_ROUTE_DRONE_TO_NODE = ActionType("RE_ROUTE_DRONE_TO_NODE", [{'name': 'drone_id', 'type': 'Drone'},
                                                                   {'name': 'destination_node_id', 'type': 'Node'}], [],
                                        False, "NetworkManager")
    DRONE_LANDING_ACTION = ActionType("DRONE_LANDING_ACTION", [{'name': 'drone_id', 'type': 'Drone'}], [], True,
                                      "NetworkManager")
    DRONE_TO_CHARGING_STATION = ActionType("DRONE_TO_CHARGING_STATION", [{'name': 'drone_id', 'type': 'Drone'},
                                                                         {'name': 'station_id', 'type': 'Node'}],
                                           [VehicleAvailableConstraint], True, "NetworkManager")
    NO_OPERATION = ActionType("NO_OPERATION", [], [], False, None)

    @classmethod
    def get_all_actions(cls):
        return [getattr(cls, attr) for attr in dir(cls) if isinstance(getattr(cls, attr), ActionType)]


# -------------------------------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Validating Class-based Action Blueprint ---")

    for action in SimulationActions.get_all_actions():
        print(f"\nAction: {action.name}")
        print(f"  - ID: {action.id}")
        print(f"  - Handler: {action.handler}")
        print(f"  - Is Automatic: {action.is_automatic}")

        print("  - Parameters:")
        if action.params:
            for param in action.params:
                print(f"    - {param['name']} (Type: {param['type']})")
        else:
            print("    - None")

        print("  - Constraints:")
        if action.constraints:
            for constraint_class in action.constraints:
                print(f"    - {constraint_class.__name__}")
        else:
            print("    - None")

    print("\n--- Validation Complete ---")
