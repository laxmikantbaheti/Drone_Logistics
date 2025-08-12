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
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]:
        pass


# --- Concrete Constraint Implementations (Placeholders for now) ---

class OrderAssignableConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]: return {}


class VehicleAvailableConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]: return {}


class VehicleCapacityConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]: return {}


class HubIsActiveConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]: return {}


class VehicleAtNodeConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]: return {}


class OrderAtNodeConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]: return {}


class OrderInCargoConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]: return {}


class DroneBatteryConstraint(Constraint):
    def get_invalidations(self, global_state: 'GlobalState', action_index: 'ActionIndex',
                          p_actions_to_check: Set['ActionType']) -> Dict[Tuple, Set[int]]: return {}


# -------------------------------------------------------------------------------------------------
# -- Part 2: ActionIndex (The "Database")
# -------------------------------------------------------------------------------------------------

class ActionIndex:
    # ... (Implementation would go here) ...
    pass


# -------------------------------------------------------------------------------------------------
# -- Part 3: Class-based Action Blueprint (The Central Source of Truth)
# -------------------------------------------------------------------------------------------------

class ActionType:
    """
    A simple data class to hold the blueprint for a single action type.
    """
    _id_counter = 0
    _id_map = {}

    def __init__(self, name: str, params: List, constraints: List, is_automatic: bool, handler: str,
                 active: bool = True):
        self.id = ActionType._id_counter
        self.name = name
        self.params = params
        self.constraints = constraints
        self.is_automatic = is_automatic
        self.handler = handler
        self.active = active

        ActionType._id_map[self.id] = self
        ActionType._id_counter += 1

    @classmethod
    def get_by_id(cls, action_id: int):
        return cls._id_map.get(action_id)


class SimulationActions:
    """
    A namespace class that holds all action blueprints. The 'active' flag
    determines which actions are included in the action map for a given scenario.
    """
    # ---------------------------------------------------------------------------------------------
    # -- Core Actions (Active for Demonstration)
    # ---------------------------------------------------------------------------------------------
    ACCEPT_ORDER = ActionType("ACCEPT_ORDER", [{'name': 'order_id', 'type': 'Order'}], [OrderAssignableConstraint],
                              False, "SupplyChainManager")
    ASSIGN_ORDER_TO_TRUCK = ActionType("ASSIGN_ORDER_TO_TRUCK",
                                       [{'name': 'order_id', 'type': 'Order'}, {'name': 'truck_id', 'type': 'Truck'}],
                                       [OrderAssignableConstraint, VehicleAvailableConstraint,
                                        VehicleCapacityConstraint], False, "SupplyChainManager")
    ASSIGN_ORDER_TO_DRONE = ActionType("ASSIGN_ORDER_TO_DRONE",
                                       [{'name': 'order_id', 'type': 'Order'}, {'name': 'drone_id', 'type': 'Drone'}],
                                       [OrderAssignableConstraint, VehicleAvailableConstraint,
                                        VehicleCapacityConstraint], False, "SupplyChainManager")
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
    TRUCK_TO_NODE = ActionType("TRUCK_TO_NODE",
                               [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'destination_node_id', 'type': 'Node'}],
                               [VehicleAvailableConstraint], True, "NetworkManager")
    DRONE_TO_NODE = ActionType("DRONE_TO_NODE",
                               [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'destination_node_id', 'type': 'Node'}],
                               [VehicleAvailableConstraint], True, "NetworkManager")
    LAUNCH_DRONE = ActionType("LAUNCH_DRONE",
                              [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'order_id', 'type': 'Order'}],
                              [VehicleAvailableConstraint, VehicleAtNodeConstraint, OrderInCargoConstraint,
                               DroneBatteryConstraint], True, "NetworkManager")
    DRONE_LANDING_ACTION = ActionType("DRONE_LANDING_ACTION", [{'name': 'drone_id', 'type': 'Drone'}], [], True,
                                      "NetworkManager")

    # ---------------------------------------------------------------------------------------------
    # -- Secondary / Inactive Actions
    # ---------------------------------------------------------------------------------------------
    PRIORITIZE_ORDER = ActionType("PRIORITIZE_ORDER",
                                  [{'name': 'order_id', 'type': 'Order'}, {'name': 'priority', 'type': 'int'}], [],
                                  False, "SupplyChainManager", active=False)
    CANCEL_ORDER = ActionType("CANCEL_ORDER", [{'name': 'order_id', 'type': 'Order'}], [], False, "SupplyChainManager",
                              active=False)
    FLAG_FOR_RE_DELIVERY = ActionType("FLAG_FOR_RE_DELIVERY", [{'name': 'order_id', 'type': 'Order'}], [], False,
                                      "SupplyChainManager", active=False)
    ASSIGN_ORDER_TO_MICRO_HUB = ActionType("ASSIGN_ORDER_TO_MICRO_HUB", [{'name': 'order_id', 'type': 'Order'},
                                                                         {'name': 'micro_hub_id', 'type': 'MicroHub'}],
                                           [OrderAssignableConstraint, HubIsActiveConstraint], False,
                                           "SupplyChainManager", active=False)
    REASSIGN_ORDER = ActionType("REASSIGN_ORDER",
                                [{'name': 'order_id', 'type': 'Order'}, {'name': 'vehicle_id', 'type': 'Vehicle'}],
                                [OrderAssignableConstraint, VehicleAvailableConstraint, VehicleCapacityConstraint],
                                False, "SupplyChainManager", active=False)
    CONSOLIDATE_FOR_TRUCK = ActionType("CONSOLIDATE_FOR_TRUCK", [{'name': 'truck_id', 'type': 'Truck'}],
                                       [VehicleAvailableConstraint, VehicleAtNodeConstraint], False,
                                       "SupplyChainManager", active=False)
    CONSOLIDATE_FOR_DRONE = ActionType("CONSOLIDATE_FOR_DRONE", [{'name': 'drone_id', 'type': 'Drone'}],
                                       [VehicleAvailableConstraint, VehicleAtNodeConstraint], False,
                                       "SupplyChainManager", active=False)
    DRONE_CHARGE_ACTION = ActionType("DRONE_CHARGE_ACTION",
                                     [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'duration', 'type': 'int'}],
                                     [VehicleAtNodeConstraint], True, "ResourceManager", active=False)
    ACTIVATE_MICRO_HUB = ActionType("ACTIVATE_MICRO_HUB", [{'name': 'micro_hub_id', 'type': 'MicroHub'}], [], False,
                                    "ResourceManager", active=False)
    DEACTIVATE_MICRO_HUB = ActionType("DEACTIVATE_MICRO_HUB", [{'name': 'micro_hub_id', 'type': 'MicroHub'}], [], False,
                                      "ResourceManager", active=False)
    ADD_TO_CHARGING_QUEUE = ActionType("ADD_TO_CHARGING_QUEUE", [{'name': 'micro_hub_id', 'type': 'MicroHub'},
                                                                 {'name': 'drone_id', 'type': 'Drone'}],
                                       [HubIsActiveConstraint, VehicleAtNodeConstraint], True, "ResourceManager",
                                       active=False)
    FLAG_VEHICLE_FOR_MAINTENANCE = ActionType("FLAG_VEHICLE_FOR_MAINTENANCE",
                                              [{'name': 'vehicle_id', 'type': 'Vehicle'}], [], False, "ResourceManager",
                                              active=False)
    FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB = ActionType("FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB",
                                                             [{'name': 'micro_hub_id', 'type': 'MicroHub'},
                                                              {'name': 'service_type', 'type': 'str'}], [], False,
                                                             "ResourceManager", active=False)
    RE_ROUTE_TRUCK_TO_NODE = ActionType("RE_ROUTE_TRUCK_TO_NODE", [{'name': 'truck_id', 'type': 'Truck'},
                                                                   {'name': 'destination_node_id', 'type': 'Node'}], [],
                                        False, "NetworkManager", active=False)
    RE_ROUTE_DRONE_TO_NODE = ActionType("RE_ROUTE_DRONE_TO_NODE", [{'name': 'drone_id', 'type': 'Drone'},
                                                                   {'name': 'destination_node_id', 'type': 'Node'}], [],
                                        False, "NetworkManager", active=False)
    DRONE_TO_CHARGING_STATION = ActionType("DRONE_TO_CHARGING_STATION", [{'name': 'drone_id', 'type': 'Drone'},
                                                                         {'name': 'station_id', 'type': 'Node'}],
                                           [VehicleAvailableConstraint], True, "NetworkManager", active=False)

    # ---------------------------------------------------------------------------------------------
    # -- Special Actions
    # ---------------------------------------------------------------------------------------------
    NO_OPERATION = ActionType("NO_OPERATION", [], [], False, None)

    @classmethod
    def get_all_actions(cls):
        return [getattr(cls, attr) for attr in dir(cls) if isinstance(getattr(cls, attr), ActionType)]


# -------------------------------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Validating Class-based Action Blueprint ---")

    active_actions = []
    inactive_actions = []

    for action in SimulationActions.get_all_actions():
        if action.active:
            active_actions.append(action.name)
        else:
            inactive_actions.append(action.name)

    print(f"\nFound {len(active_actions)} ACTIVE actions:")
    pprint(active_actions)

    print(f"\nFound {len(inactive_actions)} INACTIVE actions:")
    pprint(inactive_actions)

    print("\n--- Validation Complete ---")
