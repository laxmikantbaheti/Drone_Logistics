from typing import List, Dict, Any, Callable, Tuple, Type, Set
from abc import ABC, abstractmethod
from collections import defaultdict
from pprint import pprint
import itertools
# MLPro Imports (for validation block)
from mlpro.bf.systems import System
from mlpro.bf.events import Event, EventManager


class GlobalState: pass

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
                # if entity_type == "Order":
                entity_id = action_tuple[i + 1]
                self.actions_involving_entity[(entity_type, entity_id)].add(action_index)

    def get_actions_of_type(self, action_types: List['ActionType']) -> Set[int]:
        ids = set()
        for action_type in action_types:
            ids.update(self.actions_by_type[action_type])
        return ids

    def _handle_new_entity(self):
        pass



# -------------------------------------------------------------------------------------------------
# -- Part 3: Class-based Action Blueprint (The Central Source of Truth)
# -------------------------------------------------------------------------------------------------

class ActionType:
    """
    A simple data class to hold the blueprint for a single action type.
    """
    _id_counter = 1
    _id_map = {}

    def __init__(self, name: str, params: List, is_automatic: bool, handler: str, active: bool = True):
        self.id = ActionType._id_counter
        self.name = name
        self.params = params
        self.is_automatic = is_automatic
        self.handler = handler
        self.active = active

        ActionType._id_map[self.id] = self
        ActionType._id_counter += 1

    @classmethod
    def get_by_id(cls, action_id: int):
        return cls._id_map.get(action_id)

    def __repr__(self):
        return self.name


class SimulationActions:
    """
    A namespace class that holds all action blueprints. The 'active' flag
    determines which actions are included in the action map for a given scenario.
    """
    # ---------------------------------------------------------------------------------------------
    # -- Core Actions (Active for Demonstration)
    # ---------------------------------------------------------------------------------------------

    ACCEPT_ORDER = ActionType(name="ACCEPT_ORDER",
                              # params=[{'name': 'order_id', 'type': 'Order'}],
                              params=[{'name':'pick_up_drop', "type":"Node Pair"}],
                              is_automatic=True,
                              handler="SupplyChainManager",
                              active=False)

    ASSIGN_ORDER_TO_TRUCK = ActionType(name="ASSIGN_ORDER_TO_TRUCK",
                                       params=[{'name': 'pick_up_drop', 'type': 'Node Pair'},
                                               {'name': 'truck_id', 'type': 'Truck'}],
                                       is_automatic=False,
                                       handler="SupplyChainManager")

    ASSIGN_ORDER_TO_DRONE = ActionType(name="ASSIGN_ORDER_TO_DRONE",
                                       params=[{'name': 'pick_up_drop', 'type': 'Node Pair'},
                                               {'name': 'drone_id', 'type': 'Drone'}],
                                       is_automatic=False,
                                       handler="SupplyChainManager")

    LOAD_TRUCK_ACTION = ActionType(name="LOAD_TRUCK_ACTION",
                                   params=[{'name': 'truck_id', 'type': 'Truck'},
                                           {'name': 'order_id', 'type': 'Order'}],
                                   is_automatic=True,
                                   handler="ResourceManager")

    UNLOAD_TRUCK_ACTION = ActionType(name="UNLOAD_TRUCK_ACTION",
                                     params=[{'name': 'truck_id', 'type': 'Truck'},
                                             {'name': 'order_id', 'type': 'Order'}],
                                     is_automatic=True,
                                     handler="ResourceManager")

    LOAD_DRONE_ACTION = ActionType(name="LOAD_DRONE",
                                   params=[{'name': 'drone_id', 'type': 'Drone'},
                                           {'name': 'order_id', 'type': 'Order'}],
                                   is_automatic=True,
                                   handler="ResourceManager")

    UNLOAD_DRONE_ACTION = ActionType(name="UNLOAD_DRONE",
                                     params=[{'name': 'drone_id', 'type': 'Drone'},
                                             {'name': 'order_id', 'type': 'Order'}],
                                     is_automatic=True,
                                     handler="ResourceManager")

    TRUCK_TO_NODE = ActionType(name="TRUCK_TO_NODE",
                               params=[{'name': 'truck_id', 'type': 'Truck'},
                                       {'name': 'destination_node_id', 'type': 'Node'}],
                               is_automatic=True,
                               handler="NetworkManager",
                               active=False)

    DRONE_TO_NODE = ActionType(name="DRONE_TO_NODE",
                               params=[{'name': 'drone_id', 'type': 'Drone'},
                                       {'name': 'destination_node_id', 'type': 'Node'}],
                               is_automatic=True,
                               handler="NetworkManager",
                               active=False)

    DRONE_LAUNCH = ActionType(name="LAUNCH_DRONE",
                              params=[{'name': 'drone_id', 'type': 'Drone'},
                                      {'name': 'order_id', 'type': 'Order'}],
                              is_automatic=True,
                              handler="NetworkManager")

    DRONE_LAND = ActionType(name="LAND_DRONE",
                            params=[{'name': 'drone_id', 'type': 'Drone'}],
                            is_automatic=True,
                            handler="NetworkManager")

    CONSOLIDATE_FOR_TRUCK = ActionType(name="CONSOLIDATE_FOR_TRUCK",
                                       params=[{'name': 'truck_id', 'type': 'Truck'}],
                                       is_automatic=False,
                                       handler="SupplyChainManager", active=True)

    CONSOLIDATE_FOR_DRONE = ActionType(name="CONSOLIDATE_FOR_DRONE",
                                       params=[{'name': 'drone_id', 'type': 'Drone'}],
                                       is_automatic=False,
                                       handler="SupplyChainManager",
                                       active=True)

    # ---------------------------------------------------------------------------------------------
    # -- Secondary / Inactive Actions
    # ---------------------------------------------------------------------------------------------
    PRIORITIZE_ORDER = ActionType("PRIORITIZE_ORDER",
                                  [{'name': 'order_id', 'type': 'Order'}, {'name': 'priority', 'type': 'int'}], False,
                                  "SupplyChainManager", active=False)
    CANCEL_ORDER = ActionType("CANCEL_ORDER", [{'name': 'order_id', 'type': 'Order'}], False, "SupplyChainManager",
                              active=False)
    FLAG_FOR_RE_DELIVERY = ActionType("FLAG_FOR_RE_DELIVERY", [{'name': 'order_id', 'type': 'Order'}], False,
                                      "SupplyChainManager", active=False)
    ASSIGN_ORDER_TO_MICRO_HUB = ActionType("ASSIGN_ORDER_TO_MICRO_HUB", [{'name': 'order_id', 'type': 'Order'},
                                                                         {'name': 'micro_hub_id', 'type': 'MicroHub'}],
                                           False, "SupplyChainManager", active=False)
    REASSIGN_ORDER = ActionType("REASSIGN_ORDER",
                                [{'name': 'order_id', 'type': 'Order'}, {'name': 'vehicle_id', 'type': 'Vehicle'}],
                                False, "SupplyChainManager", active=False)
    DRONE_CHARGE_ACTION = ActionType("DRONE_CHARGE_ACTION",
                                     [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'duration', 'type': 'int'}], True,
                                     "ResourceManager", active=False)
    ACTIVATE_MICRO_HUB = ActionType("ACTIVATE_MICRO_HUB", [{'name': 'micro_hub_id', 'type': 'MicroHub'}], False,
                                    "ResourceManager", active=False)
    DEACTIVATE_MICRO_HUB = ActionType("DEACTIVATE_MICRO_HUB", [{'name': 'micro_hub_id', 'type': 'MicroHub'}], False,
                                      "ResourceManager", active=False)
    ADD_TO_CHARGING_QUEUE = ActionType("ADD_TO_CHARGING_QUEUE", [{'name': 'micro_hub_id', 'type': 'MicroHub'},
                                                                 {'name': 'drone_id', 'type': 'Drone'}], True,
                                       "ResourceManager", active=False)
    FLAG_VEHICLE_FOR_MAINTENANCE = ActionType("FLAG_VEHICLE_FOR_MAINTENANCE",
                                              [{'name': 'vehicle_id', 'type': 'Vehicle'}], False, "ResourceManager",
                                              active=False)
    FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB = ActionType("FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB",
                                                             [{'name': 'micro_hub_id', 'type': 'MicroHub'},
                                                              {'name': 'service_type', 'type': 'str'}], False,
                                                             "ResourceManager", active=False)
    RE_ROUTE_TRUCK_TO_NODE = ActionType("RE_ROUTE_TRUCK_TO_NODE", [{'name': 'truck_id', 'type': 'Truck'},
                                                                   {'name': 'destination_node_id', 'type': 'Node'}],
                                        False, "NetworkManager", active=False)
    RE_ROUTE_DRONE_TO_NODE = ActionType("RE_ROUTE_DRONE_TO_NODE", [{'name': 'drone_id', 'type': 'Drone'},
                                                                   {'name': 'destination_node_id', 'type': 'Node'}],
                                        False, "NetworkManager", active=False)
    DRONE_TO_CHARGING_STATION = ActionType("DRONE_TO_CHARGING_STATION", [{'name': 'drone_id', 'type': 'Drone'},
                                                                         {'name': 'station_id', 'type': 'Node'}], True,
                                           "NetworkManager", active=False)

    # ---------------------------------------------------------------------------------------------
    # -- Special Actions
    # ---------------------------------------------------------------------------------------------
    NO_OPERATION = ActionType("NO_OPERATION", [], False, None)

    def __init__(self):
        self.actions = self.get_all_actions()
        self.action_map = None
        self.action_space_size = None

    @classmethod
    def get_all_actions(cls):
        all_actions = [getattr(cls, attr) for attr in dir(cls)
                       if (isinstance(getattr(cls, attr), ActionType) and getattr(cls, attr).active)]
        all_actions.sort(key = lambda x: x.id)
        return all_actions

    @classmethod
    def get_actions_by_manager(cls, p_manager_name):
        actions_by_manager = [getattr(cls, attr) for attr in dir(cls)
                              if (isinstance(getattr(cls, attr), ActionType)
                                  and getattr(cls, attr).active and (getattr(cls,attr).handler == p_manager_name))]
        actions_by_manager.sort(key=lambda x: x.id)
        return actions_by_manager

    def generate_action_map(self, global_state: 'GlobalState') -> Tuple[Dict[Tuple, int], int]:
        """
        Programmatically generates the global flattened action map and action space size
        at runtime based on the entities that actually exist in the global_state.
        This version generates the COMPLETE map, ignoring the 'active' flag, to ensure
        a static action space size for any given scenario configuration.
        """
        action_map = {}
        current_index = 0

        # 1. Get the actual ID ranges from the global_state
        entity_id_ranges = {
            'Order': list(global_state.orders.keys()),
            'Truck': list(global_state.trucks.keys()),
            'Drone': list(global_state.drones.keys()),
            'Node': list(global_state.nodes.keys()),
            'MicroHub': list(global_state.micro_hubs.keys()),
            'Vehicle': list(global_state.trucks.keys()) + list(global_state.drones.keys()),
            'Node Pair': global_state.node_pairs
        }

        # 2. Iterate through each action defined in our blueprint
        for action_type in self.get_all_actions():
            # This loop now includes ALL actions to ensure a static action map size
            if not action_type.params:
                action_tuple = (action_type,)
                if action_tuple not in action_map:
                    action_map[action_tuple] = current_index
                    current_index += 1
                continue

            # 3. Get the ranges for each parameter for this action
            param_ranges = []
            possible = True
            for param in action_type.params:
                if 'range' in param:
                    param_ranges.append(param['range'])
                else:
                    param_type = param['type']
                    ids = entity_id_ranges.get(param_type, [])
                    if not ids:
                        possible = False
                        break
                    param_ranges.append(ids)

            if not possible:
                continue

            # 4. Generate all unique combinations of parameter values
            # Handling special invalid case of same-node delivery requests.
            param_combinations = list(itertools.product(*param_ranges))

            for combo in param_combinations:
                action_tuple = (action_type,) + combo
                if action_tuple not in action_map:
                    action_map[action_tuple] = current_index
                    current_index += 1

        action_space_size = len(action_map)
        self.action_map = action_map
        self.action_space_size = action_space_size
        return action_map, action_space_size



#
# # -------------------------------------------------------------------------------------------------
# # -- Validation Block
# # -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    class MockEntity(System):
        def __init__(self, p_id, status, cargo_count=0, capacity=1, battery=1.0, current_node=None):
            super().__init__(p_id=p_id)
            self.status = status
            self.cargo_manifest = list(range(cargo_count))
            self.max_payload_capacity = capacity
            self.battery_level = battery
            self.current_node_id = current_node
            self.packages_held = []

        @staticmethod
        def setup_spaces(): return None, None


    class MockGlobalState(GlobalState):
        def __init__(self):
            self.orders = {0: MockEntity(p_id=0, status='pending'), 1: MockEntity(p_id=1, status='delivered')}
            self.trucks = {101: MockEntity(p_id=101, status='idle', current_node=5)}
            self.drones = {201: MockEntity(p_id=201, status='idle', battery=0.1)}
            self.nodes = {5: MockEntity(p_id=5, status='active'), 6: MockEntity(p_id=6, status="active")}
            self.micro_hubs = {}
            self.nodes[5].packages_held = [0]
            self.node_pairs = self.get_node_pairs()
            def get_all_entities(self, type):
                return getattr(self, type + 's', {})

            self.get_all_entities = get_all_entities
            GlobalState.__init__(self)

        def get_node_pairs(self):
            node_pairs = []
            node_ids = self.nodes.keys()
            node_pairs = list(itertools.permutations(node_ids))
            return node_pairs


    mock_gs = MockGlobalState()
#
#     print("--- Validating Unified Constraint Logic ---")
#
#     # Test VehicleAtNodeConstraint
#     van_constraint = VehicleAtNodeConstraint()
#     action_tuple_valid = (SimulationActions.LOAD_TRUCK_ACTION, 101, 0)
#     mock_gs.trucks[101].current_node_id = 8  # Truck is at the wrong node
#     assert van_constraint.is_invalid(mock_gs, action_tuple_valid) == True
#     mock_gs.trucks[101].current_node_id = 5  # Truck is at the correct node
#     assert van_constraint.is_invalid(mock_gs, action_tuple_valid) == False
#     print("VehicleAtNodeConstraint: PASSED")
#
#     # Test OrderAtNodeConstraint
#     oan_constraint = OrderAtNodeConstraint()
#     mock_gs.nodes[5].packages_held = []  # Order is not at the node
#     assert oan_constraint.is_invalid(mock_gs, action_tuple_valid) == True
#     mock_gs.nodes[5].packages_held = [0]  # Order is at the node
#     assert oan_constraint.is_invalid(mock_gs, action_tuple_valid) == False
#     print("OrderAtNodeConstraint: PASSED")
#
#     # Test OrderInCargoConstraint
#     oic_constraint = OrderInCargoConstraint()
#     action_tuple_unload = (SimulationActions.UNLOAD_TRUCK_ACTION, 101, 0)
#     mock_gs.trucks[101].cargo_manifest = []  # Order is not in cargo
#     assert oic_constraint.is_invalid(mock_gs, action_tuple_unload) == True
#     mock_gs.trucks[101].cargo_manifest = [0]  # Order is in cargo
#     assert oic_constraint.is_invalid(mock_gs, action_tuple_unload) == False
#     print("OrderInCargoConstraint: PASSED")
#
#     # Test DroneBatteryConstraint
#     db_constraint = DroneBatteryConstraint()
#     action_tuple_launch = (SimulationActions.LAUNCH_DRONE, 201, 0)
#     mock_gs.drones[201].battery_level = 0.1  # Low battery
#     assert db_constraint.is_invalid(mock_gs, action_tuple_launch) == True
#     mock_gs.drones[201].battery_level = 0.9  # High battery
#     assert db_constraint.is_invalid(mock_gs, action_tuple_launch) == False
#     print("DroneBatteryConstraint: PASSED")
#
#     print("\n--- Validation Complete ---")

    actions = SimulationActions()

    maps,action_space_size = actions.generate_action_map(mock_gs)
    [print(f"{item[0][0].name}{item[0][1:]}", item[1:]) for item in maps.items()]
    print(action_space_size)
    action_index = ActionIndex(mock_gs, maps)
    print(action_index.get_actions_of_type([SimulationActions.ASSIGN_ORDER_TO_TRUCK]))
    print(action_index.get_actions_of_type([SimulationActions.DRONE_TO_NODE]))
    print(action_index.actions_involving_entity["Order", 0])
    print([action.name for action in actions.get_actions_by_manager(p_manager_name='NetworkManager')])
