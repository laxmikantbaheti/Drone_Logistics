from ddls_src.actions.base import ActionType
import itertools
from typing import Tuple, List, Dict
class SimulationActions:
    """
    A namespace class that holds all action blueprints. The 'active' flag
    determines which actions are included in the action map for a given scenario.
    """
    # ---------------------------------------------------------------------------------------------
    # -- Core Actions (Active for Demonstration)
    # ---------------------------------------------------------------------------------------------

    # ACCEPT_ORDER = ActionType(name="ACCEPT_ORDER",
    #                           # params=[{'name': 'order_id', 'type': 'Order'}],
    #                           params=[{'name':'pick_up_drop', "type":"Node Pair"}],
    #                           is_automatic=True,
    #                           handler="SupplyChainManager",
    #                           active=False)
    #
    # ASSIGN_ORDER_TO_TRUCK = ActionType(name="ASSIGN_ORDER_TO_TRUCK",
    #                                    params=[{'name': 'pick_up_drop', 'type': 'Node Pair'},
    #                                            {'name': 'truck_id', 'type': 'Truck'}],
    #                                    is_automatic=False,
    #                                    handler="SupplyChainManager")
    #
    # ASSIGN_ORDER_TO_DRONE = ActionType(name="ASSIGN_ORDER_TO_DRONE",
    #                                    params=[{'name': 'pick_up_drop', 'type': 'Node Pair'},
    #                                            {'name': 'drone_id', 'type': 'Drone'}],
    #                                    is_automatic=False,
    #                                    handler="SupplyChainManager")
    ASSIGN_ORDER_TO_RESOURCE = ActionType(name = "ASSIGN_ORDER_TO_RESOURCE",
                                          params=[{"name":'pick_up_drop', "type":"Node Pair"}],
                                          is_automatic=False,
                                          handler = "Logistic")
    SELECT_TRUCK = ActionType(name = "SELECT_TRUCK",
                                 params = [{"name" : "truck_id", "type":"Truck"}],
                                 is_automatic=True,
                                 handler = "Logistic")
    SELECT_DRONE = ActionType(name = "SELECT_DRONE",
                                 params = [{"name" : "drone_id", "type":"Drone"}],
                                 is_automatic=True,
                                 handler = "Logistic")
    SELECT_MICROHUB = ActionType(name = "SELECT_MICROHUB",
                                 params = [{"name" : "micro_hub_id", "type":"MicroHub"}],
                                 is_automatic=True,
                                 handler = "Logistic")
    CONSOLIDATE = ActionType(name="CONSOLIDATE_FOR_VEHICLE",
                             params=[],
                             is_automatic=True,
                             handler = "Logistic")
    #
    # ASSIGN_ORDER_TO_MICRO_HUB = ActionType(name="ASSIGN_ORDER_TO_MICRO_HUB",
    #                                        params=[{'name': 'pick_up_drop', 'type': 'Node Pair'},
    #                                                {'name': 'micro_hub_id', 'type': 'MicroHub'}],
    #                                        is_automatic=False,
    #                                        handler= "SupplyChainManager",
    #                                        active=True)
    #
    LOAD_TRUCK_ACTION = ActionType(name="LOAD_TRUCK_ACTION",
                                   params=[{'name': 'truck_id', 'type': 'Truck'}],
                                   is_automatic=True,
                                   handler="Logistic")

    UNLOAD_TRUCK_ACTION = ActionType(name="UNLOAD_TRUCK_ACTION",
                                     params=[{'name': 'truck_id', 'type': 'Truck'}],
                                     is_automatic=True,
                                     handler="Logistic")

    LOAD_DRONE_ACTION = ActionType(name="LOAD_DRONE",
                                   params=[{'name': 'drone_id', 'type': 'Drone'}],
                                   is_automatic=True,
                                   handler="Logistic")

    UNLOAD_DRONE_ACTION = ActionType(name="UNLOAD_DRONE",
                                     params=[{'name': 'drone_id', 'type': 'Drone'}],
                                     is_automatic=True,
                                     handler="Logistic")
    #
    # PHASE_ONE_DEFAULT = ActionType("Phase One Default",
    #                                [],
    #                                False,
    #                                "Logistic")
    #
    # PHASE_TWO_DEFAULT = ActionType("Phase Two Default",
    #                                [],
    #                                False,
    #                                "Logistic")
    #
    # TRUCK_TO_NODE = ActionType(name="TRUCK_TO_NODE",
    #                            params=[{'name': 'truck_id', 'type': 'Truck'},
    #                                    {'name': 'destination_node_id', 'type': 'Node'}],
    #                            is_automatic=True,
    #                            handler="NetworkManager",
    #                            active=False)
    #
    # DRONE_TO_NODE = ActionType(name="DRONE_TO_NODE",
    #                            params=[{'name': 'drone_id', 'type': 'Drone'},
    #                                    {'name': 'destination_node_id', 'type': 'Node'}],
    #                            is_automatic=True,
    #                            handler="NetworkManager",
    #                            active=False)
    #
    # DRONE_LAUNCH = ActionType(name="LAUNCH_DRONE",
    #                           params=[{'name': 'drone_id', 'type': 'Drone'},
    #                                   {'name': 'order_id', 'type': 'Order'}],
    #                           is_automatic=True,
    #                           active = False,
    #                           handler="NetworkManager")
    #
    # DRONE_LAND = ActionType(name="LAND_DRONE",
    #                         params=[{'name': 'drone_id', 'type': 'Drone'}],
    #                         is_automatic=True,
    #                         active = False,
    #                         handler="NetworkManager")
    #
    # CONSOLIDATE_FOR_TRUCK = ActionType(name="CONSOLIDATE_FOR_TRUCK",
    #                                    params=[{'name': 'truck_id', 'type': 'Truck'}],
    #                                    is_automatic=False,
    #                                    handler="SupplyChainManager", active=True)
    #
    # CONSOLIDATE_FOR_DRONE = ActionType(name="CONSOLIDATE_FOR_DRONE",
    #                                    params=[{'name': 'drone_id', 'type': 'Drone'}],
    #                                    is_automatic=False,
    #                                    handler="SupplyChainManager",
    #                                    active=True)
    #
    # # ---------------------------------------------------------------------------------------------
    # # -- Secondary / Inactive Actions
    # # ---------------------------------------------------------------------------------------------
    # PRIORITIZE_ORDER = ActionType("PRIORITIZE_ORDER",
    #                               [{'name': 'order_id', 'type': 'Order'}, {'name': 'priority', 'type': 'int'}], False,
    #                               "SupplyChainManager", active=False)
    # CANCEL_ORDER = ActionType("CANCEL_ORDER", [{'name': 'order_id', 'type': 'Order'}], False, "SupplyChainManager",
    #                           active=False)
    # FLAG_FOR_RE_DELIVERY = ActionType("FLAG_FOR_RE_DELIVERY", [{'name': 'order_id', 'type': 'Order'}], False,
    #                                   "SupplyChainManager", active=False)
    # # ASSIGN_ORDER_TO_MICRO_HUB = ActionType("ASSIGN_ORDER_TO_MICRO_HUB", [{'name': 'order_id', 'type': 'Order'},
    # #                                                                      {'name': 'micro_hub_id', 'type': 'MicroHub'}],
    # #                                        False, "SupplyChainManager", active=False)
    # REASSIGN_ORDER = ActionType("REASSIGN_ORDER",
    #                             [{'name': 'order_id', 'type': 'Order'}, {'name': 'vehicle_id', 'type': 'Vehicle'}],
    #                             False, "SupplyChainManager", active=False)
    # DRONE_CHARGE_ACTION = ActionType("DRONE_CHARGE_ACTION",
    #                                  [{'name': 'drone_id', 'type': 'Drone'}, {'name': 'duration', 'type': 'int'}], True,
    #                                  "ResourceManager", active=False)
    # ACTIVATE_MICRO_HUB = ActionType("ACTIVATE_MICRO_HUB", [{'name': 'micro_hub_id', 'type': 'MicroHub'}], False,
    #                                 "ResourceManager", active=False)
    # DEACTIVATE_MICRO_HUB = ActionType("DEACTIVATE_MICRO_HUB", [{'name': 'micro_hub_id', 'type': 'MicroHub'}], False,
    #                                   "ResourceManager", active=False)
    # ADD_TO_CHARGING_QUEUE = ActionType("ADD_TO_CHARGING_QUEUE", [{'name': 'micro_hub_id', 'type': 'MicroHub'},
    #                                                              {'name': 'drone_id', 'type': 'Drone'}], True,
    #                                    "ResourceManager", active=False)
    # FLAG_VEHICLE_FOR_MAINTENANCE = ActionType("FLAG_VEHICLE_FOR_MAINTENANCE",
    #                                           [{'name': 'vehicle_id', 'type': 'Vehicle'}], False, "ResourceManager",
    #                                           active=False)
    # FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB = ActionType("FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB",
    #                                                          [{'name': 'micro_hub_id', 'type': 'MicroHub'},
    #                                                           {'name': 'service_type', 'type': 'str'}], False,
    #                                                          "ResourceManager", active=False)
    # RE_ROUTE_TRUCK_TO_NODE = ActionType("RE_ROUTE_TRUCK_TO_NODE", [{'name': 'truck_id', 'type': 'Truck'},
    #                                                                {'name': 'destination_node_id', 'type': 'Node'}],
    #                                     False, "NetworkManager", active=False)
    # RE_ROUTE_DRONE_TO_NODE = ActionType("RE_ROUTE_DRONE_TO_NODE", [{'name': 'drone_id', 'type': 'Drone'},
    #                                                                {'name': 'destination_node_id', 'type': 'Node'}],
    #                                     False, "NetworkManager", active=False)
    # DRONE_TO_CHARGING_STATION = ActionType("DRONE_TO_CHARGING_STATION", [{'name': 'drone_id', 'type': 'Drone'},
    #                                                                      {'name': 'station_id', 'type': 'Node'}], True,
    #                                        "NetworkManager", active=False)
    #
    # # ---------------------------------------------------------------------------------------------
    # # -- Special Actions
    # # ---------------------------------------------------------------------------------------------
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
        at runtime. Populates 'associated_action_indexes' on entities for O(1) constraint checking.
        """
        action_map = {}
        current_index = 0

        # --- [MODIFICATION 1] ---
        # 0. Clear existing associations and flags on all entities
        # [FIX] Added .values() to ensure we iterate over the entity dictionaries, not just keys
        all_entities = global_state.get_all_entities()
        iterator = all_entities.values() if isinstance(all_entities, dict) else all_entities

        for entity_dict in iterator:
            for entity in entity_dict.values():
                if hasattr(entity, 'associated_actions'):
                    entity.associated_actions.clear()
                if hasattr(entity, 'associated_action_indexes'):
                    entity.associated_action_indexes.clear()
                if hasattr(entity, 'action_operability'):
                    entity.action_operability.clear()
        # --- [END OF MODIFICATION 1] ---

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

        # [NEW] Helper mapping to get actual entity objects
        entity_objects_map = {
            'Order': global_state.orders,
            'Truck': global_state.trucks,
            'Drone': global_state.drones,
            'Node': global_state.nodes,
            'MicroHub': global_state.micro_hubs,
        }

        # 2. Iterate through each action defined in our blueprint
        for action_type in self.get_all_actions():

            # --- [MODIFICATION 2] ---
            # [NEW] Populate GENERIC properties (associated_actions, operability)
            if action_type.params:
                for param in action_type.params:
                    param_type = param['type']
                    target_collections = []

                    if param_type in entity_objects_map:
                        target_collections.append(entity_objects_map[param_type].values())
                    elif param_type == 'Vehicle':
                        target_collections.append(global_state.trucks.values())
                        target_collections.append(global_state.drones.values())
                    elif param_type == "Resource":
                        target_collections.append(global_state.trucks.values())
                        target_collections.append(global_state.drones.values())
                        target_collections.append(global_state.micro_hubs.values())
                        pass

                    for collection in target_collections:
                        for entity in collection:
                            entity.associated_actions.add(action_type)
                            entity.action_operability[action_type] = True
            # --- [END OF MODIFICATION 2] ---

            if not action_type.params:
                action_tuple = (action_type,)
                if action_tuple not in action_map:
                    action_map[action_tuple] = current_index
                    current_index += 1
                continue

            # 3. Get ranges
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

            # 4. Generate combinations
            param_combinations = list(itertools.product(*param_ranges))

            # Filter MicroHub assignments
            if action_type.name == "ASSIGN_ORDER_TO_MICRO_HUB":
                filtered_combinations = []
                for combo in param_combinations:
                    node_pair, micro_hub_id = combo
                    pickup_node_id, delivery_node_id = node_pair
                    if micro_hub_id != pickup_node_id and micro_hub_id != delivery_node_id:
                        filtered_combinations.append(combo)
                param_combinations = filtered_combinations

            # 5. Assign Indexes and Map to Entities
            for combo in param_combinations:
                action_tuple = (action_type,) + combo

                if action_tuple not in action_map:
                    # A. Register the action
                    action_map[action_tuple] = current_index

                    # --- [START OF MODIFICATION 3] ---
                    # [NEW] Map this SPECIFIC action index (int) to the SPECIFIC entity instances involved.
                    # This enables O(1) lookup in constraints: "p_entity.associated_action_indexes"
                    if action_type.params:
                        for i, param_val in enumerate(combo):
                            param_type = action_type.params[i]['type']
                            target_entity = None

                            # Resolve ID to Object
                            if param_type in entity_objects_map:
                                target_entity = entity_objects_map[param_type].get(param_val)
                            elif param_type == 'Vehicle':
                                # Try both fleets
                                if param_val in global_state.trucks:
                                    target_entity = global_state.trucks[param_val]
                                elif param_val in global_state.drones:
                                    target_entity = global_state.drones[param_val]

                            # Assign Index
                            if target_entity is not None and hasattr(target_entity, 'associated_action_indexes'):
                                target_entity.associated_action_indexes.add(current_index)
                    # --- [END OF MODIFICATION 3] ---

                    current_index += 1

        action_space_size = len(action_map)
        self.action_map = action_map
        self.action_space_size = action_space_size
        return action_map, action_space_size