from ddls_src.actions.base import ActionType
import itertools
from typing import Tuple, List, Dict
import numpy as np


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
                                 is_automatic=False,
                                 handler = "Logistic",
                              active=False)
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

    # ---------------------------------------------------------------------------------------------
    # -- [NEW ADDITIVE INTERFACE]: 3-Map Vectorized Builders (Option B Architecture)
    # ---------------------------------------------------------------------------------------------
    def build_action_registries(self, global_state: 'GlobalState', action_map: Dict[Tuple, int] = None,
                                action_space_size: int = None):
        """
        Builds static NumPy boolean lookup matrices (Map 1 & Map 3) and Option B
        pre-extracted 1D int32 array tables (Map 2), attaching them to global_state.action_registry.
        """
        act_map = action_map if action_map is not None else self.action_map
        act_space_sz = action_space_size if action_space_size is not None else self.action_space_size

        if act_map is None or act_space_sz is None:
            raise ValueError("Action map must be generated before building action registries.")

        # Ensure deterministic integer indexing exists on entities
        if not hasattr(global_state, 'truck_id_to_idx') or len(global_state.truck_id_to_idx) == 0:
            global_state.initialize_entity_indexing()

        active_action_types = self.get_all_actions()
        action_type_to_idx = {act.name: i for i, act in enumerate(active_action_types)}

        num_trucks = len(global_state.trucks)
        num_drones = len(global_state.drones)
        num_mhs = len(global_state.micro_hubs)
        num_nps = len(global_state.nodepair_to_idx)
        num_types = len(active_action_types)

        # ------------------------------------------------------------------
        # Map 1: Action Type -> Action Mask Matrix (num_types, A)
        # ------------------------------------------------------------------
        map_1_action_type = np.zeros((num_types, act_space_sz), dtype=bool)

        # ------------------------------------------------------------------
        # Map 3: Entity Category -> Action Mask Matrices (num_entities, A)
        # ------------------------------------------------------------------
        map_3_truck = np.zeros((num_trucks, act_space_sz), dtype=bool)
        map_3_drone = np.zeros((num_drones, act_space_sz), dtype=bool)
        map_3_microhub = np.zeros((num_mhs, act_space_sz), dtype=bool)
        map_3_nodepair = np.zeros((num_nps, act_space_sz), dtype=bool)

        # Scalar coordinates for zero-parameter actions
        scalar_actions = {
            "CONSOLIDATE": act_map.get((self.CONSOLIDATE,), None),
            "NO_OPERATION": act_map.get((self.NO_OPERATION,), None),
        }

        # Populate Map 1 and Map 3 using dictionary coordinate mappings
        for action_tuple, act_idx in act_map.items():
            act_type = action_tuple[0]

            # Mark Map 1
            if act_type.name in action_type_to_idx:
                t_idx = action_type_to_idx[act_type.name]
                map_1_action_type[t_idx, act_idx] = True

            # Mark Map 3 based on parameters
            params = action_tuple[1:]
            for param_val in params:
                # 1. Truck
                if param_val in global_state.truck_id_to_idx:
                    t_idx = global_state.truck_id_to_idx[param_val]
                    map_3_truck[t_idx, act_idx] = True

                # 2. Drone
                elif param_val in global_state.drone_id_to_idx:
                    d_idx = global_state.drone_id_to_idx[param_val]
                    map_3_drone[d_idx, act_idx] = True

                # 3. MicroHub (Uses isolated microhub coordinate, avoiding node-index collision)
                if param_val in global_state.microhub_id_to_idx:
                    m_idx = global_state.microhub_id_to_idx[param_val]
                    map_3_microhub[m_idx, act_idx] = True

                # 4. Node Pair parameter resolution
                if param_val in global_state.nodepair_to_idx:
                    p_idx = global_state.nodepair_to_idx[param_val]
                    map_3_nodepair[p_idx, act_idx] = True

        # Attach raw Map 1 & Map 3 matrices to global_state
        global_state.action_registry = {
            "map_1_action_type": map_1_action_type,
            "action_type_to_idx": action_type_to_idx,
            "map_3_truck": map_3_truck,
            "map_3_drone": map_3_drone,
            "map_3_microhub": map_3_microhub,
            "map_3_nodepair": map_3_nodepair,
            "scalar_actions": scalar_actions,
            "action_space_size": act_space_sz
        }

    # ---------------------------------------------------------------------------------------------
    # -- [NEW ADDITIVE HELPER]: Map 2 Generator (Option B Table Builder)
    # ---------------------------------------------------------------------------------------------
    @classmethod
    def build_common_actions_table(cls,
                                   global_state: 'GlobalState',
                                   action_types: List[ActionType],
                                   entity_category: str) -> List[np.ndarray]:
        """
        Option B Generator:
        Given an entity category ('truck', 'drone', 'microhub', 'node_pair') and a list of
        affected ActionType blueprints, pre-extracts and returns a 1D list/array of 1D NumPy int32 arrays.
        Index i in the returned list corresponds strictly to entity.int_id == i.
        """
        reg = getattr(global_state, 'action_registry', None)
        if reg is None:
            raise RuntimeError("global_state.action_registry has not been built yet.")

        # 1. Combine Map 1 masks for all requested action types via bitwise OR
        c_mask = np.zeros(reg["action_space_size"], dtype=bool)
        for act_type in action_types:
            if act_type.name in reg["action_type_to_idx"]:
                t_idx = reg["action_type_to_idx"][act_type.name]
                c_mask |= reg["map_1_action_type"][t_idx]

        # 2. Select corresponding Map 3 entity matrix
        cat_key = f"map_3_{entity_category.lower()}"
        entity_matrix = reg.get(cat_key)
        if entity_matrix is None:
            raise KeyError(f"Unknown entity category: '{entity_category}'. Expected truck, drone, microhub, or nodepair.")

        # 3. Pre-extract the int32 action IDs for each entity row (Map 2)
        num_entities = entity_matrix.shape[0]
        common_actions_table = [None] * num_entities
        for e_idx in range(num_entities):
            common_mask = entity_matrix[e_idx] & c_mask
            common_actions_table[e_idx] = np.flatnonzero(common_mask).astype(np.int32)

        return common_actions_table