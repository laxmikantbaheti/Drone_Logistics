from typing import Dict, Any, Tuple

# Local Imports
from ..actions.action_enums import SimulationAction
from ..core.basics import LogisticsAction  # <-- IMPORT the new custom action class


# Forward declarations
class GlobalState: pass


class ActionMasker: pass


class SupplyChainManager: pass


class ResourceManager: pass


class NetworkManager: pass


class ActionManager:
    """
    Receives a global action tuple, validates it, and routes it to the
    appropriate high-level manager system using the custom LogisticsAction class.
    """

    def __init__(self,
                 global_state: 'GlobalState',
                 managers: Dict[str, Any],
                 action_map: Dict[Tuple, int],
                 action_masker: 'ActionMasker'):

        self.global_state = global_state
        self.action_map = action_map
        self.action_masker = action_masker

        self.supply_chain_manager: 'SupplyChainManager' = managers.get('supply_chain_manager')
        self.resource_manager: 'ResourceManager' = managers.get('resource_manager')
        self.network_manager: 'NetworkManager' = managers.get('network_manager')

        self._reverse_action_map: Dict[int, Tuple] = {idx: act_tuple for act_tuple, idx in action_map.items()}

        self._setup_action_sets()
        self._setup_dispatch_mappings()

        print("ActionManager (Refactored with LogisticsAction) initialized.")

    def _setup_action_sets(self):
        """Groups global actions by the manager responsible for them."""
        self.SCM_ACTIONS = {
            SimulationAction.ACCEPT_ORDER, SimulationAction.CANCEL_ORDER,
            SimulationAction.ASSIGN_ORDER_TO_TRUCK, SimulationAction.ASSIGN_ORDER_TO_DRONE,
            SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB
        }
        self.RM_ACTIONS = {
            SimulationAction.LOAD_TRUCK_ACTION, SimulationAction.UNLOAD_TRUCK_ACTION,
            SimulationAction.DRONE_LOAD_ACTION, SimulationAction.DRONE_UNLOAD_ACTION,
            SimulationAction.ACTIVATE_MICRO_HUB, SimulationAction.DEACTIVATE_MICRO_HUB
        }
        self.NM_ACTIONS = {
            SimulationAction.TRUCK_TO_NODE, SimulationAction.RE_ROUTE_TRUCK_TO_NODE,
            SimulationAction.LAUNCH_DRONE, SimulationAction.DRONE_TO_CHARGING_STATION
        }

    def _setup_dispatch_mappings(self):
        """Creates mappings for local action values and required parameters."""
        self.SCM_ACTION_MAP = {
            SimulationAction.ACCEPT_ORDER: 0, SimulationAction.CANCEL_ORDER: 1,
            SimulationAction.ASSIGN_ORDER_TO_TRUCK: 2, SimulationAction.ASSIGN_ORDER_TO_DRONE: 3,
            SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB: 4
        }
        self.RM_ACTION_MAP = {
            SimulationAction.LOAD_TRUCK_ACTION: 0, SimulationAction.UNLOAD_TRUCK_ACTION: 1,
            SimulationAction.DRONE_LOAD_ACTION: 2, SimulationAction.DRONE_UNLOAD_ACTION: 3,
            SimulationAction.ACTIVATE_MICRO_HUB: 4, SimulationAction.DEACTIVATE_MICRO_HUB: 5
        }
        self.NM_ACTION_MAP = {
            SimulationAction.TRUCK_TO_NODE: 0, SimulationAction.RE_ROUTE_TRUCK_TO_NODE: 1,
            SimulationAction.LAUNCH_DRONE: 2, SimulationAction.DRONE_TO_CHARGING_STATION: 3
        }

        self.PARAM_MAP = {
            SimulationAction.ACCEPT_ORDER: ['order_id'],
            SimulationAction.CANCEL_ORDER: ['order_id'],
            SimulationAction.ASSIGN_ORDER_TO_TRUCK: ['order_id', 'truck_id'],
            SimulationAction.ASSIGN_ORDER_TO_DRONE: ['order_id', 'drone_id'],
            SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB: ['order_id', 'micro_hub_id'],
            SimulationAction.LOAD_TRUCK_ACTION: ['truck_id', 'order_id'],
            SimulationAction.UNLOAD_TRUCK_ACTION: ['truck_id', 'order_id'],
            SimulationAction.DRONE_LOAD_ACTION: ['drone_id', 'order_id'],
            SimulationAction.DRONE_UNLOAD_ACTION: ['drone_id', 'order_id'],
            SimulationAction.ACTIVATE_MICRO_HUB: ['micro_hub_id'],
            SimulationAction.DEACTIVATE_MICRO_HUB: ['micro_hub_id'],
            SimulationAction.TRUCK_TO_NODE: ['truck_id', 'destination_node_id'],
            SimulationAction.RE_ROUTE_TRUCK_TO_NODE: ['truck_id', 'new_destination_node_id'],
            SimulationAction.LAUNCH_DRONE: ['drone_id', 'order_id'],
            SimulationAction.DRONE_TO_CHARGING_STATION: ['drone_id', 'charging_station_id']
        }

    def execute_action(self, action_tuple: Tuple, current_mask) -> bool:
        """
        Decodes the global action tuple and dispatches it to the correct manager system.
        """
        action_type_enum = action_tuple[0]
        params = self._get_params_from_tuple(action_tuple)


        if action_type_enum in self.SCM_ACTIONS:
            action_value = self.SCM_ACTION_MAP[action_type_enum]
                # REFACTORED: Use LogisticsAction
            scm_action = LogisticsAction(p_action_space=self.supply_chain_manager.get_action_space(),
                                             p_values=[action_value],
                                             **params)
            return self.supply_chain_manager.process_action(scm_action)

        elif action_type_enum in self.RM_ACTIONS:
            action_value = self.RM_ACTION_MAP[action_type_enum]
            # REFACTORED: Use LogisticsAction
            rm_action = LogisticsAction(p_action_space=self.resource_manager.get_action_space(),
                                            p_values=[action_value],
                                            **params)
            return self.resource_manager.process_action(rm_action)

        elif action_type_enum in self.NM_ACTIONS:
            action_value = self.NM_ACTION_MAP[action_type_enum]
                # REFACTORED: Use LogisticsAction
            nm_action = LogisticsAction(p_action_space=self.network_manager.get_action_space(),
                                            p_values=[action_value],
                                            **params)
            return self.network_manager.process_action(nm_action)


        # print(f"ActionManager dispatch error for action {action_tuple}: {e}")

        return False


    def _get_params_from_tuple(self, action_tuple: Tuple) -> Dict[str, Any]:
        """Creates a kwargs dictionary from the action tuple using the PARAM_MAP."""
        action_type = action_tuple[0]
        param_names = self.PARAM_MAP.get(action_type, [])

        params = {}
        for i, param_name in enumerate(param_names):
            params[param_name] = action_tuple[i + 1]

        return params
