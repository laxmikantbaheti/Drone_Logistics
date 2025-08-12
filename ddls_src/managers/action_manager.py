from typing import Dict, Any, Tuple

# Local Imports
from ddls_src.actions.base import SimulationActions, ActionType
from ddls_src.core.basics import LogisticsAction
from mlpro.bf.math import MSpace, Dimension  # For validation block


# Forward declarations
class GlobalState: pass


class ActionMasker: pass


class SupplyChainManager: pass


class ResourceManager: pass


class NetworkManager: pass


class ActionManager:
    """
    Receives a global action tuple, validates it, and routes it to the
    appropriate high-level manager system. It self-configures its dispatching
    logic by reading the action blueprints from actions/base.py.
    """

    def __init__(self,
                 global_state: 'GlobalState',
                 managers: Dict[str, Any],
                 action_map: Dict[Tuple, int]):

        self.global_state = global_state
        self.action_map = action_map
        self._reverse_action_map: Dict[int, Tuple] = {idx: act_tuple for act_tuple, idx in action_map.items()}

        # Store references to the manager systems
        self._managers = managers

        # Self-configure the dispatch and parameter maps from the blueprint
        self._dispatch_map: Dict[ActionType, Any] = {}
        self._param_map: Dict[ActionType, Any] = {}
        self._build_maps()

        print("ActionManager (Self-Configuring) initialized.")

    def _build_maps(self):
        """
        Programmatically builds the internal dispatch and parameter maps by
        reading the SimulationActions blueprint.
        """
        for action in SimulationActions.get_all_actions():
            handler_name = action.handler
            if handler_name and handler_name in self._managers:
                self._dispatch_map[action] = self._managers[handler_name]
                self._param_map[action] = [p['name'] for p in action.params]

    def execute_action(self, action_tuple: Tuple) -> bool:
        """
        Decodes the global action tuple and dispatches it to the correct manager system.
        """
        action_type = action_tuple[0]

        # 1. Find the target manager from the self-configured dispatch map
        target_manager = self._dispatch_map.get(action_type)
        if not target_manager:
            # self.log(self.C_LOG_TYPE_W, f"No handler defined or found for action type {action_type.name}")
            return False

        # 2. Create the parameter dictionary from the self-configured param map
        param_names = self._param_map.get(action_type, [])
        params = {name: value for name, value in zip(param_names, action_tuple[1:])}

        # 3. Create the LogisticsAction object for the target manager
        manager_action_space = target_manager.get_action_space()

        # We pass the global action's ID as the value for the manager's action space
        action_obj = LogisticsAction(p_action_space=manager_action_space,
                                     p_values=[action_type.id],
                                     **params)

        # 4. Dispatch the action
        try:
            return target_manager.process_action(action_obj)
        except Exception as e:
            print(f"ActionManager dispatch error for action {action_tuple}: {e}")
            return False


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # This block requires mock managers to run

    class MockManager:
        def __init__(self, name):
            self.name = name

        def get_action_space(self):
            space = MSpace()
            space.add_dim(Dimension(p_name_short="mock_action_dimension"))
            return space

        def process_action(self, action):
            print(
                f"  - MockManager '{self.name}' received action with ID {action.get_sorted_values()[0]} and data {action.data}")
            return True


    mock_managers = {
        "SupplyChainManager": MockManager("SCM"),
        "ResourceManager": MockManager("RM"),
        "NetworkManager": MockManager("NM")
    }

    # A small, representative action_map for this demo
    mock_action_map = {
        (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 101): 0,
        (SimulationActions.TRUCK_TO_NODE, 101, 5): 1
    }

    print("--- Validating Self-Configuring ActionManager ---")

    # 1. Instantiate the ActionManager
    action_manager = ActionManager(global_state=None, managers=mock_managers, action_map=mock_action_map)

    print("\n[A] Internal Dispatch Map (auto-generated):")
    for k, v in action_manager._dispatch_map.items():
        print(f"  - Action '{k.name}' -> Handler '{v.name}'")

    # 2. Demonstrate dispatching
    print("\n[B] Dispatching Demo:")

    print("\n  - Dispatching an ASSIGN_ORDER_TO_TRUCK action...")
    action_manager.execute_action((SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 101))

    print("\n  - Dispatching a TRUCK_TO_NODE action...")
    action_manager.execute_action((SimulationActions.TRUCK_TO_NODE, 101, 5))

    print("\n--- Validation Complete ---")
