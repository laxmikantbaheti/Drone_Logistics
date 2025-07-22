from typing import Dict, Any

# MLPro Imports
from mlpro.bf.systems import Action

# Local Imports
from ddls_src.actions.action_enums import SimulationAction


# Forward declarations
class GlobalState: pass


class SupplyChainManager: pass


class ActionManager:
    """
    Acts as a translator between the agent's simplified action space (the research design)
    and the framework's internal, hierarchical action system.
    """

    def __init__(self,
                 global_state: 'GlobalState',
                 supply_chain_manager: 'SupplyChainManager',
                 num_orders: int,
                 num_vehicles: int):

        self.global_state = global_state
        self.supply_chain_manager = supply_chain_manager

        self._num_orders_agent = num_orders
        self._num_vehicles_agent = num_vehicles
        self._vehicle_idx_to_id = self._create_vehicle_map()

        print("ActionManager (Translator) initialized.")

    def _create_vehicle_map(self) -> Dict[int, int]:
        """Creates a map from a simple index (0, 1, 2...) to the actual truck/drone ID."""
        vehicle_map = {}
        idx = 0
        # The order here (trucks then drones) must be consistent!
        for truck_id in sorted(self.global_state.trucks.keys()):
            vehicle_map[idx] = truck_id
            idx += 1
        for drone_id in sorted(self.global_state.drones.keys()):
            vehicle_map[idx] = drone_id
            idx += 1
        return vehicle_map

    def process_agent_action(self, agent_action_index: int) -> bool:
        """
        Translates a single integer from the agent's action space into a specific
        action for the SupplyChainManager and dispatches it.
        """
        try:
            # 1. Decode the agent's action index
            order_id = agent_action_index // self._num_vehicles_agent
            vehicle_idx = agent_action_index % self._num_vehicles_agent
            vehicle_id = self._vehicle_idx_to_id[vehicle_idx]
        except (KeyError, ZeroDivisionError):
            print(f"ActionManager: Invalid agent action index {agent_action_index}.")
            return False

        # 2. Determine the specific action and parameters
        params = {'order_id': order_id}
        scm_action_value = -1

        if vehicle_id in self.global_state.trucks:
            params['truck_id'] = vehicle_id
            scm_action_value = 2  # SCM action for ASSIGN_TO_TRUCK
        elif vehicle_id in self.global_state.drones:
            params['drone_id'] = vehicle_id
            scm_action_value = 3  # SCM action for ASSIGN_TO_DRONE
        else:
            return False

        # 3. Create and dispatch the formal MLPro Action
        scm_action = Action(p_action_space=self.supply_chain_manager.get_action_space(),
                            p_values=[scm_action_value],
                            **params)

        print(
            f"ActionManager: Agent action {agent_action_index} -> Dispatching SCM action for Order {order_id} to Vehicle {vehicle_id}")
        return self.supply_chain_manager.process_action(scm_action)
