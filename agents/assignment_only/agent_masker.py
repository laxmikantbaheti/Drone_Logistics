import numpy as np
from typing import Dict, Any, Tuple

# Local Imports
from ddls_src.actions.action_enums import SimulationAction

# Forward declarations
class GlobalState: pass

class AgentMasker:
    """
    A dedicated masker for the "assignment-only" agent. It is responsible for:
    1. Creating the mapping from the agent's simplified action space to the
       global system action space.
    2. Generating the final agent-facing action mask from the global system mask.
    """
    def __init__(self,
                 global_state: 'GlobalState',
                 action_map: Dict[Tuple, int],
                 agent_action_space_config: Dict):

        self.global_state = global_state
        self.action_map = action_map
        self.agent_action_space_config = agent_action_space_config

        # Pre-calculate the agent-to-system action mapping for efficiency
        self._agent_to_system_map = self._build_agent_to_system_map()
        print("AgentMasker (for assignment-only agent) initialized.")

    def _build_agent_to_system_map(self) -> Dict[int, int]:
        """
        Creates a fast lookup table from an agent action index to a global system action index.
        """
        mapping = {}
        num_orders = self.agent_action_space_config['num_orders']
        num_vehicles = self.agent_action_space_config['num_vehicles']
        vehicle_map = self.agent_action_space_config['vehicle_map']

        agent_action_space_size = num_orders * num_vehicles
        if agent_action_space_size == 0:
            return {}

        for agent_idx in range(agent_action_space_size):
            order_id = agent_idx // num_vehicles
            vehicle_idx = agent_idx % num_vehicles
            vehicle_id = vehicle_map.get(vehicle_idx)

            if vehicle_id is None: continue

            action_enum = SimulationAction.ASSIGN_ORDER_TO_TRUCK if vehicle_id in self.global_state.trucks else SimulationAction.ASSIGN_ORDER_TO_DRONE
            global_tuple = (action_enum, order_id, vehicle_id)
            global_idx = self.action_map.get(global_tuple)

            if global_idx is not None:
                mapping[agent_idx] = global_idx

        return mapping

    def generate_agent_mask(self, system_mask: np.ndarray) -> np.ndarray:
        """
        Generates the high-level mask for the agent's simplified action space,
        derived from the full system mask using the pre-computed map.
        """
        agent_action_space_size = len(self._agent_to_system_map)
        if agent_action_space_size == 0:
            return np.array([], dtype=bool)

        agent_mask = np.zeros(agent_action_space_size, dtype=bool)

        for agent_idx, system_idx in self._agent_to_system_map.items():
            # Ensure the system index is within the bounds of the current system mask
            if system_idx < len(system_mask) and system_mask[system_idx]:
                agent_mask[agent_idx] = True

        return agent_mask
