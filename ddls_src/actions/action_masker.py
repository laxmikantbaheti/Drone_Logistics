import numpy as np
from typing import Dict, Any, Tuple, Set, List

# Local Imports
from ..actions.action_enums import SimulationAction
from .state_action_mapper import StateActionMapper, Constraint


# Forward declarations
class GlobalState: pass


class ActionMasker:
    """
    A lightweight interface that uses a StateActionMapper to generate both
    system-level and agent-level action masks.
    """

    def __init__(self,
                 global_state: 'GlobalState',
                 action_map: Dict[Tuple, int],
                 constraints: List[Constraint],
                 agent_action_space_config: Dict):
        """
        Initializes the ActionMasker.

        Args:
            global_state (GlobalState): Reference to the central GlobalState.
            action_map (Dict[Tuple, int]): The global action map.
            constraints (List[Constraint]): A list of pluggable constraint objects.
            agent_action_space_config (Dict): Config for the agent's action space.
        """
        self.global_state = global_state
        self.action_map = action_map
        self.agent_action_space_config = agent_action_space_config

        # Instantiate the powerful StateActionMapper
        self.mapper = StateActionMapper(global_state, action_map, constraints)

        # Pre-calculate the agent-to-system action mapping for efficiency
        self._agent_to_system_map = self._build_agent_to_system_map()

        print("ActionMasker initialized.")

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

    def generate_system_mask(self) -> np.ndarray:
        """
        Generates the full, low-level mask for the entire system action space.
        This is a direct call to the StateActionMapper.
        """
        return self.mapper.generate_mask()

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
            if system_mask[system_idx]:
                agent_mask[agent_idx] = True

        return agent_mask
