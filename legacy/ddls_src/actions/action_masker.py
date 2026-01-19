import numpy as np
from typing import Dict, Any, Tuple, Set, List
import unittest
from unittest.mock import MagicMock, patch


# --- Mocking necessary external dependencies for testing ---
# In a real project, these would be imported from their respective files.

class SimulationAction:
    ASSIGN_ORDER_TO_TRUCK = "assign_order_to_truck"
    ASSIGN_ORDER_TO_DRONE = "assign_order_to_drone"

class Constraint:
    """Mock Constraint class for testing purposes."""
    def __init__(self, name: str = "MockConstraint"):
        self.name = name

    def is_action_allowed(self, global_state: Any, action_tuple: Tuple) -> bool:
        """Mock method for constraint checking."""
        # For testing, let's allow all actions by default unless specified otherwise
        return True

class GlobalState:
    """Mock GlobalState class for testing purposes."""
    def __init__(self, trucks: Set = None, drones: Set = None):
        self.trucks = trucks if trucks is not None else set()
        self.drones = drones if drones is not None else set()
        self.orders = {} # Mock orders, if needed for future tests
        self.vehicles = {} # Mock vehicles, if needed for future tests

class StateActionMapper:
    """Mock StateActionMapper class for testing purposes."""
    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int], constraints: List[Constraint]):
        self.global_state = global_state
        self.action_map = action_map
        self.constraints = constraints

    def generate_mask(self) -> np.ndarray:
        """
        Mock implementation of generate_mask.
        For simplicity in testing, let's assume all actions are valid initially,
        then apply mock constraints or specific test logic.
        """
        mask = np.ones(len(self.action_map), dtype=bool)
        # In a real scenario, this would apply actual constraint logic.
        # For this test, we'll rely on the test case to define the expected system mask.
        return mask

# --- Original ActionMasker Script (as provided by the user) ---

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


# --- Test Script ---
if __name__ == "__main__":
    print("--- Starting ActionMasker Demonstration ---")

    # 1. Set up Mock Global State
    # Imagine we have two trucks and one drone available.
    mock_global_state = GlobalState(trucks={"truck_1", "truck_2"}, drones={"drone_1"})
    print(f"\nMock Global State: Trucks={mock_global_state.trucks}, Drones={mock_global_state.drones}")

    # 2. Define a sample Global Action Map
    # This maps specific (action_type, order_id, vehicle_id) tuples to a global system index.
    global_action_map = {
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, "truck_1"): 0,  # System Action 0
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, "truck_2"): 1,  # System Action 1
        (SimulationAction.ASSIGN_ORDER_TO_DRONE, 0, "drone_1"): 2,  # System Action 2
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, "truck_1"): 3,  # System Action 3
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 1, "truck_2"): 4,  # System Action 4
        (SimulationAction.ASSIGN_ORDER_TO_DRONE, 1, "drone_1"): 5,  # System Action 5
    }
    print("\nGlobal Action Map:")
    for action_tuple, idx in global_action_map.items():
        print(f"  {action_tuple} -> Global Index {idx}")

    # 3. Define Constraints (Mocked for this example)
    mock_constraints = [Constraint("TimeWindowConstraint"), Constraint("CapacityConstraint")]
    print(f"\nMock Constraints: {[c.name for c in mock_constraints]}")

    # 4. Configure Agent's Action Space
    # An agent might perceive actions as (order, vehicle_slot).
    # Here, 'vehicle_map' converts the agent's vehicle_idx to a real vehicle_id.
    agent_action_space_config = {
        'num_orders': 2,    # Agent can consider assigning 2 orders (Order 0, Order 1)
        'num_vehicles': 3,  # Agent can consider 3 vehicle slots (Vehicle 0, Vehicle 1, Vehicle 2)
        'vehicle_map': {0: "truck_1", 1: "truck_2", 2: "drone_1"} # Mapping of agent's vehicle_idx to actual vehicle_id
    }
    print(f"\nAgent Action Space Configuration:")
    for k, v in agent_action_space_config.items():
        print(f"  {k}: {v}")

    # 5. Initialize ActionMasker
    action_masker = ActionMasker(
        global_state=mock_global_state,
        action_map=global_action_map,
        constraints=mock_constraints,
        agent_action_space_config=agent_action_space_config
    )

    # 6. Generate System Mask
    # In a real scenario, this would apply complex logic from StateActionMapper.
    # For this demo, let's manually define a system mask to show how it affects the agent mask.
    # True means allowed, False means disallowed.
    # Let's say:
    # (Order 0, truck_1) is allowed (idx 0)
    # (Order 0, truck_2) is DISALLOWED (idx 1) - e.g., truck_2 is broken
    # (Order 0, drone_1) is allowed (idx 2)
    # (Order 1, truck_1) is DISALLOWED (idx 3) - e.g., order 1 is too big for truck_1
    # (Order 1, truck_2) is allowed (idx 4)
    # (Order 1, drone_1) is DISALLOWED (idx 5) - e.g., drone_1 battery low
    mock_system_mask = np.array([True, False, True, False, True, False], dtype=bool)
    print(f"\nGenerated System Mask (True = Allowed): {mock_system_mask}")
    print(f"  System actions: {[list(global_action_map.keys())[i] for i, val in enumerate(mock_system_mask) if val]}")

    # 7. Generate Agent Mask based on the System Mask
    agent_mask = action_masker.generate_agent_mask(mock_system_mask)
    print(f"\nGenerated Agent Mask (True = Allowed): {agent_mask}")

    # 8. Interpret the Agent Mask
    print("\nInterpretation of Agent Mask (Agent's Perspective):")
    # Mapping agent index back to (order_id, vehicle_id) for clarity
    for agent_idx, is_allowed in enumerate(agent_mask):
        order_id = agent_idx // agent_action_space_config['num_vehicles']
        vehicle_idx = agent_idx % agent_action_space_config['num_vehicles']
        vehicle_id = agent_action_space_config['vehicle_map'].get(vehicle_idx, "UNKNOWN_VEHICLE")

        # Find the corresponding global action tuple for better context
        action_type = SimulationAction.ASSIGN_ORDER_TO_TRUCK if vehicle_id in mock_global_state.trucks else \
                      (SimulationAction.ASSIGN_ORDER_TO_DRONE if vehicle_id in mock_global_state.drones else "UNKNOWN_TYPE")
        global_action_tuple = (action_type, order_id, vehicle_id)
        global_idx = global_action_map.get(global_action_tuple, "N/A")

        status = "ALLOWED" if is_allowed else "DISALLOWED"
        print(f"  Agent Action {agent_idx} (Order {order_id}, Vehicle {vehicle_idx}/{vehicle_id}, "
              f"System Index {global_idx}): {status}")

    print("\n--- ActionMasker Demonstration Complete ---")