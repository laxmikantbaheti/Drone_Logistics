import itertools
from ddls_src.actions.base import SimulationAction
from typing import Dict, Tuple, Any


# Forward declaration for type hinting
class GlobalState: pass


def generate_action_map(global_state: 'GlobalState') -> Tuple[Dict[Tuple, int], int]:
    """
    Programmatically generates the global flattened action map and action space size
    at runtime based on the entities that actually exist in the global_state.

    Args:
        global_state (GlobalState): The fully populated global state object for the current scenario.

    Returns:
        Tuple[Dict[Tuple, int], int]: A tuple containing the generated ACTION_MAP and the ACTION_SPACE_SIZE.
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
        # Add other entity types or special ranges as needed
        'int': [1, 2, 3],  # Example for priority
        'str': ['CHARGING', 'SORTING']  # Example for service type
    }

    # 2. Iterate through each action defined in our blueprint
    for action_enum in SimulationAction:
        if not action_enum.params:
            # Handle actions with no parameters, like NO_OPERATION
            action_tuple = (action_enum,)
            if action_tuple not in action_map:
                action_map[action_tuple] = current_index
                current_index += 1
            continue

        # 3. Get the dynamic ranges for each parameter for this action
        param_ranges = []
        for param in action_enum.params:
            entity_type = param['type']
            ids = entity_id_ranges.get(entity_type, [])
            if not ids:
                param_ranges = []
                break
            param_ranges.append(ids)

        if not param_ranges:
            continue

        # 4. Generate all unique combinations of parameter values
        param_combinations = list(itertools.product(*param_ranges))

        for combo in param_combinations:
            action_tuple = (action_enum,) + combo
            if action_tuple not in action_map:
                action_map[action_tuple] = current_index
                current_index += 1

    action_space_size = len(action_map)
    return action_map, action_space_size


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # This block requires a mock global_state to run

    class MockEntity:
        def __init__(self, id):
            self.id = id


    class MockGlobalState:
        def __init__(self):
            self.orders = {0: MockEntity(0), 1: MockEntity(1)}
            self.trucks = {101: MockEntity(101), 102: MockEntity(102)}
            self.drones = {201: MockEntity(201)}
            self.nodes = {i: MockEntity(i) for i in range(5)}
            self.micro_hubs = {0: MockEntity(0)}


    mock_gs = MockGlobalState()

    print("--- Validating Dynamic Action Map Generator ---")

    ACTION_MAP, ACTION_SPACE_SIZE = generate_action_map(mock_gs)

    print(f"Successfully generated ACTION_MAP with {len(ACTION_MAP)} entries for the mock state.")
    print(f"Calculated ACTION_SPACE_SIZE: {ACTION_SPACE_SIZE}")

    print("\nExample entries:")
    for i, (k, v) in enumerate(ACTION_MAP.items()):
        if i >= 10: break
        print(f"  Index {v}: {k[0].name}{k[1:]}")
