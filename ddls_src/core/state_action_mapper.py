import numpy as np
from typing import Dict, Any, Tuple, Set, List
from collections import defaultdict
from pprint import pprint

# Local Imports
from ddls_src.actions.base import SimulationAction
from ddls_src.actions.constraints.base import Constraint, OrderAssignableConstraint, VehicleAvailableConstraint


# Forward declarations
class GlobalState: pass


# -------------------------------------------------------------------------
# -- ActionIndex (The "Database")
# -------------------------------------------------------------------------

class ActionIndex:
    """
    Pre-processes the global action_map into a structured database of groups
    and subgroups for hyper-efficient lookups.
    """

    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        self.actions_by_type: Dict[SimulationAction, Set[int]] = defaultdict(set)
        self.actions_involving_entity: Dict[Tuple, Set[int]] = defaultdict(set)

        print("ActionIndex: Building action database...")
        self._build_indexes(global_state, action_map)
        print("ActionIndex: Database built successfully.")

    def _build_indexes(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        """Parses the action_map to create structured groups of action indices."""
        for action_tuple, action_index in action_map.items():
            action_type = action_tuple[0]

            # Index by action type
            self.actions_by_type[action_type].add(action_index)

            # Index by the entities involved in the action
            if not action_type.params: continue
            for i, param_def in enumerate(action_type.params):
                entity_type = param_def['type']
                entity_id = action_tuple[i + 1]
                self.actions_involving_entity[(entity_type, entity_id)].add(action_index)


# -------------------------------------------------------------------------
# -- StateActionMapper (Now Self-Configuring)
# -------------------------------------------------------------------------

class StateActionMapper:
    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int], constraints: List[Constraint]):
        self.global_state = global_state
        self.action_map = action_map
        self.constraints = constraints
        self.action_index = ActionIndex(global_state, action_map)
        self._invalidation_map: Dict[Tuple, Set[int]] = {}

        print("StateActionMapper: Building invalidation map from blueprint...")
        self._build_map()
        print("StateActionMapper: Invalidation map built successfully.")

    def _build_map(self):
        """
        Builds the map by iterating through the provided constraints and merging the results.
        """
        self._invalidation_map.clear()

        for constraint in self.constraints:
            rule_map = constraint.get_invalidations(self.global_state, self.action_index)
            for state_tuple, action_indices in rule_map.items():
                if state_tuple not in self._invalidation_map:
                    self._invalidation_map[state_tuple] = set()
                self._invalidation_map[state_tuple].update(action_indices)

    def generate_mask(self) -> np.ndarray:
        mask = np.ones(len(self.action_map), dtype=bool)
        invalid_indices = set()

        all_entities = list(self.global_state.orders.values()) + \
                       list(self.global_state.trucks.values()) + \
                       list(self.global_state.drones.values())

        for entity in all_entities:
            entity_type_name = entity.__class__.__name__.replace("Mock", "")
            state_tuple = (entity_type_name, entity.id, 'status', entity.status)
            invalid_indices.update(self._invalidation_map.get(state_tuple, set()))

        if invalid_indices:
            valid_indices_to_update = [idx for idx in invalid_indices if idx < len(mask)]
            if valid_indices_to_update:
                mask[list(valid_indices_to_update)] = False

        return mask


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Create Mock Objects for the test
    class MockEntity:
        def __init__(self, id, status):
            self.id = id
            self.status = status
            self.cargo_manifest = []
            self.max_payload_capacity = 2


    class MockGlobalState:
        def __init__(self):
            self.orders = {0: MockEntity(0, 'pending')}
            self.trucks = {101: MockEntity(101, 'idle')}
            self.drones = {}


    mock_gs = MockGlobalState()

    mock_action_map = {
        (SimulationAction.ASSIGN_ORDER_TO_TRUCK, 0, 101): 0
    }

    constraints_to_use = [
        OrderAssignableConstraint(),
        VehicleAvailableConstraint()
    ]

    print("--- Validating StateActionMapper ---")

    # 2. Instantiate the StateActionMapper
    mapper = StateActionMapper(mock_gs, mock_action_map, constraints_to_use)

    print("\n[A] Generated Invalidation Map (Rulebook):")
    pprint(mapper._invalidation_map)

    # 3. Demonstrate Mask Generation
    print("\n[B] Mask Generation Demo:")
    print(f"\n  - Initial State: Truck 101 is '{mock_gs.trucks[101].status}'")
    mask1 = mapper.generate_mask()
    print(f"  - Generated Mask: {mask1.astype(int)}")
    print(f"  - Action 0 (Assign Order 0 to Truck 101) is VALID: {mask1[0]}")

    # 4. Change the state and regenerate the mask
    print("\n  --- Changing state: Truck 101 becomes 'en_route' ---")
    mock_gs.trucks[101].status = 'en_route'

    print(f"\n  - New State: Truck 101 is '{mock_gs.trucks[101].status}'")
    mask2 = mapper.generate_mask()
    print(f"  - Generated Mask: {mask2.astype(int)}")
    print(f"  - Action 0 (Assign Order 0 to Truck 101) is now VALID: {mask2[0]}")

    print("\n--- Validation Complete ---")
