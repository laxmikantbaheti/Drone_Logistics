import numpy as np
from typing import Dict, Any, Tuple, Set, List
from collections import defaultdict
from pprint import pprint

# Local Imports
from ddls_src.actions.base import SimulationActions, Constraint, ActionIndex
from mlpro.bf.systems import System
from mlpro.bf.events import Event


# Forward declarations
class GlobalState: pass


# -------------------------------------------------------------------------
# -- StateActionMapper (Now Fully Self-Configuring)
# -------------------------------------------------------------------------

class StateActionMapper:
    def __init__(self, global_state: 'GlobalState', action_map: Dict[Tuple, int]):
        self.global_state = global_state
        self.action_map = action_map
        self.action_index = ActionIndex(global_state, action_map)
        self._invalidation_map: Dict[Tuple, Set[int]] = {}

        print("StateActionMapper: Building invalidation map from blueprint...")
        self._build_map()
        print("StateActionMapper: Invalidation map built successfully.")

    def _build_map(self):
        """
        Builds the map by discovering all unique constraint classes from the
        SimulationActions blueprint, instantiating them, and merging their results.
        """
        self._invalidation_map.clear()

        # 1. Discover all unique constraint classes from the blueprint using a set
        unique_constraint_classes = set()
        for action in SimulationActions.get_all_actions():
            unique_constraint_classes.update(action.constraints)

        # 2. Instantiate each unique constraint and get its full invalidation map
        for constraint_class in unique_constraint_classes:
            constraint_instance = constraint_class()
            rule_map = constraint_instance.get_invalidations(self.global_state, self.action_index)

            # 3. Merge the results into the main map
            for state_tuple, action_indices in rule_map.items():
                if state_tuple not in self._invalidation_map:
                    self._invalidation_map[state_tuple] = set()
                self._invalidation_map[state_tuple].update(action_indices)

    def update_for_new_order(self, order_id: int):
        """Dynamically updates the maps and indexes for a newly arrived order."""
        self.action_index.add_actions_for_new_order(order_id, self.action_map)
        print("StateActionMapper: Re-building invalidation map to include new order...")
        self._build_map()
        print("StateActionMapper: Invalidation map updated.")

    def handle_new_order_event(self, p_event_id, p_event_object: Event):
        """
        Event handler method that subscribes to 'NEW_ORDER_CREATED' events.
        """
        print(f"StateActionMapper: Received event '{p_event_id}'.")
        new_order = p_event_object.get_data().get('order')
        if new_order:
            self.update_for_new_order(new_order.get_id())

    def generate_mask(self) -> np.ndarray:
        mask = np.ones(len(self.action_map), dtype=bool)
        invalid_indices = set()

        all_entities = list(self.global_state.orders.values()) + \
                       list(self.global_state.trucks.values()) + \
                       list(self.global_state.drones.values())

        for entity in all_entities:
            entity_type_name = entity.__class__.__name__.replace("Mock", "")
            state_tuple = (entity_type_name, entity.get_id(), 'status', entity.status)
            invalid_indices.update(self._invalidation_map.get(state_tuple, set()))

            if hasattr(entity, 'max_payload_capacity') and hasattr(entity, 'cargo_manifest') and len(
                    entity.cargo_manifest) >= entity.max_payload_capacity:
                capacity_state_tuple = (entity_type_name, entity.get_id(), 'capacity', 'full')
                invalid_indices.update(self._invalidation_map.get(capacity_state_tuple, set()))

        if invalid_indices:
            valid_indices_to_update = [idx for idx in invalid_indices if idx < len(mask)]
            if valid_indices_to_update:
                mask[list(valid_indices_to_update)] = False

        return mask


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Create a more comprehensive set of Mock Objects for the test
    class MockEntity(System):
        def __init__(self, p_id, status, cargo_count=0, capacity=1):
            super().__init__(p_id=p_id)
            self.status = status
            self.cargo_manifest = [0] * cargo_count
            self.max_payload_capacity = capacity

        @staticmethod
        def setup_spaces(): return None, None


    class MockGlobalState:
        def __init__(self):
            self.orders = {
                0: MockEntity(p_id=0, status='pending'),
                1: MockEntity(p_id=1, status='delivered')
            }
            self.trucks = {
                101: MockEntity(p_id=101, status='idle', cargo_count=0, capacity=2),
                102: MockEntity(p_id=102, status='en_route'),
                103: MockEntity(p_id=103, status='idle', cargo_count=2, capacity=2)  # This truck is full
            }
            self.drones = {}
            self.micro_hubs = {}


    mock_gs = MockGlobalState()

    # Create a more comprehensive mock action map for the test
    mock_action_map = {
        (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 101): 0,
        (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 1, 101): 1,
        (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 102): 2,
        (SimulationActions.ASSIGN_ORDER_TO_TRUCK, 0, 103): 3,
    }

    print("--- Validating Self-Configuring StateActionMapper (Large Instance) ---")

    # 2. Instantiate the StateActionMapper
    mapper = StateActionMapper(mock_gs, mock_action_map)

    print("\n[A] Generated Invalidation Map (Rulebook):")
    pprint(mapper._invalidation_map)

    # 3. Generate the mask and perform assertions
    print("\n[B] Generating Mask and Running Assertions...")
    final_mask = mapper.generate_mask()

    print(f"  - Final Mask: {final_mask.astype(int)}")

    # Assertions for Valid Actions
    assert final_mask[0] == True, "Test Case 1 FAILED: Assigning pending order 0 to idle truck 101 should be valid"
    print("  - PASSED: All expected valid actions are correctly marked as valid.")

    # Assertions for Invalid Actions
    assert final_mask[1] == False, "Test Case 2 FAILED: Assigning delivered order 1 should be invalid"
    assert final_mask[2] == False, "Test Case 3 FAILED: Assigning to busy truck 102 should be invalid"
    assert final_mask[3] == False, "Test Case 4 FAILED: Assigning to full truck 103 should be invalid"
    print("  - PASSED: All expected invalid actions are correctly marked as invalid.")

    print("\n--- Validation Complete ---")
