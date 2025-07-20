import numpy as np
from typing import Dict, Any, List, Tuple, Callable


# Forward declaration for LogisticsSimulation to avoid circular dependency
# This will be replaced by an actual import once LogisticsSimulation is fully defined
class LogisticsSimulation:
    pass


# Forward declaration for AgentActionType, which will be defined in action_enums.py
# For now, we'll assume NO_OPERATION is a tuple like ("NO_OP",)
NO_OPERATION = ("NO_OP",)


class ActionMasker:
    """
    Generates a boolean mask for the action space based on the current simulation state
    and predefined constraints. This class ensures that the RL agent only selects
    valid actions.
    """

    def __init__(self, simulation_instance: 'LogisticsSimulation', action_space_size: int,
                 action_map: Dict[Tuple, int]):
        """
        Initializes the ActionMasker with a reference to the main simulation,
        the total size of the flattened action space, and the action mapping.

        Args:
            simulation_instance (LogisticsSimulation): Reference to the main simulation instance.
                                                       Used to query GlobalState and other managers.
            action_space_size (int): The total number of possible flattened actions.
            action_map (Dict[Tuple, int]): A mapping from action tuple (e.g., (ActionType.MOVE, 1, 2))
                                           to its flattened integer index in the action space.
        """
        self.simulation_instance = simulation_instance
        self.action_space_size = action_space_size
        self.action_map = action_map
        # A list of callable functions, each representing a specific constraint rule.
        # Each function takes the LogisticsSimulation instance and returns a boolean numpy array
        # of the same shape as the action space, where True means valid and False means invalid.
        self.constraint_rules: List[Callable[[LogisticsSimulation], np.ndarray]] = []

        # Register all constraint rule methods during initialization
        self._register_all_default_constraints()

        print("ActionMasker initialized.")

    def _register_all_default_constraints(self) -> None:
        """
        Registers all default constraint rule methods.
        This method should be updated as new constraint rules are implemented.
        """
        # Order-related constraints
        self._register_constraint(self._mask_order_actions)
        self._register_constraint(self._mask_assign_actions)
        self._register_constraint(self._mask_consolidate_actions)

        # Truck-related constraints
        self._register_constraint(self._mask_truck_movement_actions)
        self._register_constraint(self._mask_truck_load_unload_actions)

        # Drone-related constraints
        self._register_constraint(self._mask_drone_launch_actions)
        self._register_constraint(self._mask_drone_load_unload_actions)
        self._register_constraint(self._mask_drone_landing_actions)
        self._register_constraint(self._mask_drone_charging_actions)

        # Micro-hub related constraints
        self._register_constraint(self._mask_micro_hub_actions)

        # Maintenance constraints
        self._register_constraint(self._mask_maintenance_actions)

        # Always valid action
        self._register_constraint(self._mask_no_operation_action)

    def _register_constraint(self, constraint_func: Callable[[LogisticsSimulation], np.ndarray]) -> None:
        """
        Adds a new constraint rule function to the list of active constraints.

        Args:
            constraint_func (Callable[[LogisticsSimulation], np.ndarray]): A function that takes
                                                                          the LogisticsSimulation instance
                                                                          and returns a boolean numpy array
                                                                          representing the mask for that rule.
        """
        self.constraint_rules.append(constraint_func)

    def _apply_mask_for_action(self, action_tuple: Tuple, is_valid: bool, current_mask: np.ndarray) -> None:
        """
        Helper to set a specific action's validity in the mask.

        Args:
            action_tuple (Tuple): The action tuple (e.g., (AgentActionType.TRUCK_TO_NODE, 1, 5)).
            is_valid (bool): True if the action is valid, False otherwise.
            current_mask (np.ndarray): The numpy array representing the current composite mask.
        """
        flattened_index = self.action_map.get(action_tuple)
        if flattened_index is not None and 0 <= flattened_index < self.action_space_size:
            current_mask[flattened_index] = is_valid
        # else:
        # print(f"Warning: Action tuple {action_tuple} not found in action_map or index out of bounds.")

    def generate_mask(self) -> np.ndarray:
        """
        Computes and returns the composite boolean mask by applying all registered
        constraint rules. The final mask is the logical AND of all individual rule masks.

        Returns:
            np.ndarray: A boolean numpy array of shape (action_space_size,),
                        where True indicates a valid action and False an invalid one.
        """
        # Initialize a mask where all actions are initially considered valid
        composite_mask = np.ones(self.action_space_size, dtype=bool)

        # Apply each registered constraint rule
        for constraint_func in self.constraint_rules:
            rule_mask = constraint_func(self.simulation_instance)
            # Ensure rule_mask has the correct shape
            if rule_mask.shape != composite_mask.shape: #TODO: This is not correct
                raise ValueError(f"Constraint rule {constraint_func.__name__} returned a mask "
                                 f"of shape {rule_mask.shape}, expected {composite_mask.shape}.")
            composite_mask = np.logical_and(composite_mask, rule_mask)

        # print(f"Generated composite mask (first 10 elements): {composite_mask[:10]}") # For debugging
        return composite_mask

    # --- Placeholder Constraint Rule Functions (to be implemented in Step 11) ---
    # Each of these functions will query the GlobalState via self.simulation_instance.global_state
    # and return a boolean numpy array for their specific set of actions.

    def _mask_order_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for order-related validity (PRIORITIZE_ORDER, CANCEL_ORDER,
        FLAG_FOR_RE_DELIVERY, REASSIGN_ORDER).
        Masks if order_ID is invalid or not in expected status.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Example:
        # global_state = sim.get_current_global_state()
        # for order_id, order in global_state.get_all_entities("orders").items():
        #     if order.status == 'delivered':
        #         # Mark actions related to this delivered order as invalid
        #         # This requires iterating through action_map to find relevant actions
        #         # For now, this is a placeholder.
        #         pass
        return mask

    def _mask_assign_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for assignment validity (ASSIGN_TO_TRUCK, ASSIGN_TO_DRONE, ASSIGN_TO_MICRO_HUB).
        Masks if vehicle/hub is unavailable or order already assigned.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_consolidate_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for consolidation validity (CONSOLIDATE_FOR_TRUCK/DRONE).
        Masks if vehicle is not at a consolidation point or no eligible orders.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_truck_movement_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for truck movement validity (TRUCK_TO_NODE, RE_ROUTE_TRUCK_TO_NODE).
        Masks if truck is busy, under maintenance, or destination is invalid.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_truck_load_unload_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for truck load/unload validity (LOAD_TRUCK, UNLOAD_TRUCK).
        Masks if truck not at valid load/unload node, or no cargo to unload/no space to load.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_drone_launch_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for drone launch validity (LAUNCH_DRONE).
        Masks if drone not at base, base inactive/blocked, drone battery low, or no order assigned.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_drone_load_unload_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for drone load/unload validity (DRONE_LOAD, DRONE_UNLOAD).
        Masks if drone not at valid load/unload point, or no cargo/space.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_drone_landing_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for drone landing validity (DRONE_LANDING).
        Masks if drone not in flight or no valid landing zone.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_drone_charging_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint rule for drone charging validity (DRONE_TO_CHARGING_STATION, DRONE_CHARGE).
        Masks if drone not in flight/at base, or no available slot.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_micro_hub_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint for micro-hub activation/deactivation (ACTIVATE/DEACTIVATE_MICRO_HUB,
        ADD_TO_CHARGING_QUEUE) based on hub status/capacity.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_maintenance_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Constraint for maintenance actions (FLAG_VEHICLE_FOR_MAINTENANCE,
        FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB) based on current status.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        # Implementation logic here
        return mask

    def _mask_no_operation_action(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Ensures the NO_OPERATION action is always valid.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        no_op_index = self.action_map.get(NO_OPERATION)
        if no_op_index is not None:
            mask[no_op_index] = True  # Explicitly ensure NO_OPERATION is True
        return mask

