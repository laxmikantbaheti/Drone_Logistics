from enum import Enum, auto
from typing import List, Dict, Any, Callable, Tuple


# -------------------------------------------------------------------------
# -- Placeholder Constraint Functions
# -------------------------------------------------------------------------
def check_order_assignable(global_state: Any, order_id: int) -> bool: return True


def check_vehicle_available(global_state: Any, vehicle_id: int) -> bool: return True


def check_vehicle_capacity(global_state: Any, vehicle_id: int) -> bool: return True


def check_vehicle_at_node(global_state: Any, vehicle_id: int) -> bool: return True


def check_order_in_cargo(global_state: Any, vehicle_id: int, order_id: int) -> bool: return True


def check_order_at_node(global_state: Any, node_id: int, order_id: int) -> bool: return True


def check_drone_battery_for_flight(global_state: Any, drone_id: int) -> bool: return True


def check_hub_is_active(global_state: Any, hub_id: int) -> bool: return True


# -------------------------------------------------------------------------
# -- SimulationAction Enum: The Central Source of Truth
# -------------------------------------------------------------------------
# This rich Enum is the single source of truth for all action-related
# information. It defines the structure of actions, but not their specific instances.
# -------------------------------------------------------------------------

class SimulationAction(Enum):

    def __init__(self,
                 id_val: int,
                 params: List[Dict[str, Any]],
                 constraints: List[Callable],
                 is_automatic: bool,
                 handler: str):
        self._id_val = id_val
        self.params = params
        self.constraints = constraints
        self.is_automatic = is_automatic
        self.handler = handler

    @property
    def id(self):
        return self.value[0]

    # --- Action Definitions ---
    # The 'range' key has been removed. The generator will now get the valid IDs
    # from the global_state at runtime.

    # -- Supply Chain Manager Actions --
    ACCEPT_ORDER = (auto(),
                    [{'name': 'order_id', 'type': 'Order'}],
                    [check_order_assignable], False, "SupplyChainManager"
                    )
    ASSIGN_ORDER_TO_TRUCK = (auto(),
                             [{'name': 'order_id', 'type': 'Order'}, {'name': 'truck_id', 'type': 'Truck'}],
                             [check_order_assignable, check_vehicle_available, check_vehicle_capacity], False,
                             "SupplyChainManager"
                             )
    ASSIGN_ORDER_TO_DRONE = (auto(),
                             [{'name': 'order_id', 'type': 'Order'}, {'name': 'drone_id', 'type': 'Drone'}],
                             [check_order_assignable, check_vehicle_available, check_vehicle_capacity], False,
                             "SupplyChainManager"
                             )
    TRUCK_TO_NODE = (auto(),
                     [{'name': 'truck_id', 'type': 'Truck'}, {'name': 'destination_node_id', 'type': 'Node'}],
                     [check_vehicle_available], True, "NetworkManager"
                     )

    # ... (definitions for all other actions would follow the same pattern) ...

    NO_OPERATION = (auto(), [], [], False, None)


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Validating Action Blueprints ---")

    for action in SimulationAction:
        print(f"\nAction: {action.name}")
        print(f"  - ID: {action.id}")
        print(f"  - Handler: {action.handler}")
        print(f"  - Is Automatic: {action.is_automatic}")

        print("  - Parameters:")
        if action.params:
            for param in action.params:
                print(f"    - {param['name']} (Type: {param['type']})")
        else:
            print("    - None")

        print("  - Constraints:")
        if action.constraints:
            for constraint_func in action.constraints:
                print(f"    - {constraint_func.__name__}")
        else:
            print("    - None")

    print("\n--- Validation Complete ---")
