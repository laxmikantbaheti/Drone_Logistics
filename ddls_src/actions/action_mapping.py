from typing import Dict, Tuple, Any
from .action_enums import SimulationAction

# This dictionary maps each unique action tuple to a flattened integer index.
# The action tuple format is (SimulationAction.MAIN_ACTION, param1, param2, ...)
# The parameters can be entity IDs, specific sub-action enums, or other values.

ACTION_MAP: Dict[Tuple[Any, ...], int] = {}
_current_index = 0

def _add_action(action_tuple: Tuple[Any, ...]):
    """Helper function to add an action tuple to the map and assign an index."""
    global _current_index
    if action_tuple not in ACTION_MAP:
        ACTION_MAP[action_tuple] = _current_index
        _current_index += 1

# --- Define all possible action tuples and map them to indices ---

# 1. Order-Related Actions
# (order_id: int)
_ORDER_IDS = range(0, 10) # Example: Assuming up to 10 orders. Adjust as needed.
_PRIORITY_LEVELS = [1, 2, 3] # Example: Low, Medium, High priority

for order_id in _ORDER_IDS:
    _add_action((SimulationAction.ACCEPT_ORDER, order_id))
    for priority in _PRIORITY_LEVELS:
        _add_action((SimulationAction.PRIORITIZE_ORDER, order_id, priority))
    _add_action((SimulationAction.CANCEL_ORDER, order_id))
    _add_action((SimulationAction.FLAG_FOR_RE_DELIVERY, order_id))

# 2. Assignment Actions
# (order_id: int, truck_id: int)
# (order_id: int, drone_id: int)
# (order_id: int, micro_hub_id: int)
_TRUCK_IDS = range(0, 5) # Example: Assuming up to 5 trucks
_DRONE_IDS = range(0, 5) # Example: Assuming up to 5 drones
_MICRO_HUB_IDS = range(0, 3) # Example: Assuming up to 3 micro-hubs

for order_id in _ORDER_IDS:
    for truck_id in _TRUCK_IDS:
        _add_action((SimulationAction.ASSIGN_ORDER_TO_TRUCK, order_id, truck_id))
    for drone_id in _DRONE_IDS:
        _add_action((SimulationAction.ASSIGN_ORDER_TO_DRONE, order_id, drone_id))
    for micro_hub_id in _MICRO_HUB_IDS:
        _add_action((SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB, order_id, micro_hub_id))

# 3. Consolidation Actions
# (truck_id: int)
# (drone_id: int)
for truck_id in _TRUCK_IDS:
    _add_action((SimulationAction.CONSOLIDATE_FOR_TRUCK, truck_id))
for drone_id in _DRONE_IDS:
    _add_action((SimulationAction.CONSOLIDATE_FOR_DRONE, drone_id))

# 4. Reassignment Action
# (order_id: int, new_vehicle_id: int)
_ALL_VEHICLE_IDS = list(_TRUCK_IDS) + list(_DRONE_IDS) # Combine truck and drone IDs
for order_id in _ORDER_IDS:
    for vehicle_id in _ALL_VEHICLE_IDS:
        _add_action((SimulationAction.REASSIGN_ORDER, order_id, vehicle_id))

# 5. Truck Movement Actions
# (truck_id: int, destination_node_id: int)
_NODE_IDS = range(0, 20) # Example: Assuming up to 20 nodes
for truck_id in _TRUCK_IDS:
    for dest_node_id in _NODE_IDS:
        _add_action((SimulationAction.TRUCK_TO_NODE, truck_id, dest_node_id))
        _add_action((SimulationAction.RE_ROUTE_TRUCK_TO_NODE, truck_id, dest_node_id))

# 6. Truck Resource Actions (Loading/Unloading)
# (truck_id: int, order_id: int)
for truck_id in _TRUCK_IDS:
    for order_id in _ORDER_IDS:
        _add_action((SimulationAction.LOAD_TRUCK_ACTION, truck_id, order_id))
        _add_action((SimulationAction.UNLOAD_TRUCK_ACTION, truck_id, order_id))

# 7. Drone Movement Actions
# (drone_id: int, order_id: int) for LAUNCH_DRONE
# (drone_id: int) for DRONE_LANDING_ACTION
# (drone_id: int, charging_station_id: int) for DRONE_TO_CHARGING_STATION
for drone_id in _DRONE_IDS:
    for order_id in _ORDER_IDS:
        _add_action((SimulationAction.LAUNCH_DRONE, drone_id, order_id))
    _add_action((SimulationAction.DRONE_LANDING_ACTION, drone_id))
    for charging_station_id in _NODE_IDS: # Assuming charging stations are nodes
        _add_action((SimulationAction.DRONE_TO_CHARGING_STATION, drone_id, charging_station_id))

# 8. Drone Resource Actions (Loading/Unloading/Charging)
# (drone_id: int, order_id: int) for LOAD/UNLOAD
# (drone_id: int, delta_time: float) for CHARGE_ACTION (simplified, often continuous)
_CHARGE_DURATIONS = [10, 30, 60] # Example: Charge for 10, 30, or 60 time units
for drone_id in _DRONE_IDS:
    for order_id in _ORDER_IDS:
        _add_action((SimulationAction.DRONE_LOAD_ACTION, drone_id, order_id))
        _add_action((SimulationAction.DRONE_UNLOAD_ACTION, drone_id, order_id))
    for duration in _CHARGE_DURATIONS:
        _add_action((SimulationAction.DRONE_CHARGE_ACTION, drone_id, duration))


# 9. Micro-Hub Actions
# (micro_hub_id: int) for ACTIVATE/DEACTIVATE
# (micro_hub_id: int, drone_id: int) for ADD_TO_CHARGING_QUEUE
for micro_hub_id in _MICRO_HUB_IDS:
    _add_action((SimulationAction.ACTIVATE_MICRO_HUB, micro_hub_id))
    _add_action((SimulationAction.DEACTIVATE_MICRO_HUB, micro_hub_id))
    for drone_id in _DRONE_IDS:
        _add_action((SimulationAction.ADD_TO_CHARGING_QUEUE, micro_hub_id, drone_id))

# 10. Maintenance Actions
# (vehicle_id: int) for FLAG/RELEASE_VEHICLE_FOR_MAINTENANCE
# (micro_hub_id: int, service_type: SimulationAction) for FLAG/RELEASE_UNAVAILABILITY
for vehicle_id in _ALL_VEHICLE_IDS:
    _add_action((SimulationAction.FLAG_VEHICLE_FOR_MAINTENANCE, vehicle_id))

_RESOURCE_TYPES = [
    SimulationAction.RESOURCE_CHARGING_SLOT,
    SimulationAction.RESOURCE_PACKAGE_SORTING_SERVICE,
    SimulationAction.RESOURCE_LAUNCHES,
    SimulationAction.RESOURCE_RECOVERIES
]
for micro_hub_id in _MICRO_HUB_IDS:
    for service_type in _RESOURCE_TYPES:
        _add_action((SimulationAction.FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB, micro_hub_id, service_type))


# 11. Special Action
_add_action((SimulationAction.NO_OPERATION,))

# The total size of the flattened action space
ACTION_SPACE_SIZE = len(ACTION_MAP)

print(f"Action mapping generated. Total action space size: {ACTION_SPACE_SIZE}")
# print("Example Action Map entries (first 5):")
# for i, (action_tuple, index) in enumerate(ACTION_MAP.items()):
#     if i >= 5: break
#     print(f"  {action_tuple} -> {index}")

