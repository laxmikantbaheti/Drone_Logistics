import sys
import os
import numpy as np

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import necessary classes
from ddls_src.actions.constraints.base import ActionMasker
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.actions.action_mapping import ACTION_MAP, ACTION_SPACE_SIZE

# --- Mock Classes for ActionMasker Demonstration ---
# These mocks are minimal and only provide attributes/methods
# that ActionMasker directly queries from GlobalState or LogisticsSimulation.
class MockNode:
    def __init__(self, id, coords, type, is_loadable=False, is_unloadable=False, is_charging_station=False, packages_held=None):
        self.id = id
        self.coords = coords
        self.type = type
        self.is_loadable = is_loadable
        self.is_unloadable = is_unloadable
        self.is_charging_station = is_charging_station
        self.packages_held = packages_held if packages_held is not None else []

class MockOrder:
    def __init__(self, id, customer_node_id, status="pending", assigned_vehicle_id=None, assigned_micro_hub_id=None):
        self.id = id
        self.customer_node_id = customer_node_id
        self.status = status
        self.assigned_vehicle_id = assigned_vehicle_id
        self.assigned_micro_hub_id = assigned_micro_hub_id
    def get_SLA_remaining(self, current_time): return 100.0 # Dummy value

class MockTruck:
    def __init__(self, id, current_node_id, max_payload_capacity, status="idle", cargo_manifest=None):
        self.id = id
        self.current_node_id = current_node_id
        self.max_payload_capacity = max_payload_capacity
        self.status = status
        self.cargo_manifest = cargo_manifest if cargo_manifest is not None else []
    def get_cargo(self): return self.cargo_manifest

class MockDrone:
    def __init__(self, id, current_node_id, max_payload_capacity, battery_level, status="idle", cargo_manifest=None):
        self.id = id
        self.current_node_id = current_node_id
        self.max_payload_capacity = max_payload_capacity
        self.battery_level = battery_level
        self.max_battery_capacity = 1.0
        self.status = status
        self.cargo_manifest = cargo_manifest if cargo_manifest is not None else []

class MockMicroHub(MockNode):
    def __init__(self, id, coords, num_charging_slots, type="micro_hub", operational_status="inactive", is_blocked_for_launches=False, is_blocked_for_recoveries=False, is_package_transfer_unavailable=False, packages_held=None):
        super().__init__(id, coords, type, is_loadable=True, is_unloadable=True, is_charging_station=True, packages_held=packages_held)
        self.operational_status = operational_status
        self.charging_slots = {i: None for i in range(num_charging_slots)}
        self.is_blocked_for_launches = is_blocked_for_launches
        self.is_blocked_for_recoveries = is_blocked_for_recoveries
        self.is_package_transfer_unavailable = is_package_transfer_unavailable
    def get_available_charging_slots(self): return [s for s, d in self.charging_slots.items() if d is None]


class MockNetwork:
    def __init__(self, nodes, edges=None):
        self.nodes = nodes
        self.edges = edges if edges is not None else {}
    def calculate_shortest_path(self, start_node_id, end_node_id, vehicle_type):
        # Mock pathfinding: always return a path if nodes exist.
        if start_node_id in self.nodes and end_node_id in self.nodes:
            return [start_node_id, end_node_id] # Simple direct path
        return []

class MockGlobalState:
    def __init__(self):
        self.nodes = {
            0: MockNode(id=0, coords=(0.0, 0.0), type="depot", is_loadable=True, is_unloadable=True, is_charging_station=True, packages_held=[1003]),
            1: MockNode(id=1, coords=(10.0, 5.0), type="customer", is_unloadable=True),
            2: MockNode(id=2, coords=(15.0, 15.0), type="customer", is_unloadable=True),
            3: MockMicroHub(id=3, coords=(5.0, 20.0), num_charging_slots=2, operational_status="active", is_blocked_for_launches=False, packages_held=[1004])
        }
        self.orders = {
            1001: MockOrder(id=1001, customer_node_id=1, status="pending"),
            1002: MockOrder(id=1002, customer_node_id=1, status="delivered"), # Delivered order
            1003: MockOrder(id=1003, customer_node_id=1, status="accepted"), # Order at depot
            1004: MockOrder(id=1004, customer_node_id=2, status="pending") # Order at micro_hub
        }
        self.trucks = {
            101: MockTruck(id=101, current_node_id=0, max_payload_capacity=2, status="idle", cargo_manifest=[]),
            102: MockTruck(id=102, current_node_id=0, max_payload_capacity=2, status="maintenance", cargo_manifest=[1005]) # Truck in maintenance with cargo
        }
        self.drones = {
            201: MockDrone(id=201, current_node_id=0, max_payload_capacity=1, battery_level=0.9, status="idle"), # High battery
            202: MockDrone(id=202, current_node_id=3, max_payload_capacity=1, battery_level=0.1, status="idle") # Low battery drone at microhub
        }
        self.micro_hubs = {
            3: self.nodes[3]
        }
        self.edges = {} # Not strictly needed for these masks, but good to have
        self.network = MockNetwork(self.nodes, self.edges)
        self.current_time = 0.0

    def get_entity(self, entity_type: str, entity_id: int):
        if entity_type == "node": return self.nodes.get(entity_id)
        if entity_type == "order": return self.orders.get(entity_id)
        if entity_type == "truck": return self.trucks.get(entity_id)
        if entity_type == "drone": return self.drones.get(entity_id)
        if entity_type == "micro_hub": return self.micro_hubs.get(entity_id)
        raise KeyError(f"MockGlobalState: Unknown entity type: {entity_type}")

    def get_all_entities(self, entity_type: str):
        if entity_type == "node": return self.nodes
        if entity_type == "order": return self.orders
        if entity_type == "truck": return self.trucks
        if entity_type == "drone": return self.drones
        if entity_type == "micro_hub": return self.micro_hubs
        return {}

class MockLogisticsSimulation:
    def __init__(self, global_state, action_map, action_space_size):
        self.global_state = global_state
        self.action_map = action_map
        self.action_space_size = action_space_size
        self.network = global_state.network # Reference to the mock network
    def get_current_global_state(self): return self.global_state


def demonstrate_action_masker_functionality():
    """
    Demonstrates how ActionMasker generates a mask based on a mock simulation state.
    """
    print("--- Demonstrating ActionMasker Class Functionality ---")

    # 1. Setup Mock Global State
    print("\n1. Setting up Mock Global State with specific conditions:")
    mock_global_state = MockGlobalState()
    print(f"  - Order 1001 status: {mock_global_state.orders[1001].status} (pending)")
    print(f"  - Order 1002 status: {mock_global_state.orders[1002].status} (delivered)")
    print(f"  - Truck 101 status: {mock_global_state.trucks[101].status} (idle)")
    print(f"  - Truck 102 status: {mock_global_state.trucks[102].status} (maintenance)")
    print(f"  - Drone 201 battery: {mock_global_state.drones[201].battery_level*100:.1f}% (high)")
    print(f"  - Drone 202 battery: {mock_global_state.drones[202].battery_level*100:.1f}% (low)")
    print(f"  - Node 0 packages: {mock_global_state.nodes[0].packages_held}")
    print(f"  - MicroHub 3 operational status: {mock_global_state.micro_hubs[3].operational_status}")
    print(f"  - MicroHub 3 launches blocked: {mock_global_state.micro_hubs[3].is_blocked_for_launches}")


    # 2. Setup Mock Logistics Simulation (needed by ActionMasker)
    mock_logistics_sim = MockLogisticsSimulation(mock_global_state, ACTION_MAP, ACTION_SPACE_SIZE)

    # 3. Instantiate ActionMasker and generate the mask
    print("\n3. Instantiating ActionMasker and generating the mask:")
    action_masker = ActionMasker(mock_logistics_sim, ACTION_SPACE_SIZE, ACTION_MAP)
    current_mask = action_masker.generate_mask()
    print(f"  Generated mask (total size: {len(current_mask)})")

    # 4. Test specific actions against the generated mask
    print("\n4. Testing specific actions against the mask (True = Valid, False = Invalid):")

    # Order-related actions
    action_accept_order_valid = (SimulationAction.ACCEPT_ORDER, 1001) # Pending order
    action_accept_order_invalid = (SimulationAction.ACCEPT_ORDER, 1002) # Delivered order
    action_cancel_order_valid = (SimulationAction.CANCEL_ORDER, 1001) # Pending order
    action_cancel_order_invalid = (SimulationAction.CANCEL_ORDER, 1002) # Delivered order

    # Truck movement actions
    action_truck_move_valid = (SimulationAction.TRUCK_TO_NODE, 101, 1) # Idle truck, path exists
    action_truck_move_invalid_status = (SimulationAction.TRUCK_TO_NODE, 102, 1) # Maintenance truck
    action_truck_move_invalid_node = (SimulationAction.TRUCK_TO_NODE, 101, 999) # Non-existent node

    # Truck load/unload actions
    action_load_truck_valid = (SimulationAction.LOAD_TRUCK_ACTION, 101, 1003) # Idle truck at depot, order 1003 at depot
    action_load_truck_invalid_no_order = (SimulationAction.LOAD_TRUCK_ACTION, 101, 1001) # Order 1001 not at depot
    action_unload_truck_invalid_no_cargo = (SimulationAction.UNLOAD_TRUCK_ACTION, 101, 1005) # Order 1005 not in truck 101

    # Drone launch actions
    action_launch_drone_valid = (SimulationAction.LAUNCH_DRONE, 201, 1001) # Drone 201 (high battery) at depot, order 1001
    mock_global_state.drones[201].cargo_manifest.append(1001) # Manually add order to drone cargo for launch to be valid
    action_launch_drone_invalid_battery = (SimulationAction.LAUNCH_DRONE, 202, 1004) # Drone 202 (low battery)
    mock_global_state.drones[202].cargo_manifest.append(1004) # Manually add order to drone cargo

    # Drone charging actions
    action_drone_to_charging_valid = (SimulationAction.DRONE_TO_CHARGING_STATION, 201, 3) # Drone 201 to MicroHub 3
    action_drone_to_charging_invalid_status = (SimulationAction.DRONE_TO_CHARGING_STATION, 202, 3) # Drone 202 (low battery, but also status 'idle' is fine)
    # Let's make drone 202 status 'broken_down' to test invalid status
    mock_global_state.drones[202].status = "broken_down"
    action_drone_to_charging_invalid_status_broken = (SimulationAction.DRONE_TO_CHARGING_STATION, 202, 3)

    action_drone_charge_valid = (SimulationAction.DRONE_CHARGE_ACTION, 202, 30) # Drone 202 at MicroHub 3, low battery (set status to charging)
    mock_global_state.drones[202].status = "charging" # Set status to charging for this test
    action_drone_charge_invalid_full = (SimulationAction.DRONE_CHARGE_ACTION, 201, 30) # Drone 201 (high battery)

    # Micro-hub actions
    action_activate_hub_invalid = (SimulationAction.ACTIVATE_MICRO_HUB, 3) # MicroHub 3 is already active
    action_deactivate_hub_valid = (SimulationAction.DEACTIVATE_MICRO_HUB, 3) # MicroHub 3 is active

    # Maintenance actions
    action_flag_truck_maintenance_invalid = (SimulationAction.FLAG_VEHICLE_FOR_MAINTENANCE, 102) # Truck 102 already in maintenance
    action_flag_truck_maintenance_valid = (SimulationAction.FLAG_VEHICLE_FOR_MAINTENANCE, 101) # Truck 101 is idle


    actions_to_test = [
        ("ACCEPT_ORDER (Valid)", action_accept_order_valid),
        ("ACCEPT_ORDER (Invalid - Delivered)", action_accept_order_invalid),
        ("CANCEL_ORDER (Valid)", action_cancel_order_valid),
        ("CANCEL_ORDER (Invalid - Delivered)", action_cancel_order_invalid),
        ("TRUCK_TO_NODE (Valid)", action_truck_move_valid),
        ("TRUCK_TO_NODE (Invalid - Maintenance)", action_truck_move_invalid_status),
        ("TRUCK_TO_NODE (Invalid - Non-existent Node)", action_truck_move_invalid_node),
        ("LOAD_TRUCK_ACTION (Valid)", action_load_truck_valid),
        ("LOAD_TRUCK_ACTION (Invalid - Order not at node)", action_load_truck_invalid_no_order),
        ("UNLOAD_TRUCK_ACTION (Invalid - No cargo)", action_unload_truck_invalid_no_cargo),
        ("LAUNCH_DRONE (Valid)", action_launch_drone_valid),
        ("LAUNCH_DRONE (Invalid - Low Battery)", action_launch_drone_invalid_battery),
        ("DRONE_TO_CHARGING_STATION (Valid)", action_drone_to_charging_valid),
        ("DRONE_TO_CHARGING_STATION (Invalid - Broken)", action_drone_to_charging_invalid_status_broken),
        ("DRONE_CHARGE_ACTION (Valid)", action_drone_charge_valid),
        ("DRONE_CHARGE_ACTION (Invalid - Full Battery)", action_drone_charge_invalid_full),
        ("ACTIVATE_MICRO_HUB (Invalid - Already Active)", action_activate_hub_invalid),
        ("DEACTIVATE_MICRO_HUB (Valid)", action_deactivate_hub_valid),
        ("FLAG_VEHICLE_FOR_MAINTENANCE (Invalid - Already in Maintenance)", action_flag_truck_maintenance_invalid),
        ("FLAG_VEHICLE_FOR_MAINTENANCE (Valid)", action_flag_truck_maintenance_valid),
        ((SimulationAction.NO_OPERATION,), "NO_OPERATION (Always Valid)") # Special case for NO_OPERATION
    ]

    for description, action_tuple in actions_to_test:
        idx = ACTION_MAP.get(action_tuple)
        if idx is not None:
            # Safely check if the index is within bounds before accessing the mask
            if idx < len(current_mask):
                print(f"  {description:<50}: {current_mask[idx]}")
            else:
                # This case indicates a mismatch between ACTION_MAP size and actual mask size
                print(f"  {description:<50}: Index {idx} out of mask bounds ({len(current_mask)}). Action mapping mismatch.")
        else:
            # This handles the case where action_tuple is not found in ACTION_MAP
            print(f"  {description:<50}: Action tuple NOT FOUND in ACTION_MAP. Check action_mapping.py and its ID ranges.")


    print("\n--- ActionMasker Class Functionality Demonstration Complete ---")

if __name__ == "__main__":
    demonstrate_action_masker_functionality()

