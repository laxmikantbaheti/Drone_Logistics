import sys
import os
import numpy as np

# Add the project root to the Python path to allow importing ddls_src
# Assuming this script is in project_root/examples/
project_root = "C:\\Users\\SHAIK RIFSHU\\PycharmProjects\\Drone_Logistics\\ddls_src"
sys.path.insert(0, project_root)

# Import necessary classes
from ddls_src.managers.action_manager import ActionManager
from ddls_src.actions.constraints.base import ActionMasker
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.actions.action_mapping import ACTION_MAP, ACTION_SPACE_SIZE


# Mock classes for demonstration purposes
class MockNode:
    def __init__(self, id, coords, type, is_loadable=False, is_unloadable=False, is_charging_station=False):
        self.id = id
        self.coords = coords
        self.type = type
        self.is_loadable = is_loadable
        self.is_unloadable = is_unloadable
        self.is_charging_station = is_charging_station
        self.packages_held = []


class MockOrder:
    def __init__(self, id, customer_node_id, status="pending", assigned_vehicle_id=None, assigned_micro_hub_id=None):
        self.id = id
        self.customer_node_id = customer_node_id
        self.status = status
        self.assigned_vehicle_id = assigned_vehicle_id
        self.assigned_micro_hub_id = assigned_micro_hub_id

    def update_status(self, new_status): self.status = new_status


class MockTruck:
    def __init__(self, id, start_node_id, max_payload_capacity, status="idle", cargo_manifest=None):
        self.id = id
        self.current_node_id = start_node_id
        self.max_payload_capacity = max_payload_capacity
        self.status = status
        self.cargo_manifest = cargo_manifest if cargo_manifest is not None else []

    def get_cargo(self): return self.cargo_manifest

    def set_status(self, new_status): self.status = new_status


class MockDrone:
    def __init__(self, id, start_node_id, max_payload_capacity, initial_battery, status="idle", cargo_manifest=None):
        self.id = id
        self.current_node_id = start_node_id
        self.max_payload_capacity = max_payload_capacity
        self.battery_level = initial_battery
        self.max_battery_capacity = 1.0
        self.status = status
        self.cargo_manifest = cargo_manifest if cargo_manifest is not None else []

    def get_cargo(self): return self.cargo_manifest

    def set_status(self, new_status): self.status = new_status


class MockMicroHub(MockNode):
    def __init__(self, id, coords, num_charging_slots, type="micro_hub", operational_status="inactive",
                 is_blocked_for_launches=False, is_blocked_for_recoveries=False, is_package_transfer_unavailable=False):
        super().__init__(id, coords, type, is_loadable=True, is_unloadable=True, is_charging_station=True)
        self.operational_status = operational_status
        self.charging_slots = {i: None for i in range(num_charging_slots)}
        self.is_blocked_for_launches = is_blocked_for_launches
        self.is_blocked_for_recoveries = is_blocked_for_recoveries
        self.is_package_transfer_unavailable = is_package_transfer_unavailable

    def get_available_charging_slots(self): return [s for s, d in self.charging_slots.items() if d is None]


class MockGlobalState:
    def __init__(self):
        self.nodes = {
            0: MockNode(id=0, coords=(0.0, 0.0), type="depot", is_loadable=True, is_unloadable=True,
                        is_charging_station=True),
            1: MockNode(id=1, coords=(10.0, 5.0), type="customer", is_unloadable=True),
            3: MockMicroHub(id=3, coords=(5.0, 20.0), num_charging_slots=2, operational_status="active")
        }
        self.orders = {
            1001: MockOrder(id=1001, customer_node_id=1, status="pending"),
            1002: MockOrder(id=1002, customer_node_id=1, status="delivered")  # Delivered order for masking demo
        }
        self.trucks = {
            101: MockTruck(id=101, start_node_id=0, max_payload_capacity=2, status="idle"),
            102: MockTruck(id=102, start_node_id=0, max_payload_capacity=2, status="maintenance")
        }
        self.drones = {
            201: MockDrone(id=201, start_node_id=0, max_payload_capacity=1, initial_battery=0.9, status="idle"),
            202: MockDrone(id=202, start_node_id=3, max_payload_capacity=1, initial_battery=0.1, status="idle")
            # Low battery drone at microhub
        }
        self.micro_hubs = {
            3: self.nodes[3]  # Microhub is also a node
        }
        self.edges = {}  # Not strictly needed for these specific masks, but good to have
        self.network = MockNetwork(self.nodes, self.edges)  # Mock Network instance
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


class MockNetwork:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def calculate_shortest_path(self, start_node_id, end_node_id, vehicle_type):
        # Simple mock: always return a path if nodes exist and are not blocked
        if start_node_id in self.nodes and end_node_id in self.nodes:
            # For this example, assume direct path always exists and is not blocked
            return [start_node_id, end_node_id]
        return []


class MockLogisticsSimulation:
    def __init__(self, global_state, action_map, action_space_size):
        self.global_state = global_state
        self.action_map = action_map
        self.action_space_size = action_space_size
        self.network = global_state.network  # Reference to the mock network

    def get_current_global_state(self): return self.global_state


# Mock Managers needed by ActionManager (they won't actually do anything in this demo)
class MockSupplyChainManager:
    def __init__(self, global_state): self.global_state = global_state

    def accept_order(self, *args): print(f"  MockSCM: Accepted order {args[0]}")

    def prioritize_order(self, *args): print(f"  MockSCM: Prioritized order {args[0]}")

    def cancel_order(self, *args): print(f"  MockSCM: Cancelled order {args[0]}")

    def flag_for_re_delivery(self, *args): print(f"  MockSCM: Flagged order {args[0]} for re-delivery")

    def assign_order_to_truck(self, *args): print(f"  MockSCM: Assigned order {args[0]} to truck {args[1]}")

    def assign_order_to_drone(self, *args): print(f"  MockSCM: Assigned order {args[0]} to drone {args[1]}")

    def assign_order_to_micro_hub(self, *args): print(f"  MockSCM: Assigned order {args[0]} to micro_hub {args[1]}")

    def consolidate_for_truck(self, *args): print(f"  MockSCM: Consolidated for truck {args[0]}")

    def consolidate_for_drone(self, *args): print(f"  MockSCM: Consolidated for drone {args[0]}")

    def reassign_order(self, *args): print(f"  MockSCM: Reassigned order {args[0]} to vehicle {args[1]}")


class MockResourceManager:
    def __init__(self, global_state):
        self.global_state = global_state
        self.fleet_manager = MockFleetManager(global_state)
        self.micro_hubs_manager = MockMicroHubsManager(global_state)

    def flag_vehicle_for_maintenance(self, *args): print(f"  MockRM: Flagged vehicle {args[0]} for maintenance")

    def release_vehicle_from_maintenance(self, *args): print(f"  MockRM: Released vehicle {args[0]} from maintenance")

    def flag_unavailability_of_service_at_micro_hub(self, *args): print(
        f"  MockRM: Flagged hub {args[0]} service {args[1]} unavailable")

    def release_unavailability_of_service_at_micro_hub(self, *args): print(
        f"  MockRM: Released hub {args[0]} service {args[1]} available")


class MockFleetManager:
    def __init__(self, global_state): self.global_state = global_state

    def load_truck(self, *args): print(f"  MockFM: Loaded truck {args[0]} with order {args[1]}")

    def unload_truck(self, *args): print(f"  MockFM: Unloaded truck {args[0]} order {args[1]}")

    def drone_load(self, *args): print(f"  MockFM: Loaded drone {args[0]} with order {args[1]}")

    def drone_unload(self, *args): print(f"  MockFM: Unloaded drone {args[0]} order {args[1]}")

    def drone_charge(self, *args): print(f"  MockFM: Charged drone {args[0]} for {args[1]}s")


class MockMicroHubsManager:
    def __init__(self, global_state): self.global_state = global_state

    def activate_micro_hub(self, *args): print(f"  MockMHM: Activated hub {args[0]}")

    def deactivate_micro_hub(self, *args): print(f"  MockMHM: Deactivated hub {args[0]}")

    def add_to_charging_queue(self, *args): print(f"  MockMHM: Added drone {args[1]} to hub {args[0]} charging queue")


class MockNetworkManager:
    def __init__(self, global_state, network): self.global_state = global_state; self.network = network

    def truck_to_node(self, *args): print(f"  MockNM: Truck {args[0]} to node {args[1]}")

    def re_route_truck_to_node(self, *args): print(f"  MockNM: Re-routed truck {args[0]} to node {args[1]}")

    def launch_drone(self, *args): print(f"  MockNM: Launched drone {args[0]} for order {args[1]}")

    def drone_landing(self, *args): print(f"  MockNM: Drone {args[0]} landing")

    def drone_to_charging_station(self, *args): print(f"  MockNM: Drone {args[0]} to charging station {args[1]}")


def demonstrate_action_manager_masker_functionality():
    """
    Demonstrates the functionalities of ActionManager and ActionMasker.
    """
    print("--- Demonstrating ActionManager and ActionMasker Functionality ---")

    # 1. Setup Mock Global State
    print("\n1. Setting up Mock Global State:")
    mock_global_state = MockGlobalState()
    print(
        f"  Mock Global State has {len(mock_global_state.trucks)} trucks, {len(mock_global_state.drones)} drones, {len(mock_global_state.orders)} orders.")
    print(f"  Truck 101 status: {mock_global_state.trucks[101].status}")
    print(f"  Truck 102 status: {mock_global_state.trucks[102].status}")
    print(f"  Order 1001 status: {mock_global_state.orders[1001].status}")
    print(f"  Order 1002 status: {mock_global_state.orders[1002].status}")
    print(f"  Drone 202 battery: {mock_global_state.drones[202].battery_level * 100:.1f}%")
    print(f"  MicroHub 3 status: {mock_global_state.micro_hubs[3].operational_status}")

    # 2. Setup Mock Logistics Simulation (needed by ActionMasker)
    mock_logistics_sim = MockLogisticsSimulation(mock_global_state, ACTION_MAP, ACTION_SPACE_SIZE)

    # 3. Instantiate ActionMasker
    print("\n3. Instantiating ActionMasker and generating initial mask:")
    action_masker = ActionMasker(mock_logistics_sim, ACTION_SPACE_SIZE, ACTION_MAP)
    initial_mask = action_masker.generate_mask()
    print(f"  Initial Mask (first 10 valid indices): {np.where(initial_mask)[0][:10]}")

    # 4. Instantiate ActionManager with Mock Managers
    print("\n4. Instantiating ActionManager with Mock Managers:")
    mock_managers = {
        'supply_chain_manager': MockSupplyChainManager(mock_global_state),
        'resource_manager': MockResourceManager(mock_global_state),
        'network_manager': MockNetworkManager(mock_global_state, mock_global_state.network)
    }
    action_manager = ActionManager(mock_global_state, mock_managers, ACTION_MAP, action_masker)

    # 5. Demonstrate Action Masking Logic (before dispatch)
    print("\n5. Demonstrating Action Masking Logic:")

    # Test an action that should be valid (e.g., ACCEPT_ORDER for pending order)
    action_accept_order = (SimulationAction.ACCEPT_ORDER, 1001)
    idx_accept_order = ACTION_MAP.get(action_accept_order)
    if idx_accept_order is not None:
        print(f"  Is ACCEPT_ORDER 1001 valid? {initial_mask[idx_accept_order]} (Expected True)")

    # Test an action that should be invalid (e.g., ACCEPT_ORDER for delivered order)
    action_accept_delivered_order = (SimulationAction.ACCEPT_ORDER, 1002)
    idx_accept_delivered_order = ACTION_MAP.get(action_accept_delivered_order)
    if idx_accept_delivered_order is not None:
        print(f"  Is ACCEPT_ORDER 1002 (delivered) valid? {initial_mask[idx_accept_delivered_order]} (Expected False)")

    # Test an action that should be invalid (e.g., TRUCK_TO_NODE for maintenance truck)
    action_truck_move_maintenance = (SimulationAction.TRUCK_TO_NODE, 102, 1)
    idx_truck_move_maintenance = ACTION_MAP.get(action_truck_move_maintenance)
    if idx_truck_move_maintenance is not None:
        print(
            f"  Is TRUCK_TO_NODE 102 (maintenance) to Node 1 valid? {initial_mask[idx_truck_move_maintenance]} (Expected False)")

    # Test an action that should be invalid (e.g., DRONE_LAUNCH for low battery)
    # Drone 202 is at MicroHub 3 with low battery (0.1)
    action_drone_launch_low_battery = (SimulationAction.LAUNCH_DRONE, 202, 1001)
    idx_drone_launch_low_battery = ACTION_MAP.get(action_drone_launch_low_battery)
    if idx_drone_launch_low_battery is not None:
        print(
            f"  Is LAUNCH_DRONE 202 (low battery) valid? {initial_mask[idx_drone_launch_low_battery]} (Expected False)")

    # Test an action that should be invalid (e.g., ADD_TO_CHARGING_QUEUE for full battery)
    # Drone 201 has full battery (0.9)
    action_add_to_charging_full_battery = (SimulationAction.ADD_TO_CHARGING_QUEUE, 0,
                                           201)  # Node 0 is a depot/charging station
    idx_add_to_charging_full_battery = ACTION_MAP.get(action_add_to_charging_full_battery)
    if idx_add_to_charging_full_battery is not None:
        print(
            f"  Is ADD_TO_CHARGING_QUEUE 0 Drone 201 (full battery) valid? {initial_mask[idx_add_to_charging_full_battery]} (Expected False)")

    # 6. Demonstrate Action Dispatching
    print("\n6. Demonstrating Action Dispatching (using ActionManager):")

    # Valid Action: ACCEPT_ORDER
    print("\n  Attempting to execute ACCEPT_ORDER 1001 (should be valid):")
    mock_global_state.orders[1001].status = "pending"  # Ensure it's pending
    current_mask_updated = action_masker.generate_mask()  # Re-generate mask after potential state change
    executed = action_manager.execute_action(action_accept_order, current_mask_updated)
    print(f"  Execution result: {executed}")  # Should be True

    # Invalid Action: TRUCK_TO_NODE for maintenance truck
    print("\n  Attempting to execute TRUCK_TO_NODE 102 to Node 1 (should be invalid due to mask):")
    executed = action_manager.execute_action(action_truck_move_maintenance, current_mask_updated)
    print(f"  Execution result: {executed}")  # Should be False

    # Valid Action: TRUCK_TO_NODE for idle truck
    action_truck_move_valid = (SimulationAction.TRUCK_TO_NODE, 101, 1)
    print("\n  Attempting to execute TRUCK_TO_NODE 101 to Node 1 (should be valid):")
    executed = action_manager.execute_action(action_truck_move_valid, current_mask_updated)
    print(f"  Execution result: {executed}")  # Should be True

    # Valid Action: ACTIVATE_MICRO_HUB (MicroHub 3 is active, so this should be masked out for activation)
    action_activate_hub = (SimulationAction.ACTIVATE_MICRO_HUB, 3)
    idx_activate_hub = ACTION_MAP.get(action_activate_hub)
    if idx_activate_hub is not None:
        print(
            f"\n  Is ACTIVATE_MICRO_HUB 3 valid? {current_mask_updated[idx_activate_hub]} (Expected False, as it's already active)")

    print("\n--- ActionManager and ActionMasker Functionality Demonstration Complete ---")


if __name__ == "__main__":
    demonstrate_action_manager_masker_functionality()

