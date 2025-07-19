from typing import List, Dict, Any, Optional


# Forward declaration for GlobalState to avoid circular dependency
class GlobalState:
    pass


# Forward declarations for Vehicle, Truck, Drone, Node, MicroHub for type hinting
class Vehicle: pass


class Truck: pass


class Drone: pass


class Node: pass


class MicroHub: pass


class FleetManager:
    """
    Manages operations related to the vehicle fleet (trucks and drones),
    including loading, unloading, and drone charging.
    It interacts with GlobalState to modify vehicle and node states.
    """

    def __init__(self, global_state: 'GlobalState'):
        """
        Initializes the FleetManager.

        Args:
            global_state (GlobalState): Reference to the central GlobalState.
        """
        self.global_state = global_state
        print("FleetManager initialized.")

    def load_truck(self, truck_id: int, order_id: int) -> bool:
        """
        Handles package transfer from a node to a truck.
        Assumes the truck is at the node where the package is located.

        Args:
            truck_id (int): The ID of the truck.
            order_id (int): The ID of the order/package to load.

        Returns:
            bool: True if package was successfully loaded, False otherwise.
        """
        try:
            truck = self.global_state.get_entity("truck", truck_id)
            order = self.global_state.get_entity("order", order_id)

            if truck.current_node_id is None:
                print(f"FleetManager: Truck {truck_id} is not at a node to load package {order_id}.")
                return False

            current_node = self.global_state.get_entity("node", truck.current_node_id)

            if not current_node.is_loadable:
                print(f"FleetManager: Node {current_node.id} is not a loadable point for truck {truck_id}.")
                return False

            if order_id not in current_node.get_packages():
                print(f"FleetManager: Package {order_id} not found at Node {current_node.id}.")
                return False

            if len(truck.cargo_manifest) >= truck.max_payload_capacity:
                print(f"FleetManager: Truck {truck_id} is full, cannot load package {order_id}.")
                return False

            truck.add_cargo(order_id)
            current_node.remove_package(order_id)
            order.update_status("in_transit")  # Order is now in transit
            truck.set_status("loading")  # Update truck status temporarily
            print(f"FleetManager: Loaded package {order_id} onto Truck {truck_id} at Node {current_node.id}.")
            return True
        except KeyError as e:
            print(f"FleetManager: Load truck failed - {e}.")
            return False
        except ValueError as e:
            print(f"FleetManager: Load truck failed - {e}.")
            return False

    def unload_truck(self, truck_id: int, order_id: int) -> bool:
        """
        Handles package transfer from a truck to a node.
        Assumes the truck is at the node where the package is to be delivered/dropped.

        Args:
            truck_id (int): The ID of the truck.
            order_id (int): The ID of the order/package to unload.

        Returns:
            bool: True if package was successfully unloaded, False otherwise.
        """
        try:
            truck = self.global_state.get_entity("truck", truck_id)
            order = self.global_state.get_entity("order", order_id)

            if truck.current_node_id is None:
                print(f"FleetManager: Truck {truck_id} is not at a node to unload package {order_id}.")
                return False

            current_node = self.global_state.get_entity("node", truck.current_node_id)

            if not current_node.is_unloadable:
                print(f"FleetManager: Node {current_node.id} is not an unloadable point for truck {truck_id}.")
                return False

            if order_id not in truck.get_cargo():
                print(f"FleetManager: Package {order_id} not found in Truck {truck_id}'s cargo.")
                return False

            truck.remove_cargo(order_id)
            current_node.add_package(order_id)

            # Determine if it's a final delivery or transfer to micro-hub
            if order.customer_node_id == current_node.id:
                order.update_status("delivered")
                order.delivery_time = self.global_state.current_time  # Record delivery time
                print(
                    f"FleetManager: Delivered package {order_id} by Truck {truck_id} to customer at Node {current_node.id}.")
            else:
                order.update_status("at_node")  # Or "at_micro_hub" if current_node is a micro_hub
                print(f"FleetManager: Unloaded package {order_id} from Truck {truck_id} to Node {current_node.id}.")

            truck.set_status("unloading")  # Update truck status temporarily
            return True
        except KeyError as e:
            print(f"FleetManager: Unload truck failed - {e}.")
            return False
        except ValueError as e:
            print(f"FleetManager: Unload truck failed - {e}.")
            return False

    def drone_load(self, drone_id: int, order_id: int) -> bool:
        """
        Handles package transfer from a node/micro-hub to a drone.
        Assumes the drone is at the node where the package is located.

        Args:
            drone_id (int): The ID of the drone.
            order_id (int): The ID of the order/package to load.

        Returns:
            bool: True if package was successfully loaded, False otherwise.
        """
        try:
            drone = self.global_state.get_entity("drone", drone_id)
            order = self.global_state.get_entity("order", order_id)

            if drone.current_node_id is None:
                print(f"FleetManager: Drone {drone_id} is not at a node to load package {order_id}.")
                return False

            current_node = self.global_state.get_entity("node", drone.current_node_id)

            if not current_node.is_loadable:
                print(f"FleetManager: Node {current_node.id} is not a loadable point for drone {drone_id}.")
                return False

            if order_id not in current_node.get_packages():
                print(f"FleetManager: Package {order_id} not found at Node {current_node.id}.")
                return False

            if len(drone.cargo_manifest) >= drone.max_payload_capacity:
                print(f"FleetManager: Drone {drone_id} is full, cannot load package {order_id}.")
                return False

            drone.add_cargo(order_id)
            current_node.remove_package(order_id)
            order.update_status("in_transit")  # Order is now in transit
            drone.set_status("loading")  # Update drone status temporarily
            print(f"FleetManager: Loaded package {order_id} onto Drone {drone_id} at Node {current_node.id}.")
            return True
        except KeyError as e:
            print(f"FleetManager: Drone load failed - {e}.")
            return False
        except ValueError as e:
            print(f"FleetManager: Drone load failed - {e}.")
            return False

    def drone_unload(self, drone_id: int, order_id: int) -> bool:
        """
        Handles package transfer from a drone to a node/micro-hub.
        Assumes the drone is at the node where the package is to be delivered/dropped.

        Args:
            drone_id (int): The ID of the drone.
            order_id (int): The ID of the order/package to unload.

        Returns:
            bool: True if package was successfully unloaded, False otherwise.
        """
        try:
            drone = self.global_state.get_entity("drone", drone_id)
            order = self.global_state.get_entity("order", order_id)

            if drone.current_node_id is None:
                print(f"FleetManager: Drone {drone_id} is not at a node to unload package {order_id}.")
                return False

            current_node = self.global_state.get_entity("node", drone.current_node_id)

            if not current_node.is_unloadable:
                print(f"FleetManager: Node {current_node.id} is not an unloadable point for drone {drone_id}.")
                return False

            if order_id not in drone.get_cargo():
                print(f"FleetManager: Package {order_id} not found in Drone {drone_id}'s cargo.")
                return False

            drone.remove_cargo(order_id)
            current_node.add_package(order_id)

            # Determine if it's a final delivery or transfer
            if order.customer_node_id == current_node.id:
                order.update_status("delivered")
                order.delivery_time = self.global_state.current_time  # Record delivery time
                print(
                    f"FleetManager: Delivered package {order_id} by Drone {drone_id} to customer at Node {current_node.id}.")
            else:
                order.update_status("at_node")  # Or "at_micro_hub" if current_node is a micro_hub
                print(f"FleetManager: Unloaded package {order_id} from Drone {drone_id} to Node {current_node.id}.")

            drone.set_status("unloading")  # Update drone status temporarily
            return True
        except KeyError as e:
            print(f"FleetManager: Drone unload failed - {e}.")
            return False
        except ValueError as e:
            print(f"FleetManager: Drone unload failed - {e}.")
            return False

    def drone_charge(self, drone_id: int, delta_time: float) -> bool:
        """
        Manages drone battery charge. Assumes the drone is at a charging station.

        Args:
            drone_id (int): The ID of the drone to charge.
            delta_time (float): The duration for which to charge the drone.

        Returns:
            bool: True if drone was found and charged, False otherwise.
        """
        try:
            drone = self.global_state.get_entity("drone", drone_id)

            if drone.current_node_id is None:
                print(f"FleetManager: Drone {drone_id} is not at a node to charge.")
                return False

            current_node = self.global_state.get_entity("node", drone.current_node_id)

            if not current_node.is_charging_station:
                print(f"FleetManager: Node {current_node.id} is not a charging station for drone {drone_id}.")
                return False

            # If the node is a MicroHub, check its operational status and slot availability
            if current_node.type == 'micro_hub':
                micro_hub = self.global_state.get_entity("micro_hub", current_node.id)
                if micro_hub.operational_status != "active":
                    print(f"FleetManager: MicroHub {micro_hub.id} is not active for charging.")
                    return False
                # Check if drone is assigned to a slot. This would typically be handled by MicroHubsManager.
                # For now, assume if it's at a charging station and status is charging, it's fine.

            if drone.status != "charging":
                print(f"FleetManager: Drone {drone_id} is not in 'charging' status. Set status to 'charging' first.")
                return False

            drone.charge_battery(delta_time)
            print(
                f"FleetManager: Drone {drone_id} charged for {delta_time:.2f} time units. Battery: {drone.battery_level * 100:.1f}%")
            return True
        except KeyError as e:
            print(f"FleetManager: Drone charge failed - {e}.")
            return False
        except ValueError as e:
            print(f"FleetManager: Drone charge failed - {e}.")
            return False

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes fleet-specific plotting data.
        This method will likely defer to individual vehicle's initialize_plot_data methods.
        """
        print("FleetManager: Initializing plot data.")
        # Iterate through all vehicles and call their plotting initialization
        for truck in self.global_state.get_all_entities("truck").values():
            truck.initialize_plot_data(figure_data)
        for drone in self.global_state.get_all_entities("drone").values():
            drone.initialize_plot_data(figure_data)

        # Any fleet-level initial data (e.g., total vehicle count)
        if 'fleet_overview' not in figure_data:
            figure_data['fleet_overview'] = {}
        figure_data['fleet_overview']['initial_total_vehicles'] = len(self.global_state.trucks) + len(
            self.global_state.drones)

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates fleet-specific plotting data for the current simulation step.
        This method will likely defer to individual vehicle's update_plot_data methods.
        """
        print("FleetManager: Updating plot data.")
        # Iterate through all vehicles and call their plotting update
        for truck in self.global_state.get_all_entities("truck").values():
            truck.update_plot_data(figure_data)
        for drone in self.global_state.get_all_entities("drone").values():
            drone.update_plot_data(figure_data)

        # Any fleet-level dynamic data (e.g., vehicles by status)
        if 'fleet_overview' not in figure_data:
            figure_data['fleet_overview'] = {}  # Defensive check

        vehicle_status_counts = {}
        for vehicle_type in ["truck", "drone"]:
            for vehicle in self.global_state.get_all_entities(vehicle_type).values():
                vehicle_status_counts[vehicle.status] = vehicle_status_counts.get(vehicle.status, 0) + 1
        figure_data['fleet_overview']['current_vehicle_status_counts'] = vehicle_status_counts
