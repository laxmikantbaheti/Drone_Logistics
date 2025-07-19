from typing import List, Dict, Any, Tuple, Optional



class NetworkManager:
    """
    Manages all operations related to the simulation network, including
    vehicle movement, route calculation, and handling arrivals/departures.
    It interacts with GlobalState and the Network graph structure.
    """

    def __init__(self, global_state: 'GlobalState', network: 'Network'):
        """
        Initializes the NetworkManager.

        Args:
            global_state (GlobalState): Reference to the central GlobalState.
            network (Network): Reference to the Network graph structure.
        """
        self.global_state = global_state
        self.network = network  # The graph structure with nodes and edges
        print("NetworkManager initialized.")

    def truck_to_node(self, truck_id: int, destination_node_id: int) -> bool:
        """
        Calculates a route for a truck to a destination node and sets the truck's path.

        Args:
            truck_id (int): The ID of the truck.
            destination_node_id (int): The ID of the node the truck should travel to.

        Returns:
            bool: True if a valid route was found and set, False otherwise.
        """
        try:
            truck = self.global_state.get_entity("truck", truck_id)
            if truck.status not in ["idle", "loading", "unloading"]:
                print(f"NetworkManager: Truck {truck_id} is not available to set a new route (status: {truck.status}).")
                return False
            if truck.current_node_id is None:
                print(
                    f"NetworkManager: Truck {truck_id} current location is not a node. Cannot set route from mid-segment.")
                return False

            # Use the Network class to calculate the shortest path
            path = self.network.calculate_shortest_path(truck.current_node_id, destination_node_id, 'truck')

            if not path:
                print(
                    f"NetworkManager: No valid path found for Truck {truck_id} from {truck.current_node_id} to {destination_node_id}.")
                return False

            truck.set_route(path)
            print(f"NetworkManager: Truck {truck_id} route set to {path}.")
            return True
        except KeyError as e:
            print(f"NetworkManager: Truck-to-node failed - {e}.")
            return False
        except ValueError as e:
            print(f"NetworkManager: Truck-to-node failed - {e}.")
            return False

    def re_route_truck_to_node(self, truck_id: int, new_destination_node_id: int) -> bool:
        """
        Recalculates and updates the route for an actively moving truck.
        The new route will start from the truck's current node (if at node)
        or the next node in its current path (if en-route).

        Args:
            truck_id (int): The ID of the truck.
            new_destination_node_id (int): The ID of the new destination node.

        Returns:
            bool: True if a valid new route was found and set, False otherwise.
        """
        try:
            truck = self.global_state.get_entity("truck", truck_id)

            start_node_for_reroute = truck.current_node_id
            if start_node_for_reroute is None:
                # If truck is mid-segment, find the next node in its current route
                if truck.current_route and len(truck.current_route) > 1:
                    start_node_for_reroute = truck.current_route[1]  # Next node in current route
                    print(f"NetworkManager: Re-routing Truck {truck_id} from next node {start_node_for_reroute}.")
                else:
                    print(
                        f"NetworkManager: Truck {truck_id} is not at a node and has no valid current route to re-route from.")
                    return False

            path = self.network.calculate_shortest_path(start_node_for_reroute, new_destination_node_id, 'truck')

            if not path:
                print(
                    f"NetworkManager: No valid re-route found for Truck {truck_id} from {start_node_for_reroute} to {new_destination_node_id}.")
                return False

            truck.set_route(path)
            print(f"NetworkManager: Truck {truck_id} re-routed to {path}.")
            return True
        except KeyError as e:
            print(f"NetworkManager: Re-route truck failed - {e}.")
            return False
        except ValueError as e:
            print(f"NetworkManager: Re-route truck failed - {e}.")
            return False

    def launch_drone(self, drone_id: int, order_id: int) -> bool:
        """
        Calculates a drone flight path from its current base (node) to the order's
        customer destination, and sets the drone's path.
        Assumes the order is already loaded onto the drone.

        Args:
            drone_id (int): The ID of the drone.
            order_id (int): The ID of the order being delivered by the drone.

        Returns:
            bool: True if a valid flight path was found and set, False otherwise.
        """
        try:
            drone = self.global_state.get_entity("drone", drone_id)
            order = self.global_state.get_entity("order", order_id)

            if drone.status not in ["idle", "loading", "unloading",
                                    "charging"]:  # Drones can launch from charging if battery is enough
                print(f"NetworkManager: Drone {drone_id} is not available to launch (status: {drone.status}).")
                return False
            if drone.current_node_id is None:
                print(f"NetworkManager: Drone {drone_id} is not at a node to launch from.")
                return False
            if order_id not in drone.cargo_manifest:
                print(f"NetworkManager: Order {order_id} is not in Drone {drone_id}'s cargo manifest.")
                return False

            # Check if current node is a valid launch point (e.g., depot or active micro-hub)
            current_node = self.global_state.get_entity("node", drone.current_node_id)
            if current_node.type == 'micro_hub':
                micro_hub = self.global_state.get_entity("micro_hub", current_node.id)
                if micro_hub.operational_status != "active" or micro_hub.is_blocked_for_launches:
                    print(f"NetworkManager: MicroHub {micro_hub.id} is not active or blocked for launches.")
                    return False
            # Add other launch point validations (e.g., drone battery level for flight duration)
            # These are typically handled by ActionMasker but can be duplicated here for robustness.

            path = self.network.calculate_shortest_path(drone.current_node_id, order.customer_node_id, 'drone')

            if not path:
                print(
                    f"NetworkManager: No valid flight path found for Drone {drone_id} to customer {order.customer_node_id}.")
                return False

            drone.set_route(path)
            print(f"NetworkManager: Drone {drone_id} launched with Order {order_id} on route {path}.")
            return True
        except KeyError as e:
            print(f"NetworkManager: Drone launch failed - {e}.")
            return False
        except ValueError as e:
            print(f"NetworkManager: Drone launch failed - {e}.")
            return False

    def drone_landing(self, drone_id: int) -> bool:
        """
        Handles drone landing sequence. If the drone is at its customer destination
        and has the package, it triggers delivery. Otherwise, it lands at its current node.

        Args:
            drone_id (int): The ID of the drone.

        Returns:
            bool: True if landing sequence initiated/completed, False otherwise.
        """
        try:
            drone = self.global_state.get_entity("drone", drone_id)

            if drone.status != "en_route" and drone.status != "idle":
                print(f"NetworkManager: Drone {drone_id} is not en-route or idle, cannot initiate landing.")
                return False

            if drone.current_node_id is None:
                # This implies drone is mid-flight. It should land at the next node in its route.
                if drone.current_route and len(drone.current_route) > 1:
                    landing_node_id = drone.current_route[1]  # Land at next planned node
                    print(f"NetworkManager: Drone {drone_id} will attempt to land at next node {landing_node_id}.")
                    # Drone's move_along_route will handle arrival at this node.
                    # This action effectively tells the drone to proceed to land.
                    # For a direct "land now" from anywhere, more complex logic is needed.
                    # For simplicity, this action implies "proceed to next planned landing spot or current node".
                    drone.set_status("landing_approach")  # Set an intermediate status
                    return True
                else:
                    print(f"NetworkManager: Drone {drone_id} is mid-flight with no clear landing node in route.")
                    return False

            # If already at a node, set status to idle and check for delivery
            landing_node = self.global_state.get_entity("node", drone.current_node_id)

            # Check if current node is a valid recovery point (e.g., depot or active micro-hub)
            if landing_node.type == 'micro_hub':
                micro_hub = self.global_state.get_entity("micro_hub", landing_node.id)
                if micro_hub.operational_status != "active" or micro_hub.is_blocked_for_recoveries:
                    print(f"NetworkManager: MicroHub {micro_hub.id} is not active or blocked for recoveries.")
                    return False

            drone.set_status("idle")  # Drone has landed
            drone.current_route = []  # Clear route after landing
            drone.route_progress = 0.0

            # Check for package delivery (if drone has cargo and is at customer node)
            delivered_orders = []
            for order_id in list(drone.cargo_manifest):  # Iterate over a copy
                order = self.global_state.get_entity("order", order_id)
                if order.customer_node_id == landing_node.id:
                    # Trigger unload/delivery via FleetManager or directly here
                    # For now, we'll directly update order status and remove cargo
                    drone.remove_cargo(order_id)
                    landing_node.add_package(order_id)  # Package is now at customer node
                    order.update_status("delivered")
                    order.delivery_time = self.global_state.current_time
                    delivered_orders.append(order_id)
                    print(
                        f"NetworkManager: Drone {drone_id} delivered Order {order_id} to customer at Node {landing_node.id}.")

            if delivered_orders:
                print(
                    f"NetworkManager: Drone {drone_id} landed at Node {landing_node.id} and delivered {len(delivered_orders)} packages.")
            else:
                print(f"NetworkManager: Drone {drone_id} landed at Node {landing_node.id}.")

            return True
        except KeyError as e:
            print(f"NetworkManager: Drone landing failed - {e}.")
            return False
        except ValueError as e:
            print(f"NetworkManager: Drone landing failed - {e}.")
            return False

    def drone_to_charging_station(self, drone_id: int, charging_station_id: int) -> bool:
        """
        Calculates a flight path for a drone to a charging station and sets its route.

        Args:
            drone_id (int): The ID of the drone.
            charging_station_id (int): The ID of the charging station node.

        Returns:
            bool: True if a valid path was found and set, False otherwise.
        """
        try:
            drone = self.global_state.get_entity("drone", drone_id)
            charging_node = self.global_state.get_entity("node", charging_station_id)

            if drone.status not in ["idle", "en_route"]:  # Drones can be rerouted to charge
                print(
                    f"NetworkManager: Drone {drone_id} is not available to route to charging station (status: {drone.status}).")
                return False
            if not charging_node.is_charging_station:
                print(f"NetworkManager: Node {charging_station_id} is not a charging station.")
                return False

            start_node_for_route = drone.current_node_id
            if start_node_for_route is None:  # If mid-flight, route from next node in path
                if drone.current_route and len(drone.current_route) > 1:
                    start_node_for_route = drone.current_route[1]
                else:
                    print(
                        f"NetworkManager: Drone {drone_id} is mid-flight with no clear starting node for charging route.")
                    return False

            path = self.network.calculate_shortest_path(start_node_for_route, charging_station_id, 'drone')

            if not path:
                print(
                    f"NetworkManager: No valid flight path found for Drone {drone_id} to charging station {charging_station_id}.")
                return False

            drone.set_route(path)
            # Set drone status to en_route. When it arrives, MicroHubsManager will handle charging.
            print(
                f"NetworkManager: Drone {drone_id} route set to charging station {charging_station_id} on path {path}.")
            return True
        except KeyError as e:
            print(f"NetworkManager: Route to charging station failed - {e}.")
            return False
        except ValueError as e:
            print(f"NetworkManager: Route to charging station failed - {e}.")
            return False

    def handle_vehicle_arrival(self, vehicle_id: int, arrived_node_id: int) -> None:
        """
        Internal method called by Vehicle.move_along_route when a vehicle arrives at a node.
        This method can trigger further actions based on the vehicle's purpose or destination.
        """
        # This method is primarily for internal coordination.
        # For example, if a truck arrives at a micro-hub to drop off packages,
        # or a drone arrives at a customer node for delivery.
        # The logic for package transfer/delivery is largely handled by FleetManager.
        # This method might log the arrival or schedule an event.
        print(f"NetworkManager: Vehicle {vehicle_id} arrived at Node {arrived_node_id}.")
        # Additional logic can be added here, e.g.:
        # if vehicle has cargo and arrived at customer_node_id of one of its orders, trigger delivery
        # if vehicle is a drone and arrived at charging station, change status to charging (handled by MicroHubsManager)
        pass

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes network-specific plotting data.
        This method will likely defer to the Network class's initialize_plot_data.
        """
        print("NetworkManager: Initializing plot data.")
        # The Network class itself holds the static graph structure for plotting
        self.network.initialize_plot_data(figure_data)

        # Any network-level initial data (e.g., total edges, total nodes)
        if 'network_overview' not in figure_data:
            figure_data['network_overview'] = {}
        figure_data['network_overview']['total_nodes'] = len(self.global_state.nodes)
        figure_data['network_overview']['total_edges'] = len(self.global_state.edges)

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates network-specific plotting data for the current simulation step.
        This method will likely defer to the Network class's update_plot_data.
        """
        print("NetworkManager: Updating plot data.")
        # The Network class itself updates dynamic graph properties like traffic
        self.network.update_plot_data(figure_data)

        # Any network-level dynamic data (e.g., number of blocked edges)
        if 'network_overview' not in figure_data:
            figure_data['network_overview'] = {}  # Defensive check

        blocked_edges_count = sum(1 for edge in self.global_state.edges.values() if edge.is_blocked)
        figure_data['network_overview']['current_blocked_edges'] = blocked_edges_count
