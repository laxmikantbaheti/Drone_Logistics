from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional


# Forward declaration for NetworkManager to avoid circular dependency
class NetworkManager:
    pass


class Vehicle(ABC):
    """
    Abstract base class for all vehicles in the simulation (e.g., Trucks, Drones).
    Defines common attributes and abstract methods that concrete vehicle types must implement.
    """

    def __init__(self, id: int, type: str, start_node_id: int,
                 max_payload_capacity: float, max_speed: float):
        """
        Initializes a Vehicle.

        Args:
            id (int): Unique identifier for the vehicle.
            type (str): The type of the vehicle (e.g., 'truck', 'drone').
            start_node_id (int): The ID of the node where the vehicle starts.
            max_payload_capacity (float): Maximum weight/volume of cargo the vehicle can carry.
            max_speed (float): Maximum speed of the vehicle (e.g., in km/h or units/minute).
        """
        self.id: int = id
        self.type: str = type
        self.current_node_id: Optional[int] = start_node_id  # Current node if at a node, None if en-route
        self.current_location_coords: Tuple[float, float] = (0.0, 0.0)  # Will be updated by NetworkManager
        self.status: str = "idle"  # e.g., "idle", "en_route", "loading", "unloading", "charging", "maintenance"
        self.cargo_manifest: List[int] = []  # List of order IDs currently carried by the vehicle
        self.max_payload_capacity: float = max_payload_capacity
        self.max_speed: float = max_speed
        self.current_route: List[int] = []  # List of node IDs representing the planned route
        self.route_progress: float = 0.0  # Progress along the current segment (0.0 to 1.0)
        self.distance_traveled_current_segment: float = 0.0  # Distance covered on current segment

        print(f"Vehicle {self.id} (Type: {self.type}) initialized at node {self.current_node_id}.")

    @abstractmethod
    def update_energy(self, delta_amount: float) -> None:
        """
        Abstract method to update the vehicle's energy level (fuel or battery).
        Must be implemented by concrete vehicle types.

        Args:
            delta_amount (float): The amount to change the energy level by.
                                  Positive for gain (e.g., charging), negative for consumption.
        """
        pass

    def move_along_route(self, delta_time: float, network_manager: 'NetworkManager') -> None:
        """
        Advances the vehicle along its current route based on delta_time.
        Updates current_location_coords and current_node_id if an arrival occurs.
        This method relies on NetworkManager for route segment details and node coordinates.

        Args:
            delta_time (float): The time duration to simulate movement for.
            network_manager (NetworkManager): Reference to the NetworkManager for route calculations.
        """
        if self.status != "en_route" or not self.current_route or len(self.current_route) < 2:
            return  # Vehicle not en-route or no valid route

        # Get current segment
        start_node_id = self.current_route[0]
        end_node_id = self.current_route[1]

        # Get edge information from NetworkManager (which uses the Network graph)
        # Assuming NetworkManager has a method to get edge details or calculate segment time
        # For now, we'll directly use the Network class (via GlobalState's Network reference)
        # This will require GlobalState to have a Network instance, and Network to have get_edge_between_nodes
        global_state = network_manager.global_state  # NetworkManager has global_state
        network = global_state.network  # GlobalState has network

        edge = network.get_edge_between_nodes(start_node_id, end_node_id)
        if not edge:
            print(f"Vehicle {self.id}: No edge found between {start_node_id} and {end_node_id}. Stopping movement.")
            self.status = "idle"  # Or "stuck"
            self.current_route = []
            return

        # Calculate travel time for this segment based on vehicle type
        if self.type == 'truck':
            segment_travel_time = edge.get_current_travel_time()
        elif self.type == 'drone':
            segment_travel_time = edge.get_drone_flight_time()
        else:
            print(f"Vehicle {self.id}: Unknown vehicle type '{self.type}' for movement calculation.")
            return

        if segment_travel_time == float('inf') or segment_travel_time <= 0:
            print(
                f"Vehicle {self.id}: Segment {start_node_id}-{end_node_id} is impassable or has zero travel time. Stopping movement.")
            self.status = "idle"  # Or "stuck"
            self.current_route = []
            return

        # Calculate how much of the segment can be covered in delta_time
        time_to_cover_remaining_segment = (1.0 - self.route_progress) * segment_travel_time
        time_to_move = min(delta_time, time_to_cover_remaining_segment)

        # Update route progress
        progress_increment = time_to_move / segment_travel_time
        self.route_progress += progress_increment

        # Update energy based on movement
        # Assuming energy consumption rate is per unit of time or distance
        # For now, we'll assume consumption is proportional to time spent moving.
        # Concrete classes will implement update_energy
        # self.update_energy(-self.energy_consumption_rate * time_to_move) # Example

        # Update current location coordinates
        start_coords = global_state.get_entity("node", start_node_id).coords
        end_coords = global_state.get_entity("node", end_node_id).coords

        # Linear interpolation for coordinates
        self.current_location_coords = (
            start_coords[0] + (end_coords[0] - start_coords[0]) * self.route_progress,
            start_coords[1] + (end_coords[1] - start_coords[1]) * self.route_progress
        )

        # Check for arrival at the next node
        if self.route_progress >= 1.0:
            self.current_node_id = end_node_id  # Vehicle has arrived at the next node
            self.current_route.pop(0)  # Remove the node just arrived at

            if not self.current_route:  # Route completed
                self.status = "idle"
                self.route_progress = 0.0
                self.distance_traveled_current_segment = 0.0
                # print(f"Vehicle {self.id} arrived at final destination node {self.current_node_id}.")
            else:  # Move to the next segment
                self.route_progress = 0.0  # Reset progress for the new segment
                self.distance_traveled_current_segment = 0.0
                # print(f"Vehicle {self.id} arrived at intermediate node {self.current_node_id}. Continuing to {self.current_route[1]}.")

            # Any arrival events (e.g., package delivery) would be triggered here or by NetworkManager
            # For now, NetworkManager is responsible for this.
            network_manager.handle_vehicle_arrival(self.id, self.current_node_id)

    def set_status(self, new_status: str) -> None:
        """
        Sets the current status of the vehicle.

        Args:
            new_status (str): The new status (e.g., "idle", "en_route", "loading").
        """
        valid_statuses = ["idle", "en_route", "loading", "unloading", "charging", "maintenance", "broken_down"]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid vehicle status: {new_status}. Must be one of {valid_statuses}")
        self.status = new_status
        # print(f"Vehicle {self.id}: Status set to {new_status}.")

    def add_cargo(self, order_id: int) -> None:
        """
        Adds an order ID to the vehicle's cargo manifest.

        Args:
            order_id (int): The ID of the order to add.
        """
        if len(self.cargo_manifest) >= self.max_payload_capacity:  # Simple count-based capacity
            print(f"Vehicle {self.id}: Cannot add cargo {order_id}, payload capacity full.")
            return
        if order_id not in self.cargo_manifest:
            self.cargo_manifest.append(order_id)
            # print(f"Vehicle {self.id}: Added cargo {order_id}. Manifest: {self.cargo_manifest}")

    def remove_cargo(self, order_id: int) -> None:
        """
        Removes an order ID from the vehicle's cargo manifest.

        Args:
            order_id (int): The ID of the order to remove.
        """
        if order_id in self.cargo_manifest:
            self.cargo_manifest.remove(order_id)
            # print(f"Vehicle {self.id}: Removed cargo {order_id}. Manifest: {self.cargo_manifest}")
        # else:
        # print(f"Vehicle {self.id}: Cargo {order_id} not found to remove.")

    def get_cargo(self) -> List[int]:
        """
        Returns a list of order IDs currently in the vehicle's cargo manifest.

        Returns:
            List[int]: A list of integer order IDs.
        """
        return list(self.cargo_manifest)  # Return a copy

    def set_route(self, route_nodes: List[int]) -> None:
        """
        Sets the planned route for the vehicle as a list of node IDs.
        Resets route progress and sets status to 'en_route'.

        Args:
            route_nodes (List[int]): A list of node IDs representing the route.
                                     Must contain at least two nodes (start and end).
        """
        if not route_nodes or len(route_nodes) < 2:
            print(f"Vehicle {self.id}: Invalid route provided. Must have at least two nodes.")
            self.current_route = []
            self.status = "idle"
            return

        if self.current_node_id != route_nodes[0]:
            # This can happen if vehicle is already en-route and re-routing,
            # or if it's idle at a different node.
            # For simplicity, we assume the first node in the route is the current location.
            # More complex logic would involve pathfinding from current coords to route_nodes[0]
            print(
                f"Vehicle {self.id}: Warning - Route starts at {route_nodes[0]} but vehicle is at {self.current_node_id}.")
            # For now, we'll force the vehicle to the start of the new route if it's idle.
            if self.status == "idle":
                self.current_node_id = route_nodes[0]
                # Update current_location_coords based on the node's coordinates
                # This requires access to GlobalState, which is typically via NetworkManager or LogisticsSimulation
                # For now, assume it's handled by the calling manager.
                # self.current_location_coords = global_state.get_entity("node", self.current_node_id).coords

        self.current_route = list(route_nodes)
        self.route_progress = 0.0
        self.distance_traveled_current_segment = 0.0
        self.status = "en_route"
        # print(f"Vehicle {self.id}: Route set to {self.current_route}. Status: {self.status}")

    def get_current_route_segment(self) -> Optional[Tuple[int, int]]:
        """
        Returns the current segment of the route (start_node_id, end_node_id).

        Returns:
            Optional[Tuple[int, int]]: A tuple of (start_node_id, end_node_id) or None if no active route.
        """
        if self.current_route and len(self.current_route) >= 2:
            return (self.current_route[0], self.current_route[1])
        return None

    def is_at_node(self, node_id: int) -> bool:
        """
        Checks if the vehicle is currently located at a specific node.

        Args:
            node_id (int): The ID of the node to check against.

        Returns:
            bool: True if the vehicle's current_node_id matches the given node_id.
        """
        return self.current_node_id == node_id

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes the plotting data for this specific vehicle.
        This is typically called once at the start of the simulation.
        """
        print(f"Vehicle {self.id}: Initializing plot data.")
        # This will contribute to a 'vehicles' layer in figure_data
        if 'vehicles' not in figure_data:
            figure_data['vehicles'] = {}

        figure_data['vehicles'][self.id] = {
            'type': self.type,
            'initial_coords': self.current_location_coords,  # Or lookup from start_node_id
            'status': self.status,
            'cargo_count': len(self.cargo_manifest)
        }

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates the plotting data for this specific vehicle, reflecting its current state
        (position, status, cargo).
        This method is called at each simulation timestep.
        """
        print(f"Vehicle {self.id}: Updating plot data. Status: {self.status}, Coords: {self.current_location_coords}")
        if 'vehicles' not in figure_data:
            figure_data['vehicles'] = {}  # Defensive check

        # Update dynamic properties
        vehicle_data = figure_data['vehicles'].get(self.id, {})
        vehicle_data['current_coords'] = self.current_location_coords
        vehicle_data['status'] = self.status
        vehicle_data['cargo_count'] = len(self.cargo_manifest)
        vehicle_data['current_node_id'] = self.current_node_id  # Useful for snapping to node if arrived

        # Specific energy level update will be handled by Truck/Drone update_plot_data

        figure_data['vehicles'][self.id] = vehicle_data
