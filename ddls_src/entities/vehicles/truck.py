from typing import List, Tuple, Any, Dict, Optional
from .base import Vehicle  # Import the base Vehicle class


class Truck(Vehicle):
    """
    Represents a truck vehicle in the simulation.
    Inherits from Vehicle and manages its fuel level.
    """

    def __init__(self, id: int, start_node_id: int, max_payload_capacity: float,
                 max_speed: float, initial_fuel: float, fuel_consumption_rate: float):
        """
        Initializes a Truck.

        Args:
            id (int): Unique identifier for the truck.
            start_node_id (int): The ID of the node where the truck starts.
            max_payload_capacity (float): Maximum weight/volume of cargo the truck can carry.
            max_speed (float): Maximum speed of the truck.
            initial_fuel (float): The starting fuel level of the truck.
            fuel_consumption_rate (float): Rate of fuel consumption per unit of time (e.g., liters/minute).
        """
        super().__init__(id, 'truck', start_node_id, max_payload_capacity, max_speed)
        self.fuel_level: float = initial_fuel
        self.fuel_consumption_rate: float = fuel_consumption_rate  # Rate per unit of time (e.g., per minute)
        self.max_fuel_capacity: float = initial_fuel * 1.5  # Assuming a max capacity, e.g., 1.5x initial fuel

        print(f"Truck {self.id} initialized with {self.fuel_level} fuel.")

    def update_energy(self, delta_amount: float) -> None:
        """
        Updates the truck's fuel level. Positive delta_amount for refueling,
        negative for consumption.

        Args:
            delta_amount (float): The amount to change the fuel level by.
        """
        self.fuel_level += delta_amount
        # Ensure fuel level does not go below zero or exceed max capacity
        self.fuel_level = max(0.0, min(self.fuel_level, self.max_fuel_capacity))

        if self.fuel_level <= 0.0 and self.status == "en_route":
            self.status = "broken_down"  # Or "out_of_fuel"
            print(f"Truck {self.id} ran out of fuel and is now {self.status}.")

    def consume_fuel(self, delta_time: float) -> None:
        """
        Consumes fuel based on the truck's fuel_consumption_rate and delta_time.
        This method will be called during continuous dynamics in LogisticsSimulation.

        Args:
            delta_time (float): The time duration over which fuel is consumed.
        """
        fuel_consumed = self.fuel_consumption_rate * delta_time
        self.update_energy(-fuel_consumed)
        # print(f"Truck {self.id}: Consumed {fuel_consumed:.2f} fuel. Remaining: {self.fuel_level:.2f}")

    # Override move_along_route to include fuel consumption
    # We can call super().move_along_route and then consume fuel based on time_to_move
    def move_along_route(self, delta_time: float, network_manager: 'NetworkManager') -> None:
        """
        Advances the truck along its current route, consuming fuel.
        """
        if self.status != "en_route" or not self.current_route or len(self.current_route) < 2:
            return

        # Get current segment travel time (before movement)
        start_node_id = self.current_route[0]
        end_node_id = self.current_route[1]
        global_state = network_manager.global_state
        network = global_state.network
        edge = network.get_edge_between_nodes(start_node_id, end_node_id)

        if not edge:
            super().move_along_route(delta_time, network_manager)  # Let base class handle stopping
            return

        segment_travel_time = edge.get_current_travel_time()
        if segment_travel_time == float('inf') or segment_travel_time <= 0:
            super().move_along_route(delta_time, network_manager)  # Let base class handle stopping
            return

        # Calculate actual time that will be spent moving in this step
        time_to_cover_remaining_segment = (1.0 - self.route_progress) * segment_travel_time
        time_to_move = min(delta_time, time_to_cover_remaining_segment)

        # Consume fuel for the time spent moving
        self.consume_fuel(time_to_move)

        # Only move if there's still fuel
        if self.fuel_level > 0:
            super().move_along_route(delta_time, network_manager)
        else:
            self.status = "broken_down"  # Or "out_of_fuel"
            print(f"Truck {self.id} ran out of fuel during movement and is now {self.status}.")
            self.current_route = []  # Stop movement

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes the plotting data for this specific truck.
        Calls the base class method and adds truck-specific initial data.
        """
        super().initialize_plot_data(figure_data)  # Call base Vehicle's initializer
        print(f"Truck {self.id}: Initializing truck-specific plot data.")

        # Add truck-specific initial data to the existing vehicle entry
        if 'vehicles' in figure_data and self.id in figure_data['vehicles']:
            figure_data['vehicles'][self.id]['initial_fuel_level'] = self.fuel_level
            figure_data['vehicles'][self.id]['max_fuel_capacity'] = self.max_fuel_capacity

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates the plotting data for this specific truck, reflecting its current state
        (position, status, cargo, and fuel level).
        Calls the base class method and adds truck-specific dynamic data.
        """
        super().update_plot_data(figure_data)  # Call base Vehicle's updater
        print(f"Truck {self.id}: Updating truck-specific plot data. Fuel: {self.fuel_level:.2f}")

        # Update truck-specific dynamic data
        if 'vehicles' in figure_data and self.id in figure_data['vehicles']:
            figure_data['vehicles'][self.id]['current_fuel_level'] = self.fuel_level
