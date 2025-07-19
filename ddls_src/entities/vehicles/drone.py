from typing import List, Tuple, Any, Dict, Optional
from .base import Vehicle  # Import the base Vehicle class


# Forward declaration for NetworkManager to avoid circular dependency
class NetworkManager:
    pass


class Drone(Vehicle):
    """
    Represents a drone vehicle in the simulation.
    Inherits from Vehicle and manages its battery level.
    """

    def __init__(self, id: int, start_node_id: int, max_payload_capacity: float,
                 max_speed: float, initial_battery: float, battery_drain_rate_flying: float,
                 battery_drain_rate_idle: float, battery_charge_rate: float):
        """
        Initializes a Drone.

        Args:
            id (int): Unique identifier for the drone.
            start_node_id (int): The ID of the node where the drone starts.
            max_payload_capacity (float): Maximum weight/volume of cargo the drone can carry.
            max_speed (float): Maximum speed of the drone.
            initial_battery (float): The starting battery level of the drone (0.0 to 1.0, or percentage).
            battery_drain_rate_flying (float): Rate of battery drain per unit of time when flying.
            battery_drain_rate_idle (float): Rate of battery drain per unit of time when idle/at rest.
            battery_charge_rate (float): Rate of battery charge per unit of time.
        """
        super().__init__(id, 'drone', start_node_id, max_payload_capacity, max_speed)
        self.battery_level: float = initial_battery
        self.battery_drain_rate_flying: float = battery_drain_rate_flying
        self.battery_drain_rate_idle: float = battery_drain_rate_idle
        self.battery_charge_rate: float = battery_charge_rate
        self.max_battery_capacity: float = 1.0  # Assuming battery level is normalized 0.0 to 1.0

        # Additional drone-specific states
        self.max_flight_time: float = (
                    initial_battery / battery_drain_rate_flying) if battery_drain_rate_flying > 0 else float('inf')

        print(f"Drone {self.id} initialized with {self.battery_level * 100:.1f}% battery.")

    def update_energy(self, delta_amount: float) -> None:
        """
        Updates the drone's battery level. Positive delta_amount for charging,
        negative for draining.

        Args:
            delta_amount (float): The amount to change the battery level by.
        """
        self.battery_level += delta_amount
        # Ensure battery level does not go below zero or exceed max capacity
        self.battery_level = max(0.0, min(self.battery_level, self.max_battery_capacity))

        if self.battery_level <= 0.0 and self.status == "en_route":
            self.status = "broken_down"  # Or "out_of_battery"
            print(f"Drone {self.id} ran out of battery and is now {self.status}.")

    def drain_battery(self, delta_time: float) -> None:
        """
        Drains the battery based on the drone's current status (flying or idle).

        Args:
            delta_time (float): The time duration over which battery is drained.
        """
        if self.status == "en_route":
            drain_rate = self.battery_drain_rate_flying
        else:  # idle, loading, unloading, etc.
            drain_rate = self.battery_drain_rate_idle

        battery_consumed = drain_rate * delta_time
        self.update_energy(-battery_consumed)
        # print(f"Drone {self.id}: Drained {battery_consumed*100:.1f}% battery. Remaining: {self.battery_level*100:.1f}%")

    def charge_battery(self, delta_time: float) -> None:
        """
        Charges the battery based on the drone's battery_charge_rate and delta_time.
        This method will be called during continuous dynamics in LogisticsSimulation
        when the drone is in a 'charging' status.

        Args:
            delta_time (float): The time duration over which battery is charged.
        """
        if self.status != "charging":
            # print(f"Drone {self.id}: Not in 'charging' status, cannot charge.")
            return

        battery_charged = self.battery_charge_rate * delta_time
        self.update_energy(battery_charged)
        # print(f"Drone {self.id}: Charged {battery_charged*100:.1f}% battery. Remaining: {self.battery_level*100:.1f}%")

    # Override move_along_route to include battery consumption
    def move_along_route(self, delta_time: float, network_manager: 'NetworkManager') -> None:
        """
        Advances the drone along its current route, consuming battery.
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

        segment_travel_time = edge.get_drone_flight_time()
        if segment_travel_time == float('inf') or segment_travel_time <= 0:
            super().move_along_route(delta_time, network_manager)  # Let base class handle stopping
            return

        # Calculate actual time that will be spent moving in this step
        time_to_cover_remaining_segment = (1.0 - self.route_progress) * segment_travel_time
        time_to_move = min(delta_time, time_to_cover_remaining_segment)

        # Consume battery for the time spent moving (flying drain rate)
        self.drain_battery(time_to_move)

        # Only move if there's still battery
        if self.battery_level > 0:
            super().move_along_route(delta_time, network_manager)
        else:
            self.status = "broken_down"  # Or "out_of_battery"
            print(f"Drone {self.id} ran out of battery during movement and is now {self.status}.")
            self.current_route = []  # Stop movement

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes the plotting data for this specific drone.
        Calls the base class method and adds drone-specific initial data.
        """
        super().initialize_plot_data(figure_data)  # Call base Vehicle's initializer
        print(f"Drone {self.id}: Initializing drone-specific plot data.")

        # Add drone-specific initial data to the existing vehicle entry
        if 'vehicles' in figure_data and self.id in figure_data['vehicles']:
            figure_data['vehicles'][self.id]['initial_battery_level'] = self.battery_level
            figure_data['vehicles'][self.id]['max_battery_capacity'] = self.max_battery_capacity
            figure_data['vehicles'][self.id]['max_flight_time'] = self.max_flight_time

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates the plotting data for this specific drone, reflecting its current state
        (position, status, cargo, and battery level).
        Calls the base class method and adds drone-specific dynamic data.
        """
        super().update_plot_data(figure_data)  # Call base Vehicle's updater
        print(f"Drone {self.id}: Updating drone-specific plot data. Battery: {self.battery_level * 100:.1f}%")

        # Update drone-specific dynamic data
        if 'vehicles' in figure_data and self.id in figure_data['vehicles']:
            figure_data['vehicles'][self.id]['current_battery_level'] = self.battery_level
