from typing import List, Tuple, Dict, Any, Optional
from .node import Node  # Import the base Node class


class MicroHub(Node):
    """
    Represents a micro-hub in the simulation network.
    Micro-hubs are specialized nodes that can activate/deactivate,
    provide charging slots for drones, and hold packages for transfer/consolidation.
    It inherits from Node.
    """

    def __init__(self, id: int, coords: Tuple[float, float], num_charging_slots: int, type: str = 'micro_hub'):
        """
        Initializes a MicroHub.

        Args:
            id (int): Unique identifier for the micro-hub.
            coords (Tuple[float, float]): (x, y) coordinates of the micro-hub.
            num_charging_slots (int): The total number of drone charging slots available at this hub.
            type (str): The type of the node. Defaults to 'micro_hub'.
        """
        # MicroHubs are typically loadable, unloadable, and charging stations
        super().__init__(id, coords, type, is_loadable=True, is_unloadable=True, is_charging_station=True)

        self.operational_status: str = "inactive"  # e.g., "active", "inactive"
        # Dictionary to track charging slots: {slot_id: drone_id | None}
        self.charging_slots: Dict[int, Optional[int]] = {i: None for i in range(num_charging_slots)}

        # Flags for service unavailability due to maintenance or other issues
        self.is_blocked_for_launches: bool = False
        self.is_blocked_for_recoveries: bool = False
        self.is_package_transfer_unavailable: bool = False  # For package sorting/transfer service

        print(f"MicroHub {self.id} initialized at {self.coords} with {num_charging_slots} charging slots.")

    def activate(self) -> None:
        """Sets the operational status of the micro-hub to 'active'."""
        self.operational_status = 'active'
        print(f"MicroHub {self.id} activated.")

    def deactivate(self) -> None:
        """Sets the operational status of the micro-hub to 'inactive'."""
        self.operational_status = 'inactive'
        print(f"MicroHub {self.id} deactivated.")

    def assign_charging_slot(self, slot_id: int, drone_id: int) -> bool:
        """
        Assigns a drone to a specific charging slot.

        Args:
            slot_id (int): The ID of the charging slot to assign.
            drone_id (int): The ID of the drone to assign to the slot.

        Returns:
            bool: True if the slot was successfully assigned, False otherwise.
        """
        if slot_id not in self.charging_slots:
            print(f"MicroHub {self.id}: Slot {slot_id} does not exist.")
            return False
        if self.charging_slots[slot_id] is not None:
            print(f"MicroHub {self.id}: Slot {slot_id} is already occupied by drone {self.charging_slots[slot_id]}.")
            return False

        self.charging_slots[slot_id] = drone_id
        # print(f"MicroHub {self.id}: Drone {drone_id} assigned to slot {slot_id}.")
        return True

    def release_charging_slot(self, slot_id: int) -> bool:
        """
        Releases a specific charging slot, making it available.

        Args:
            slot_id (int): The ID of the charging slot to release.

        Returns:
            bool: True if the slot was successfully released, False otherwise.
        """
        if slot_id not in self.charging_slots:
            print(f"MicroHub {self.id}: Slot {slot_id} does not exist.")
            return False
        if self.charging_slots[slot_id] is None:
            print(f"MicroHub {self.id}: Slot {slot_id} is already free.")
            return False

        released_drone_id = self.charging_slots[slot_id]
        self.charging_slots[slot_id] = None
        # print(f"MicroHub {self.id}: Slot {slot_id} released (was occupied by drone {released_drone_id}).")
        return True

    def get_available_charging_slots(self) -> List[int]:
        """
        Returns a list of IDs of currently available (unoccupied) charging slots.

        Returns:
            List[int]: A list of available slot IDs.
        """
        return [slot_id for slot_id, drone_id in self.charging_slots.items() if drone_id is None]

    # Override add_package and remove_package to reflect micro-hub's role in holding packages
    def add_package_to_holding(self, order_id: int) -> None:
        """
        Adds a package (order ID) to the micro-hub's holding area.
        This is an alias for the base Node's add_package.
        """
        super().add_package(order_id)
        # print(f"MicroHub {self.id}: Package {order_id} added to holding. Total: {len(self.packages_held)}")

    def remove_package_from_holding(self, order_id: int) -> None:
        """
        Removes a package (order ID) from the micro-hub's holding area.
        This is an alias for the base Node's remove_package.
        """
        super().remove_package(order_id)
        # print(f"MicroHub {self.id}: Package {order_id} removed from holding. Total: {len(self.packages_held)}")

    def flag_service_unavailable(self, service_type: str) -> None:
        """
        Sets a specific service at the micro-hub as unavailable.

        Args:
            service_type (str): The type of service to flag (e.g., 'charging', 'package_transfer', 'launches', 'recoveries').
        """
        if service_type == 'charging':
            # This would block new drone assignments to charging slots
            # Existing charging drones might continue or be interrupted based on more detailed logic
            pass  # No direct flag for 'charging' service, managed by slot availability and operational status
        elif service_type == 'package_transfer':
            self.is_package_transfer_unavailable = True
        elif service_type == 'launches':
            self.is_blocked_for_launches = True
        elif service_type == 'recoveries':
            self.is_blocked_for_recoveries = True
        else:
            print(f"MicroHub {self.id}: Unknown service type to flag unavailable: {service_type}")
        # print(f"MicroHub {self.id}: Service '{service_type}' flagged as unavailable.")

    def release_service_available(self, service_type: str) -> None:
        """
        Releases a specific service at the micro-hub, making it available again.

        Args:
            service_type (str): The type of service to release.
        """
        if service_type == 'charging':
            pass
        elif service_type == 'package_transfer':
            self.is_package_transfer_unavailable = False
        elif service_type == 'launches':
            self.is_blocked_for_launches = False
        elif service_type == 'recoveries':
            self.is_blocked_for_recoveries = False
        else:
            print(f"MicroHub {self.id}: Unknown service type to release: {service_type}")
        # print(f"MicroHub {self.id}: Service '{service_type}' released as available.")

    def block_launches(self) -> None:
        """Sets the micro-hub as blocked for drone launches."""
        self.is_blocked_for_launches = True
        # print(f"MicroHub {self.id}: Blocked for launches.")

    def unblock_launches(self) -> None:
        """Unsets the micro-hub as blocked for drone launches."""
        self.is_blocked_for_launches = False
        # print(f"MicroHub {self.id}: Unblocked for launches.")

    def block_recoveries(self) -> None:
        """Sets the micro-hub as blocked for drone recoveries (landings)."""
        self.is_blocked_for_recoveries = True
        # print(f"MicroHub {self.id}: Blocked for recoveries.")

    def unblock_recoveries(self) -> None:
        """Unsets the micro-hub as blocked for drone recoveries (landings)."""
        self.is_blocked_for_recoveries = False
        # print(f"MicroHub {self.id}: Unblocked for recoveries.")

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes the plotting data for this specific micro-hub.
        Calls the base Node's method and adds micro-hub specific initial data.
        """
        super().initialize_plot_data(figure_data)  # Call base Node's initializer
        print(f"MicroHub {self.id}: Initializing micro-hub specific plot data.")

        # Add micro-hub specific initial data to the existing node entry
        # Assuming 'node_details' is where base Node data is stored
        if 'node_details' in figure_data and self.id in figure_data['node_details']:
            figure_data['node_details'][self.id]['operational_status'] = self.operational_status
            figure_data['node_details'][self.id]['num_charging_slots'] = len(self.charging_slots)
            figure_data['node_details'][self.id]['initial_available_slots'] = len(self.get_available_charging_slots())

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates the plotting data for this specific micro-hub, reflecting its current state
        (operational status, charging slot occupancy, package count, blocked status).
        Calls the base Node's method and adds micro-hub specific dynamic data.
        """
        super().update_plot_data(figure_data)  # Call base Node's updater
        print(f"MicroHub {self.id}: Updating micro-hub specific plot data. Status: {self.operational_status}")

        # Update micro-hub specific dynamic data
        # Assuming 'node_dynamic_data' is where base Node dynamic data is stored
        if 'node_dynamic_data' in figure_data and self.id in figure_data['node_dynamic_data']:
            figure_data['node_dynamic_data'][self.id]['operational_status'] = self.operational_status
            figure_data['node_dynamic_data'][self.id]['occupied_charging_slots'] = len(self.charging_slots) - len(
                self.get_available_charging_slots())
            figure_data['node_dynamic_data'][self.id]['is_blocked_for_launches'] = self.is_blocked_for_launches
            figure_data['node_dynamic_data'][self.id]['is_blocked_for_recoveries'] = self.is_blocked_for_recoveries
            figure_data['node_dynamic_data'][self.id][
                'is_package_transfer_unavailable'] = self.is_package_transfer_unavailable
            # The visualization layer would then interpret these values to draw the micro-hub
