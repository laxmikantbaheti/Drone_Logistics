from typing import List, Dict, Any, Optional


# Forward declaration for GlobalState to avoid circular dependency
class GlobalState:
    pass


# Forward declarations for FleetManager and MicroHubsManager, which will be
# instantiated and managed by ResourceManager
class FleetManager:
    pass


class MicroHubsManager:
    pass


class ResourceManager:
    """
    Manages various resources within the simulation, including vehicle maintenance
    and the availability of services at micro-hubs. It orchestrates actions
    related to fleet and micro-hub resources.
    """

    def __init__(self, global_state: 'GlobalState'):
        """
        Initializes the ResourceManager.

        Args:
            global_state (GlobalState): Reference to the central GlobalState.
        """
        self.global_state = global_state

        # Instantiate sub-managers. These will handle more granular resource operations.
        # They are initialized here, but their methods will be called by ResourceManager
        # or directly by ActionManager if the action dispatch is granular enough.
        self.fleet_manager: FleetManager = FleetManager(global_state)
        self.micro_hubs_manager: MicroHubsManager = MicroHubsManager(global_state)

        print("ResourceManager initialized.")

    def flag_vehicle_for_maintenance(self, vehicle_id: int) -> bool:
        """
        Flags a vehicle for maintenance, changing its status.

        Args:
            vehicle_id (int): The ID of the vehicle to flag.

        Returns:
            bool: True if the vehicle was found and flagged, False otherwise.
        """
        try:
            vehicle = self.global_state.get_vehicle_status(vehicle_id)  # Use generic getter for vehicle status

            # Determine if it's a truck or drone to get the actual object
            if vehicle_id in self.global_state.trucks:
                vehicle_obj = self.global_state.get_entity("truck", vehicle_id)
            elif vehicle_id in self.global_state.drones:
                vehicle_obj = self.global_state.get_entity("drone", vehicle_id)
            else:
                print(f"ResourceManager: Vehicle {vehicle_id} not found for maintenance flagging.")
                return False

            if vehicle_obj.status not in ["maintenance", "broken_down"]:
                vehicle_obj.set_status("maintenance")
                # If vehicle was en-route, it should stop movement. This is handled by set_status.
                # Any cargo should be handled (e.g., dropped at current node or transferred).
                # For now, just status change.
                print(f"ResourceManager: Vehicle {vehicle_id} flagged for maintenance.")
                return True
            else:
                print(f"ResourceManager: Vehicle {vehicle_id} is already in maintenance or broken down.")
                return False
        except KeyError as e:
            print(f"ResourceManager: Failed to flag vehicle {vehicle_id} for maintenance - {e}.")
            return False

    def release_vehicle_from_maintenance(self, vehicle_id: int) -> bool:
        """
        Releases a vehicle from maintenance, changing its status back to 'idle'.

        Args:
            vehicle_id (int): The ID of the vehicle to release.

        Returns:
            bool: True if the vehicle was found and released, False otherwise.
        """
        try:
            # Determine if it's a truck or drone to get the actual object
            if vehicle_id in self.global_state.trucks:
                vehicle_obj = self.global_state.get_entity("truck", vehicle_id)
            elif vehicle_id in self.global_state.drones:
                vehicle_obj = self.global_state.get_entity("drone", vehicle_id)
            else:
                print(f"ResourceManager: Vehicle {vehicle_id} not found for maintenance release.")
                return False

            if vehicle_obj.status == "maintenance":
                vehicle_obj.set_status("idle")  # Or a more appropriate initial status
                print(f"ResourceManager: Vehicle {vehicle_id} released from maintenance.")
                return True
            else:
                print(f"ResourceManager: Vehicle {vehicle_id} is not currently in maintenance.")
                return False
        except KeyError as e:
            print(f"ResourceManager: Failed to release vehicle {vehicle_id} from maintenance - {e}.")
            return False

    def flag_unavailability_of_service_at_micro_hub(self, micro_hub_id: int, service_type: str) -> bool:
        """
        Flags a specific service at a micro-hub as unavailable.

        Args:
            micro_hub_id (int): The ID of the micro-hub.
            service_type (str): The type of service to flag (e.g., 'charging', 'package_transfer', 'launches', 'recoveries').

        Returns:
            bool: True if the micro-hub was found and service flagged, False otherwise.
        """
        try:
            micro_hub = self.global_state.get_entity("micro_hub", micro_hub_id)
            micro_hub.flag_service_unavailable(service_type)
            print(f"ResourceManager: MicroHub {micro_hub_id} service '{service_type}' flagged unavailable.")
            return True
        except KeyError:
            print(f"ResourceManager: MicroHub {micro_hub_id} not found.")
            return False
        except Exception as e:  # Catching potential errors from micro_hub.flag_service_unavailable
            print(f"ResourceManager: Error flagging service at MicroHub {micro_hub_id}: {e}")
            return False

    def release_unavailability_of_service_at_micro_hub(self, micro_hub_id: int, service_type: str) -> bool:
        """
        Releases a specific service at a micro-hub, making it available again.

        Args:
            micro_hub_id (int): The ID of the micro-hub.
            service_type (str): The type of service to release.

        Returns:
            bool: True if the micro-hub was found and service released, False otherwise.
        """
        try:
            micro_hub = self.global_state.get_entity("micro_hub", micro_hub_id)
            micro_hub.release_service_available(service_type)
            print(f"ResourceManager: MicroHub {micro_hub_id} service '{service_type}' released.")
            return True
        except KeyError:
            print(f"ResourceManager: MicroHub {micro_hub_id} not found.")
            return False
        except Exception as e:  # Catching potential errors from micro_hub.release_service_available
            print(f"ResourceManager: Error releasing service at MicroHub {micro_hub_id}: {e}")
            return False

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes any resource management specific plotting data.
        This might include initial vehicle statuses or micro-hub service availability.
        It also calls the sub-managers' plotting initialization.
        """
        print("ResourceManager: Initializing plot data.")
        # Call sub-managers' plotting initializers
        self.fleet_manager.initialize_plot_data(figure_data)
        self.micro_hubs_manager.initialize_plot_data(figure_data)

        # Add any top-level ResourceManager specific data here if needed
        # For example, a summary of all vehicles currently in maintenance
        if 'resource_overview' not in figure_data:
            figure_data['resource_overview'] = {}

        vehicles_in_maintenance = [
                                      v.id for v in self.global_state.get_all_entities("truck").values() if
                                      v.status == "maintenance"
                                  ] + [
                                      v.id for v in self.global_state.get_all_entities("drone").values() if
                                      v.status == "maintenance"
                                  ]
        figure_data['resource_overview']['initial_vehicles_in_maintenance'] = vehicles_in_maintenance

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates resource management specific plotting data for the current simulation step.
        This might include real-time vehicle status, micro-hub service status, etc.
        It also calls the sub-managers' plotting updates.
        """
        print("ResourceManager: Updating plot data.")
        # Call sub-managers' plotting updaters
        self.fleet_manager.update_plot_data(figure_data)
        self.micro_hubs_manager.update_plot_data(figure_data)

        # Update any top-level ResourceManager specific data here
        if 'resource_overview' not in figure_data:
            figure_data['resource_overview'] = {}  # Defensive check

        current_vehicles_in_maintenance = [
                                              v.id for v in self.global_state.get_all_entities("truck").values() if
                                              v.status == "maintenance"
                                          ] + [
                                              v.id for v in self.global_state.get_all_entities("drone").values() if
                                              v.status == "maintenance"
                                          ]
        figure_data['resource_overview']['current_vehicles_in_maintenance'] = current_vehicles_in_maintenance
