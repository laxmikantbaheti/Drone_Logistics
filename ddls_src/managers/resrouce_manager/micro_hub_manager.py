from typing import List, Dict, Any, Optional


# Forward declaration for GlobalState to avoid circular dependency
class GlobalState:
    pass


# Forward declarations for MicroHub, Drone for type hinting
class MicroHub: pass


class Drone: pass


class MicroHubsManager:
    """
    Manages operations related to micro-hubs, including their activation/deactivation,
    and the assignment of drones to charging queues/slots.
    It interacts with GlobalState to modify micro-hub and drone states.
    """

    def __init__(self, global_state: 'GlobalState'):
        """
        Initializes the MicroHubsManager.

        Args:
            global_state (GlobalState): Reference to the central GlobalState.
        """
        self.global_state = global_state
        print("MicroHubsManager initialized.")

    def activate_micro_hub(self, micro_hub_id: int) -> bool:
        """
        Activates a specific micro-hub, changing its operational status to 'active'.

        Args:
            micro_hub_id (int): The ID of the micro-hub to activate.

        Returns:
            bool: True if the micro-hub was found and activated, False otherwise.
        """
        try:
            micro_hub = self.global_state.get_entity("micro_hub", micro_hub_id)
            if micro_hub.operational_status == "inactive":
                micro_hub.activate()
                print(f"MicroHubsManager: MicroHub {micro_hub_id} activated.")
                return True
            else:
                print(f"MicroHubsManager: MicroHub {micro_hub_id} is already active.")
                return False
        except KeyError as e:
            print(f"MicroHubsManager: MicroHub {micro_hub_id} not found for activation - {e}.")
            return False

    def deactivate_micro_hub(self, micro_hub_id: int) -> bool:
        """
        Deactivates a specific micro-hub, changing its operational status to 'inactive'.
        This might involve removing drones from charging slots, or moving packages.

        Args:
            micro_hub_id (int): The ID of the micro-hub to deactivate.

        Returns:
            bool: True if the micro-hub was found and deactivated, False otherwise.
        """
        try:
            micro_hub = self.global_state.get_entity("micro_hub", micro_hub_id)
            if micro_hub.operational_status == "active":
                # Release any drones currently charging at this hub
                for slot_id, drone_id in list(micro_hub.charging_slots.items()):  # Iterate over a copy
                    if drone_id is not None:
                        micro_hub.release_charging_slot(slot_id)
                        # Optionally, change drone status from 'charging' to 'idle'
                        try:
                            drone = self.global_state.get_entity("drone", drone_id)
                            if drone.status == "charging":
                                drone.set_status("idle")
                                print(
                                    f"MicroHubsManager: Drone {drone_id} released from charging due to hub deactivation.")
                        except KeyError:
                            print(
                                f"MicroHubsManager: Warning - Charging drone {drone_id} not found during hub deactivation.")

                micro_hub.deactivate()
                print(f"MicroHubsManager: MicroHub {micro_hub_id} deactivated.")
                return True
            else:
                print(f"MicroHubsManager: MicroHub {micro_hub_id} is already inactive.")
                return False
        except KeyError as e:
            print(f"MicroHubsManager: MicroHub {micro_hub_id} not found for deactivation - {e}.")
            return False

    def add_to_charging_queue(self, micro_hub_id: int, drone_id: int, priority_level: int = 0) -> bool:
        """
        Manages charging slot assignment for a drone at a micro-hub.
        This method attempts to assign a drone to an available slot.

        Args:
            micro_hub_id (int): The ID of the micro-hub.
            drone_id (int): The ID of the drone requesting a charge.
            priority_level (int): Priority for charging (higher value = higher priority).

        Returns:
            bool: True if the drone was successfully assigned a charging slot, False otherwise.
        """
        try:
            micro_hub = self.global_state.get_entity("micro_hub", micro_hub_id)
            drone = self.global_state.get_entity("drone", drone_id)

            if micro_hub.operational_status != "active":
                print(f"MicroHubsManager: MicroHub {micro_hub_id} is not active for charging.")
                return False
            if not micro_hub.is_charging_station:
                print(f"MicroHubsManager: MicroHub {micro_hub_id} is not configured as a charging station.")
                return False
            if drone.current_node_id != micro_hub_id:
                print(f"MicroHubsManager: Drone {drone_id} is not at MicroHub {micro_hub_id} to charge.")
                return False

            available_slots = micro_hub.get_available_charging_slots()
            if not available_slots:
                print(f"MicroHubsManager: No available charging slots at MicroHub {micro_hub_id}.")
                return False

            # Assign to the first available slot (simple logic for now)
            slot_to_assign = available_slots[0]
            if micro_hub.assign_charging_slot(slot_to_assign, drone_id):
                drone.set_status("charging")  # Set drone status to charging
                print(
                    f"MicroHubsManager: Drone {drone_id} assigned to charging slot {slot_to_assign} at MicroHub {micro_hub_id}.")
                return True
            return False  # Should not happen if assign_charging_slot logic is correct after checks

        except KeyError as e:
            print(f"MicroHubsManager: Charging assignment failed - {e}.")
            return False
        except ValueError as e:
            print(f"MicroHubsManager: Charging assignment failed - {e}.")
            return False

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes micro-hub specific plotting data.
        This method will likely defer to individual micro-hub's initialize_plot_data methods.
        """
        print("MicroHubsManager: Initializing plot data.")
        # Iterate through all micro-hubs and call their plotting initialization
        for micro_hub in self.global_state.get_all_entities("micro_hub").values():
            micro_hub.initialize_plot_data(figure_data)

        # Any micro-hub level initial data (e.g., total active hubs)
        if 'micro_hub_overview' not in figure_data:
            figure_data['micro_hub_overview'] = {}

        initial_active_hubs = [
            h.id for h in self.global_state.get_all_entities("micro_hub").values() if h.operational_status == "active"
        ]
        figure_data['micro_hub_overview']['initial_active_hubs'] = initial_active_hubs

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates micro-hub specific plotting data for the current simulation step.
        This method will likely defer to individual micro-hub's update_plot_data methods.
        """
        print("MicroHubsManager: Updating plot data.")
        # Iterate through all micro-hubs and call their plotting update
        for micro_hub in self.global_state.get_all_entities("micro_hub").values():
            micro_hub.update_plot_data(figure_data)

        # Any micro-hub level dynamic data (e.g., active hubs, occupied slots)
        if 'micro_hub_overview' not in figure_data:
            figure_data['micro_hub_overview'] = {}  # Defensive check

        current_active_hubs = [
            h.id for h in self.global_state.get_all_entities("micro_hub").values() if h.operational_status == "active"
        ]
        figure_data['micro_hub_overview']['current_active_hubs'] = current_active_hubs

        total_occupied_slots = 0
        for micro_hub in self.global_state.get_all_entities("micro_hub").values():
            total_occupied_slots += (len(micro_hub.charging_slots) - len(micro_hub.get_available_charging_slots()))
        figure_data['micro_hub_overview']['total_occupied_charging_slots'] = total_occupied_slots
