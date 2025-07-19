from typing import List, Dict, Any, Optional


# Forward declaration for GlobalState to avoid circular dependency
class GlobalState:
    pass


class SupplyChainManager:
    """
    Manages the lifecycle and assignment of orders within the simulation.
    It interacts with GlobalState to modify order and vehicle assignments.
    """

    def __init__(self, global_state: 'GlobalState'):
        """
        Initializes the SupplyChainManager.

        Args:
            global_state (GlobalState): Reference to the central GlobalState.
        """
        self.global_state = global_state
        print("SupplyChainManager initialized.")

    def accept_order(self, order_id: int) -> bool:
        """
        Accepts a new order, typically changing its status from 'pending' to 'accepted'
        or making it available for assignment.

        Args:
            order_id (int): The ID of the order to accept.

        Returns:
            bool: True if the order was found and status updated, False otherwise.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            if order.status == "pending":
                order.update_status("accepted")  # Assuming 'accepted' is a valid intermediate status
                print(f"SupplyChainManager: Order {order_id} accepted.")
                return True
            else:
                print(f"SupplyChainManager: Order {order_id} is not pending (current status: {order.status}).")
                return False
        except KeyError:
            print(f"SupplyChainManager: Order {order_id} not found.")
            return False

    def prioritize_order(self, order_id: int, new_priority: int) -> bool:
        """
        Changes the priority of an existing order.

        Args:
            order_id (int): The ID of the order to prioritize.
            new_priority (int): The new priority level.

        Returns:
            bool: True if the order was found and priority updated, False otherwise.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            order.priority = new_priority
            print(f"SupplyChainManager: Order {order_id} priority set to {new_priority}.")
            return True
        except KeyError:
            print(f"SupplyChainManager: Order {order_id} not found.")
            return False

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancels an order, changing its status to 'cancelled'.
        If the order was assigned, it also unassigns the vehicle.

        Args:
            order_id (int): The ID of the order to cancel.

        Returns:
            bool: True if the order was found and cancelled, False otherwise.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            if order.status != "delivered" and order.status != "cancelled":
                if order.assigned_vehicle_id is not None:
                    # In a full implementation, you'd also need to tell the vehicle to drop this cargo
                    # For now, just unassign from the order's perspective.
                    order.unassign_vehicle()
                order.update_status("cancelled")
                print(f"SupplyChainManager: Order {order_id} cancelled.")
                return True
            else:
                print(f"SupplyChainManager: Order {order_id} cannot be cancelled (current status: {order.status}).")
                return False
        except KeyError:
            print(f"SupplyChainManager: Order {order_id} not found.")
            return False

    def flag_for_re_delivery(self, order_id: int) -> bool:
        """
        Flags an order for re-delivery, typically if a previous delivery attempt failed.

        Args:
            order_id (int): The ID of the order to flag.

        Returns:
            bool: True if the order was found and flagged, False otherwise.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            if order.status != "delivered" and order.status != "cancelled":
                order.update_status("flagged_re_delivery")
                if order.assigned_vehicle_id is not None:
                    order.unassign_vehicle()  # Unassign if it was previously assigned
                print(f"SupplyChainManager: Order {order_id} flagged for re-delivery.")
                return True
            else:
                print(
                    f"SupplyChainManager: Order {order_id} cannot be flagged for re-delivery (current status: {order.status}).")
                return False
        except KeyError:
            print(f"SupplyChainManager: Order {order_id} not found.")
            return False

    def assign_order_to_truck(self, order_id: int, truck_id: int) -> bool:
        """
        Assigns an order to a specific truck.

        Args:
            order_id (int): The ID of the order to assign.
            truck_id (int): The ID of the truck to assign the order to.

        Returns:
            bool: True if assignment was successful, False otherwise.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            truck = self.global_state.get_entity("truck", truck_id)

            if order.status not in ["pending", "accepted", "flagged_re_delivery"]:
                print(f"SupplyChainManager: Order {order_id} (status: {order.status}) cannot be assigned.")
                return False
            if truck.status not in ["idle", "loading", "unloading"]:  # Truck must be available
                print(f"SupplyChainManager: Truck {truck_id} (status: {truck.status}) is not available for assignment.")
                return False
            if len(truck.cargo_manifest) >= truck.max_payload_capacity:
                print(f"SupplyChainManager: Truck {truck_id} is full.")
                return False

            order.assign_vehicle(truck_id)
            truck.add_cargo(order_id)  # Add to truck's manifest
            print(f"SupplyChainManager: Order {order_id} assigned to Truck {truck_id}.")
            return True
        except KeyError as e:
            print(f"SupplyChainManager: Assignment failed - {e}.")
            return False
        except ValueError as e:
            print(f"SupplyChainManager: Assignment failed - {e}.")
            return False

    def assign_order_to_drone(self, order_id: int, drone_id: int) -> bool:
        """
        Assigns an order to a specific drone.

        Args:
            order_id (int): The ID of the order to assign.
            drone_id (int): The ID of the drone to assign the order to.

        Returns:
            bool: True if assignment was successful, False otherwise.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            drone = self.global_state.get_entity("drone", drone_id)

            if order.status not in ["pending", "accepted", "flagged_re_delivery"]:
                print(f"SupplyChainManager: Order {order_id} (status: {order.status}) cannot be assigned to drone.")
                return False
            if drone.status not in ["idle", "loading", "unloading"]:  # Drone must be available
                print(f"SupplyChainManager: Drone {drone_id} (status: {drone.status}) is not available for assignment.")
                return False
            if len(drone.cargo_manifest) >= drone.max_payload_capacity:
                print(f"SupplyChainManager: Drone {drone_id} is full.")
                return False

            # Additional drone-specific checks (e.g., battery level, proximity to micro-hub)
            # These might be better handled by ActionMasker, but can be duplicated here for robustness.
            # if drone.battery_level < MIN_DRONE_BATTERY_FOR_ASSIGNMENT: # Example
            #     print(f"Drone {drone_id} battery too low for assignment.")
            #     return False

            order.assign_vehicle(drone_id)
            drone.add_cargo(order_id)  # Add to drone's manifest
            print(f"SupplyChainManager: Order {order_id} assigned to Drone {drone_id}.")
            return True
        except KeyError as e:
            print(f"SupplyChainManager: Assignment failed - {e}.")
            return False
        except ValueError as e:
            print(f"SupplyChainManager: Assignment failed - {e}.")
            return False

    def assign_order_to_micro_hub(self, order_id: int, micro_hub_id: int) -> bool:
        """
        Assigns an order to a specific micro-hub for temporary storage or transfer.
        This implies the order is physically moved to the micro-hub.

        Args:
            order_id (int): The ID of the order to assign.
            micro_hub_id (int): The ID of the micro-hub to assign the order to.

        Returns:
            bool: True if assignment was successful, False otherwise.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            micro_hub = self.global_state.get_entity("micro_hub", micro_hub_id)

            if order.status not in ["pending", "accepted", "flagged_re_delivery"]:
                print(f"SupplyChainManager: Order {order_id} (status: {order.status}) cannot be assigned to micro-hub.")
                return False
            if micro_hub.operational_status != "active":
                print(f"SupplyChainManager: MicroHub {micro_hub_id} is not active.")
                return False
            if micro_hub.is_package_transfer_unavailable:
                print(f"SupplyChainManager: MicroHub {micro_hub_id} package transfer service is unavailable.")
                return False

            order.assign_micro_hub(micro_hub_id)
            micro_hub.add_package_to_holding(order_id)  # Add to micro-hub's held packages
            order.update_status("at_micro_hub")  # New status for orders at micro-hub
            print(f"SupplyChainManager: Order {order_id} assigned to MicroHub {micro_hub_id}.")
            return True
        except KeyError as e:
            print(f"SupplyChainManager: Assignment failed - {e}.")
            return False
        except ValueError as e:
            print(f"SupplyChainManager: Assignment failed - {e}.")
            return False

    def consolidate_for_truck(self, truck_id: int) -> List[int]:
        """
        Internal logic to select and mark orders for truck consolidation.
        This method would typically identify eligible orders at the truck's current node
        and add them to the truck's cargo manifest, updating their status.

        Args:
            truck_id (int): The ID of the truck for which to consolidate orders.

        Returns:
            List[int]: A list of order IDs that were consolidated.
        """
        consolidated_orders = []
        try:
            truck = self.global_state.get_entity("truck", truck_id)
            if truck.status != "idle" and truck.status != "loading":
                print(f"SupplyChainManager: Truck {truck_id} is not idle or loading, cannot consolidate.")
                return []

            # Assuming truck is at a node where packages are available
            if truck.current_node_id is None:
                print(f"SupplyChainManager: Truck {truck_id} is not at a node for consolidation.")
                return []

            current_node = self.global_state.get_entity("node", truck.current_node_id)

            # Logic to select orders for consolidation:
            # Iterate through packages at the current_node and available orders
            eligible_order_ids_at_node = current_node.get_packages()

            for order_id in eligible_order_ids_at_node:
                order = self.global_state.get_entity("order", order_id)
                # Check if order is pending/accepted/flagged and not already assigned
                if order.status in ["pending", "accepted", "flagged_re_delivery"] and order.assigned_vehicle_id is None:
                    if len(truck.cargo_manifest) < truck.max_payload_capacity:
                        # Assign order to truck and update status
                        order.assign_vehicle(truck_id)
                        truck.add_cargo(order_id)
                        current_node.remove_package(order_id)  # Remove from node
                        consolidated_orders.append(order_id)
                        print(f"SupplyChainManager: Consolidated Order {order_id} onto Truck {truck_id}.")
                    else:
                        print(f"SupplyChainManager: Truck {truck_id} is full, stopped consolidating.")
                        break  # Truck is full

        except KeyError as e:
            print(f"SupplyChainManager: Consolidation failed - {e}.")
        return consolidated_orders

    def consolidate_for_drone(self, drone_id: int) -> List[int]:
        """
        Internal logic to select and mark orders for drone consolidation.
        Similar to truck consolidation, but for drones, typically at micro-hubs.

        Args:
            drone_id (int): The ID of the drone for which to consolidate orders.

        Returns:
            List[int]: A list of order IDs that were consolidated.
        """
        consolidated_orders = []
        try:
            drone = self.global_state.get_entity("drone", drone_id)
            if drone.status != "idle" and drone.status != "loading":
                print(f"SupplyChainManager: Drone {drone_id} is not idle or loading, cannot consolidate.")
                return []

            # Drones typically consolidate at MicroHubs or Depots
            if drone.current_node_id is None:
                print(f"SupplyChainManager: Drone {drone_id} is not at a node for consolidation.")
                return []

            current_node = self.global_state.get_entity("node", drone.current_node_id)

            # Logic to select orders for consolidation:
            eligible_order_ids_at_node = current_node.get_packages()

            for order_id in eligible_order_ids_at_node:
                order = self.global_state.get_entity("order", order_id)
                if order.status in ["pending", "accepted", "flagged_re_delivery"] and order.assigned_vehicle_id is None:
                    if len(drone.cargo_manifest) < drone.max_payload_capacity:
                        order.assign_vehicle(drone_id)
                        drone.add_cargo(order_id)
                        current_node.remove_package(order_id)
                        consolidated_orders.append(order_id)
                        print(f"SupplyChainManager: Consolidated Order {order_id} onto Drone {drone_id}.")
                    else:
                        print(f"SupplyChainManager: Drone {drone_id} is full, stopped consolidating.")
                        break
        except KeyError as e:
            print(f"SupplyChainManager: Consolidation failed - {e}.")
        return consolidated_orders

    def reassign_order(self, order_id: int, new_vehicle_id: int) -> bool:
        """
        Reassigns an order from its current vehicle (or unassigned state) to a new vehicle.
        This involves removing it from the old vehicle's manifest (if any) and adding to the new.

        Args:
            order_id (int): The ID of the order to reassign.
            new_vehicle_id (int): The ID of the new vehicle (truck or drone).

        Returns:
            bool: True if reassignment was successful, False otherwise.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            new_vehicle = None
            if new_vehicle_id in self.global_state.trucks:
                new_vehicle = self.global_state.get_entity("truck", new_vehicle_id)
            elif new_vehicle_id in self.global_state.drones:
                new_vehicle = self.global_state.get_entity("drone", new_vehicle_id)
            else:
                print(f"SupplyChainManager: New vehicle {new_vehicle_id} not found.")
                return False

            if new_vehicle.status not in ["idle", "loading", "unloading"]:
                print(f"SupplyChainManager: New vehicle {new_vehicle_id} is not available for reassignment.")
                return False
            if len(new_vehicle.cargo_manifest) >= new_vehicle.max_payload_capacity:
                print(f"SupplyChainManager: New vehicle {new_vehicle_id} is full.")
                return False

            # Remove from old vehicle if assigned
            if order.assigned_vehicle_id is not None:
                old_vehicle_id = order.assigned_vehicle_id
                old_vehicle = None
                if old_vehicle_id in self.global_state.trucks:
                    old_vehicle = self.global_state.get_entity("truck", old_vehicle_id)
                elif old_vehicle_id in self.global_state.drones:
                    old_vehicle = self.global_state.get_entity("drone", old_vehicle_id)

                if old_vehicle:
                    old_vehicle.remove_cargo(order_id)
                    print(f"SupplyChainManager: Order {order_id} removed from old vehicle {old_vehicle_id}.")
                else:
                    print(f"SupplyChainManager: Warning - Old vehicle {old_vehicle_id} not found for order {order_id}.")

            # Assign to new vehicle
            order.assign_vehicle(new_vehicle_id)
            new_vehicle.add_cargo(order_id)
            order.update_status("assigned")  # Revert to assigned status after reassignment

            print(f"SupplyChainManager: Order {order_id} reassigned to vehicle {new_vehicle_id}.")
            return True
        except KeyError as e:
            print(f"SupplyChainManager: Reassignment failed - {e}.")
            return False
        except ValueError as e:
            print(f"SupplyChainManager: Reassignment failed - {e}.")
            return False

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes any supply chain specific plotting data.
        This might include initial order statuses or overall supply chain metrics.
        """
        print("SupplyChainManager: Initializing plot data.")
        # Example: Initial count of orders by status
        if 'supply_chain_overview' not in figure_data:
            figure_data['supply_chain_overview'] = {}

        initial_order_counts = {}
        for order in self.global_state.get_all_entities("order").values():
            initial_order_counts[order.status] = initial_order_counts.get(order.status, 0) + 1

        figure_data['supply_chain_overview']['initial_order_counts'] = initial_order_counts
        # Other static supply chain related data can be added here

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates supply chain specific plotting data for the current simulation step.
        This might include real-time order statuses, delivery progress, or SLA breaches.
        """
        print("SupplyChainManager: Updating plot data.")
        if 'supply_chain_overview' not in figure_data:
            figure_data['supply_chain_overview'] = {}  # Defensive check

        current_order_counts = {}
        for order in self.global_state.get_all_entities("order").values():
            current_order_counts[order.status] = current_order_counts.get(order.status, 0) + 1

        figure_data['supply_chain_overview']['current_order_counts'] = current_order_counts

        # Example: Track SLA breaches
        sla_breaches_count = 0
        current_time = self.global_state.current_time  # Assuming GlobalState holds current_time
        for order in self.global_state.get_all_entities("order").values():
            if order.status != "delivered" and order.get_SLA_remaining(current_time) < 0:
                sla_breaches_count += 1
        figure_data['supply_chain_overview']['sla_breaches_count'] = sla_breaches_count

        # This data can be used to render dashboards or summary plots.
