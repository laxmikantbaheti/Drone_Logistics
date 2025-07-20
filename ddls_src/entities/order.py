from typing import Optional, List, Any, Dict


class Order:
    """
    Represents a customer order for package delivery within the simulation.
    Orders have a lifecycle (status), delivery deadlines, and can be assigned to vehicles or micro-hubs.
    """

    def __init__(self, id: int, customer_node_id: int, time_received: float,
                 SLA_deadline: float, priority: int = 1):
        """
        Initializes an Order.

        Args:
            id (int): Unique identifier for the order.
            customer_node_id (int): The ID of the node where the customer is located (delivery destination).
            time_received (float): The simulation time when the order was received.
            SLA_deadline (float): The Service Level Agreement deadline for delivery.
            priority (int): Priority level of the order (higher number = higher priority).
        """
        self.id: int = id
        self.customer_node_id: int = customer_node_id
        self.status: str = "pending"  # e.g., "pending", "accepted", "assigned", "in_transit", "delivered", "cancelled", "flagged_re_delivery", "at_micro_hub"
        self.assigned_vehicle_id: Optional[int] = None  # ID of the truck or drone assigned
        self.assigned_micro_hub_id: Optional[int] = None  # ID of the micro-hub assigned for consolidation/transfer
        self.time_received: float = time_received
        self.SLA_deadline: float = SLA_deadline
        self.priority: int = priority
        self.delivery_time: Optional[float] = None  # Actual time of delivery

        print(
            f"Order {self.id} (Customer Node: {self.customer_node_id}) received at {self.time_received}. SLA: {self.SLA_deadline}")

    def update_status(self, new_status: str) -> None:
        """
        Updates the status of the order.

        Args:
            new_status (str): The new status for the order.
        """
        # UPDATED: Added "accepted" and "at_micro_hub" to valid_statuses
        valid_statuses = ["pending", "accepted", "assigned", "in_transit", "delivered", "cancelled",
                          "flagged_re_delivery", "at_micro_hub", "at_node"]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid order status: {new_status}. Must be one of {valid_statuses}")
        self.status = new_status
        # print(f"Order {self.id}: Status updated to {new_status}.")
        if new_status == "delivered":
            # This should ideally be set by the delivery mechanism (e.g., NetworkManager)
            # but providing a placeholder here for completeness.
            # The actual delivery time would come from the simulation's current_time.
            pass  # self.delivery_time = current_simulation_time (passed from outside)

    def assign_vehicle(self, vehicle_id: int) -> None:
        """
        Assigns this order to a specific vehicle (truck or drone).

        Args:
            vehicle_id (int): The ID of the vehicle being assigned.
        """
        self.assigned_vehicle_id = vehicle_id
        # When assigned to a vehicle, status should be "assigned"
        self.status = "assigned"
        # print(f"Order {self.id}: Assigned to vehicle {vehicle_id}.")

    def unassign_vehicle(self) -> None:
        """
        Unassigns any vehicle from this order.
        """
        self.assigned_vehicle_id = None
        # If it was assigned, revert to a state that allows re-assignment or re-delivery
        if self.status == "assigned" or self.status == "in_transit":
            self.status = "flagged_re_delivery"  # Or "pending" if it was never picked up
        # print(f"Order {self.id}: Vehicle unassigned.")

    def assign_micro_hub(self, micro_hub_id: int) -> None:
        """
        Assigns this order to a specific micro-hub for consolidation or transfer.

        Args:
            micro_hub_id (int): The ID of the micro-hub being assigned.
        """
        self.assigned_micro_hub_id = micro_hub_id
        # Status change to 'at_micro_hub' is typically handled by SupplyChainManager
        # or FleetManager when the package is physically moved.
        # print(f"Order {self.id}: Assigned to micro-hub {micro_hub_id}.")

    def get_SLA_remaining(self, current_time: float) -> float:
        """
        Calculates the remaining time until the SLA deadline.

        Args:
            current_time (float): The current simulation time.

        Returns:
            float: The time remaining until SLA deadline. Negative if deadline passed.
        """
        return self.SLA_deadline - current_time

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes the plotting data for this specific order.
        This might involve adding a representation of the order at its initial location.
        """
        print(f"Order {self.id}: Initializing plot data.")
        if 'orders' not in figure_data:
            figure_data['orders'] = {}

        figure_data['orders'][self.id] = {
            'customer_node_id': self.customer_node_id,
            'initial_location_coords': None,  # This would need to be looked up from GlobalState.nodes
            'status': self.status,
            'priority': self.priority,
            'SLA_deadline': self.SLA_deadline
        }
        # The coordinates would ideally be fetched from the Node entity
        # This will be handled by a higher-level plotting orchestrator (e.g., GlobalState or RenderEngine)
        # that has access to both Order and Node objects.

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates the plotting data for this specific order, reflecting its current status
        and location (if it's in transit or delivered).
        """
        print(f"Order {self.id}: Updating plot data. Status: {self.status}")
        if 'orders' not in figure_data:
            figure_data['orders'] = {}  # Should be initialized by initialize_plot_data, but defensive check

        # Update dynamic properties
        order_data = figure_data['orders'].get(self.id, {})
        order_data['status'] = self.status
        order_data['assigned_vehicle_id'] = self.assigned_vehicle_id
        order_data['assigned_micro_hub_id'] = self.assigned_micro_hub_id
        order_data['delivery_time'] = self.delivery_time

        # If the order is in a vehicle or at a node, its location for plotting
        # would be derived from the vehicle's or node's current location.
        # This update might trigger a change in its visual representation (e.g., color, icon).

        figure_data['orders'][self.id] = order_data
