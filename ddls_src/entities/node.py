from typing import List, Tuple, Any, Dict


class Node:
    """
    Represents a location in the simulation network. Nodes can be various types
    like depots, customer locations, micro-hubs, or charging stations.
    """

    def __init__(self, id: int, coords: Tuple[float, float], type: str,
                 is_loadable: bool = False, is_unloadable: bool = False,
                 is_charging_station: bool = False):
        """
        Initializes a Node.

        Args:
            id (int): Unique identifier for the node.
            coords (Tuple[float, float]): (x, y) coordinates of the node in the simulation space.
            type (str): The type of the node (e.g., 'depot', 'customer', 'micro_hub', 'charging_station').
            is_loadable (bool): True if vehicles can load packages at this node.
            is_unloadable (bool): True if vehicles can unload packages at this node.
            is_charging_station (bool): True if drones can charge at this node.
        """
        self.id: int = id
        self.coords: Tuple[float, float] = coords
        self.type: str = type
        self.packages_held: List[int] = []  # List of order IDs currently at this node
        self.is_loadable: bool = is_loadable
        self.is_unloadable: bool = is_unloadable
        self.is_charging_station: bool = is_charging_station

        print(f"Node {self.id} (Type: {self.type}) initialized at {self.coords}.")

    def add_package(self, order_id: int) -> None:
        """
        Adds a package (order ID) to the node's held packages.

        Args:
            order_id (int): The ID of the order/package to add.
        """
        if order_id not in self.packages_held:
            self.packages_held.append(order_id)
            # print(f"Node {self.id}: Added package {order_id}. Packages held: {self.packages_held}")
        # else:
        # print(f"Node {self.id}: Package {order_id} already exists.")

    def remove_package(self, order_id: int) -> None:
        """
        Removes a package (order ID) from the node's held packages.

        Args:
            order_id (int): The ID of the order/package to remove.
        """
        if order_id in self.packages_held:
            self.packages_held.remove(order_id)
            # print(f"Node {self.id}: Removed package {order_id}. Packages held: {self.packages_held}")
        # else:
        # print(f"Node {self.id}: Package {order_id} not found to remove.")

    def get_packages(self) -> List[int]:
        """
        Returns a list of order IDs currently held at this node.

        Returns:
            List[int]: A list of integer order IDs.
        """
        return list(self.packages_held)  # Return a copy to prevent external modification

    def set_loadable(self, status: bool) -> None:
        """
        Sets whether this node is a valid loading point.

        Args:
            status (bool): True to make it loadable, False otherwise.
        """
        self.is_loadable = status
        # print(f"Node {self.id}: Set is_loadable to {status}.")

    def set_unloadable(self, status: bool) -> None:
        """
        Sets whether this node is a valid unloading point.

        Args:
            status (bool): True to make it unloadable, False otherwise.
        """
        self.is_unloadable = status
        # print(f"Node {self.id}: Set is_unloadable to {status}.")

    def set_charging_station(self, status: bool) -> None:
        """
        Sets whether this node is a drone charging station.

        Args:
            status (bool): True to make it a charging station, False otherwise.
        """
        self.is_charging_station = status
        # print(f"Node {self.id}: Set is_charging_station to {status}.")

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes the plotting data for this specific node within the shared figure_data.
        This method is called once at the start of the simulation.
        It contributes to the 'network_nodes' layer managed by GlobalState or Network.
        """
        # Node-specific initial plot data can be added here if needed,
        # but often, static node data is handled by the overall Network/GlobalState
        # to draw the base graph.
        # This method might be more relevant for dynamic node properties like color based on status.

        # For now, print a message. The actual data population for nodes
        # is primarily handled by GlobalState's initialize_plot_data.
        print(f"Node {self.id}: Initializing plot data (often handled by GlobalState/Network).")

        # Example: If a node needs a specific static symbol or label
        if 'node_details' not in figure_data:
            figure_data['node_details'] = {}
        figure_data['node_details'][self.id] = {
            'coords': self.coords,
            'type': self.type,
            'initial_packages_count': len(self.packages_held)
        }

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates the plotting data for this specific node, reflecting its current state.
        This method is called at each simulation timestep.
        It modifies the shared figure_data dictionary.
        """
        # Update dynamic properties like the number of packages held, or change color based on status
        # For example, if a node has packages, its visual representation might change.

        # Ensure the 'node_dynamic_data' key exists
        if 'node_dynamic_data' not in figure_data:
            figure_data['node_dynamic_data'] = {}

        # Update current packages held at this node
        figure_data['node_dynamic_data'][self.id] = {
            'packages_held_count': len(self.packages_held),
            'has_packages': bool(self.packages_held)
            # Add other dynamic attributes like current status, if applicable
        }
        # print(f"Node {self.id}: Updating plot data. Packages: {len(self.packages_held)}")

