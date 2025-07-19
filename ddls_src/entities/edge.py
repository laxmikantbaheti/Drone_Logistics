from typing import Tuple, Dict, Any


class Edge:
    """
    Represents a connection between two nodes in the simulation network.
    Edges have properties like length, base travel time, and dynamic factors
    like traffic and drone flight impact.
    """

    def __init__(self, id: int, start_node_id: int, end_node_id: int,
                 length: float, base_travel_time: float):
        """
        Initializes an Edge.

        Args:
            id (int): Unique identifier for the edge.
            start_node_id (int): The ID of the starting node of the edge.
            end_node_id (int): The ID of the ending node of the edge.
            length (float): The physical length of the edge (e.g., in kilometers).
            base_travel_time (float): The ideal travel time across this edge
                                      under normal conditions (e.g., in minutes).
        """
        self.id: int = id
        self.start_node_id: int = start_node_id
        self.end_node_id: int = end_node_id
        self.length: float = length
        self.base_travel_time: float = base_travel_time
        self.current_traffic_factor: float = 1.0  # Multiplier for base_travel_time (1.0 means no impact)
        self.is_blocked: bool = False  # True if the edge is impassable
        self.drone_flight_impact_factor: float = 1.0  # Multiplier for drone travel time

        print(f"Edge {self.id} (from Node {self.start_node_id} to Node {self.end_node_id}) initialized.")

    def get_current_travel_time(self) -> float:
        """
        Calculates the current travel time across this edge for trucks,
        considering the base travel time and current traffic factor.

        Returns:
            float: The current estimated travel time.
        """
        if self.is_blocked:
            return float('inf')  # Impassable
        return self.base_travel_time * self.current_traffic_factor

    def get_drone_flight_time(self) -> float:
        """
        Calculates the current flight time across this edge for drones,
        considering the base travel time and drone flight impact factor.

        Returns:
            float: The current estimated drone flight time.
        """
        if self.is_blocked:  # Drones might also be affected by blocked paths (e.g., restricted airspace)
            return float('inf')
        return self.base_travel_time * self.drone_flight_impact_factor

    def set_traffic_factor(self, factor: float) -> None:
        """
        Sets the current traffic factor for this edge.

        Args:
            factor (float): The new traffic multiplier (e.g., 1.5 for 50% slower).
        """
        if factor < 0:
            raise ValueError("Traffic factor cannot be negative.")
        self.current_traffic_factor = factor
        # print(f"Edge {self.id}: Traffic factor set to {factor}.")

    def set_blocked(self, status: bool) -> None:
        """
        Sets whether this edge is blocked (impassable).

        Args:
            status (bool): True if blocked, False if clear.
        """
        self.is_blocked = status
        # print(f"Edge {self.id}: Blocked status set to {status}.")

    def set_drone_flight_impact_factor(self, factor: float) -> None:
        """
        Sets the impact factor on drone flight time for this edge.
        This could represent weather conditions, restricted airspace, etc.

        Args:
            factor (float): The new drone flight impact multiplier.
        """
        if factor < 0:
            raise ValueError("Drone flight impact factor cannot be negative.")
        self.drone_flight_impact_factor = factor
        # print(f"Edge {self.id}: Drone flight impact factor set to {factor}.")

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes the plotting data for this specific edge within the shared figure_data.
        This is typically called once at the start of the simulation.
        It contributes to the 'network_edges' layer managed by GlobalState or Network.
        """
        # Static edge data (like its existence and base position) is often handled
        # by the overall Network/GlobalState's initialize_plot_data.
        # This method might be used for edge-specific static labels or initial styling.
        print(f"Edge {self.id}: Initializing plot data (often handled by GlobalState/Network).")
        # Example: If an edge needs a specific static symbol or label
        if 'edge_details' not in figure_data:
            figure_data['edge_details'] = {}
        figure_data['edge_details'][self.id] = {
            'start_node_id': self.start_node_id,
            'end_node_id': self.end_node_id,
            'length': self.length,
            'base_travel_time': self.base_travel_time
        }

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates the plotting data for this specific edge, reflecting its current state.
        This method is called at each simulation timestep.
        It modifies the shared figure_data dictionary.
        """
        # Update dynamic properties like traffic factor or blocked status,
        # which could translate to changing the edge's color or thickness in visualization.
        print(f"Edge {self.id}: Updating plot data (traffic/blocked status).")
        if 'edge_dynamic_data' not in figure_data:
            figure_data['edge_dynamic_data'] = {}

        figure_data['edge_dynamic_data'][self.id] = {
            'current_traffic_factor': self.current_traffic_factor,
            'is_blocked': self.is_blocked,
            'drone_flight_impact_factor': self.drone_flight_impact_factor
            # The visualization layer would then interpret these values to draw the edge
        }
