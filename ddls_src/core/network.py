from typing import Dict, Any, List, Tuple, Optional
import heapq  # For Dijkstra's algorithm


# Forward declarations for type hinting
class GlobalState:
    pass


class Node:
    pass


class Edge:
    pass


class Network:
    """
    Encapsulates the graph structure (nodes and edges) and provides efficient methods
    for network queries and pathfinding data. It acts as the underlying graph data
    structure that NetworkManager operates on.
    """

    def __init__(self, global_state: 'GlobalState'):
        """
        Initializes the Network graph structure from GlobalState's nodes and edges.

        Args:
            global_state (GlobalState): A reference to the central GlobalState.
        """
        self.global_state = global_state
        # References to the actual dictionaries in GlobalState
        self.nodes: Dict[int, Node] = global_state.nodes
        self.edges: Dict[int, Edge] = global_state.edges
        self.adjacency_list: Dict[int, List[Tuple[int, int]]] = {}  # {node_id: [(neighbor_node_id, edge_id), ...]}

        self._build_adjacency_list()
        print("Network class initialized.")

    def _build_adjacency_list(self) -> None:
        """
        Builds the adjacency list based on the current nodes and edges in GlobalState.
        This should be called during initialization and potentially if the network topology changes.
        """
        self.adjacency_list = {node_id: [] for node_id in self.nodes.keys()}
        for edge_id, edge in self.edges.items():
            if edge.start_node_id in self.adjacency_list:
                self.adjacency_list[edge.start_node_id].append((edge.end_node_id, edge_id))
            # Assuming edges are defined bidirectionally in the initial data if they are two-way.
            # If an edge (A,B) exists, and we want (B,A) to also be traversable,
            # it must be explicitly defined in the initial data or added here.
            # For now, we only add edges as they appear in the self.edges dictionary.

        print("Network adjacency list built.")

    def get_neighbors(self, node_id: int) -> List[Tuple[int, int]]:
        """
        Returns a list of (neighbor_node_id, edge_id) tuples for a given node.
        """
        return self.adjacency_list.get(node_id, [])

    def get_edge_between_nodes(self, node1_id: int, node2_id: int) -> Optional[Edge]:
        """
        Returns the Edge object connecting node1_id to node2_id, if it exists.
        Assumes unique edge between two nodes in a given direction.
        """
        for edge in self.edges.values():
            if edge.start_node_id == node1_id and edge.end_node_id == node2_id:
                return edge
        return None

    def calculate_shortest_path(self, start_node_id: int, end_node_id: int, vehicle_type: str) -> List[int]:
        """
        Calculates the shortest path between two nodes using Dijkstra's algorithm.
        Considers travel times based on vehicle_type (truck vs. drone).

        Args:
            start_node_id (int): The ID of the starting node.
            end_node_id (int): The ID of the destination node.
            vehicle_type (str): 'truck' or 'drone', to determine travel time calculation.

        Returns:
            List[int]: A list of node IDs representing the shortest path,
                       or an empty list if no path exists.
        """
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            print(f"Pathfinding error: Start or end node not found ({start_node_id} -> {end_node_id}).")
            return []

        # Dijkstra's algorithm implementation
        distances = {node_id: float('inf') for node_id in self.nodes.keys()}
        previous_nodes = {node_id: None for node_id in self.nodes.keys()}
        distances[start_node_id] = 0
        priority_queue = [(0, start_node_id)]  # (distance, node_id)

        while priority_queue:
            current_distance, current_node_id = heapq.heappop(priority_queue)

            if current_distance > distances[current_node_id]:
                continue

            if current_node_id == end_node_id:
                break  # Found the shortest path to the target

            for neighbor_node_id, edge_id in self.get_neighbors(current_node_id):
                edge = self.edges.get(edge_id)
                if not edge:
                    continue  # Should not happen if adjacency list is built correctly

                # Calculate travel time based on vehicle type
                if vehicle_type == 'truck':
                    travel_time = edge.get_current_travel_time()
                elif vehicle_type == 'drone':
                    travel_time = edge.get_drone_flight_time()
                else:
                    raise ValueError(f"Unknown vehicle type for pathfinding: {vehicle_type}")

                # Check if edge is blocked
                if edge.is_blocked:
                    continue

                new_distance = current_distance + travel_time

                if new_distance < distances[neighbor_node_id]:
                    distances[neighbor_node_id] = new_distance
                    previous_nodes[neighbor_node_id] = current_node_id
                    heapq.heappush(priority_queue, (new_distance, neighbor_node_id))

        # Reconstruct path
        path = []
        current = end_node_id
        while current is not None:
            path.insert(0, current)
            current = previous_nodes[current]

        if not path or path[0] != start_node_id:  # Path not found or reconstruction failed
            return []

        print(f"Path calculated for {vehicle_type} from {start_node_id} to {end_node_id}: {path}")
        return path

    def calculate_distance(self, p_node_1, p_node_2):
        return 10

    # --- Plotting Methods ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes the network graph data (nodes and edges) within the shared figure_data.
        This is typically called once at the start of the simulation.
        """
        print("Network: Initializing plot data.")
        # Assuming nodes have 'coords' and edges have 'start_node_id', 'end_node_id'

        node_coords = [node.coords for node in self.nodes.values()]
        node_ids = [node.id for node in self.nodes.values()]
        node_types = [node.type for node in self.nodes.values()]

        edge_segments = []
        edge_ids = []
        for edge in self.edges.values():
            start_coords = self.global_state.nodes[edge.start_node_id].coords  # Access coords from global_state.nodes
            end_coords = self.global_state.nodes[edge.end_node_id].coords  # Access coords from global_state.nodes
            edge_segments.append((start_coords, end_coords))
            edge_ids.append(edge.id)

        figure_data['network_nodes'] = {
            'coords': node_coords,
            'ids': node_ids,
            'types': node_types
        }
        figure_data['network_edges'] = {
            'segments': edge_segments,
            'ids': edge_ids
        }
        print("Network: Initial plot data added to figure_data.")

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates the dynamic aspects of the network plot data (e.g., traffic on edges).
        This is typically called each simulation timestep.
        """
        print("Network: Updating plot data (e.g., traffic conditions).")
        # Example: Update edge colors based on traffic factor or blocked status
        updated_edge_info = []
        for edge in self.edges.values():
            start_coords = self.global_state.nodes[edge.start_node_id].coords
            end_coords = self.global_state.nodes[edge.end_node_id].coords
            # Determine color/style based on traffic or blocked status
            color = 'gray'
            if edge.is_blocked:
                color = 'red'
            elif edge.current_traffic_factor > 1.5:
                color = 'orange'
            updated_edge_info.append({
                'id': edge.id,
                'segment': (start_coords, end_coords),
                'color': color
            })
        figure_data['network_edges_dynamic'] = updated_edge_info
        print("Network: Dynamic plot data updated in figure_data.")
