# In file: ddls_src/core/network.py
import heapq
# --- Imports for visualization ---
import matplotlib.pyplot as plt
import networkx as nx
from ast import Param
from mlpro.bf.exceptions import ParamError
from typing import Dict, Any, List, Tuple, Optional
from xmlrpc.client import Error


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
    for network queries, pathfinding, and live visualization.
    """
    C_NETWORK_AIR = "Air"
    C_NETWORK_GROUND = "Land"
    def __init__(self, global_state: 'GlobalState', movement_mode: str = 'network', land_distance_matrix: Dict = None, air_distance_matrix: Dict = None):
        """
        Initializes the Network class.

        Args:
            global_state (GlobalState): The global state of the simulation.
            movement_mode (str): The movement mode ('network' or 'matrix').
            land_distance_matrix (Dict): The distance matrix for matrix-based movement.
        """
        self.custom_log = False
        self.global_state = global_state
        self.nodes: Dict[int, Node] = global_state.nodes
        self.edges: Dict[int, Edge] = global_state.edges
        self.adjacency_list: Dict[int, List[Tuple[int, int]]] = {}
        self._build_adjacency_list()

        # New attributes for distance matrix mode
        self.movement_mode = movement_mode
        if self.movement_mode:
            if land_distance_matrix is None or air_distance_matrix is None:
                raise ParamError("Please provide distance matrix when using matrix movement mode.")
            self.land_distance_matrix = land_distance_matrix
            self.air_distance_matrix = air_distance_matrix
        else:
            self.land_distance_matrix = {}
            self.air_distance_matrix = {}


        # Visualization attributes
        self.fig = None
        self.ax = None
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.node_colors = {}
        if self.custom_log:
            print("Network class initialized.")

    def _build_adjacency_list(self) -> None:
        self.adjacency_list = {node_id: [] for node_id in self.nodes.keys()}
        for edge_id, edge in self.edges.items():
            if edge.start_node_id in self.adjacency_list:
                self.adjacency_list[edge.start_node_id].append((edge.end_node_id, edge_id))

    def get_neighbors(self, node_id: int) -> List[Tuple[int, int]]:
        return self.adjacency_list.get(node_id, [])

    def get_edge_between_nodes(self, node1_id: int, node2_id: int) -> Optional[Edge]:
        for edge in self.edges.values():
            if edge.start_node_id == node1_id and edge.end_node_id == node2_id:
                return edge
        return None

    def get_travel_time(self, start_node_id: int, end_node_id: int, network_type = None) -> Optional[float]:
        """Gets the travel time between two nodes based on the current movement mode."""
        if self.movement_mode == 'matrix':
            if network_type == None:
                raise ValueError("Please provide a valide network type to get distance when in matrix mode")
            elif network_type == self.C_NETWORK_AIR:
                try:
                    # Ensure keys are strings for JSON compatibility
                    return self.air_distance_matrix[str(start_node_id)][str(end_node_id)]
                except KeyError:
                    return None
            elif network_type == self.C_NETWORK_GROUND:
                try:
                    # Ensure keys are strings for JSON compatibility
                    return self.land_distance_matrix[str(start_node_id)][str(end_node_id)]
                except KeyError:
                    return None
        else: # network mode
            edge = self.get_edge_between_nodes(start_node_id, end_node_id)
            return edge.get_current_travel_time() if edge else None

    def calculate_shortest_path(self, start_node_id: int, end_node_id: int, vehicle_type: str, network_type=None) -> List[int]:
        if self.movement_mode == 'matrix':
            # For matrix mode, we assume a direct path.
            # You could implement a more complex pathfinding algorithm here if needed.
            if network_type is None:
                raise ValueError("Please provide network type when working with dist matrix, to get distance.")
            elif network_type == self.C_NETWORK_AIR:
                if str(start_node_id) in self.air_distance_matrix and str(end_node_id) in self.air_distance_matrix[str(start_node_id)]:
                    return [start_node_id, end_node_id]
            elif network_type == self.C_NETWORK_GROUND:
                if str(start_node_id) in self.land_distance_matrix and str(end_node_id) in self.land_distance_matrix[str(start_node_id)]:
                    return [start_node_id, end_node_id]
            else:
                return []
        else: # network mode
            # Dijkstra's algorithm implementation remains the same
            if start_node_id not in self.nodes or end_node_id not in self.nodes: return []
            distances = {node_id: float('inf') for node_id in self.nodes.keys()}
            previous_nodes = {node_id: None for node_id in self.nodes.keys()}
            distances[start_node_id] = 0
            priority_queue = [(0, start_node_id)]
            while priority_queue:
                dist, current_node_id = heapq.heappop(priority_queue)
                if dist > distances[current_node_id]: continue
                if current_node_id == end_node_id: break
                for neighbor_id, edge_id in self.get_neighbors(current_node_id):
                    edge = self.edges.get(edge_id)
                    if not edge or edge.is_blocked: continue
                    travel_time = edge.get_current_travel_time() if vehicle_type == 'truck' else edge.get_drone_flight_time()
                    if travel_time == float('inf'): continue
                    new_dist = dist + travel_time
                    if new_dist < distances[neighbor_id]:
                        distances[neighbor_id] = new_dist
                        previous_nodes[neighbor_id] = current_node_id
                        heapq.heappush(priority_queue, (new_dist, neighbor_id))
            path = []
            current = end_node_id
            while current is not None:
                path.insert(0, current)
                current = previous_nodes.get(current)
            return path if path and path[0] == start_node_id else []

    def calculate_distance(self, p_node_1, p_node_2):
        return 10

    # --- Plotting Methods ---
    def setup_visualization(self):
        """Initializes the plot figure and axis."""
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        plt.ion()
        color_map = {'depot': 'green', 'customer': 'blue', 'micro_hub': 'orange', 'junction': 'grey'}
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id)
            self.node_positions[node_id] = node.coords
            self.node_colors[node_id] = color_map.get(node.type_of_node, 'grey')
        for edge in self.edges.values():
            self.graph.add_edge(edge.start_node_id, edge.end_node_id)
        print("Visualization setup complete.")

    def update_plot(self):
        """Clears and redraws the plot with the current state."""
        if not self.fig: return
        self.ax.clear()

        # 1. Draw the network graph
        # node_color_list = [self.node_colors.get(node, 'gray') for node in self.graph.nodes()]
        nx.draw(self.graph, self.node_positions, ax=self.ax, node_color='grey',
                with_labels=True, node_size=100, font_size=8, arrows=True)

        # 2. Draw vehicles from GlobalState
        # Trucks
        truck_coords = [t.get_current_location() for t in self.global_state.trucks.values() if
                        t.get_current_location()]
        if truck_coords:
            x_trucks, y_trucks = zip(*truck_coords)
            self.ax.scatter(x_trucks, y_trucks, c='red', marker='s', s=120, label='Trucks', zorder=5)
        # Drones
        drone_coords = [d.get_current_location() for d in self.global_state.drones.values() if
                        d.get_current_location()]
        if drone_coords:
            x_drones, y_drones = zip(*drone_coords)
            self.ax.scatter(x_drones, y_drones, c='purple', marker='^', s=120, label='Drones', zorder=5)

        # 3. Update plot titles and legend
        self.ax.set_title(f"Logistics Simulation (Time: {self.global_state.current_time:.2f}s)")
        self.ax.legend(loc="upper right")

        # 4. Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

    def close_plot(self):
        """Turns off interactive mode and shows the final plot."""
        if self.fig:
            plt.ioff()
            plt.show()