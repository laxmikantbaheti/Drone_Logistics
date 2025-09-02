# In file: ddls_src/core/network.py

from typing import Dict, Any, List, Tuple, Optional
import heapq

# --- Imports for visualization ---
import matplotlib.pyplot as plt
import networkx as nx


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

    def __init__(self, global_state: 'GlobalState'):
        self.global_state = global_state
        self.nodes: Dict[int, Node] = global_state.nodes
        self.edges: Dict[int, Edge] = global_state.edges
        self.adjacency_list: Dict[int, List[Tuple[int, int]]] = {}
        self._build_adjacency_list()

        # Visualization attributes
        self.fig = None
        self.ax = None
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.node_colors = {}
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

    def calculate_shortest_path(self, start_node_id: int, end_node_id: int, vehicle_type: str) -> List[int]:
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
        node_color_list = [self.node_colors.get(node, 'gray') for node in self.graph.nodes()]
        nx.draw(self.graph, self.node_positions, ax=self.ax, node_color=node_color_list,
                with_labels=True, node_size=250, font_size=8, arrows=True)

        # 2. Draw vehicles from GlobalState
        # Trucks
        truck_coords = [t.current_location_coords for t in self.global_state.trucks.values() if
                        t.current_location_coords]
        if truck_coords:
            x_trucks, y_trucks = zip(*truck_coords)
            self.ax.scatter(x_trucks, y_trucks, c='red', marker='s', s=120, label='Trucks', zorder=5)
        # Drones
        drone_coords = [d.current_location_coords for d in self.global_state.drones.values() if
                        d.current_location_coords]
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