from typing import Dict, Any, List, Tuple, Optional

# Import entity classes
# Assuming these are available from ddls_src/entities/ and its subdirectories
from ...entities.node import Node
from ...entities.edge import Edge
from ...entities.order import Order
from ...entities.vehicles.base import Vehicle  # Base class, though we'll instantiate Truck/Drone
from ...entities.vehicles.truck import Truck
from ...entities.vehicles.drone import Drone
from ...entities.micro_hub import MicroHub


class ScenarioGenerator:  # Renamed from SimulationBuilder
    """
    Provides a standard API to construct and populate simulation entities (Nodes, Edges,
    Vehicles, Orders, MicroHubs) from raw configuration data or programmatic calls.
    It instantiates the actual entity objects from dictionaries to form a complete scenario.
    """

    def __init__(self, raw_entity_data: Optional[Dict[str, Any]] = None):
        """
        Initializes the ScenarioGenerator.

        Args:
            raw_entity_data (Optional[Dict[str, Any]]): A dictionary containing raw data
                                                         for entities (e.g., loaded from JSON).
                                                         Expected keys: 'nodes', 'edges', 'trucks',
                                                         'drones', 'micro_hubs', 'orders'.
        """
        self._raw_entity_data = raw_entity_data if raw_entity_data is not None else {}

        # Dictionaries to hold instantiated entity objects, keyed by ID
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}
        self.trucks: Dict[int, Truck] = {}
        self.drones: Dict[int, Drone] = {}
        self.micro_hubs: Dict[int, MicroHub] = {}
        self.orders: Dict[int, Order] = {}

        # Store initial time if present in raw data, otherwise default
        self.initial_time: float = self._raw_entity_data.get('initial_time', 0.0)

        print("ScenarioGenerator initialized.")  # Updated print

    def add_node(self, id: int, coords: Tuple[float, float], type: str,
                 is_loadable: bool = False, is_unloadable: bool = False,
                 is_charging_station: bool = False) -> Node:
        """Programmatic API to add a Node entity."""
        node = Node(id, coords, type, is_loadable, is_unloadable, is_charging_station)
        if node.id in self.nodes:
            raise ValueError(f"Node with ID {node.id} already exists.")
        self.nodes[node.id] = node
        return node

    def add_edge(self, id: int, start_node_id: int, end_node_id: int,
                 length: float, base_travel_time: float) -> Edge:
        """Programmatic API to add an Edge entity."""
        edge = Edge(id, start_node_id, end_node_id, length, base_travel_time)
        if edge.id in self.edges:
            raise ValueError(f"Edge with ID {edge.id} already exists.")
        self.edges[edge.id] = edge
        return edge

    def add_truck(self, id: int, start_node_id: int, max_payload_capacity: float,
                  max_speed: float, initial_fuel: float, fuel_consumption_rate: float) -> Truck:
        """Programmatic API to add a Truck entity."""
        truck = Truck(id, start_node_id, max_payload_capacity, max_speed, initial_fuel, fuel_consumption_rate)
        if truck.id in self.trucks:
            raise ValueError(f"Truck with ID {truck.id} already exists.")
        self.trucks[truck.id] = truck
        return truck

    def add_drone(self, id: int, start_node_id: int, max_payload_capacity: float,
                  max_speed: float, initial_battery: float, battery_drain_rate_flying: float,
                  battery_drain_rate_idle: float, battery_charge_rate: float) -> Drone:
        """Programmatic API to add a Drone entity."""
        drone = Drone(id, start_node_id, max_payload_capacity, max_speed, initial_battery,
                      battery_drain_rate_flying, battery_drain_rate_idle, battery_charge_rate)
        if drone.id in self.drones:
            raise ValueError(f"Drone with ID {drone.id} already exists.")
        self.drones[drone.id] = drone
        return drone

    def add_micro_hub(self, id: int, coords: Tuple[float, float], num_charging_slots: int,
                      type: str = 'micro_hub', operational_status: str = "inactive",  # Added default status
                      is_blocked_for_launches: bool = False, is_blocked_for_recoveries: bool = False,
                      is_package_transfer_unavailable: bool = False) -> MicroHub:  # Added default flags
        """Programmatic API to add a MicroHub entity."""
        micro_hub = MicroHub(id, coords, num_charging_slots, type)  # Pass all args
        if micro_hub.id in self.micro_hubs:
            raise ValueError(f"MicroHub with ID {micro_hub.id} already exists.")
        self.micro_hubs[micro_hub.id] = micro_hub
        self.nodes[micro_hub.id] = micro_hub  # Also add to the general nodes dictionary
        return micro_hub

    def add_order(self, id: int, customer_node_id: int, time_received: float,
                  SLA_deadline: float, priority: int = 1) -> Order:
        """Programmatic API to add an Order entity."""
        order = Order(id, customer_node_id, time_received, SLA_deadline, priority)
        if order.id in self.orders:
            raise ValueError(f"Order with ID {order.id} already exists.")
        self.orders[order.id] = order
        return order

    def build_entities(self) -> Dict[str, Dict[int, Any]]:
        """
        Instantiates all simulation entity objects from the raw_entity_data
        provided during initialization (or added programmatically).

        Returns:
            Dict[str, Dict[int, Any]]: A dictionary containing dictionaries of
                                       instantiated entity objects, ready for GlobalState.
                                       Keys: 'nodes', 'edges', 'trucks', 'drones', 'micro_hubs', 'orders'.
        """
        print("ScenarioGenerator: Building entities from raw data...")  # Updated print

        # Instantiate Nodes and MicroHubs based on their type
        for node_data in self._raw_entity_data.get('nodes', []):
            node_type = node_data.get('type')
            packages_held_at_node = node_data.pop('packages_held', [])  # Extract and remove packages_held

            if node_type == 'micro_hub':
                # Extract specific arguments for MicroHub, including default optional ones
                micro_hub_args = {
                    'id': node_data['id'],
                    'coords': tuple(node_data['coords']),
                    'num_charging_slots': node_data['num_charging_slots'],
                    'type': node_type,

                }
                instantiated_node = self.add_micro_hub(**micro_hub_args)
            else:
                # For generic nodes, ensure only Node.__init__ arguments are passed
                # Filter out 'num_charging_slots' if it somehow appears in non-micro_hub node data
                node_args = {k: v for k, v in node_data.items() if k != 'num_charging_slots'}
                # Ensure coords is a tuple if it's a list from JSON
                if 'coords' in node_args and isinstance(node_args['coords'], list):
                    node_args['coords'] = tuple(node_args['coords'])
                instantiated_node = self.add_node(**node_args)

            # Add packages to the instantiated node after creation
            for order_id in packages_held_at_node:
                instantiated_node.add_package(order_id)

        # Instantiate Edges
        for edge_data in self._raw_entity_data.get('edges', []):
            self.add_edge(**edge_data)

        # Instantiate Trucks
        for truck_data in self._raw_entity_data.get('trucks', []):
            self.add_truck(**truck_data)

        # Instantiate Drones
        for drone_data in self._raw_entity_data.get('drones', []):
            self.add_drone(**drone_data)

        # Instantiate Orders
        for order_data in self._raw_entity_data.get('orders', []):
            self.add_order(**order_data)

        print("ScenarioGenerator: All entities instantiated.")  # Updated print

        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'trucks': self.trucks,
            'drones': self.drones,
            'micro_hubs': self.micro_hubs,  # This will now contain only MicroHub objects created from 'nodes' list
            'orders': self.orders,
            'initial_time': self.initial_time  # Pass initial time from raw data
        }

    # --- Plotting Methods (Placeholder) ---
    def initialize_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Initializes plot data related to the scenario setup, if any.
        (e.g., indicating the number of entities created).
        """
        print("ScenarioGenerator: Initializing plot data.")  # Updated print
        if 'scenario_setup_info' not in figure_data:
            figure_data['scenario_setup_info'] = {}
        figure_data['scenario_setup_info']['num_nodes'] = len(self.nodes)
        figure_data['scenario_setup_info']['num_edges'] = len(self.edges)
        figure_data['scenario_setup_info']['num_trucks'] = len(self.trucks)
        figure_data['scenario_setup_info']['num_drones'] = len(self.drones)
        figure_data['scenario_setup_info']['num_micro_hubs'] = len(self.micro_hubs)
        figure_data['scenario_setup_info']['num_orders'] = len(self.orders)

    def update_plot_data(self, figure_data: Dict[str, Any]) -> None:
        """
        Updates plot data related to the scenario setup. (Less dynamic for a generator).
        """
        print("ScenarioGenerator: Updating plot data (no dynamic changes expected).")  # Updated print
        # No dynamic updates typically for a builder once entities are built.
