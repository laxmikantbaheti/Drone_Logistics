from typing import Dict, Any, List, Tuple, Optional

# Import entity classes
from ddls_src.entities.node import Node
from ddls_src.entities.edge import Edge
from ddls_src.entities.order import Order
from ddls_src.entities.vehicles.base import Vehicle
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities.micro_hub import MicroHub


class ScenarioGenerator:
    """
    Instantiates entity objects from raw configuration data.
    This version uses a two-phase initialization to handle dependencies.
    """

    def __init__(self, raw_entity_data: Optional[Dict[str, Any]] = None):
        self._raw_entity_data = raw_entity_data if raw_entity_data is not None else {}
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}
        self.trucks: Dict[int, Truck] = {}
        self.drones: Dict[int, Drone] = {}
        self.micro_hubs: Dict[int, MicroHub] = {}
        self.orders: Dict[int, Order] = {}
        self.initial_time: float = self._raw_entity_data.get('initial_time', 0.0)
        print("ScenarioGenerator initialized.")

    def _prepare_kwargs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates keys from the simple JSON format to the formal MLPro constructor parameters.
        Specifically, it renames 'id' to 'p_id'.
        """
        if 'id' in data:
            data['p_id'] = data.pop('id')
        return data

    def add_node(self, **p_kwargs) -> Node:
        node = Node(**p_kwargs)
        self.nodes[node.id] = node
        return node

    def add_micro_hub(self, **p_kwargs) -> MicroHub:
        micro_hub = MicroHub(**p_kwargs)
        self.micro_hubs[micro_hub.id] = micro_hub
        self.nodes[micro_hub.id] = micro_hub
        return micro_hub

    def build_entities(self) -> Dict[str, Dict[int, Any]]:
        """
        Phase 1 of initialization: Instantiates all entity objects from raw data.
        """
        print("ScenarioGenerator: Building entities from raw data...")

        for node_data in self._raw_entity_data.get('nodes', []):
            node_data = self._prepare_kwargs(node_data)  # <-- FIX: Rename 'id' to 'p_id'
            node_type = node_data.get('type')
            packages_held = node_data.pop('packages_held', [])

            if 'coords' in node_data and isinstance(node_data['coords'], list):
                node_data['coords'] = tuple(node_data['coords'])

            if node_type == 'micro_hub':
                instantiated_node = self.add_micro_hub(**node_data)
            else:
                instantiated_node = self.add_node(**node_data)

            instantiated_node.temp_packages = packages_held

        for edge_data in self._raw_entity_data.get('edges', []):
            edge_data = self._prepare_kwargs(edge_data)  # <-- FIX
            self.edges[edge_data['p_id']] = Edge(**edge_data)

        for truck_data in self._raw_entity_data.get('trucks', []):
            truck_data = self._prepare_kwargs(truck_data)  # <-- FIX
            self.trucks[truck_data['p_id']] = Truck(**truck_data)

        for drone_data in self._raw_entity_data.get('drones', []):
            drone_data = self._prepare_kwargs(drone_data)  # <-- FIX
            self.drones[drone_data['p_id']] = Drone(**drone_data)

        for order_data in self._raw_entity_data.get('orders', []):
            order_data = self._prepare_kwargs(order_data)  # <-- FIX
            self.orders[order_data['p_id']] = Order(**order_data)

        print("ScenarioGenerator: All entities instantiated (Phase 1 complete).")

        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'trucks': self.trucks,
            'drones': self.drones,
            'micro_hubs': self.micro_hubs,
            'orders': self.orders,
            'initial_time': self.initial_time
        }
