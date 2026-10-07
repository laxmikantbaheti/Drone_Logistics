import itertools
from collections import defaultdict
from ddls_src.entities.order import PseudoOrder
from ddls_src.functions.data_manager_obsolete import DataManager
from typing import Dict, Any, List, Tuple
from ddls_src.functions.event_logger import EventLogger
from datetime import datetime

# Forward declarations for entities to avoid circular imports.
class Node: pass
class Edge: pass
class Order: pass
class Truck: pass
class Drone: pass
class MicroHub: pass
class Network: pass
class OrderRequests: pass


class GlobalState:
    """
    A single source of truth for all simulation data, providing controlled access.
    All managers and entities will interact with the simulation state through this class.
    """

    def __init__(self, initial_entities: Dict[str, Dict[int, Any]], movement_mode, custom_log = False):
        self.micro_hub_phase = True
        self.active_resource = None
        self.custom_log = custom_log
        self.entity_dicts = {}
        self.nodes: Dict[int, Node] = initial_entities.get('nodes', {})
        self.entity_dicts["Node"] = self.nodes
        self.edges: Dict[int, Edge] = initial_entities.get('edges', {})
        self.entity_dicts["Edge"] = self.edges
        self.orders: Dict[int, Order] = initial_entities.get('orders', {})
        self.entity_dicts["Order"] = self.orders
        self.pseudo_orders: Dict[int, PseudoOrder] = initial_entities.get('pseudo_orders', {})
        self.entity_dicts["Pseudo Order"] = self.pseudo_orders
        self.trucks: Dict[int, Truck] = initial_entities.get('trucks', {})
        self.entity_dicts["Truck"] = self.trucks
        self.drones: Dict[int, Drone] = initial_entities.get('drones', {})
        self.entity_dicts["Drone"] = self.drones
        self.micro_hubs: Dict[int, MicroHub] = initial_entities.get('micro_hubs', {})
        self.entity_dicts["MicroHub"] = self.micro_hubs
        self.current_time: float = initial_entities.get('initial_time', 0.0)
        self.network: Network = None
        self.node_pairs = initial_entities.get('node_pairs', {})
        self.entity_dicts["Node Pair"] = self.node_pairs
        self.entities_by_type = {"Node", "Edge", "Order", "Pseudo Order", "Truck", "Drone", "Micro Hub", "Node Pair"}
        self.orders_by_nodes = self.setup_order_by_node_pairs()
        self.capacity_demands = self.setup_capacity_demands()
        self.movement_mode = movement_mode
        self.microhub_routes = self.get_microhub_routes()
        self.agent_masks = []
        self.evaluation_deck = []
        self.computational_profile = 0
        self.constraint_latency = datetime.now()
        self.constraint_latency_per_step = []
        # Initialize the centralized DataManager
        self.data_manager = DataManager()
        self.event_logger = EventLogger()

        # -----------------------------------------------------------------------------------------
        # [NEW]: Deterministic Integer Indexing Registries (Parallel Non-Breaking Buffers)
        # -----------------------------------------------------------------------------------------
        self.truck_id_to_idx: Dict[Any, int] = {}
        self.drone_id_to_idx: Dict[Any, int] = {}
        self.microhub_id_to_idx: Dict[Any, int] = {}
        self.node_id_to_idx: Dict[Any, int] = {}
        self.nodepair_to_idx: Dict[Any, int] = {}

        self.trucks_by_idx: List[Any] = []
        self.drones_by_idx: List[Any] = []
        self.microhubs_by_idx: List[Any] = []
        self.nodes_by_idx: List[Any] = []
        self.nodepairs_by_idx: List[Any] = []

        if self.custom_log:
            print(f"GlobalState initialized with provided entities. Movement mode set to: '{self.movement_mode}'.")

    # ---------------------------------------------------------------------------------------------
    # [NEW INTERFACE]: Entity Deterministic Indexing Setup
    # ---------------------------------------------------------------------------------------------
    def initialize_entity_indexing(self):
        """
        Assigns deterministic, contiguous zero-based integer indices
        to physical entities without modifying existing .id or .get_id() attributes.
        Populates bidirectional lookup structures in GlobalState.
        """
        # 1. Trucks (0 to V_truck - 1)
        sorted_trucks = sorted(self.trucks.values(), key=lambda t: str(t.get_id()))
        self.truck_id_to_idx = {}
        self.trucks_by_idx = sorted_trucks
        for idx, truck in enumerate(sorted_trucks):
            truck.int_id = idx
            truck.truck_int_id = idx
            self.truck_id_to_idx[truck.get_id()] = idx
            if hasattr(truck, 'id'):
                self.truck_id_to_idx[truck.id] = idx
        # Also map dictionary keys directly
        for k, truck in self.trucks.items():
            self.truck_id_to_idx[k] = truck.int_id

        # 2. Drones (0 to V_drone - 1)
        sorted_drones = sorted(self.drones.values(), key=lambda d: str(d.get_id()))
        self.drone_id_to_idx = {}
        self.drones_by_idx = sorted_drones
        for idx, drone in enumerate(sorted_drones):
            drone.int_id = idx
            drone.drone_int_id = idx
            self.drone_id_to_idx[drone.get_id()] = idx
            if hasattr(drone, 'id'):
                self.drone_id_to_idx[drone.id] = idx
        for k, drone in self.drones.items():
            self.drone_id_to_idx[k] = drone.int_id

        # 3. Micro-Hubs (0 to M - 1)
        sorted_mhs = sorted(self.micro_hubs.values(), key=lambda mh: str(mh.get_id()))
        self.microhub_id_to_idx = {}
        self.microhubs_by_idx = sorted_mhs
        for idx, mh in enumerate(sorted_mhs):
            mh.int_id = idx
            mh.mh_int_id = idx
            self.microhub_id_to_idx[mh.get_id()] = idx
            if hasattr(mh, 'id'):
                self.microhub_id_to_idx[mh.id] = idx
        for k, mh in self.micro_hubs.items():
            self.microhub_id_to_idx[k] = mh.mh_int_id

        # 4. Physical Nodes (0 to N - 1)
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: str(n.get_id()))
        self.node_id_to_idx = {}
        self.nodes_by_idx = sorted_nodes
        for idx, node in enumerate(sorted_nodes):
            node.node_int_id = idx
            # Do not overwrite mh.int_id if node is also a micro-hub
            if not hasattr(node, 'mh_int_id'):
                node.int_id = idx
            self.node_id_to_idx[node.get_id()] = idx
            if hasattr(node, 'id'):
                self.node_id_to_idx[node.id] = idx
        for k, node in self.nodes.items():
            self.node_id_to_idx[k] = getattr(node, 'node_int_id', node.int_id)

        # 5. Node Pairs (0 to P - 1)
        self.nodepair_to_idx = {}
        self.nodepairs_by_idx = []
        if isinstance(self.node_pairs, dict):
            sorted_pairs = sorted(self.node_pairs.items(), key=lambda item: str(item[0]))
            for idx, (pair_key, pair_val) in enumerate(sorted_pairs):
                self.nodepair_to_idx[pair_key] = idx
                self.nodepairs_by_idx.append(pair_val)
                if hasattr(pair_val, '__dict__'):
                    pair_val.int_id = idx
                    pair_val.pair_int_id = idx
                if hasattr(pair_val, 'get_id'):
                    self.nodepair_to_idx[pair_val.get_id()] = idx
        else:
            sorted_pairs = sorted(self.node_pairs, key=lambda p: str(getattr(p, 'id', p)))
            self.nodepairs_by_idx = sorted_pairs
            for idx, pair_val in enumerate(sorted_pairs):
                pair_key = getattr(pair_val, 'id', pair_val)
                self.nodepair_to_idx[pair_key] = idx
                if hasattr(pair_val, '__dict__'):
                    pair_val.int_id = idx
                    pair_val.pair_int_id = idx

    def setup_node_pairs(self):
        node_ids = list(self.nodes.keys())
        node_pairs_list = list(itertools.permutations(node_ids, 2))
        node_pairs = {node_pair:(self.nodes[node_pair[0]], self.nodes[node_pair[1]]) for node_pair in node_pairs_list}
        return node_pairs

    def get_entity(self, entity_type: str, entity_id: int) -> Any:
        """
        Generic getter for any entity by type and ID.
        Raises KeyError if entity_type or entity_id is invalid.
        """
        entities_dict = getattr(self, entity_type + 's', None)  # e.g., 'nodes' for 'node'
        if entities_dict is None:
            raise KeyError(f"Unknown entity type: {entity_type}")
        if entity_id not in entities_dict:
            raise KeyError(f"Entity of type '{entity_type}' with ID '{entity_id}' not found.")
        return entities_dict[entity_id]

    def get_all_entities_by_type(self, entity_type: str) -> Dict[int, Any]:
        """
        Generic getter for all entities of a specific type.
        """
        entities_dict = getattr(self, entity_type + 's', None)
        if entities_dict is None:
            raise KeyError(f"Unknown entity type: {entity_type}")
        return entities_dict

    def update_entity_attribute(self, entity_type: str, entity_id: int, attribute: str, value: Any):
        """
        Internal method for state modification. Modifies a specific attribute of an entity.
        Managers should ideally call specific entity methods that internally call this.
        """
        entity = self.get_entity(entity_type, entity_id)
        if hasattr(entity, attribute):
            setattr(entity, attribute, value)
        else:
            raise AttributeError(f"Entity {entity_type} (ID: {entity_id}) does not have attribute '{attribute}'.")

    def add_entity(self, entity_obj: Any):
        """
        Adds a new entity instance to the state.
        Determines entity type from the object's class name (e.g., 'Truck' -> 'trucks').
        """
        entity_type_plural = entity_obj.__class__.__name__.lower() + 's'  # e.g., 'trucks'
        if not hasattr(self, entity_type_plural):
            raise ValueError(f"Cannot add entity of unknown type: {entity_obj.__class__.__name__}")

        target_dict = getattr(self, entity_type_plural)
        if entity_obj.id in target_dict:
            raise ValueError(f"Entity of type {entity_obj.__class__.__name__} with ID {entity_obj.id} already exists.")
        target_dict[entity_obj.id] = entity_obj

    def remove_entity(self, entity_type: str, entity_id: int):
        """
        Removes an entity instance from the state.
        """
        entities_dict = getattr(self, entity_type + 's', None)
        if entities_dict is None:
            raise KeyError(f"Unknown entity type: {entity_type}")
        if entity_id not in entities_dict:
            raise KeyError(f"Entity of type '{entity_type}' with ID '{entity_id}' not found for removal.")
        del entities_dict[entity_id]

    def add_vehicles(self, p_vehicles:[]):
        pass

    def remove_vehicles(self, p_vehicles:[]):
        pass

    def add_nodes(self, p_nodes:[Node]):
        pass

    def remove_node(self, p_nodes:[Node]):
        pass

    def add_edge(self, p_edges:[Node]):
        pass

    def remove_edge(self, p_edges:[]):
        pass

    def add_orders(self, p_orders:[Order]):
        pass

    def remove_orders(self, p_orders:[]):
        pass

    def add_micro_hubs(self, p_micro_hubs:[]):
        pass

    def remove_micro_hub(self, p_micro_hubs:[]):
        pass

    # --- Specific Getters ---

    def get_truck_location(self, truck_id: int) -> int:
        """Returns current node ID of truck."""
        truck = self.get_entity("truck", truck_id)
        return truck.current_node_id

    def is_node_loadable(self, node_id: int) -> bool:
        """Checks if a node is a valid loading point."""
        node = self.get_entity("node", node_id)
        return node.is_loadable

    def get_order_status(self, order_id: int) -> str:
        """Returns order status."""
        order = self.get_entity("order", order_id)
        return order.status

    def get_vehicle(self, p_vehicle_id):
        if p_vehicle_id in self.trucks:
            return self.trucks[p_vehicle_id]
        elif p_vehicle_id in self.drones:
            return self.drones[p_vehicle_id]
        else:
            raise ValueError("Vehicle does not exist in the keys of the global state.")

    def get_vehicle_status(self, vehicle_id: int) -> str:
        """Returns vehicle status (can be truck or drone)."""
        if vehicle_id in self.trucks:
            return self.trucks[vehicle_id].status
        elif vehicle_id in self.drones:
            return self.drones[vehicle_id].status
        else:
            raise KeyError(f"Vehicle with ID '{vehicle_id}' not found in trucks or drones.")

    def get_drone_battery_level(self, drone_id: int) -> float:
        """Returns drone battery."""
        drone = self.get_entity("drone", drone_id)
        return drone.battery_level

    def get_micro_hub_status(self, hub_id: int) -> str:
        """Returns micro-hub status."""
        hub = self.get_entity("micro_hub", hub_id)
        return hub.operational_status

    def get_packages_at_node(self, node_id: int) -> List[int]:
        """Returns list of order IDs at a node."""
        node = self.get_entity("node", node_id)
        return node.packages_held

    def initialize_plot_data(self, figure_data: dict):
        print("GlobalState: Initializing plot data...")
        figure_data['network_nodes'] = {
            'coords': [node.coords for node in self.nodes.values()],
            'ids': [node.id for node in self.nodes.values()],
            'types': [node.type for node in self.nodes.values()]
        }
        figure_data['network_edges'] = {
            'segments': [(self.nodes[edge.start_node_id].coords, self.nodes[edge.end_node_id].coords) for edge in
                         self.edges.values()],
            'ids': [edge.id for edge in self.edges.values()]
        }
        print("GlobalState: Initial plot data placeholder added to figure_data.")

    def update_plot_data(self, figure_data: dict):
        print(f"GlobalState: Updating plot data at time {self.current_time}...")
        figure_data['vehicle_positions'] = {
            'trucks': [{'id': t.id, 'coords': t.current_location_coords, 'status': t.status} for t in
                       self.trucks.values()],
            'drones': [{'id': d.id, 'coords': d.current_location_coords, 'status': d.status, 'battery': d.battery_level}
                       for d in self.drones.values()]
        }
        figure_data['parcel_locations'] = {
            'at_nodes': {node_id: node.packages_held for node_id, node in self.nodes.items() if node.packages_held},
            'in_trucks': {truck_id: truck.cargo_manifest for truck_id, truck in self.trucks.items() if
                          truck.cargo_manifest},
            'in_drones': {drone_id: drone.cargo_manifest for drone_id, drone in self.drones.items() if
                          drone.cargo_manifest}
        }
        print("GlobalState: Update plot data placeholder added to figure_data.")

    def setup_order_by_node_pairs(self):
        order_requests = {key:[] for key in self.node_pairs.keys()}
        for ids,order in self.orders.items() :
            node_pick_up = order.get_pickup_node_id()
            node_delivery = order.get_delivery_node_id()
            order_requests[node_pick_up,node_delivery].append(order)
        return order_requests

    def get_order_requests(self):
        order_requests = defaultdict(list)
        for order in self.orders.values():
            if order.get_state_value_by_dim_name(order.C_DIM_DELIVERY_STATUS[0]) == order.C_STATUS_PLACED:
                key = (order.get_pickup_node_id(), order.get_delivery_node_id())
                order_requests[key].append(order)
        return dict(order_requests)

    def get_orders(self):
        return self.orders

    def add_dynamic_orders(self, p_orders:list):
        for ordr in p_orders:
            self.orders[ordr.get_id()] = ordr
            self.orders_by_nodes[ordr.get_pickup_node_id(), ordr.get_delivery_node_id()].append(ordr)
            p_id, d_id = ordr.get_pickup_node_id(), ordr.get_delivery_node_id()
            if (p_id, d_id) not in self.capacity_demands:
                self.capacity_demands[p_id, d_id] = [ordr.size]
            else:
                self.capacity_demands[ordr.get_pickup_node_id(), ordr.get_delivery_node_id()].append(ordr.size)
            if isinstance(ordr, PseudoOrder):
                self.pseudo_orders[ordr.get_id()] = ordr

    def get_all_entities(self):
        return [self.node_pairs, self.orders, self.trucks, self.drones, self.micro_hubs, self.nodes]

    def reset(self, entities):
        self.current_time = 0
        for ps_order in self.pseudo_orders.keys():
            self.orders.pop(ps_order)
        self.pseudo_orders = {}
        self.data_manager.reset()
        self.event_logger.reset()

    def get_available_capacities(self):
        caps = {v.get_id():v.get_remaining_capacity() for v in (self.trucks | self.drones).values()}
        return caps

    def get_pending_demands(self, all_node_pairs=False):
        if all_node_pairs:
            return {self.node_pairs[key].get_id(): [o.size for o in value] for key, value in self.orders_by_nodes.items()}
        caps = self.setup_capacity_demands()
        return caps

    def get_next_demands(self, except_micro_hubs=True):
        if except_micro_hubs:
            return [o[0].size for np,o in self.get_order_requests().items() if len(o) and np not in self.microhub_routes.keys()]
        else:
            return [o[0].size for np,o in self.get_order_requests().items() if len(o) and np[0] not in self.micro_hubs.keys()]

    def setup_capacity_demands(self):
        caps = {self.node_pairs[key].get_id(): [o.size for o in value] for key, value in self.get_order_requests().items()}
        return caps

    def get_total_distance(self):
        tot_dist = sum([v.distance_travelled for v in (self.trucks|self.drones).values()])
        return tot_dist

    def get_vehicles(self):
        trucks = list(self.trucks.values())
        drones = list(self.drones.values())
        vehicles = trucks + drones
        return vehicles

    def get_resources(self):
        vehicles = self.get_vehicles()
        micro_hubs = list(self.micro_hubs.values())
        resources = vehicles + micro_hubs
        return resources

    def get_microhub_orders(self):
        micro_hubs = self.micro_hubs
        requests = self.get_order_requests()
        pickups = {key:value for key,value in requests.items() if key[0] in micro_hubs.keys()}
        deliveries = {key:value for key, value in requests.items() if key[1] in micro_hubs.keys()}
        return deliveries, pickups

    def get_microhub_routes(self):
        micro_hubs = self.micro_hubs
        node_pairs = self.node_pairs
        routes = {key:value for key,value in node_pairs.items() if key[0] in micro_hubs.keys() or key[1] in micro_hubs.keys()}
        return routes

    def get_microhub(self, param):
        mh = self.micro_hubs[param]
        return mh