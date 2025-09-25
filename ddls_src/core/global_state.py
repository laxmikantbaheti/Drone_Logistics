import itertools
from typing import Dict, Any, List, Tuple


# Forward declarations for entities to avoid circular imports.
# These will be replaced by actual imports once the entity classes are defined.
class Node: pass


class Edge: pass


class Order: pass


class Truck: pass


class Drone: pass


class MicroHub: pass


class Network:  # Forward declaration for Network class
    pass


class OrderRequests:
    pass


class GlobalState:
    """
    A single source of truth for all simulation data, providing controlled access.
    All managers and entities will interact with the simulation state through this class.
    """

    def __init__(self, initial_entities: Dict[str, Dict[int, Any]]):  # UPDATED: Takes instantiated entities
        """
        Initializes all entities based on pre-instantiated entity objects.

        Args:
            initial_entities (Dict[str, Dict[int, Any]]): A dictionary containing
                                                          dictionaries of instantiated
                                                          entity objects, e.g.:
                                                          {
                                                              'nodes': {id: Node_obj, ...},
                                                              'edges': {id: Edge_obj, ...},
                                                              ...
                                                          }
        """
        self.nodes: Dict[int, Node] = initial_entities.get('nodes', {})
        self.edges: Dict[int, Edge] = initial_entities.get('edges', {})
        self.orders: Dict[int, Order] = initial_entities.get('orders', {})
        self.trucks: Dict[int, Truck] = initial_entities.get('trucks', {})
        self.drones: Dict[int, Drone] = initial_entities.get('drones', {})
        self.micro_hubs: Dict[int, MicroHub] = initial_entities.get('micro_hubs', {})
        self.current_time: float = initial_entities.get('initial_time', 0.0)  # Can be passed from builder config
        self.network: Network = None  # Reference to the Network graph structure, populated after entities
        self.node_pairs = self.setup_node_pairs()
        self.orders_by_nodes = self.setup_order_by_node_pairs()
        print("GlobalState initialized with provided entities.")

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

    def get_all_entities(self, entity_type: str) -> Dict[int, Any]:
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
        # print(f"Added {entity_obj.__class__.__name__} with ID {entity_obj.id}")

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
        # print(f"Removed {entity_type} with ID {entity_id}")

    def add_vehicles(self, p_vehicles:[]):
        # TODO: Do this
        pass

    def remove_vehicles(self, p_vehicles:[]):
        pass

    def add_nodes(self, p_nodes:[Node]):
        # Todo: Do this
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

    # --- Specific Getters (as per plan) ---

    def get_truck_location(self, truck_id: int) -> int:
        """Returns current node ID of truck."""
        truck = self.get_entity("truck", truck_id)
        # Assumes Truck class has current_node_id attribute
        return truck.current_node_id

    def is_node_loadable(self, node_id: int) -> bool:
        """Checks if a node is a valid loading point."""
        node = self.get_entity("node", node_id)
        # Assumes Node class has is_loadable attribute
        return node.is_loadable

    def get_order_status(self, order_id: int) -> str:
        """Returns order status."""
        order = self.get_entity("order", order_id)
        # Assumes Order class has status attribute
        return order.status

    def get_vehicle_status(self, vehicle_id: int) -> str:
        """Returns vehicle status (can be truck or drone)."""
        # This requires checking both truck and drone dictionaries or a unified vehicle type handling
        if vehicle_id in self.trucks:
            return self.trucks[vehicle_id].status
        elif vehicle_id in self.drones:
            return self.drones[vehicle_id].status
        else:
            raise KeyError(f"Vehicle with ID '{vehicle_id}' not found in trucks or drones.")

    def get_drone_battery_level(self, drone_id: int) -> float:
        """Returns drone battery."""
        drone = self.get_entity("drone", drone_id)
        # Assumes Drone class has battery_level attribute
        return drone.battery_level

    def get_micro_hub_status(self, hub_id: int) -> str:
        """Returns micro-hub status."""
        hub = self.get_entity("micro_hub", hub_id)
        # Assumes MicroHub class has operational_status attribute
        return hub.operational_status

    def get_packages_at_node(self, node_id: int) -> List[int]:
        """Returns list of order IDs at a node."""
        node = self.get_entity("node", node_id)
        # Assumes Node class has packages_held attribute
        return node.packages_held

    # --- Plotting Methods (Placeholders) ---
    def initialize_plot_data(self, figure_data: dict):
        """
        Sets up the initial plotting data for GlobalState-level elements (e.g., the network graph).
        Modifies the passed figure_data dictionary.
        This method will typically be called once at the start of a simulation.
        """
        print("GlobalState: Initializing plot data...")
        # This method will call similar initialization methods on its contained entities
        # or aggregate their initial data.
        # For example, it might iterate self.nodes and call node.initialize_plot_data() if nodes handle their own drawing.
        # Or it might directly add network nodes/edges data here.

        # Placeholder for network plot (nodes and edges)
        # Assumes Node has 'coords' and 'type', Edge has 'start_node_id' and 'end_node_id'
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
        # Other initial plot data as needed.
        print("GlobalState: Initial plot data placeholder added to figure_data.")

    def update_plot_data(self, figure_data: dict):
        """
        Updates the plotting data for GlobalState-level elements for the current simulation step.
        Modifies the passed figure_data dictionary.
        This method will typically be called after each main simulation timestep.
        """
        print(f"GlobalState: Updating plot data at time {self.current_time}...")
        # For dynamic elements like vehicles and parcels, their current positions/statuses
        # will need to be updated in the figure_data.

        # For example, vehicles will have their own update_plot_data methods
        # Or GlobalState can aggregate data for all vehicles and update a single 'vehicles' layer in figure_data

        # Placeholder for vehicle positions
        # Assumes Truck has 'current_location_coords', 'status', 'cargo_manifest'
        # Assumes Drone has 'current_location_coords', 'status', 'battery_level', 'cargo_manifest'
        figure_data['vehicle_positions'] = {
            'trucks': [{'id': t.id, 'coords': t.current_location_coords, 'status': t.status} for t in
                       self.trucks.values()],
            'drones': [{'id': d.id, 'coords': d.current_location_coords, 'status': d.status, 'battery': d.battery_level}
                       for d in self.drones.values()]
        }

        # Placeholder for parcels at nodes / in vehicles
        # Assumes Node has 'packages_held'
        # Assumes Truck/Drone have 'cargo_manifest'
        figure_data['parcel_locations'] = {
            'at_nodes': {node_id: node.packages_held for node_id, node in self.nodes.items() if node.packages_held},
            'in_trucks': {truck_id: truck.cargo_manifest for truck_id, truck in self.trucks.items() if
                          truck.cargo_manifest},
            'in_drones': {drone_id: drone.cargo_manifest for drone_id, drone in self.drones.items() if
                          drone.cargo_manifest}
        }

        # The actual implementation will depend on the final structure of the figure_data
        # and how the plotting library expects to receive updates (e.g., updating scatter points, line segments).
        print("GlobalState: Update plot data placeholder added to figure_data.")

    def setup_order_by_node_pairs(self):
        order_requests = {}
        for ids,order in self.orders.items() :
            node_pick_up = order.get_pickup_node_id()
            node_delivery = order.get_delivery_node_id()
            if (node_pick_up, node_delivery) not in order_requests.keys():
                order_requests[(node_pick_up, node_delivery)] = [order]
            else:
                order_requests[(node_pick_up,node_delivery)].append(order)
        return order_requests

    def get_order_requests(self):
        order_requests = {}
        for ids, order in self.orders.items():
            if order.get_state_value_by_dim_name(order.C_DIM_DELIVERY_STATUS[0]) == order.C_STATUS_PLACED:
                node_pick_up = order.get_pickup_node_id()
                node_delivery = order.get_delivery_node_id()
                if (node_pick_up, node_delivery) not in order_requests.keys():
                    order_requests[(node_pick_up, node_delivery)] = [order]
                else:
                    order_requests[(node_pick_up, node_delivery)].append(order)
        return order_requests

    def get_orders(self):
        return self.orders

