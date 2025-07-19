import random
from typing import Dict, Any, List, Tuple
from .data_generator import BaseDataGenerator  # Import the base class


class RandomDataGenerator(BaseDataGenerator):
    """
    A concrete data generator that procedurally generates random initial simulation data
    for nodes, edges, vehicles, and orders based on configurable parameters,
    with core counts scaled by a 'base_scale_factor'.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the RandomDataGenerator.

        Args:
            config (Dict[str, Any]): Configuration for random data generation. Expected keys:
                                     'base_scale_factor': int (main parameter to scale scenario size)
                                     'grid_size': Tuple[float, float] (e.g., [100.0, 100.0]),
                                     'max_edges_per_node': int,
                                     'max_travel_time_per_unit_length': float,
                                     'truck_payload_range': Tuple[int, int],
                                     'drone_payload_range': Tuple[int, int],
                                     'truck_speed_range': Tuple[float, float],
                                     'drone_speed_range': Tuple[float, float],
                                     'initial_fuel_range': Tuple[float, float],
                                     'initial_battery_range': Tuple[float, float],
                                     'sla_min_hours': float,
                                     'sla_max_hours': float,
                                     'priority_distribution': Dict[int, float]
                                     'scaling_factors': Dict[str, float] (e.g., {'nodes': 2.0, 'customers': 0.7, ...})
        """
        super().__init__(config)

        self.base_scale_factor = config.get('base_scale_factor', 5)  # Default scale factor

        # Define default scaling factors for various entities relative to base_scale_factor
        default_scaling_factors = {
            'nodes': 2.0,
            'depots': 0.1,  # New: scaling factor for number of depots
            'customers': 1.5,
            'micro_hubs': 0.2,
            'trucks': 0.1,
            'drones': 0.4,
            'initial_orders': 1.0
        }
        self.scaling_factors = config.get('scaling_factors', default_scaling_factors)

        # Derived counts based on base_scale_factor and scaling_factors
        self.num_depots = max(1, int(self.base_scale_factor * self.scaling_factors.get('depots', 0.1)))
        self.num_customers = max(1, int(self.base_scale_factor * self.scaling_factors.get('customers', 1.5)))
        self.num_micro_hubs = max(0, int(self.base_scale_factor * self.scaling_factors.get('micro_hubs', 0.2)))
        self.num_trucks = max(1, int(self.base_scale_factor * self.scaling_factors.get('trucks', 0.1)))
        self.num_drones = max(0, int(self.base_scale_factor * self.scaling_factors.get('drones', 0.4)))
        self.num_initial_orders = max(1, int(self.base_scale_factor * self.scaling_factors.get('initial_orders', 1.0)))

        # Ensure num_depots + num_customers + num_micro_hubs does not exceed num_nodes
        total_special_nodes = self.num_depots + self.num_customers + self.num_micro_hubs
        self.num_nodes = max(total_special_nodes + 2, int(self.base_scale_factor * self.scaling_factors.get('nodes',
                                                                                                            2.0)))  # Ensure enough nodes for special types + buffer

        self.grid_size = tuple(config.get('grid_size', [50.0, 50.0]))  # (width, height)
        self.max_edges_per_node = config.get('max_edges_per_node', 3)
        self.max_travel_time_per_unit_length = config.get('max_travel_time_per_unit_length',
                                                          60.0)  # seconds per unit length
        self.truck_payload_range = tuple(config.get('truck_payload_range', [3, 10]))
        self.drone_payload_range = tuple(config.get('drone_payload_range', [1, 2]))
        self.truck_speed_range = tuple(config.get('truck_speed_range', [40.0, 80.0]))  # units per hour
        self.drone_speed_range = tuple(config.get('drone_speed_range', [20.0, 40.0]))  # units per hour
        self.initial_fuel_range = tuple(config.get('initial_fuel_range', [80.0, 120.0]))
        self.initial_battery_range = tuple(config.get('initial_battery_range', [0.8, 1.0]))  # 0.0 to 1.0
        self.sla_min_seconds = config.get('sla_min_hours', 1.0) * 3600
        self.sla_max_seconds = config.get('sla_max_hours', 4.0) * 3600
        self.priority_distribution = config.get('priority_distribution', {1: 1.0})

        # Fixed rates for vehicles (can be made configurable)
        self.truck_fuel_consumption_rate = config.get('truck_fuel_consumption_rate', 0.1)  # per minute
        self.drone_battery_drain_rate_flying = config.get('drone_battery_drain_rate_flying', 0.005)  # per minute
        self.drone_battery_drain_rate_idle = config.get('drone_battery_drain_rate_idle', 0.001)  # per minute
        self.drone_battery_charge_rate = config.get('drone_battery_charge_rate', 0.01)  # per minute

        print(f"RandomDataGenerator initialized with base_scale_factor={self.base_scale_factor}.")
        print(
            f"  Derived counts: Nodes={self.num_nodes}, Depots={self.num_depots}, Customers={self.num_customers}, MicroHubs={self.num_micro_hubs}, "
            f"Trucks={self.num_trucks}, Drones={self.num_drones}, InitialOrders={self.num_initial_orders}")

    def generate_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates random initial simulation data based on the configured scale factor.
        """
        print("RandomDataGenerator: Generating random simulation data...")
        data = {
            "nodes": [],
            "edges": [],
            "trucks": [],
            "drones": [],
            "micro_hubs": [],
            "orders": [],
            "initial_time": 0.0
        }

        node_coords = {}
        depot_ids = []
        customer_ids = []
        micro_hub_ids = []

        current_node_id = 0

        # Generate Depots
        for i in range(self.num_depots):
            coords = (random.uniform(0, self.grid_size[0] / 4),
                      random.uniform(0, self.grid_size[1] / 4))  # Depots in a corner region
            data["nodes"].append({"id": current_node_id, "coords": list(coords), "type": "depot",
                                  "is_loadable": True, "is_unloadable": True, "is_charging_station": True})
            node_coords[current_node_id] = coords
            depot_ids.append(current_node_id)
            current_node_id += 1

        # Generate Customers
        for i in range(self.num_customers):
            coords = (random.uniform(0, self.grid_size[0]), random.uniform(0, self.grid_size[1]))
            data["nodes"].append({"id": current_node_id, "coords": list(coords), "type": "customer",
                                  "is_loadable": False, "is_unloadable": True, "is_charging_station": False})
            node_coords[current_node_id] = coords
            customer_ids.append(current_node_id)
            current_node_id += 1

        # Generate Micro-hubs
        for i in range(self.num_micro_hubs):
            coords = (random.uniform(0, self.grid_size[0]), random.uniform(0, self.grid_size[1]))
            num_slots = random.randint(1, 3)  # Random number of charging slots
            data["nodes"].append({"id": current_node_id, "coords": list(coords), "type": "micro_hub",
                                  "is_loadable": True, "is_unloadable": True, "is_charging_station": True,
                                  "num_charging_slots": num_slots})
            node_coords[current_node_id] = coords
            micro_hub_ids.append(current_node_id)
            current_node_id += 1

        # Generate generic nodes if num_nodes is greater than sum of specific nodes
        num_generic_nodes = self.num_nodes - len(depot_ids) - self.num_customers - self.num_micro_hubs
        for i in range(num_generic_nodes):
            coords = (random.uniform(0, self.grid_size[0]), random.uniform(0, self.grid_size[1]))
            data["nodes"].append({"id": current_node_id, "coords": list(coords), "type": "junction",
                                  "is_loadable": False, "is_unloadable": False, "is_charging_station": False})
            node_coords[current_node_id] = coords
            current_node_id += 1

        all_node_ids = list(node_coords.keys())
        if not all_node_ids:
            print("RandomDataGenerator: No nodes generated. Cannot create edges or entities.")
            return data

        # Generate Edges (simple connectivity for now)
        edge_id_counter = 0
        for i in range(len(all_node_ids)):
            node1_id = all_node_ids[i]

            num_connections = random.randint(1, self.max_edges_per_node)

            # Ensure each node has at least one outgoing edge if possible
            if not [e for e in data["edges"] if e["start_node_id"] == node1_id]:
                num_connections = max(1, num_connections)

            for _ in range(num_connections):
                possible_targets = [n for n in all_node_ids if n != node1_id]
                current_node_outgoing_edges = [e["end_node_id"] for e in data["edges"] if
                                               e["start_node_id"] == node1_id]
                possible_targets = [n for n in possible_targets if n not in current_node_outgoing_edges]

                if not possible_targets:
                    break

                node2_id = random.choice(possible_targets)

                coords1 = node_coords[node1_id]
                coords2 = node_coords[node2_id]
                length = ((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2) ** 0.5
                base_travel_time = length * self.max_travel_time_per_unit_length

                # Check for existing edge (bidirectional check)
                exists = False
                for existing_edge in data["edges"]:
                    if (existing_edge["start_node_id"] == node1_id and existing_edge["end_node_id"] == node2_id) or \
                            (existing_edge["start_node_id"] == node2_id and existing_edge["end_node_id"] == node1_id):
                        exists = True
                        break
                if not exists:
                    data["edges"].append({"id": edge_id_counter, "start_node_id": node1_id, "end_node_id": node2_id,
                                          "length": length, "base_travel_time": base_travel_time})
                    edge_id_counter += 1
                    # Add reverse edge for bidirectionality
                    data["edges"].append({"id": edge_id_counter, "start_node_id": node2_id, "end_node_id": node1_id,
                                          "length": length, "base_travel_time": base_travel_time})
                    edge_id_counter += 1

        # Generate Trucks
        if not depot_ids:
            print("Warning: No depots generated. Trucks will start at a random node.")
            truck_start_nodes = all_node_ids
        else:
            truck_start_nodes = depot_ids

        for i in range(self.num_trucks):
            start_node = random.choice(truck_start_nodes)
            data["trucks"].append({
                "id": 100 + i,
                "start_node_id": start_node,
                "max_payload_capacity": random.randint(*self.truck_payload_range),
                "max_speed": random.uniform(*self.truck_speed_range),
                "initial_fuel": random.uniform(*self.initial_fuel_range),
                "fuel_consumption_rate": self.truck_fuel_consumption_rate
            })

        # Generate Drones
        drone_start_nodes = []
        if depot_ids: drone_start_nodes.extend(depot_ids)
        if micro_hub_ids: drone_start_nodes.extend(micro_hub_ids)
        if not drone_start_nodes:
            print("Warning: No depots or micro-hubs. Drones will start at a random node.")
            drone_start_nodes = all_node_ids  # Fallback to any node

        for i in range(self.num_drones):
            start_node = random.choice(drone_start_nodes)
            data["drones"].append({
                "id": 200 + i,
                "start_node_id": start_node,
                "max_payload_capacity": random.randint(*self.drone_payload_range),
                "max_speed": random.uniform(*self.drone_speed_range),
                "initial_battery": random.uniform(*self.initial_battery_range),
                "battery_drain_rate_flying": self.drone_battery_drain_rate_flying,
                "battery_drain_rate_idle": self.drone_battery_drain_rate_idle,
                "battery_charge_rate": self.drone_battery_charge_rate
            })

        # Generate Initial Orders
        order_id_counter = 1000
        for i in range(self.num_initial_orders):
            if not customer_ids:
                print("Warning: No customer nodes available to generate orders.")
                break
            customer_node_id = random.choice(customer_ids)
            time_received = 0.0  # All initial orders received at t=0
            sla_deadline = time_received + random.uniform(self.sla_min_seconds, self.sla_max_seconds)

            priorities, weights = zip(*self.priority_distribution.items())
            priority = random.choices(priorities, weights=weights, k=1)[0]

            data["orders"].append({
                "id": order_id_counter + i,
                "customer_node_id": customer_node_id,
                "time_received": time_received,
                "SLA_deadline": sla_deadline,
                "priority": priority
            })

        print(f"RandomDataGenerator: Generated {len(data['nodes'])} nodes, {len(data['edges'])} edges, "
              f"{len(data['trucks'])} trucks, {len(data['drones'])} drones, {len(data['micro_hubs'])} micro-hubs, "
              f"{len(data['orders'])} orders.")
        return data

