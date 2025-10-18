import random
from typing import Dict, Any, List, Tuple
from .data_generator import BaseDataGenerator # Import the base class

class DistanceMatrixDataGenerator(BaseDataGenerator):
    """
    Generates simulation data based on a predefined or randomly generated distance matrix,
    without using explicit node coordinates. Edges are not generated as travel times
    are derived directly from the matrix.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DistanceMatrixDataGenerator.

        Args:
            config (Dict[str, Any]): Configuration for data generation. Expected keys:
                                     'base_scale_factor': int
                                     'num_nodes': int (Total number of nodes)
                                     'distance_matrix': Dict[str, Dict[str, float]] (Optional, pre-defined matrix)
                                     'max_travel_time': float (Optional, if generating matrix)
                                     # Other keys similar to RandomDataGenerator for scaling, vehicles, orders
                                     'scaling_factors': Dict[str, float]
                                     'truck_payload_range': Tuple[int, int]
                                     'drone_payload_range': Tuple[int, int]
                                     'truck_speed_range': Tuple[float, float] # Speed might be less relevant here
                                     'drone_speed_range': Tuple[float, float] # Speed might be less relevant here
                                     'initial_fuel_range': Tuple[float, float]
                                     'initial_battery_range': Tuple[float, float]
                                     'sla_min_hours': float
                                     'sla_max_hours': float
                                     'priority_distribution': Dict[int, float]
                                     'truck_fuel_consumption_rate': float
                                     'drone_battery_drain_rate_flying': float
                                     'drone_battery_drain_rate_idle': float
                                     'drone_battery_charge_rate': float
        """
        super().__init__(config)

        self.base_scale_factor = config.get('base_scale_factor', 5)

        # Scaling factors (similar to RandomDataGenerator)
        default_scaling_factors = {
            'nodes': 2.0, # Used if num_nodes is not directly provided
            'depots': 0.1,
            'customers': 1.5,
            'micro_hubs': 0.2,
            'trucks': 0.1,
            'drones': 0.4,
            'initial_orders': 1.0
        }
        self.scaling_factors = config.get('scaling_factors', default_scaling_factors)

        # Use num_nodes if provided, otherwise derive from scale factor
        self.num_nodes = config.get('num_nodes',
                                    max(10, int(self.base_scale_factor * self.scaling_factors.get('nodes', 2.0))))

        # Derived counts
        self.num_depots = max(1, int(self.base_scale_factor * self.scaling_factors.get('depots', 0.1)))
        self.num_customers = max(1, int(self.base_scale_factor * self.scaling_factors.get('customers', 1.5)))
        self.num_micro_hubs = max(0, int(self.base_scale_factor * self.scaling_factors.get('micro_hubs', 0.2)))
        self.num_trucks = max(1, int(self.base_scale_factor * self.scaling_factors.get('trucks', 0.1)))
        self.num_drones = max(0, int(self.base_scale_factor * self.scaling_factors.get('drones', 0.4)))
        self.num_initial_orders = max(1, int(self.base_scale_factor * self.scaling_factors.get('initial_orders', 1.0)))

        # Ensure we don't try to create more special nodes than total nodes
        total_special_nodes = self.num_depots + self.num_customers + self.num_micro_hubs
        if total_special_nodes > self.num_nodes:
            # Adjust counts proportionally or prioritize depots/customers if needed
            print(f"Warning: Requested special nodes ({total_special_nodes}) exceed total nodes ({self.num_nodes}). Adjusting...")
            scale_down = self.num_nodes / total_special_nodes
            self.num_customers = max(1, int(self.num_customers * scale_down))
            self.num_micro_hubs = int(self.num_micro_hubs * scale_down)
            self.num_depots = max(1, self.num_nodes - self.num_customers - self.num_micro_hubs) # Ensure depot gets priority

        # Distance Matrix
        self.distance_matrix = config.get('distance_matrix', None)
        self.max_travel_time = config.get('max_travel_time', 3600.0) # Max seconds between any two nodes if generating

        # Vehicle/Order parameters (similar to RandomDataGenerator)
        self.truck_payload_range = tuple(config.get('truck_payload_range', [3, 10]))
        self.drone_payload_range = tuple(config.get('drone_payload_range', [1, 2]))
        # Speed ranges might be less critical if times come directly from the matrix
        self.truck_speed_range = tuple(config.get('truck_speed_range', [40.0, 80.0]))
        self.drone_speed_range = tuple(config.get('drone_speed_range', [20.0, 40.0]))
        self.initial_fuel_range = tuple(config.get('initial_fuel_range', [80.0, 120.0]))
        self.initial_battery_range = tuple(config.get('initial_battery_range', [0.8, 1.0]))
        self.sla_min_seconds = config.get('sla_min_hours', 1.0) * 3600
        self.sla_max_seconds = config.get('sla_max_hours', 4.0) * 3600
        self.priority_distribution = config.get('priority_distribution', {1: 1.0})
        self.truck_fuel_consumption_rate = config.get('truck_fuel_consumption_rate', 0.1)
        self.drone_battery_drain_rate_flying = config.get('drone_battery_drain_rate_flying', 0.005)
        self.drone_battery_drain_rate_idle = config.get('drone_battery_drain_rate_idle', 0.001)
        self.drone_battery_charge_rate = config.get('drone_battery_charge_rate', 0.01)

        print(f"DistanceMatrixDataGenerator initialized.")
        print(f"  Node counts: Total={self.num_nodes}, Depots={self.num_depots}, Customers={self.num_customers}, MicroHubs={self.num_micro_hubs}")
        print(f"  Vehicle/Order counts: Trucks={self.num_trucks}, Drones={self.num_drones}, Orders={self.num_initial_orders}")
        if self.distance_matrix:
             print("  Using pre-defined distance matrix.")
        else:
             print("  Will generate a random distance matrix.")


    def _generate_random_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        """Generates a symmetrical distance matrix with random travel times."""
        matrix = {}
        node_ids = list(range(self.num_nodes))
        for i in node_ids:
            matrix[str(i)] = {}
            for j in node_ids:
                if i == j:
                    matrix[str(i)][str(j)] = 0.0
                elif str(j) in matrix and str(i) in matrix[str(j)]:
                    # Ensure symmetry
                    matrix[str(i)][str(j)] = matrix[str(j)][str(i)]
                else:
                    # Random travel time (in seconds)
                    matrix[str(i)][str(j)] = random.uniform(60.0, self.max_travel_time)
        return matrix

    def generate_data(self) -> Dict[str, Any]:
        """
        Generates initial simulation data including nodes, vehicles, orders,
        and a distance matrix.
        """
        print("DistanceMatrixDataGenerator: Generating data...")
        data = {
            "nodes": [],
            "edges": [], # No edges needed for matrix mode
            "trucks": [],
            "drones": [],
            "micro_hubs": [], # Still generate microhub entities, they are nodes
            "orders": [],
            "initial_time": 0.0,
            "distance_matrix": {}
        }

        all_node_ids = list(range(self.num_nodes))
        depot_ids = []
        customer_ids = []
        micro_hub_ids = []

        # --- Generate Nodes (without coordinates) ---
        node_ids_pool = list(all_node_ids)
        random.shuffle(node_ids_pool)

        # Generate Depots
        for _ in range(self.num_depots):
            if not node_ids_pool: break
            node_id = node_ids_pool.pop()
            data["nodes"].append({"id": node_id, "coords": [], "type": "depot", # Coords empty
                                  "is_loadable": True, "is_unloadable": True, "is_charging_station": True})
            depot_ids.append(node_id)

        # Generate Customers
        for _ in range(self.num_customers):
            if not node_ids_pool: break
            node_id = node_ids_pool.pop()
            data["nodes"].append({"id": node_id, "coords": [], "type": "customer", # Coords empty
                                  "is_loadable": False, "is_unloadable": True, "is_charging_station": False})
            customer_ids.append(node_id)

        # Generate Micro-hubs
        for _ in range(self.num_micro_hubs):
            if not node_ids_pool: break
            node_id = node_ids_pool.pop()
            num_slots = random.randint(1, 3)
            data["nodes"].append({"id": node_id, "coords": [], "type": "micro_hub", # Coords empty
                                  "is_loadable": True, "is_unloadable": True, "is_charging_station": True,
                                  "num_charging_slots": num_slots})
            micro_hub_ids.append(node_id)
            # Add microhub object separately if needed by older parts of code, though it's redundant
            # data["micro_hubs"].append({"id": node_id, "num_charging_slots": num_slots})


        # Generate generic nodes for the remainder
        for node_id in node_ids_pool:
            data["nodes"].append({"id": node_id, "coords": [], "type": "junction", # Coords empty
                                  "is_loadable": False, "is_unloadable": False, "is_charging_station": False})

        # Sort nodes by ID for consistency (optional)
        data["nodes"].sort(key=lambda x: x['id'])

        # --- Generate Distance Matrix ---
        if self.distance_matrix:
            # Validate provided matrix (basic check)
            if not all(str(i) in self.distance_matrix for i in all_node_ids):
                 raise ValueError("Provided distance_matrix is missing node IDs or has incorrect format.")
            data["distance_matrix"] = self.distance_matrix
            print("  Using provided distance matrix.")
        else:
            data["distance_matrix"] = self._generate_random_distance_matrix()
            print("  Generated random distance matrix.")


        # --- Generate Trucks ---
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
                 "max_speed": random.uniform(*self.truck_speed_range), # Speed less critical now
                 "initial_fuel": random.uniform(*self.initial_fuel_range),
                 "fuel_consumption_rate": self.truck_fuel_consumption_rate
             })

        # --- Generate Drones ---
        drone_start_nodes = []
        if depot_ids: drone_start_nodes.extend(depot_ids)
        if micro_hub_ids: drone_start_nodes.extend(micro_hub_ids)
        if not drone_start_nodes:
             print("Warning: No depots or micro-hubs. Drones will start at random node.")
             drone_start_nodes = all_node_ids

        for i in range(self.num_drones):
             start_node = random.choice(drone_start_nodes)
             data["drones"].append({
                 "id": 200 + i,
                 "start_node_id": start_node,
                 "max_payload_capacity": random.randint(*self.drone_payload_range),
                 "max_speed": random.uniform(*self.drone_speed_range), # Speed less critical now
                 "initial_battery": random.uniform(*self.initial_battery_range),
                 "battery_drain_rate_flying": self.drone_battery_drain_rate_flying,
                 "battery_drain_rate_idle": self.drone_battery_drain_rate_idle,
                 "battery_charge_rate": self.drone_battery_charge_rate
             })

        # --- Generate Initial Orders ---
        order_id_counter = 1000
        possible_pickup_nodes = depot_ids # Where orders might originate
        possible_delivery_nodes = customer_ids # Where orders might go

        if not possible_pickup_nodes or not possible_delivery_nodes:
            print("Warning: Not enough nodes (depots/customers/hubs) to generate orders.")
        else:
            for i in range(self.num_initial_orders):
                 # Ensure pickup and delivery nodes are different
                 pickup_node_id = random.choice(possible_pickup_nodes)
                 delivery_node_id = random.choice(possible_delivery_nodes)
                 # if not delivery_node_id: # Fallback if only one possible node type exists
                 #      delivery_node_id = pickup_node_id # Allow self-delivery if needed, though unlikely

                 time_received = 0.0
                 sla_deadline = time_received + random.uniform(self.sla_min_seconds, self.sla_max_seconds)

                 priorities, weights = zip(*self.priority_distribution.items())
                 priority = random.choices(priorities, weights=weights, k=1)[0]

                 data["orders"].append({
                     "id": order_id_counter + i,
                     "p_pickup_node_id": pickup_node_id,   # Assign pickup node
                     "p_delivery_node_id": delivery_node_id, # Assign delivery node
                     "time_received": time_received,
                     "SLA_deadline": sla_deadline,
                     "priority": priority
                 })

        print(f"DistanceMatrixDataGenerator: Generated {len(data['nodes'])} nodes, {len(data['trucks'])} trucks, "
              f"{len(data['drones'])} drones, {len(data['orders'])} orders.")
        return data