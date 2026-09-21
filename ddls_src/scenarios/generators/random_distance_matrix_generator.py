import math
import random
from typing import Dict, Any, List, Tuple

from .data_generator import BaseDataGenerator  # Import the base class


class DistanceMatrixDataGenerator(BaseDataGenerator):
    """
    Generates simulation data with uniformly distributed node coordinates,
    micro-hub placement using K-Means clustering on customer locations,
    1:1 pairing between micro-hubs and drones, dual distance matrices (ground and air),
    and randomized order sizes satisfying:
      1. Each order size < minimum truck capacity.
      2. A portion of orders <= drone capacity (drone-eligible).
      3. Sum of all order sizes < sum of total truck capacities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DistanceMatrixDataGenerator.

        Args:
            config (Dict[str, Any]): Configuration for data generation.
        """
        super().__init__(config)

        self.base_scale_factor = config.get('base_scale_factor', 5)

        # Coordinate bounding box for uniform distribution
        self.area_x_range: Tuple[float, float] = tuple(config.get('area_x_range', (0.0, 100.0)))
        self.area_y_range: Tuple[float, float] = tuple(config.get('area_y_range', (0.0, 100.0)))

        # Scaling factors
        default_scaling_factors = {
            'nodes': 2.0,
            'depots': 0.1,
            'customers': 1.5,
            'micro_hubs': 0.2,
            'trucks': 0.1,
            'initial_orders': 1.0
        }
        self.scaling_factors = config.get('scaling_factors', default_scaling_factors)

        # Node counts
        self.num_nodes = config.get('num_nodes',
                                    max(10, int(self.base_scale_factor * self.scaling_factors.get('nodes', 2.0))))

        self.num_depots = max(1, int(self.base_scale_factor * self.scaling_factors.get('depots', 0.1)))
        self.num_customers = max(1, int(self.base_scale_factor * self.scaling_factors.get('customers', 1.5)))
        self.num_micro_hubs = max(0, int(self.base_scale_factor * self.scaling_factors.get('micro_hubs', 0.2)))
        self.num_trucks = max(1, int(self.base_scale_factor * self.scaling_factors.get('trucks', 0.1)))
        self.num_initial_orders = max(1, int(self.base_scale_factor * self.scaling_factors.get('initial_orders', 1.0)))

        # Constrain special nodes count to num_nodes
        total_special_nodes = self.num_depots + self.num_customers + self.num_micro_hubs
        if total_special_nodes > self.num_nodes:
            print(
                f"Warning: Requested special nodes ({total_special_nodes}) exceed total nodes ({self.num_nodes}). Adjusting...")
            scale_down = self.num_nodes / total_special_nodes
            self.num_customers = max(1, int(self.num_customers * scale_down))
            self.num_micro_hubs = int(self.num_micro_hubs * scale_down)
            self.num_depots = max(1, self.num_nodes - self.num_customers - self.num_micro_hubs)

        # 1:1 pairing: number of drones matches number of micro-hubs
        self.num_drones = self.num_micro_hubs

        # Vehicle and order configurations
        self.truck_payload_range = tuple(config.get('truck_payload_range', [3, 10]))
        self.drone_payload_range = tuple(config.get('drone_payload_range', [1, 2]))
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

        # Proportion of generated orders that must be drone-eligible
        self.drone_eligible_order_ratio = config.get('drone_eligible_order_ratio', 0.4)
        self.seed = config.get("seed", 20)
        if self.seed is not None:
            random.seed(self.seed)

        print(f"DistanceMatrixDataGenerator initialized.")
        print(
            f"  Node counts: Total={self.num_nodes}, Depots={self.num_depots}, Customers={self.num_customers}, MicroHubs={self.num_micro_hubs}")
        print(
            f"  Vehicle/Order counts: Trucks={self.num_trucks}, Drones={self.num_drones} (1:1 with MicroHubs), Orders={self.num_initial_orders}")

    def _sample_uniform_coords(self) -> Tuple[float, float]:
        """Samples uniform 2D coordinates within the configured bounding area."""
        x = random.uniform(self.area_x_range[0], self.area_x_range[1])
        y = random.uniform(self.area_y_range[0], self.area_y_range[1])
        return round(x, 2), round(y, 2)

    def _kmeans_centroids(self, points: List[Tuple[float, float]], k: int, max_iter: int = 50) -> List[
        Tuple[float, float]]:
        """
        K-Means clustering algorithm to compute k centroids from a list of 2D points.
        Used to position micro-hubs near customer clusters.
        """
        if not points or k <= 0:
            return []

        if len(points) <= k:
            centroids = list(points)
            while len(centroids) < k:
                centroids.append(self._sample_uniform_coords())
            return centroids

        centroids = random.sample(points, k)

        for _ in range(max_iter):
            clusters: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(k)}
            for px, py in points:
                closest_idx = min(
                    range(k),
                    key=lambda idx: math.hypot(px - centroids[idx][0], py - centroids[idx][1])
                )
                clusters[closest_idx].append((px, py))

            new_centroids = []
            shifted = False
            for i in range(k):
                cluster_pts = clusters[i]
                if cluster_pts:
                    mean_x = sum(p[0] for p in cluster_pts) / len(cluster_pts)
                    mean_y = sum(p[1] for p in cluster_pts) / len(cluster_pts)
                    new_centroid = (round(mean_x, 2), round(mean_y, 2))
                else:
                    new_centroid = self._sample_uniform_coords()

                if new_centroid != centroids[i]:
                    shifted = True
                new_centroids.append(new_centroid)

            centroids = new_centroids
            if not shifted:
                break

        return centroids

    def _compute_distance_matrices(self, nodes: List[Dict[str, Any]]) -> Tuple[
        Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Computes two distance matrices from node coordinates:
          1. ground_distance_matrix: Manhattan distance (|dx| + |dy|)
          2. air_distance_matrix: Euclidean distance (sqrt(dx^2 + dy^2))
        """
        ground_matrix: Dict[str, Dict[str, float]] = {}
        air_matrix: Dict[str, Dict[str, float]] = {}

        for n1 in nodes:
            id1_str = str(n1["id"])
            ground_matrix[id1_str] = {}
            air_matrix[id1_str] = {}
            x1, y1 = n1["coords"]

            for n2 in nodes:
                id2_str = str(n2["id"])
                x2, y2 = n2["coords"]

                if id1_str == id2_str:
                    ground_matrix[id1_str][id2_str] = 0.0
                    air_matrix[id1_str][id2_str] = 0.0
                else:
                    manhattan_dist = abs(x1 - x2) + abs(y1 - y2)
                    ground_matrix[id1_str][id2_str] = round(manhattan_dist, 2)

                    euclidean_dist = math.hypot(x1 - x2, y1 - y2)
                    air_matrix[id1_str][id2_str] = round(euclidean_dist, 2)

        return ground_matrix, air_matrix

    def _generate_order_sizes(self, num_orders: int, truck_capacities: List[int], drone_capacities: List[int]) -> List[
        int]:
        """
        Generates order sizes such that:
          1. Every order size < minimum truck capacity.
          2. A defined fraction of orders <= drone capacity (eligible for drone delivery).
          3. Sum of all order sizes < sum of all truck capacities.
        """
        min_truck_cap = min(truck_capacities) if truck_capacities else self.truck_payload_range[0]
        max_drone_cap = max(drone_capacities) if drone_capacities else self.drone_payload_range[1]
        total_truck_capacity = sum(truck_capacities) if truck_capacities else self.num_trucks * \
                                                                              self.truck_payload_range[0]

        # Upper bound: order size strictly less than min_truck_cap (at least 1)
        max_allowed_order_size = max(1, min_truck_cap - 1)

        # Number of orders guaranteed to be within drone capacity
        num_drone_eligible = max(1, int(num_orders * self.drone_eligible_order_ratio))

        sizes: List[int] = []
        for i in range(num_orders):
            if i < num_drone_eligible:
                # Strictly within drone limits
                drone_upper = min(max_drone_cap, max_allowed_order_size)
                size = random.randint(1, max(1, drone_upper))
            else:
                # General order size strictly less than truck capacity
                size = random.randint(1, max_allowed_order_size)
            sizes.append(size)

        # Enforce sum(order_sizes) < total_truck_capacity
        # Reserve a safety gap of at least 1 unit
        target_max_total = total_truck_capacity - 1

        if target_max_total < num_orders:
            # If total truck capacity is too small for 1 unit per order, clamp to 1
            sizes = [1] * num_orders
        elif sum(sizes) > target_max_total:
            # Proportional scale-down
            scale = target_max_total / sum(sizes)
            sizes = [max(1, int(s * scale)) for s in sizes]

            # Fine adjustment if sum still violates the strict inequality
            while sum(sizes) > target_max_total:
                reducible = [idx for idx, s in enumerate(sizes) if s > 1]
                if not reducible:
                    break
                idx_to_reduce = random.choice(reducible)
                sizes[idx_to_reduce] -= 1

        random.shuffle(sizes)
        return sizes

    def generate_data(self) -> Dict[str, Any]:
        """
        Generates initial simulation data with uniform coordinates,
        K-Means micro-hub placement, paired micro-hub/drone entities,
        dual distance matrices, and constrained randomized order sizes.
        """
        print("DistanceMatrixDataGenerator: Generating data...")
        data = {
            "nodes": [],
            "edges": [],
            "trucks": [],
            "drones": [],
            "micro_hubs": [],
            "orders": [],
            "initial_time": 0.0,
            "ground_distance_matrix": {},
            "air_distance_matrix": {}
        }

        all_node_ids = list(range(self.num_nodes))
        depot_ids = []
        customer_ids = []
        micro_hub_ids = []

        node_ids_pool = list(all_node_ids)
        random.shuffle(node_ids_pool)

        # 1. Generate Depots (Uniformly Distributed)
        for _ in range(self.num_depots):
            if not node_ids_pool: break
            node_id = node_ids_pool.pop()
            coords = list(self._sample_uniform_coords())
            data["nodes"].append({
                "id": node_id,
                "coords": coords,
                "type": "depot",
                "is_loadable": True,
                "is_unloadable": True,
                "is_charging_station": True
            })
            depot_ids.append(node_id)

        # 2. Generate Customers (Uniformly Distributed)
        customer_coords: List[Tuple[float, float]] = []
        for _ in range(self.num_customers):
            if not node_ids_pool: break
            node_id = node_ids_pool.pop()
            cx, cy = self._sample_uniform_coords()
            customer_coords.append((cx, cy))
            data["nodes"].append({
                "id": node_id,
                "coords": [cx, cy],
                "type": "customer",
                "is_loadable": False,
                "is_unloadable": True,
                "is_charging_station": False
            })
            customer_ids.append(node_id)

        # 3. Generate Micro-Hubs (Placed via K-Means & assigned unique drone ID)
        micro_hub_centroids = self._kmeans_centroids(customer_coords, self.num_micro_hubs)
        drone_id_base = 200
        micro_hub_to_drone_map: Dict[int, int] = {}

        for i in range(self.num_micro_hubs):
            if not node_ids_pool: break
            node_id = node_ids_pool.pop()
            coords = list(micro_hub_centroids[i]) if i < len(micro_hub_centroids) else list(
                self._sample_uniform_coords())
            num_slots = random.randint(1, 3)

            assigned_drone_id = drone_id_base + i
            micro_hub_to_drone_map[node_id] = assigned_drone_id

            data["nodes"].append({
                "id": node_id,
                "coords": coords,
                "type": "micro_hub",
                "is_loadable": True,
                "is_unloadable": True,
                "is_charging_station": True,
                "num_charging_slots": num_slots,
                "assigned_drone_id": assigned_drone_id
            })
            micro_hub_ids.append(node_id)

        # 4. Generate Remaining Junction Nodes (Uniformly Distributed)
        for node_id in node_ids_pool:
            coords = list(self._sample_uniform_coords())
            data["nodes"].append({
                "id": node_id,
                "coords": coords,
                "type": "junction",
                "is_loadable": False,
                "is_unloadable": False,
                "is_charging_station": False
            })

        data["nodes"].sort(key=lambda x: x['id'])

        # 5. Build Distance Matrices: Manhattan (Ground) & Euclidean (Air)
        ground_matrix, air_matrix = self._compute_distance_matrices(data["nodes"])
        data["ground_distance_matrix"] = ground_matrix
        data["air_distance_matrix"] = air_matrix

        # 6. Generate Trucks (Needed prior to Orders to obtain capacities)
        truck_start_nodes = depot_ids if depot_ids else all_node_ids
        truck_capacities: List[int] = []
        for i in range(self.num_trucks):
            start_node = random.choice(truck_start_nodes)
            payload_cap = random.randint(*self.truck_payload_range)
            truck_capacities.append(payload_cap)
            data["trucks"].append({
                "id": 100 + i,
                "start_node_id": start_node,
                "max_payload_capacity": payload_cap,
                "max_speed": random.uniform(*self.truck_speed_range),
                "initial_fuel": random.uniform(*self.initial_fuel_range),
                "fuel_consumption_rate": self.truck_fuel_consumption_rate
            })

        # 7. Generate Drones (1:1 Paired with Micro-Hubs)
        drone_capacities: List[int] = []
        for hub_node_id, drone_id in micro_hub_to_drone_map.items():
            payload_cap = random.randint(*self.drone_payload_range)
            drone_capacities.append(payload_cap)
            data["drones"].append({
                "id": drone_id,
                "start_node_id": hub_node_id,
                "max_payload_capacity": payload_cap,
                "max_speed": random.uniform(*self.drone_speed_range),
                "initial_battery": random.uniform(*self.initial_battery_range),
                "battery_drain_rate_flying": self.drone_battery_drain_rate_flying,
                "battery_drain_rate_idle": self.drone_battery_drain_rate_idle,
                "battery_charge_rate": self.drone_battery_charge_rate
            })

        # 8. Generate Initial Orders with Constrained Sizes
        order_id_counter = 1000
        possible_pickup_nodes = depot_ids
        possible_delivery_nodes = customer_ids

        if possible_pickup_nodes and possible_delivery_nodes:
            order_sizes = self._generate_order_sizes(
                num_orders=self.num_initial_orders,
                truck_capacities=truck_capacities,
                drone_capacities=drone_capacities
            )

            for i in range(self.num_initial_orders):
                pickup_node_id = random.choice(possible_pickup_nodes)
                delivery_node_id = random.choice(possible_delivery_nodes)
                time_received = 0.0
                sla_deadline = time_received + random.uniform(self.sla_min_seconds, self.sla_max_seconds)

                priorities, weights = zip(*self.priority_distribution.items())
                priority = random.choices(priorities, weights=weights, k=1)[0]

                data["orders"].append({
                    "id": order_id_counter + i,
                    "size": order_sizes[i],
                    "p_pickup_node_id": pickup_node_id,
                    "p_delivery_node_id": delivery_node_id,
                    "time_received": time_received,
                    "SLA_deadline": sla_deadline,
                    "priority": priority
                })

        print(f"DistanceMatrixDataGenerator: Generated {len(data['nodes'])} nodes, {len(data['trucks'])} trucks, "
              f"{len(data['drones'])} drones, {len(data['orders'])} orders.")
        return data