import json
import math
import numpy as np
import random
from rasterio.rio.helpers import coords


class RandomInstanceGenerator:
    def __init__(self,
                 bbox,
                 num_customers,
                 num_suppliers,
                 num_microhubs,
                 num_orders,
                 num_trucks,
                 num_drones,
                 drone_config,
                 truck_config,
                 std_dev_scale = 4):

        self.bbox = bbox
        self.num_customers = num_customers
        self.num_suppliers = num_suppliers
        self.num_orders = num_orders
        self.num_trucks = num_trucks
        self.num_drones = num_drones
        self.num_vehicles = self.num_trucks + self.num_drones
        self.num_microhubs = num_microhubs
        self.drone_config = drone_config
        self.truck_config = truck_config
        self.std_dev_scale = std_dev_scale
        self.customer_locations = self.get_customer_locations()
        self.supplier_locations = self.get_supplier_locations()
        self.microhub_locations = self.get_microhub_locations()
        self.compile_customers()
        self.compile_supplier()
        self.generate_micro_hubs()
        self.compile_nodes()
        self.generate_vehicles()

    def get_customer_locations(self):
        """
        Generates N random points normally distributed within a bounding box.

        Uses rejection sampling to ensure all points stay strictly inside the box.

        Parameters:
        - num_points: Number of coordinates to generate.
        - bbox: Tuple or list (min_x, min_y, max_x, max_y).
        - std_dev_scale: Determines how 'spread out' the points are.
                         Higher number = tighter cluster in center.
                         4 means the width covers roughly 4 standard deviations (approx 95% of points).
        """
        min_x, min_y, max_x, max_y = self.bbox

        # 1. Calculate the center (mean of distribution)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # 2. Calculate Standard Deviation based on box size
        # We divide the width/height by the scale.
        sigma_x = (max_x - min_x) / self.std_dev_scale
        sigma_y = (max_y - min_y) / self.std_dev_scale

        valid_points = []

        # 3. Generate points using Rejection Sampling
        # We generate batches until we fill the quota to ensure strict bounds
        while len(valid_points) < self.num_customers:
            # Generate a batch (estimated size needed)
            needed = self.num_customers - len(valid_points)
            # We generate slightly more (1.5x) to account for rejections
            batch_size = int(needed * 1.5)

            x = np.random.normal(loc=center_x, scale=sigma_x, size=batch_size)
            y = np.random.normal(loc=center_y, scale=sigma_y, size=batch_size)

            # Filter: Keep only points strictly inside the bounding box
            mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

            # Combine valid x and y
            batch_valid = np.column_stack((x[mask], y[mask]))

            valid_points.extend(batch_valid.tolist())

        # Trim to exact number requested and convert to array
        return np.array(valid_points[:self.num_customers])

    def get_supplier_locations(self):
        """
        Generates N random points normally distributed within a bounding box.
        Uses rejection sampling to ensure all points stay strictly inside the box.

        Parameters:
        - num_points: Number of coordinates to generate.
        - bbox: Tuple or list (min_x, min_y, max_x, max_y).
        - std_dev_scale: Determines how 'spread out' the points are.
                         Higher number = tighter cluster in center.
                         4 means the width covers roughly 4 standard deviations (approx 95% of points).
        """
        min_x, min_y, max_x, max_y = self.bbox

        # 1. Calculate the center (mean of distribution)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # 2. Calculate Standard Deviation based on box size
        # We divide the width/height by the scale.
        sigma_x = (max_x - min_x) / self.std_dev_scale
        sigma_y = (max_y - min_y) / self.std_dev_scale

        valid_points = []

        # 3. Generate points using Rejection Sampling
        # We generate batches until we fill the quota to ensure strict bounds
        while len(valid_points) < self.num_suppliers:
            # Generate a batch (estimated size needed)
            needed = self.num_suppliers - len(valid_points)
            # We generate slightly more (1.5x) to account for rejections
            batch_size = int(needed * 1.5)

            x = np.random.normal(loc=center_x, scale=sigma_x, size=batch_size)
            y = np.random.normal(loc=center_y, scale=sigma_y, size=batch_size)

            # Filter: Keep only points strictly inside the bounding box
            mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

            # Combine valid x and y
            batch_valid = np.column_stack((x[mask], y[mask]))

            valid_points.extend(batch_valid.tolist())

        # Trim to exact number requested and convert to array
        return np.array(valid_points[:self.num_suppliers])

    def compile_customers(self):

        self.customers = {}

        for i in range(self.num_customers):
            id = i+self.num_suppliers
            coords = self.customer_locations[i]
            type = "customer"
            self.customers[id] = {"id":id, "coords":coords, "type":type}

    def compile_supplier(self):

        self.suppliers = {}

        for i in range(self.num_suppliers):
            id = i
            coords = self.supplier_locations[i]
            type = "supplier"
            self.suppliers[id] = {"id":id, "coords":coords, "type":type}
        pass

    def compile_nodes(self):
        self.nodes = {**self.customers, **self.suppliers, **self.microhub_locations}

    def generate_orders(self):

        self.orders = {}

        for i in range(self.num_orders):

            idx = 1000 + i
            delivery_node_id = random.choice(list(self.customers.keys()))
            pickup_node_id = random.choice(list(self.suppliers.keys()))
            self.orders[i] = {"id": idx, "p_delivery_node_id":delivery_node_id, "p_pickup_node_id": pickup_node_id}
        return

    def generate_vehicles(self):
        self.trucks = {}
        self.drones = {}
        for i in range(self.num_trucks):
            idx = 10000 + i
            self.trucks[i] = {"id":idx,
                              "max_payload_capacity": self.truck_config['max_payload_capacity'],
                              "max_speed":self.truck_config["max_speed"],
                              "start_node_id": random.choice(list(self.suppliers.keys()))}

        drone_locations = [hub["id"] for hub in self.microhubs.values()]
        for i in range(self.num_drones):
            idx = 1000 + self.num_trucks + i
            if i>len(drone_locations-1):
                start_node_id = drone_locations[i]
            else:
                start_node_id = random.choice(list(self.customers.keys()))
            self.drones[i] = {"id":idx,
                              "max_payload_capacity": self.drone_config['max_payload_capacity'],
                              "max_speed":self.drone_config["max_speed"],
                              "start_node_id": random.choice(self.suppliers.keys())}

        return

    def generate_network(self):
        return self.generate_distance_matrices()


    def get_microhub_locations(self, **kwargs):
        """
        Generates N random points normally distributed within a bounding box.
        Uses rejection sampling to ensure all points stay strictly inside the box.

        Parameters:
        - num_points: Number of coordinates to generate.
        - bbox: Tuple or list (min_x, min_y, max_x, max_y).
        - std_dev_scale: Determines how 'spread out' the points are.
                         Higher number = tighter cluster in center.
                         4 means the width covers roughly 4 standard deviations (approx 95% of points).
        """
        if not self.num_microhubs:
            return []
        min_x, min_y, max_x, max_y = self.bbox

        # 1. Calculate the center (mean of distribution)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # 2. Calculate Standard Deviation based on box size
        # We divide the width/height by the scale.
        sigma_x = (max_x - min_x) / self.std_dev_scale
        sigma_y = (max_y - min_y) / self.std_dev_scale

        valid_points = []

        # 3. Generate points using Rejection Sampling
        # We generate batches until we fill the quota to ensure strict bounds
        while len(valid_points) < self.num_microhubs:
            # Generate a batch (estimated size needed)
            needed = self.num_microhubs - len(valid_points)
            # We generate slightly more (1.5x) to account for rejections
            batch_size = int(needed * 1.5)

            x = np.random.normal(loc=center_x, scale=sigma_x, size=batch_size)
            y = np.random.normal(loc=center_y, scale=sigma_y, size=batch_size)

            # Filter: Keep only points strictly inside the bounding box
            mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

            # Combine valid x and y
            batch_valid = np.column_stack((x[mask], y[mask]))

            valid_points.extend(batch_valid.tolist())

        # Trim to exact number requested and convert to array
        return np.array(valid_points[:self.num_microhubs])


    def generate_micro_hubs(self):

        self.microhubs = {}

        for i in range(self.num_microhubs):
            idx = self.num_customers+self.num_suppliers+i
            chargeable = random.choice([0,1])
            type = "micro_hub"
            self.microhubs[i] = {"id":idx, "coords": self.microhub_locations[i], "chargeable":chargeable, "type":type}

    def generate_distance_matrices(self):
        """
        Generates:
        1. Euclidean distance matrix
        2. Truck distance matrix (Manhattan distance)

        Parameters
        ----------
        coords : list of tuples
            [(x1, y1), (x2, y2), ...]

        Returns
        -------
        euclidean_matrix : numpy.ndarray
        truck_matrix : numpy.ndarray
        """
        n = len(self.nodes)
        coords = [node.coords for node in self.nodes]
        self.drones_distance_matrix = np.zeros((n, n))
        self.truck_distance_matrix = np.zeros((n, n))

        for i in range(n):
            x1, y1 = coords[i]
            for j in range(n):
                x2, y2 = coords[j]

                # Euclidean (straight-line)
                self.drones_distance_matrix[i, j] = math.sqrt(
                    (x2 - x1) ** 2 + (y2 - y1) ** 2
                )

                # Truck distance (Manhattan / grid-based)
                self.truck_distance_matrix[i, j] = abs(x2 - x1) + abs(y2 - y1)

        return self.drones_distance_matrix, self.truck_distance_matrix

    def get_instance_data(self):
        expected_keys = ["nodes", "edges", "trucks", "drones", "micro_hubs", "orders"]
        instance = {"nodes": self.nodes,
                    "edges": [],
                    "trucks": self.trucks,
                    "drones": self.drones,
                    "micro_hubs": self.microhubs,
                    "orders": self.orders}
        return instance