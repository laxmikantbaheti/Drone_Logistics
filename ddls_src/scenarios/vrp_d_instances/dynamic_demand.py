import copy
import math
import random
from typing import Dict, Any
from ddls_src.scenarios.vrp_d_instances.vrpd_instance_generator import VRPDBenchmarkDataGenerator
# Assuming VRPDBenchmarkDataGenerator is imported or defined above
# from ddls_src.scenarios.vrp_d_instances.vrpd_instance_generator import VRPDBenchmarkDataGenerator

class DynamicVRPDBenchmarkDataGenerator(VRPDBenchmarkDataGenerator):
    """
    Dynamic version of the VRPDBenchmarkDataGenerator using direct inheritance.
    Applies capacity-safe perturbations to the demands on every reset().
    """

    def __init__(self, config: Dict[str, Any]):
        # 1. Initialize the parent generator with the standard config
        super().__init__(config)

        # 2. Extract dynamic-specific settings
        self.demand_variance = float(config.get("demand_variance", 0.2))

        # 3. Generate the static baseline exactly once using the parent's logic
        self.baseline_data = super().generate_data()

        # 4. Extract constraints from the generated metadata to cap the perturbation
        meta = self.baseline_data.get("meta", {})
        truck_capacity = meta.get("capacity_Q", 0)
        num_trucks = meta.get("num_trucks", 1)
        self.dynamic = True

        self.total_fleet_capacity = truck_capacity * num_trucks

    def generate_data(self) -> Dict[str, Any]:
        """
        Overrides the base method. Whenever data is requested, it returns a freshly
        perturbed episode rather than rebuilding the static benchmark from the file.
        """
        return self.reset()

    def reset(self) -> Dict[str, Any]:
        """
        Creates a fresh episode payload by applying uniform noise to the baseline
        order sizes, scaling them if they exceed fleet capacity, and syncing the nodes.
        """
        self.episode_data = copy.deepcopy(self.baseline_data)

        new_total_demand = 0
        perturbed_sizes = {}

        # 5. Apply perturbation to each order's size based on the baseline
        for order in self.episode_data["orders"]:
            base_size = order["size"]
            if base_size > 0:
                noise = random.uniform(1.0 - self.demand_variance, 1.0 + self.demand_variance)
                new_size = max(1, int(round(base_size * noise)))
                perturbed_sizes[order["id"]] = new_size
                new_total_demand += new_size
            else:
                perturbed_sizes[order["id"]] = 0

        # 6. Calculate scale factor if the new total demand exceeds fleet capacity
        scale_factor = 1.0
        if new_total_demand > self.total_fleet_capacity:
            scale_factor = self.total_fleet_capacity / new_total_demand

        # 7. Apply the final sizes and synchronize the orders with the nodes
        for order in self.episode_data["orders"]:
            oid = order["id"]
            if perturbed_sizes[oid] > 0:
                # Floor the scaled demand but ensure it doesn't drop below 1
                final_size = max(1, int(math.floor(perturbed_sizes[oid] * scale_factor)))
            else:
                final_size = 0

            order["size"] = float(final_size)

            # Sync the corresponding delivery node's demand attribute
            delivery_node_id = order["p_delivery_node_id"]
            for node in self.episode_data["nodes"]:
                if node["id"] == delivery_node_id:
                    node["demand"] = final_size
                    break

        return self.episode_data