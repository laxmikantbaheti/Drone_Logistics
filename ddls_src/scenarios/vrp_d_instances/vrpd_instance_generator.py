import math
import numpy as np
import os
import random
import re
from ddls_src.scenarios.generators.data_generator import BaseDataGenerator
from typing import Dict, Any, List, Tuple, Optional


class VRPDBenchmarkDataGenerator(BaseDataGenerator):
    """
    Data generator that loads a TSPLIB-like VRP-D .vrp benchmark instance and converts it into the
    simulation JSON-like structure.

    Config is passed as a DICT (not kwargs), consistent with BaseDataGenerator.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # ------------------------------------------------------------------
        # Required
        # ------------------------------------------------------------------
        self.instance_path = config.get("instance_path")
        if not self.instance_path:
            raise ValueError("VRPDBenchmarkDataGenerator: 'instance_path' must be provided.")

        # Resolve relative paths robustly relative to THIS file (not CWD)
        if not os.path.isabs(self.instance_path):
            here = os.path.dirname(os.path.abspath(__file__))
            self.instance_path = os.path.abspath(os.path.join(here, self.instance_path))

        if not os.path.isfile(self.instance_path):
            raise FileNotFoundError(f"VRPDBenchmarkDataGenerator: instance file not found: {self.instance_path}")

        # ------------------------------------------------------------------
        # Optional config
        # ------------------------------------------------------------------
        self.num_drones = int(config.get("num_drones", 0))
        if self.num_drones < 0:
            raise ValueError("'num_drones' must be >= 0")

        self.num_microhubs = int(config.get("num_microhubs", 0))
        self.bbox = tuple(config.get("bbox", (0.0, 0.0, 100.0, 100.0)))
        self.std_dev_scale = float(config.get("std_dev_scale", 4.0))

        self.drone_capacity_ratio = float(config.get("drone_capacity_ratio", 1.0))
        self.truck_speed = float(config.get("truck_speed", 1.0))
        self.drone_speed = float(config.get("drone_speed", 1.0))

        self.seed = config.get("seed")
        self.initial_time = float(config.get("initial_time", 0.0))
        self.include_index_maps = bool(config.get("include_index_maps", True))

        # Optional override for number of trucks
        self.num_trucks_override: Optional[int] = config.get("num_trucks")
        if self.num_trucks_override is not None:
            self.num_trucks_override = int(self.num_trucks_override)

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        print(f"VRPDBenchmarkDataGenerator initialized for instance: {self.instance_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_data(self) -> Dict[str, Any]:
        parsed = self._parse_tsplib_vrp(self.instance_path)

        capacity_Q = parsed["capacity"]

        truck_config = {
            "max_payload_capacity": int(capacity_Q),
            "max_speed": self.truck_speed,
        }
        drone_config = {
            "max_payload_capacity": int(max(1, round(capacity_Q * self.drone_capacity_ratio))),
            "max_speed": self.drone_speed,
        }

        # Number of trucks
        if self.num_trucks_override is not None:
            num_trucks = self.num_trucks_override
        else:
            num_trucks = self._infer_trucks_from_filename(self.instance_path) or 1

        # Nodes
        nodes_list, depot_id, _ = self._build_nodes(parsed)

        # Micro hubs
        micro_hubs_list = self._build_micro_hubs(nodes_list)

        # All nodes for matrices
        all_nodes_list = nodes_list + micro_hubs_list

        # Vehicles
        trucks_list = self._build_trucks(num_trucks, depot_id, truck_config)
        drones_list = self._build_drones(self.num_drones, depot_id, drone_config)

        # Orders
        orders_list = self._build_orders_from_demand(parsed, depot_id)

        # Distance matrices
        ground_dmt, air_dmt = self._build_distance_matrices(all_nodes_list)

        data: Dict[str, Any] = {
            "nodes": all_nodes_list,
            "edges": [],
            "trucks": trucks_list,
            "drones": drones_list,
            "micro_hubs": micro_hubs_list,
            "orders": orders_list,
            "initial_time": self.initial_time,
            "ground_distance_matrix": ground_dmt,
            "air_distance_matrix": air_dmt,
            "meta": {
                "instance_name": parsed.get("name", os.path.basename(self.instance_path)),
                "capacity_Q": capacity_Q,
                "num_trucks": num_trucks,
                "num_drones": self.num_drones,
                "num_microhubs": self.num_microhubs,
                "source_file": os.path.basename(self.instance_path),
            },
        }

        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_tsplib_vrp(self, path: str) -> Dict[str, Any]:
        name = None
        dimension = None
        capacity = None
        coords: Dict[int, Tuple[float, float]] = {}
        demands: Dict[int, int] = {}
        mode = None

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                u = line.upper()
                if u.startswith("NODE_COORD_SECTION"):
                    mode = "COORDS"
                    continue
                if u.startswith("DEMAND_SECTION"):
                    mode = "DEMANDS"
                    continue
                if u.startswith("DEPOT_SECTION"):
                    mode = "DEPOT"
                    continue
                if u.startswith("EOF"):
                    break

                if mode is None and ":" in line:
                    k, v = [x.strip() for x in line.split(":", 1)]
                    if k.upper() == "NAME":
                        name = v
                    elif k.upper() == "DIMENSION":
                        dimension = int(float(v))
                    elif k.upper() == "CAPACITY":
                        capacity = int(float(v))
                    continue

                if mode == "COORDS":
                    i, x, y = line.split()
                    coords[int(i)] = (float(x), float(y))
                elif mode == "DEMANDS":
                    i, d = line.split()
                    demands[int(i)] = int(float(d))

        if dimension is None or capacity is None:
            raise ValueError("DIMENSION or CAPACITY missing in VRP file.")

        demands.setdefault(1, 0)

        return {
            "name": name,
            "dimension": dimension,
            "capacity": capacity,
            "coords": coords,
            "demands": demands,
        }

    def _infer_trucks_from_filename(self, path: str) -> Optional[int]:
        m = re.search(r"-k(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else None

    def _build_nodes(self, parsed: Dict[str, Any]):
        nodes = []
        depot_id = 0

        nodes.append({
            "id": depot_id,
            "coords": list(parsed["coords"][1]),
            "type": "supplier",
            "demand": 0,
        })

        for tsplib_id in range(2, parsed["dimension"] + 1):
            nodes.append({
                "id": tsplib_id - 1,
                "coords": list(parsed["coords"][tsplib_id]),
                "type": "customer",
                "demand": parsed["demands"].get(tsplib_id, 0),
            })

        return nodes, depot_id, []

    def _build_micro_hubs(self, existing_nodes):
        if self.num_microhubs <= 0:
            return []

        min_x, min_y, max_x, max_y = self.bbox
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        sx, sy = (max_x - min_x) / self.std_dev_scale, (max_y - min_y) / self.std_dev_scale

        max_id = max(n["id"] for n in existing_nodes)
        hubs = []

        while len(hubs) < self.num_microhubs:
            x = np.random.normal(cx, sx)
            y = np.random.normal(cy, sy)
            if min_x <= x <= max_x and min_y <= y <= max_y:
                hubs.append({
                    "id": max_id + len(hubs) + 1,
                    "coords": [float(x), float(y)],
                    "type": "micro_hub",
                    "chargeable": random.choice([0, 1]),
                    "demand": 0,
                })

        return hubs

    def _build_trucks(self, n, depot_id, truck_config):
        return [{
            "id": 10000 + i,
            "max_payload_capacity": truck_config["max_payload_capacity"],
            "max_speed": truck_config["max_speed"],
            "start_node_id": depot_id,
        } for i in range(n)]

    def _build_drones(self, n, depot_id, drone_config):
        return [{
            "id": 20000 + i,
            "max_payload_capacity": drone_config["max_payload_capacity"],
            "max_speed": drone_config["max_speed"],
            "start_node_id": depot_id,
        } for i in range(n)]

    def _build_orders_from_demand(self, parsed, depot_id):
        orders = []
        oid = 1000
        for tsplib_id in range(2, parsed["dimension"] + 1):
            for _ in range(parsed["demands"].get(tsplib_id, 0)):
                orders.append({
                    "id": oid,
                    "p_pickup_node_id": depot_id,
                    "p_delivery_node_id": tsplib_id - 1,
                })
                oid += 1
        return orders

    def _build_distance_matrices(self, nodes):
        """
        Returns distances as nested dictionaries keyed by node ids:

        {
          "truck_manhattan": {from_id: {to_id: dist, ...}, ...},
          "drone_euclidean": {from_id: {to_id: dist, ...}, ...}
        }
        """
        node_ids = [n["id"] for n in nodes]
        coords = {n["id"]: (float(n["coords"][0]), float(n["coords"][1])) for n in nodes}

        truck: Dict[str, Dict[str, float]] = {}
        drone: Dict[str, Dict[str, float]] = {}

        for i in node_ids:
            x1, y1 = coords[i]
            truck[str(i)] = {}
            drone[str(i)] = {}
            for j in node_ids:
                x2, y2 = coords[j]
                dx = x2 - x1
                dy = y2 - y1
                truck[str(i)][str(j)] = abs(dx) + abs(dy)
                drone[str(i)][str(j)] = math.hypot(dx, dy)

        return truck, drone



# ======================================================================
# TESTING BLOCK
# ======================================================================
if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    INSTANCE_PATH = os.path.join(HERE, "VRP-D", "A-n32-k5-20.vrp")

    config = {
        "instance_path": INSTANCE_PATH,
        "num_drones": 6,
        "num_microhubs": 2,
        "bbox": (0, 0, 100, 100),
        "drone_capacity_ratio": 0.2,
        "seed": 42,
    }

    gen = VRPDBenchmarkDataGenerator(config)
    data = gen.generate_data()

    print("Nodes:", len(data["nodes"]))
    print("Trucks:", len(data["trucks"]))
    print("Drones:", len(data["drones"]))
    print("Orders:", len(data["orders"]))
    print("Truck matrix size:", len(data["distance_matrix"]["truck_manhattan"]))
    print("Drone matrix size:", len(data["distance_matrix"]["drone_euclidean"]))
