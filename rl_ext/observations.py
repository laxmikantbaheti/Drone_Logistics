import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod


class BaseObservations(ABC):
    @abstractmethod
    def get_observation(self, global_state) -> np.ndarray:
        pass

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        pass


class DefaultObservations(BaseObservations):
    def __init__(self, max_vehicles=5, max_order_slots=20):
        self.max_vehicles = max_vehicles
        self.max_order_slots = max_order_slots

        # Mapping categorical strings to floats for RL
        self.trip_status_map = {"Idle": 0.0, "En Route": 1.0, "Halted": 2.0, "Loading": 3.0, "Unloading": 4.0}
        self.order_status_map = {"pending": 0.0, "assigned": 1.0, "En Route": 2.0, "Delivered": 3.0}

        # Size: Vehicle (6 features) + Order (4 features)
        self._obs_size = (6 * self.max_vehicles) + (4 * self.max_order_slots)

    def get_observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=np.inf, shape=(self._obs_size,), dtype=np.float32)

    def get_observation(self, global_state) -> np.ndarray:
        obs = []

        # --- 1. ENCODE VEHICLES ---
        all_vehicles = sorted(
            list(global_state.trucks.values()) + list(global_state.drones.values()),
            key=lambda v: v.get_id()
        )

        for v in all_vehicles[:self.max_vehicles]:
            # Physical Location (Direct strings from setup_spaces)
            obs.append(float(v.get_state_value_by_dim_name("loc x")))
            obs.append(float(v.get_state_value_by_dim_name("loc y")))

            # Trip Status using class attribute
            status_str = v.get_state_value_by_dim_name(v.C_DIM_TRIP_STATE[0])
            obs.append(self.trip_status_map.get(status_str, -1.0))

            # Availability and Node Boolean using class attributes
            obs.append(1.0 if v.get_state_value_by_dim_name(v.C_DIM_AVAILABLE[0]) else 0.0)
            obs.append(1.0 if v.get_state_value_by_dim_name(v.C_DIM_AT_NODE[0]) else 0.0)

            # Cargo Manifest size using class attribute
            obs.append(float(v.get_state_value_by_dim_name(v.C_DIM_CURRENT_CARGO[0])))

        # Pad remaining vehicle slots
        for _ in range(len(all_vehicles), self.max_vehicles):
            obs.extend([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])

        # --- 2. ENCODE ORDERS ---
        active_orders = [o for o in global_state.orders.values()
                         if o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) != o.C_STATUS_DELIVERED]

        for order in active_orders[:self.max_order_slots]:
            # Network Nodes using class attributes
            obs.append(float(order.get_state_value_by_dim_name(order.C_DIM_PICKUP_NODE[0])))
            obs.append(float(order.get_state_value_by_dim_name(order.C_DIM_DELIVERY_NODE[0])))

            # Weight/Size from generic Dimension
            obs.append(float(order.get_state_value_by_dim_name("w")))

            # Delivery Status using class attribute
            ord_status = order.get_state_value_by_dim_name(order.C_DIM_DELIVERY_STATUS[0])
            obs.append(self.order_status_map.get(ord_status, -1.0))

        # Pad remaining order slots
        for _ in range(len(active_orders), self.max_order_slots):
            obs.extend([-1.0, -1.0, 0.0, -1.0])

        return np.array(obs, dtype=np.float32)


import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from ddls_src.core.global_state import GlobalState


class BaseObservations(ABC):
    @abstractmethod
    def get_observation(self, global_state: GlobalState) -> np.ndarray:
        pass

    @abstractmethod
    def get_observation_space(self, global_state: GlobalState) -> spaces.Space:
        pass


class DemandCapacityObservations(BaseObservations):
    def __init__(self, num_vehicles=5, num_nodes=32, max_capacity=100.0, max_time=4000.0):
        self.num_vehicles = num_vehicles
        self.num_nodes = num_nodes

        # Scaling limits
        self.max_capacity = max_capacity
        self.max_time = max_time

    def get_observation_space(self, global_state: GlobalState) -> spaces.Box:
        # Dynamically check the length from the global state
        num_node_pairs = len(global_state.orders_by_nodes)

        # Exact Size: 1 (Time) + (Exact Vehicles * 2) + (Exact Node Pairs)
        self._obs_size = 1 + (self.num_vehicles * 2) + num_node_pairs

        low_bounds = [0.0]
        high_bounds = [self.max_time]

        # Vehicle Bounds: [Current Node ID, Remaining Cap]
        for _ in range(self.num_vehicles):
            low_bounds.extend([0.0, 0.0])
            high_bounds.extend([float(self.num_nodes), self.max_capacity])

        # Node Pair Bounds: [Total Capacity Demand for specific O-D Pair]
        low_bounds.extend([0.0] * num_node_pairs)
        high_bounds.extend([self.max_capacity * 5] * num_node_pairs)

        return spaces.Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            dtype=np.float32
        )

    def get_observation(self, global_state: GlobalState) -> np.ndarray:
        obs = []
        num_node_pairs = len(global_state.orders_by_nodes)

        # --- 1. GLOBAL STATE ---
        current_time = min(float(global_state.current_time), self.max_time)
        obs.append(current_time)

        # --- 2. VEHICLE STATE ---
        trucks = list(global_state.trucks.values())
        drones = list(global_state.drones.values())
        all_vehicles = sorted(trucks + drones, key=lambda v: v.get_id())

        assert len(
            all_vehicles) == self.num_vehicles, f"Expected exactly {self.num_vehicles} vehicles, got {len(all_vehicles)}"

        for v in all_vehicles:
            curr_node = getattr(v, 'current_node_id', -1)
            if curr_node is not None:
                obs.append(float(curr_node))
            else:
                obs.append(-1.0)
            obs.append(float(v.get_remaining_capacity()))

        # --- 3. NODE PAIR STATE (O-D DEMAND MATRIX) ---
        node_pair_dict = global_state.orders_by_nodes

        # Deterministically sort the pair tuples
        sorted_pairs = sorted(node_pair_dict.keys())

        for pair_key in sorted_pairs:
            orders_list = node_pair_dict[pair_key]

            total_pair_demand = 0.0
            for order in orders_list:
                # Filter out delivered orders so the agent sees demand drop over time
                if order.get_state_value_by_dim_name(order.C_DIM_DELIVERY_STATUS[0]) != order.C_STATUS_DELIVERED:
                    total_pair_demand += float(order.size)

            obs.append(total_pair_demand)

        return np.array(obs, dtype=np.float32)