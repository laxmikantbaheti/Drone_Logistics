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
    def __init__(self, max_vehicles=10, max_order_slots=20):
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