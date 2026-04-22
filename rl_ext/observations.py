# ddls_src/rl_extension/observations.py

import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod


class BaseObservations(ABC):
    """Abstract base class for observation variations."""

    @abstractmethod
    def get_observation(self, global_state) -> np.ndarray:
        """Translates GlobalState into a numerical array."""
        pass

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """Returns the Gymnasium space definition."""
        pass


class DefaultObservations(BaseObservations):
    """
    Standard observation variant including vehicle states, 
    routes, and pending order details.
    """

    def __init__(self, max_vehicles=11, max_route_lookahead=3, max_order_slots=20):
        self.max_vehicles = max_vehicles
        self.max_route_lookahead = max_route_lookahead
        self.max_order_slots = max_order_slots

        # Calculate size: 
        # Vehicles: (curr_node + rem_cap + lookahead) * max_vehicles
        # Orders: (pickup + delivery + size) * max_orders
        self._obs_size = ((2 + self.max_route_lookahead) * self.max_vehicles) + (3 * self.max_order_slots)

    def get_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=-1,
            high=np.inf,
            shape=(self._obs_size,),
            dtype=np.float32
        )

    def get_observation(self, global_state) -> np.ndarray:
        obs = []

        # --- 1. Encode Vehicles ---
        # Sorting ensures the agent sees vehicles in the same order every step
        all_vehicles = sorted(
            list(global_state.trucks.values()) + list(global_state.drones.values()),
            key=lambda v: v.get_id()
        )

        for v in all_vehicles:
            curr_node = v.current_node_id if v.current_node_id is not None else -1
            rem_cap = v.get_remaining_capacity()  #

            # Route lookahead (pickups then deliveries)
            route_nodes = [o.get_pickup_node_id() for o in v.get_pickup_orders()]
            route_nodes += [o.get_delivery_node_id() for o in v.get_delivery_orders()]  #

            # Pad or truncate route
            padded_route = route_nodes[:self.max_route_lookahead]
            padded_route += [-1] * (self.max_route_lookahead - len(padded_route))

            obs.extend([curr_node, rem_cap] + padded_route)

        # Pad missing vehicles
        for _ in range(len(all_vehicles), self.max_vehicles):
            obs.extend([-1, 0.0] + [-1] * self.max_route_lookahead)

        # --- 2. Encode Active Orders ---
        pending_orders = []
        order_requests = global_state.get_order_requests()  #
        for order_list in order_requests.values():
            pending_orders.extend(order_list)

        # Sort by deadline to keep observation consistent
        pending_orders.sort(key=lambda o: getattr(o, 'SLA_deadline', 0.0))

        orders_processed = 0
        for order in pending_orders[:self.max_order_slots]:
            obs.extend([
                order.get_pickup_node_id(),  #
                order.get_delivery_node_id(),
                order.size
            ])
            orders_processed += 1

        # Pad remaining order slots
        for _ in range(orders_processed, self.max_order_slots):
            obs.extend([-1, -1, 0.0])

        return np.array(obs, dtype=np.float32)