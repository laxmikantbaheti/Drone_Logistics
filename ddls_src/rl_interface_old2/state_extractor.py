import numpy as np


class StateExtractor:
    def __init__(self, max_vehicles=11, max_route_lookahead=3, max_order_slots=20):
        self.max_vehicles = max_vehicles
        self.max_route_lookahead = max_route_lookahead
        self.max_order_slots = max_order_slots

        # Calculate total size of the 1D observation array
        # Vehicle block: (Current Node + Rem Capacity + Lookahead nodes) * Max Vehicles
        # Order block: (Pickup Node + Delivery Node + Size) * Max Orders
        self.obs_size = ((2 + self.max_route_lookahead) * self.max_vehicles) + (3 * self.max_order_slots)

    def extract_observation(self, global_state) -> np.ndarray:
        """
        Translates the dynamic GlobalState into a fixed-size, 1D numerical array.
        """
        obs = []

        # --- 1. ENCODE VEHICLES ---
        # Sort to ensure consistent indexing across steps
        all_vehicles = sorted(
            list(global_state.trucks.values()) + list(global_state.drones.values()),
            key=lambda v: v.get_id()
        )

        for v in all_vehicles:
            curr_node = v.current_node_id if v.current_node_id is not None else -1
            rem_cap = v.get_remaining_capacity()

            # Extract Route (Lookahead)
            route_nodes = []
            for order in v.get_pickup_orders():
                route_nodes.append(order.get_pickup_node_id())
            for order in v.get_delivery_orders():
                route_nodes.append(order.get_delivery_node_id())

            # Pad or truncate the route
            padded_route = route_nodes[:self.max_route_lookahead]
            while len(padded_route) < self.max_route_lookahead:
                padded_route.append(-1)

            obs.extend([curr_node, rem_cap] + padded_route)

        # Pad missing vehicles if the fleet is smaller than max_vehicles
        vehicles_processed = len(all_vehicles)
        while vehicles_processed < self.max_vehicles:
            obs.extend([-1, 0.0] + [-1] * self.max_route_lookahead)
            vehicles_processed += 1

        # --- 2. ENCODE ACTIVE ORDERS ---
        order_requests = global_state.get_order_requests()
        orders_processed = 0

        pending_orders = []
        for pair_id, order_list in order_requests.items():
            for order in order_list:
                pending_orders.append(order)

        # Sort by SLA deadline to prioritize what the agent sees
        pending_orders.sort(key=lambda o: getattr(o, 'SLA_deadline', 0.0))

        for order in pending_orders:
            if orders_processed >= self.max_order_slots:
                break

            obs.extend([
                order.get_pickup_node_id(),
                order.get_delivery_node_id(),
                order.size
            ])
            orders_processed += 1

        # Pad remaining order slots
        while orders_processed < self.max_order_slots:
            obs.extend([-1, -1, 0.0])
            orders_processed += 1

        # --- 3. RETURN AS NUMPY ARRAY ---
        # Deep RL algorithms require float32
        return np.array(obs, dtype=np.float32)