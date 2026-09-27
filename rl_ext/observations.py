import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from ddls_src.entities import Truck, Drone, MicroHub, Vehicle


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
    def __init__(self, num_vehicles=10, num_nodes=32, max_capacity=100.0, max_time=10000.0):
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


class ObservationSpaceActiveResource(BaseObservations):
    def __init__(self, num_vehicles=9, num_nodes=69, max_capacity=100.0, max_time=10000.0):
        self.num_vehicles = num_vehicles
        self.num_nodes = num_nodes

        # Scaling limits
        self.max_capacity = max_capacity
        self.max_time = max_time

    def get_observation_space(self, global_state: GlobalState) -> spaces.Space:
        # Dynamically check the length from the global state
        num_node_pairs = len(global_state.node_pairs)

        # Exact Size: 1 (Active Resource) + 1 (Current Cargo Size) + 1 (Last assigned node) + num of nodes
        self._obs_size = 3 + num_node_pairs

        low_bounds = [0.0]
        high_bounds = [len(global_state.trucks|global_state.drones|global_state.micro_hubs)]

        low_bounds.extend([0.0])
        high_bounds.extend([float(self.max_capacity)])

        low_bounds.extend([0.0])
        high_bounds.extend([num_node_pairs])

        # Node Pair Bounds: [Total Capacity Demand for specific O-D Pair]
        low_bounds.extend([0.0] * num_node_pairs)
        high_bounds.extend([self.max_capacity] * num_node_pairs)

        return spaces.Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            dtype=np.float32
        )

    def get_observation(self, global_state: GlobalState) -> np.ndarray:
        obs = []
        num_node_pairs = len(global_state.orders_by_nodes)

        # --- 1. ACTIVE RESOURCE ---
        active_resource = global_state.active_resource
        if active_resource is not None:
            active_resource_id = active_resource.get_id()
        else:
            active_resource_id = -1

        # --- 2. Current Cargo Size ---
        if isinstance(active_resource, MicroHub) or (active_resource is None):
            current_cargo_size = -1
        elif isinstance(active_resource, Vehicle):
            current_cargo_size = active_resource.get_committed_cargo_size()
        else:
            raise ValueError("The active resource shall either be a vehicle or a MicroHub.")

        # --- 3. Last assigned node ---
        if isinstance(active_resource, MicroHub) or (active_resource is None):
            last_assigned_node = -1
        elif isinstance(active_resource, Vehicle):
            delivery_nodes = active_resource.delivery_node_ids
            if len(delivery_nodes):
                last_assigned_node = delivery_nodes[-1]
            else:
                last_assigned_node = -1

        else:
            raise ValueError("The active resource shall either be a vehicle or a MicroHub.")

        # --- 3. Demand at each delivery node (including the micro_hub nodes) ---
        demands = [d[0] if len(d) else 0 for d in global_state.get_pending_demands(all_node_pairs=True).values()]

        obs.append(active_resource_id)
        obs.append(current_cargo_size)
        obs.append(last_assigned_node)
        obs.extend(demands)

        return np.array(obs, dtype=np.float32)


class ActiveResourceObservation(BaseObservations):

    def __init__(self, num_vehicles=7, num_nodes=53, max_capacity=100.0, max_time=10000.0):
        self.num_vehicles = num_vehicles
        self.num_nodes = num_nodes

        # Scaling limits
        self.max_capacity = max_capacity
        self.max_time = max_time

    def get_observation_space(self, global_state: GlobalState) -> spaces.Space:
        return spaces.Box(low= -1, high = self.num_vehicles)


    def get_observation(self, global_state):
        active_resource = global_state.active_resource
        if active_resource is None:
            return 0
        else:
            return int(active_resource.get_id())+1

class MaskState(BaseObservations):

    def __init__(self, num_trucks=7, num_drones=5, num_nodes=53, num_microhubs=2, max_capacity=100.0, max_time=10000.0):
        self.num_vehicles = num_trucks + num_microhubs
        self.num_nodes = num_nodes
        self.num_microhubs = num_microhubs
        # Scaling limits
        self.max_capacity = max_capacity
        self.max_time = max_time


    def get_observation_space(self, global_state: GlobalState) -> spaces.Space:

        # Dimensions = all masks
        # num_masks = num_actions
        # num_actions = ASSIGN_ACTIONS
        node_pairs = len(global_state.node_pairs)

        return spaces.Box(low= 0, high = 1, shape=(node_pairs,), dtype=np.float32)


    def get_observation(self, global_state):

        return global_state.agent_masks[:-1]
    


class BaseObservations(ABC):
    @abstractmethod
    def get_observation(self, global_state: GlobalState) -> np.ndarray:
        pass

    @abstractmethod
    def get_observation_space(self, global_state: GlobalState) -> spaces.Space:
        pass


class PerturbationAwareActiveResourceObservation(BaseObservations):
    """
    Observation space optimized for fixed infrastructure with stochastic operational variations:
    - Robust against package weight perturbations, variable order sets, and time window shifts.
    - Features are strictly normalized to [-1.0, 1.0] or [0.0, 1.0] for stable PPO training.
    """
    def __init__(
        self,
        num_vehicles: int = 12,
        num_nodes: int = 53,
        max_capacity: float = 100.0,
        max_time: float = 7200.0,      # e.g., 2-hour operational window in seconds
        max_deadline_slack: float = 3600.0
    ):
        self.num_vehicles = num_vehicles
        self.num_nodes = num_nodes
        self.max_capacity = max_capacity
        self.max_time = max_time
        self.max_deadline_slack = max_deadline_slack

    def get_observation_space(self, global_state: GlobalState) -> spaces.Box:
        num_node_pairs = len(global_state.node_pairs)

        # 1. Global state: [normalized_current_time] -> 1
        # 2. Active resource features: -> 6
        #    - resource_type: One-hot or discrete category (0: None, 1: Truck, 2: Drone, 3: MicroHub)
        #    - current_node_id: normalized [-1, 1]
        #    - normalized_remaining_capacity: [0, 1]
        #    - normalized_battery_soc: [0, 1] (1.0 for trucks/microhubs, actual SoC for drones)
        #    - is_vehicle: {0.0, 1.0}
        #    - is_drone: {0.0, 1.0}
        # 3. Dynamic Node-Pair features (3 features per pair): -> 3 * num_node_pairs
        #    - normalized_pending_demand_size: [0, 1]
        #    - normalized_earliest_time_to_deadline: [-1, 1] (-1: expired, 0..1: normalized slack)
        #    - pending_order_count: normalized [0, 1]
        
        self._total_size = 1 + 6 + (3 * num_node_pairs)

        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._total_size,),
            dtype=np.float32
        )

    def get_observation(self, global_state: GlobalState) -> np.ndarray:
        obs = []
        curr_time = float(global_state.current_time)

        # --- 1. GLOBAL TEMPORAL STATE ---
        norm_time = np.clip(curr_time / self.max_time, 0.0, 1.0) * 2.0 - 1.0  # mapped to [-1, 1]
        obs.append(norm_time)

        # --- 2. ACTIVE RESOURCE ENCODING ---
        active_res = global_state.active_resource

        if active_res is None:
            # [res_type, current_node, rem_cap, battery, is_vehicle, is_drone]
            obs.extend([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        elif isinstance(active_res, MicroHub):
            hub_node = getattr(active_res, 'node_id', -1)
            norm_hub_node = (float(hub_node) / self.num_nodes) * 2.0 - 1.0 if hub_node >= 0 else -1.0
            obs.extend([
                1.0,           # res_type: MicroHub
                norm_hub_node, # current_node
                1.0,           # MicroHub has infinite/unbounded buffer capacity
                1.0,           # Grid-connected battery
                0.0,           # is_vehicle
                0.0            # is_drone
            ])
        elif isinstance(active_res, Vehicle):
            # Check Vehicle Subtype
            is_drone = 1.0 if isinstance(active_res, Drone) else 0.0
            res_type = -0.33 if is_drone else 0.33

            # Location
            curr_node = getattr(active_res, 'current_node_id', -1)
            if curr_node is None or curr_node < 0:
                # Check last visited delivery node
                d_nodes = getattr(active_res, 'delivery_node_ids', [])
                curr_node = d_nodes[-1] if len(d_nodes) else -1
            norm_node = (float(curr_node) / self.num_nodes) * 2.0 - 1.0 if curr_node >= 0 else -1.0

            # Capacity
            rem_cap = float(active_res.get_remaining_capacity())
            norm_rem_cap = np.clip(rem_cap / self.max_capacity, 0.0, 1.0)

            # Battery State of Charge (SoC)
            if is_drone:
                # Retrieve battery level if available, fallback to 1.0
                try:
                    soc = float(active_res.get_state_value_by_dim_name("battery level")) / 100.0
                except Exception:
                    soc = getattr(active_res, 'battery_level', 1.0)
                norm_soc = np.clip(soc, 0.0, 1.0)
            else:
                norm_soc = 1.0  # Diesel/Electric Truck considered fully operational

            obs.extend([
                res_type,
                norm_node,
                norm_rem_cap,
                norm_soc,
                1.0,       # is_vehicle
                is_drone   # is_drone
            ])
        else:
            raise ValueError(f"Unsupported active resource type: {type(active_res)}")

        # --- 3. DYNAMIC DEMAND & TIME-WINDOW PERTURBATIONS PER NODE-PAIR ---
        node_pair_dict = global_state.orders_by_nodes
        sorted_pairs = sorted(global_state.node_pairs)

        for pair_key in sorted_pairs:
            orders = node_pair_dict.get(pair_key, [])

            total_demand = 0.0
            earliest_deadline_slack = self.max_deadline_slack
            active_count = 0

            for ord_obj in orders:
                # Filter active/uncompleted orders
                status = ord_obj.get_state_value_by_dim_name(ord_obj.C_DIM_DELIVERY_STATUS[0])
                if status != ord_obj.C_STATUS_DELIVERED:
                    active_count += 1
                    total_demand += float(ord_obj.size)

                    # Extract deadline slack if defined on the order object
                    deadline = getattr(ord_obj, 'delivery_deadline', None)
                    if deadline is not None:
                        slack = float(deadline) - curr_time
                        if slack < earliest_deadline_slack:
                            earliest_deadline_slack = slack

            # 3a. Normalized Demand Size
            norm_demand = np.clip(total_demand / (self.max_capacity * 2.0), 0.0, 1.0)

            # 3b. Normalized Deadline Urgency: [-1, 1]
            # -1.0 means overdue, 0.0 means deadline is right now, 1.0 means plenty of time
            norm_urgency = np.clip(earliest_deadline_slack / self.max_deadline_slack, -1.0, 1.0)

            # 3c. Normalized Order Density
            norm_count = np.clip(active_count / 10.0, 0.0, 1.0)

            obs.extend([norm_demand, norm_urgency, norm_count])

        return np.array(obs, dtype=np.float32)