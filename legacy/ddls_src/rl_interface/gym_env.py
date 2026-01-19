import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import your framework components
from ddls_src.core.basics import LogisticsAction
from ddls_src.actions.action_enums import SimulationAction


class LogisticsEnv(gym.Env):
    """
    A Gymnasium-compatible wrapper for the Drone Logistics System.

    State Space:
        - demand: (Num_Pairs, 2) Matrix [Count, Cost]
        - supply: (Num_Vehicles, 4) Matrix [Loc, Avail, Cap, Fuel]
        - action_mask: Binary vector for valid actions

    Action Space:
        - Discrete(ACTION_SPACE_SIZE) directly mapping to system actions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, system, render_mode=None):
        self.system = system
        self.render_mode = render_mode
        self.global_state = system.global_state

        # --- 1. Cache Static Structures ---
        self.node_pair_keys = list(self.global_state.node_pairs.keys())
        self.num_pairs = len(self.node_pair_keys)

        self.vehicles = sorted(
            list(self.global_state.trucks.values()) + list(self.global_state.drones.values()),
            key=lambda x: x.id
        )
        self.num_vehicles = len(self.vehicles)

        # --- 2. Define Action Space ---
        self.action_space_size = self.system.action_space_size
        self.action_space = spaces.Discrete(self.action_space_size)

        # --- 3. Define Observation Space ---
        self.observation_space = spaces.Dict({
            "demand": spaces.Box(low=0, high=np.inf, shape=(self.num_pairs, 2), dtype=np.float32),
            "supply": spaces.Box(low=-1, high=np.inf, shape=(self.num_vehicles, 4), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.system.reset(p_seed=seed)
        else:
            self.system.reset()

        self.vehicles = sorted(
            list(self.global_state.trucks.values()) + list(self.global_state.drones.values()),
            key=lambda x: x.id
        )
        self._last_delivered_count = 0

        return self._get_observation(), {}

    def step(self, action_idx):
        # 1. Translate Integer -> LogisticsAction
        action_obj = LogisticsAction(
            p_action_space=self.system.get_action_space(),
            p_values=[action_idx]
        )

        # 2. Check if action is NO_OPERATION
        action_tuple = self.system._reverse_action_map.get(action_idx)
        is_no_op = (action_tuple and action_tuple[0] == SimulationAction.NO_OPERATION)

        # 3. Execute Decision Phase
        self.system.process_action(action_obj)

        # 4. Execute Progression Phase (Time Advance)
        if is_no_op:
            self.system.advance_time()
            if self.render_mode == "human":
                self.system.network.update_plot()

        # 5. Get Reward & Next State
        obs = self._get_observation()
        reward = self._calculate_reward(is_no_op)
        terminated = self.system.get_success()
        truncated = self.system.get_broken()

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        # --- Part A: Demand Matrix ---
        demand_matrix = np.zeros((self.num_pairs, 2), dtype=np.float32)
        current_orders_map = self.global_state.get_order_requests()

        for i, pair in enumerate(self.node_pair_keys):
            orders = current_orders_map.get(pair, [])
            demand_matrix[i, 0] = len(orders)

            cost = self.system.network.get_travel_time(pair[0], pair[1])
            demand_matrix[i, 1] = cost if cost is not None else 0.0

        # --- Part B: Supply Matrix ---
        supply_matrix = np.zeros((self.num_vehicles, 4), dtype=np.float32)

        for i, vehicle in enumerate(self.vehicles):
            loc = vehicle.current_node_id
            supply_matrix[i, 0] = float(loc) if loc is not None else -1.0
            supply_matrix[i, 1] = 1.0 if vehicle.status == "idle" else 0.0
            supply_matrix[i, 2] = float(vehicle.max_payload_capacity - len(vehicle.cargo_manifest))

            if hasattr(vehicle, 'fuel_level'):
                supply_matrix[i, 3] = vehicle.fuel_level
            elif hasattr(vehicle, 'battery_level'):
                supply_matrix[i, 3] = vehicle.battery_level

        # --- Part C: Action Mask (FIXED) ---
        # We retrieve the list from the system and force it into a NumPy array here.
        raw_mask = self.system.get_current_mask()
        mask = np.array(raw_mask, dtype=np.int8)

        return {
            "demand": demand_matrix,
            "supply": supply_matrix,
            "action_mask": mask
        }

    def _calculate_reward(self, is_no_op):
        reward = 0.0

        if is_no_op:
            step_duration = self.system.get_latency().total_seconds()
            reward -= (0.1 * step_duration)

        state = self.system.get_state()
        delivered_dim_id = state.get_related_set().get_dim_by_name("delivered_orders").get_id()
        current_delivered = state.get_value(delivered_dim_id)

        if not hasattr(self, '_last_delivered_count'):
            self._last_delivered_count = 0

        new_deliveries = current_delivered - self._last_delivered_count
        if new_deliveries > 0:
            reward += (50.0 * new_deliveries)

        self._last_delivered_count = current_delivered

        return reward