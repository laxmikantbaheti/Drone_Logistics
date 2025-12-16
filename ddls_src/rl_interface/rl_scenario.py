import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any

# Local Imports from your uploaded files
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.core.basics import LogisticsAction
# from ddls_src.actions.action_enums import SimulationAction
from ddls_src.actions.base import SimulationActions
from ddls_src.functions.plotting import plot_vehicle_gantt_chart, plot_vehicle_states


class LogisticRLScenario(gym.Env):
    """
    A Gym environment wrapper for the LogisticsSystem.
    The 'step' method strictly implements the logic loop provided by the user.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, sim_config: Dict[str, Any], visualize: bool = False):
        super().__init__()

        self.visualize = visualize

        # 1. Initialize the internal LogisticsSystem
        self._system = LogisticsSystem(
            p_id='gym_env',
            p_visualize=visualize,
            config=sim_config
        )
        # self._system.initialize_simulation()

        # 2. Define Action Space
        self.action_space = spaces.Discrete(self._system.agent_action_space_size)

        # --- Dynamic Observation Space Setup (CRITICAL FIX) ---

        # A. Create the Node ID -> Index Mapping
        # This dictionary maps your simulation's specific Node IDs (e.g., 101, 102)
        # to sequential matrix indices (0, 1) required for the Numpy observation array.
        sorted_node_ids = sorted(self._system.global_state.nodes.keys())
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}
        self.num_nodes = len(sorted_node_ids)

        # B. Vehicle Counts
        self.num_trucks = len(self._system.global_state.trucks)
        self.num_drones = len(self._system.global_state.drones)

        # C. Feature Sizes
        self.truck_feat_size = 1  # [Location, Status, Cargo]
        self.drone_feat_size = 1  # [Location, Status, Cargo, Battery]

        # D. Calculate Total Observation Size
        self.obs_size = (self.num_nodes * self.num_nodes) + \
                        (self.num_trucks * self.truck_feat_size) + \
                        (self.num_drones * self.drone_feat_size)

        # 3. Define Observation Space
        self.observation_space = spaces.Box(
            low=-1.0,
            high=9999.0,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        # Cache NO_OPERATION index
        self._no_op_idx = self._system.action_map.get((SimulationActions.NO_OPERATION,))

        # Status Mapping for Encoding
        self.status_map = {
            "idle": 0,
            "en_route": 1,
            "loading": 2,
            "unloading": 3,
            "maintenance": 4,
            "charging": 5,
            "broken_down": 6,
            "halted": 7
        }
        self.truncate_counter = 0

    def reset(self, seed=None, options = None):
        """
        Resets the simulation. Note: This calls step() with a dummy action to
        fast-forward to the first decision point if the loop logic dictates it.
        """
        super().reset(seed=seed, options = options)
        self._system.reset(p_seed=seed)
        self.truncate_counter = 0

        # To ensure we start at a valid decision point, we can trigger the logic.
        # However, standard reset just returns the initial state.
        # If the initial state has no actions, the agent's first step will trigger the loop.

        if self.visualize:
            # self._system.network.setup_visualization()
            # self._system.network.update_plot()
            pass

        return self._get_observation(), self._get_info()

    def step(self, action_idx: int):
        """
        Strict implementation of the user-defined logic loop:
        1. Checks for available actions (Auto or Agent).
        2. If NONE -> Advances Time.
        3. If Auto -> Executes Auto.
        4. If Agent -> Executes 'action_idx' and RETURNS.
        """
        self.truncate_counter += 1
        while True:
            # Check termination
            if self._check_done()[0] or self._check_done()[1]:
                print(self._system.global_state.current_time)
                return self._get_observation(), self._calculate_reward(), self._check_done()[0], self._check_done()[1], self._get_info()

            # --- Check availability ---
            auto_actions = self._system.get_automatic_actions()
            agent_mask = self._get_agent_mask()

            # Check if there are any unmasked agent actions (excluding NO_OP usually,
            # but strictly checking mask here).
            # We assume NO_OP is handled by the time advance logic if it's the only option.
            valid_agent_indices = np.where(agent_mask)[0]
            # Filter out NO_OP from "available actions" count if we want the loop to handle waiting
            meaningful_agent_actions = [i for i in valid_agent_indices if i != self._no_op_idx]
            # meaningful_agent_actions.remove(436)
            has_auto = len(auto_actions) > 0
            has_agent = len(meaningful_agent_actions) > 0
            # "checks if there is any action available (automatic or non automatic),
            #  if not the the simulation is advanced in time"
            if not has_auto and not has_agent:
                self._system.advance_time()

                if self.visualize:
                    self._system.network.update_plot()

                # "after advancing time it again checks" -> Continue loop
                continue

            # "If automatic action available we execute the automatic action"
            if has_auto:
                # Execute the first one (standard queue processing)
                self._system.action_manager.execute_action(auto_actions[0])

                if self.visualize:
                    self._system.network.update_plot()

                # "We keep repeating this again and again" -> Continue loop
                continue

            # "Once there are no automatic actions available we check if there are any
            #  unmasked actions available for the agent to take."
            if has_agent:
                sys_action_id = self._system.agent_to_system_map[action_idx]
                # "If there are the agent takes action and executes in the environment"
                action_obj = LogisticsAction(
                    p_action_space=self._system.get_action_space(),
                    p_values=[sys_action_id]
                )
                self._system.process_action(action_obj)

                if self.visualize:
                    self._system.network.update_plot()

                # "and the step function returns"
                if self._check_done()[0] or self._check_done()[1]:
                    print(self._system.global_state.current_time)
                return self._get_observation(), self._calculate_reward(), self._check_done()[0], self._check_done()[1], self._get_info()

    # --- Helpers ---

    def _get_observation(self):
        """
        Constructs a flattened state vector representing:
        1. Node Pair Order Counts (Demand) - Using get_order_requests()
        2. Truck States (Supply)
        3. Drone States (Supply)
        """
        global_state = self._system.global_state

        # 1. Node Pair Matrix (Demand) - Size: N * N
        demand_matrix = np.zeros((len(self._system.global_state.nodes), len(self._system.global_state.nodes)),
                                 dtype=np.float64)

        # Use native method to get orders grouped by (pickup, delivery)
        # Note: This method filters for orders with status == C_STATUS_PLACED
        active_requests = global_state.get_order_requests()

        for (pickup_id, delivery_id), orders in active_requests.items():
            if pickup_id in self.node_id_to_idx and delivery_id in self.node_id_to_idx:
                p_idx = self.node_id_to_idx[pickup_id]
                d_idx = self.node_id_to_idx[delivery_id]
                demand_matrix[p_idx, d_idx] = float(len(orders))

        demand_vector = demand_matrix.flatten()

        # 2. Truck States - Size: T * 3
        truck_vectors = []
        for t_id in sorted(global_state.trucks.keys()):
            truck = global_state.trucks[t_id]

            # Feature 1: Location (Node Index)
            loc_idx = -1.0
            if truck.current_node_id is not None and truck.current_node_id in self.node_id_to_idx:
                loc_idx = float(self.node_id_to_idx[truck.current_node_id])

            # Feature 2: Status
            # status_code = float(self.status_map.get(truck.status.lower(), -1))
            #
            # # Feature 3: Current Cargo Count
            # cargo_count = float(len(truck.cargo_manifest))

            truck_vectors.extend([loc_idx])

        # 3. Drone States - Size: D * 4
        drone_vectors = []
        for d_id in sorted(global_state.drones.keys()):
            drone = global_state.drones[d_id]

            # Feature 1: Location
            loc_idx = -1.0
            if drone.current_node_id is not None and drone.current_node_id in self.node_id_to_idx:
                loc_idx = float(self.node_id_to_idx[drone.current_node_id])

            # Feature 2: Status
            # status_code = float(self.status_map.get(drone.status.lower(), -1))
            #
            # # Feature 3: Cargo Count
            # cargo_count = float(len(drone.cargo_manifest))

            # Feature 4: Battery Level
            # battery = float(drone.battery_level)


            drone_vectors.extend([loc_idx])

        # Combine into single observation vector
        obs = np.concatenate([
            demand_vector,
            np.array(truck_vectors, dtype=np.float64),
            np.array(drone_vectors, dtype=np.float64)
        ])
        # print(obs)
        return obs


    def _get_agent_mask(self):
        return self._system.get_agent_mask().astype(np.int8)

    def _calculate_reward(self):
        if self._check_done()[0]:
            return -self._system.global_state.current_time
        if self._check_done()[1]:
            return -5000
        else:
            return 0

    def _check_done(self):
        success = self._system.get_success()
        broken = True if self.truncate_counter > 500 else False
        if success and self.visualize:
            plot_vehicle_gantt_chart(self._system.global_state)
            plot_vehicle_states(self._system.global_state)
        if success or broken:
            print(True)
        return success, broken

    def _get_info(self):
        return {
            "action_mask": self._get_agent_mask(),
            "current_time": self._system.global_state.current_time
        }