import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple
import copy  # Added for deepcopying the logs

# Core simulation imports
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.core.basics import LogisticsAction
from rl_ext.observations import DefaultObservations, BaseObservations
from rl_ext.rewards import DefaultRewards, BaseRewards


class LogisticsEnv(gym.Env):
    def __init__(self, sim_config: Dict[str, Any], observation_handler: BaseObservations = None,
                 rewards_handler: BaseRewards = None, custom_log: bool = False):
        super().__init__()

        self._system = LogisticsSystem(
            config=sim_config,
            p_visualize=False,
            p_logging=False,
            custom_log=custom_log
        )

        if observation_handler is not None:
            self.obs_handler = observation_handler
        else:
            self.obs_handler = DefaultObservations()

        if rewards_handler is not None:
            self.rewards_handler = rewards_handler
        else:
            self.rewards_handler = DefaultRewards()

        self.action_space = spaces.Discrete(self._system.agent_action_space_size)
        self.observation_space = self.obs_handler.get_observation_space()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._system.reset(p_seed=seed)
        self.rewards_handler.reset(self._system)
        self._proceed_simulation()
        return self._get_obs(), self._get_info()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        sys_action_id = self._system.agent_to_system_map[int(action_idx)]
        action_obj = LogisticsAction(
            p_action_space=self._system.get_action_space(),
            p_values=[sys_action_id]
        )
        self._system.process_action(action_obj)
        self._proceed_simulation()

        observation = self._get_obs()

        # --- MODIFICATION: Reward is strictly Negative Makespan ---
        reward = self._calculate_reward()

        terminated = self._system.get_success() or self._system.get_broken()
        truncated = False
        info = self._get_info()

        # --- MODIFICATION: Checkpoint the logs before SB3 auto-resets ---
        if terminated:
            logger = self._system.global_state.event_logger
            # Store a snapshot of the logs in info so the callback can find them
            info["terminal_logs"] = copy.deepcopy(logger.logs)
            info["terminal_event_count"] = logger.recorded_events_count
            info["reward"] = self._calculate_reward()
            info["makespan"] = self._system.global_state.current_time

        return observation, reward, terminated, truncated, info

    def _proceed_simulation(self):
        while True:
            if len(self._system.get_automatic_actions()) > 0:
                self._system.run_automatic_action_loop()
                continue
            if self._system.get_success() or self._system.get_broken():
                break
            agent_mask = self._system.get_agent_mask()
            if np.any(agent_mask):
                break
            self._system.advance_time()

    def _get_obs(self) -> np.ndarray:
        return self.obs_handler.get_observation(self._system.global_state)

    def _get_info(self) -> dict:
        return {
            "is_success": self._system.get_success(),
            "is_broken": self._system.get_broken(),
            "agent_mask": self._system.get_agent_mask(),
            "current_time": self._system.global_state.current_time,
            "delivered_count": sum(
                1 for o in self._system.global_state.orders.values()
                if o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) == o.C_STATUS_DELIVERED
            )
        }

    def action_masks(self) -> np.ndarray:
        return self._system.get_agent_mask().astype(bool)

    def _calculate_reward(self) -> float:
        # Kept for compatibility but not used in step() anymore per your request
        if self._system.get_success():
            return -float(self._system.global_state.current_time)
        elif self._system.get_broken():
            return -float(1000000)
        else:
            return 0