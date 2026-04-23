import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

# Core simulation imports
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.core.basics import LogisticsAction
from rl_ext.observations import DefaultObservations, BaseObservations
from rl_ext.rewards import DefaultRewards, BaseRewards

class LogisticsEnv(gym.Env):
    """
    Gymnasium environment wrapper for the LogisticsSystem simulation.
    Implements a macro-step logic that progresses time until an agent decision is required.
    """

    def __init__(self, sim_config: Dict[str, Any], observation_handler:BaseObservations = None, rewards_handler:BaseRewards = None, custom_log: bool = False):
        super().__init__()

        # 1. Initialize the underlying simulation engine
        # We pass the configuration and disable internal MLPro logging/visualization
        # to keep the RL loop clean.
        self._system = LogisticsSystem(
            config=sim_config,
            p_visualize=False,
            p_logging=False,
            custom_log=custom_log
        )

        # 2. Observation Handler
        if observation_handler is not None:
            self.obs_handler = observation_handler
        else:
            print("Taking default observation handler, since none provided")
            self.obs_handler = DefaultObservations()

        if rewards_handler is not None:
            self.rewards_handler = rewards_handler
        else:
            print("Taking default reward handler, since none provided")
            self.rewards_handler = DefaultRewards()

        # 3. Define Action Space
        # The size is determined by the number of non-automatic actions in the system.
        self.action_space = spaces.Discrete(self._system.agent_action_space_size)

        # 4. Define Observation Space
        self.observation_space = self.obs_handler.get_observation_space()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to a stable initial state."""
        super().reset(seed=seed)

        # Reset the simulation system with the provided seed
        self._system.reset(p_seed=seed)

        # Reset the stateful rewards, if any
        self.rewards_handler.reset(self._system)

        # Settle the system before the first step to resolve initial automatic logic
        self._proceed_simulation()

        return self._get_obs(), self._get_info()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one agent action and progresses the simulation until
        the next agent decision is unmasked or the episode ends.
        """
        # A. Execute the Agent's Action
        # Map the agent's discrete index to the global system action ID.
        sys_action_id = self._system.agent_to_system_map[int(action_idx)]
        action_obj = LogisticsAction(
            p_action_space=self._system.get_action_space(),
            p_values=[sys_action_id]
        )
        self._system.process_action(action_obj)

        # B. Macro-Step Resolution (The Discussion Logic)
        self._proceed_simulation()

        # C. Prepare returns
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._system.get_success() or self._system.get_broken()  #
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _proceed_simulation(self):
        """
        Internal loop that handles the priority of actions:
        Automatic Actions > Agent Decisions > Time Advancement.
        """
        while True:
            # 1. Resolve Automatic Actions first
            if len(self._system.get_automatic_actions()) > 0:
                self._system.run_automatic_action_loop()
                # Re-evaluate state immediately after automatic changes
                continue

            # 2. Check for episode termination before idling
            if self._system.get_success() or self._system.get_broken():
                break

            # 3. Check for Agent decisions
            agent_mask = self._system.get_agent_mask()
            if np.any(agent_mask):
                # Valid decisions are available; return control to agent.
                break

            # 4. Advance time if the system is stable and no decisions are possible
            self._system.advance_time()
            # The loop restarts to check for auto-actions (e.g. order generation)
            # triggered by the time jump.

    def _get_obs(self) -> np.ndarray:
        """Extracts numerical observation from the system state."""
        # This logic will eventually move to observations.py
        # state = self._system.get_state()
        # dims = state.get_related_set().get_dims()
        # return np.array([state.get_value(d.get_id()) for d in dims], dtype=np.float32)
        obs = self.obs_handler.get_observation(self._system.global_state)
        return obs

    def _get_info(self) -> dict:
        """
        Returns diagnostic information about the current simulation state.
        """
        # Directly query the system's termination status
        is_success = self._system.get_success()
        is_broken = self._system.get_broken()

        return {
            "is_success": is_success,
            "is_broken": is_broken,
            "agent_mask": self._system.get_agent_mask(),
            "current_time": self._system.global_state.current_time,
            "delivered_count": sum(
                1 for o in self._system.global_state.orders.values()
                if o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) == o.C_STATUS_DELIVERED
            )
        }

    def action_masks(self) -> np.ndarray:
        """
        Standard interface for MaskablePPO.
        Returns a boolean array where True = Valid, False = Invalid.
        """
        # We query the same system logic used in your _proceed_simulation loop
        return self._system.get_agent_mask().astype(bool)

    def _calculate_reward(self) -> float:
        """Computes the reward signal. To be expanded in rewards.py."""
        reward = self.rewards_handler.get_reward(self._system)
        return reward