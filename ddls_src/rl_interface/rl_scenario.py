import gym
from gym import spaces
import numpy as np
from typing import Dict, Any

# Local Imports from your uploaded files
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.core.basics import LogisticsAction
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.functions.plotting import plot_vehicle_gantt_chart, plot_vehicle_states


class LogisticRLScenario(gym.Env):
    """
    A Gym environment wrapper for the LogisticsSystem.
    The 'step' method strictly implements the logic loop provided by the user.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, sim_config: Dict[str, Any], visualize: bool = False):
        super(LogisticRLScenario, self).__init__()

        self.visualize = visualize

        # 1. Initialize the internal LogisticsSystem
        self._system = LogisticsSystem(
            p_id='gym_env',
            p_visualize=visualize,
            config=sim_config
        )
        # self._system.initialize_simulation()

        # 2. Define Action Space
        self.action_space = spaces.Discrete(self._system.action_space_size)

        # 3. Define Observation Space (Placeholder: [Total Orders, Delivered Orders])
        self.observation_space = spaces.Box(
            low=0,
            high=9999,
            shape=(2,),
            dtype=np.float32
        )

        # Cache NO_OPERATION index
        self._no_op_idx = self._system.action_map.get((SimulationAction.NO_OPERATION,))

    def reset(self, seed=None):
        """
        Resets the simulation. Note: This calls step() with a dummy action to
        fast-forward to the first decision point if the loop logic dictates it.
        """
        super().reset(seed=seed)
        self._system.reset(p_seed=seed)

        # To ensure we start at a valid decision point, we can trigger the logic.
        # However, standard reset just returns the initial state.
        # If the initial state has no actions, the agent's first step will trigger the loop.

        if self.visualize:
            # self._system.network.setup_visualization()
            # self._system.network.update_plot()
            pass

        return self._get_observation()

    def step(self, action_idx: int):
        """
        Strict implementation of the user-defined logic loop:
        1. Checks for available actions (Auto or Agent).
        2. If NONE -> Advances Time.
        3. If Auto -> Executes Auto.
        4. If Agent -> Executes 'action_idx' and RETURNS.
        """

        while True:
            # Check termination
            if self._check_done():
                return self._get_observation(), self._calculate_reward(), True, self._get_info()

            # --- Check availability ---
            auto_actions = self._system.get_automatic_actions()
            agent_mask = self._get_agent_mask()

            # Check if there are any unmasked agent actions (excluding NO_OP usually,
            # but strictly checking mask here).
            # We assume NO_OP is handled by the time advance logic if it's the only option.
            valid_agent_indices = np.where(agent_mask)[0]
            # Filter out NO_OP from "available actions" count if we want the loop to handle waiting
            meaningful_agent_actions = [i for i in valid_agent_indices if i != self._no_op_idx]

            has_auto = len(auto_actions) > 0
            has_agent = len(meaningful_agent_actions) > 1

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
                # "If there are the agent takes action and executes in the environment"
                action_obj = LogisticsAction(
                    p_action_space=self._system.get_action_space(),
                    p_values=[action_idx]
                )
                self._system.process_action(action_obj)

                if self.visualize:
                    self._system.network.update_plot()

                # "and the step function returns"
                return self._get_observation(), self._calculate_reward(), self._check_done(), self._get_info()

    # --- Helpers ---

    def _get_observation(self):
        mlpro_state = self._system.get_state()
        total = mlpro_state.get_value(mlpro_state.get_related_set().get_dim_by_name("total_orders").get_id())
        delivered = mlpro_state.get_value(mlpro_state.get_related_set().get_dim_by_name("delivered_orders").get_id())
        return np.array([total, delivered], dtype=np.float32)

    def _get_agent_mask(self):
        return self._system.get_agent_mask().astype(np.int8)

    def _calculate_reward(self):
        # Placeholder
        mlpro_state = self._system.get_state()
        return mlpro_state.get_value(mlpro_state.get_related_set().get_dim_by_name("delivered_orders").get_id())

    def _check_done(self):
        success = self._system.get_success()
        broken = self._system.get_broken()
        if success and self.visualize:
            plot_vehicle_gantt_chart(self._system.global_state)
            plot_vehicle_states(self._system.global_state)
        if success or broken:
            print(True)
        return success or broken

    def _get_info(self):
        return {
            "action_mask": self._get_agent_mask(),
            "current_time": self._system.global_state.current_time
        }