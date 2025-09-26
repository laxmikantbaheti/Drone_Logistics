# In file: ddls_src/scenarios/scenario.py

import numpy as np
from datetime import timedelta
from mlpro.bf.ml import Scenario
from mlpro.bf.ops import Mode
from ddls_src.core.basics import LogisticsAction
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.actions.base import SimulationActions
from agents.dummy_agent import DummyAgent
from ddls_src.functions.plotting import plot_vehicle_gantt_chart


class LogisticsScenario(Scenario):
    C_NAME = 'LogisticsScenario'

    def __init__(self, p_mode=Mode.C_MODE_SIM, p_cycle_limit=100, p_visualize: bool = False, p_logging=False,
                 **p_kwargs):
        self._config = p_kwargs.pop('config', {})
        self._system: LogisticsSystem = None
        self._logging = p_logging
        # Store the visualization flag
        self._visualize = p_visualize

        super().__init__(p_mode=p_mode, p_cycle_limit=p_cycle_limit, p_visualize=p_visualize, p_logging=p_logging,
                         **p_kwargs)

    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        self.log(self.C_LOG_TYPE_I, "Setting up scenario...")
        self._system = LogisticsSystem(p_id='logsys_001', p_visualize=p_visualize, p_logging=p_logging,
                                       config=self._config)

        # --- NEW: Setup visualization if enabled ---
        if self._visualize:
            self._system.network.setup_visualization()
            self._system.network.update_plot()
        # -----------------------------------------

        agent = DummyAgent(p_logging=p_logging)
        agent._action_space = self._system.get_action_space()
        agent._no_op_idx = self._system.action_map.get((SimulationActions.NO_OPERATION,))
        return agent

    # def _run_cycle(self):
    #     current_state = self._system.get_state()
    #     system_mask = self._system.get_current_mask()
    #     action = self._model.compute_action(p_state=current_state, p_action_mask=system_mask)
    #     action_tuple = self._system._reverse_action_map.get(action.get_sorted_values()[0])
    #     self.log(self.C_LOG_TYPE_I, f"Cycle {self.get_cycle_id()}: Agent chose action {action_tuple}")
    #     self._system.simulate_reaction(p_state=current_state, p_action=action)
    #
    #     # --- NEW: Update visualization at the end of the cycle ---
    #     if self._visualize:
    #         self._system.network.update_plot()
    #     # --------------------------------------------------------
    #
    #     new_state = self._system.get_state()
    #     return new_state.get_success(), new_state.get_broken(), False, new_state.get_terminal()

    # def _run_cycle(self):
    #     """
    #     Runs a single macro-cycle, with the decision loop now correctly using the agent-specific mask.
    #     """
    #     self.log(self.C_LOG_TYPE_I, f"--- Starting Macro-Cycle {self.get_cycle_id()} ---")
    #
    #     # 1. Decision Phase
    #     self.log(self.C_LOG_TYPE_I, "Entering Decision Phase...")
    #
    #     while not all(self._system.get_masks()):
    #         current_state = self._system.get_state()
    #         # --- MODIFIED: Get the mask for the agent ---
    #         agent_mask = self._system.get_agent_mask()
    #         # --------------------------------------------
    #         no_op_idx = self._model._no_op_idx
    #
    #         # Condition 1: Are there any valid actions left for the agent?
    #         agent_has_moves = np.any(np.delete(agent_mask, no_op_idx))
    #         if not agent_has_moves:
    #             self.log(self.C_LOG_TYPE_I, "No valid agent actions available. Ending Decision Phase.")
    #             break
    #
    #         # Agent selects an action based on its specific mask
    #         action = self._model.compute_action(p_state=current_state, p_action_mask=agent_mask)
    #         action_idx = action.get_sorted_values()[0]
    #
    #         # Condition 2: Did the agent choose NO_OPERATION?
    #         if action_idx == no_op_idx:
    #             self.log(self.C_LOG_TYPE_I, "Agent chose NO_OPERATION. Ending Decision Phase.")
    #             break
    #
    #         # Process the agent's chosen action
    #         self._system.process_action(action)
    #
    #         if self._visualize:
    #             self._system.network.update_plot()
    #
    #     # 2. Progression Phase
    #     self.log(self.C_LOG_TYPE_I, "Entering Progression Phase...")
    #     self._system.advance_time()
    #
    #     if self._visualize:
    #         self._system.network.update_plot()
    #
    #     new_state = self._system.get_state()
    #     return False, self._system.get_broken(), self._system.get_success(), False


    def _run_cycle(self):
        """
        Runs a single macro-cycle, with the decision loop now correctly using the agent-specific mask.
        """
        eof_data = False
        adapted = False
        self.log(self.C_LOG_TYPE_I, f"--- Starting Macro-Cycle {self.get_cycle_id()} ---")

        # 1. Decision Phase
        self.log(self.C_LOG_TYPE_I, "Entering Decision Phase...")

        while not all(self._system.get_masks()):
            if not len(self._system.get_automatic_actions()):

                current_state = self._system.get_state()
                # --- MODIFIED: Get the mask for the agent ---
                agent_mask = self._system.get_agent_mask()
                # --------------------------------------------
                no_op_idx = self._model._no_op_idx

                # Condition 1: Are there any valid actions left for the agent?
                agent_has_moves = np.any(np.delete(agent_mask, no_op_idx))
                if not agent_has_moves:
                    self.log(self.C_LOG_TYPE_I, "No valid agent actions available. Ending Decision Phase.")
                    break

                # Agent selects an action based on its specific mask
                action = self._model.compute_action(p_state=current_state, p_action_mask=agent_mask)
                action_idx = action.get_sorted_values()[0]

                # Condition 2: Did the agent choose NO_OPERATION?
                if action_idx == no_op_idx:
                    self.log(self.C_LOG_TYPE_I, "Agent chose NO_OPERATION. Ending Decision Phase.")
                    break

            # Process the agent's chosen action
                self._system.process_action(action)
            else:
                self._system.run_automatic_action_loop()

            if self._visualize:
                self._system.network.update_plot()

        # 2. Progression Phase
        self.log(self.C_LOG_TYPE_I, "Entering Progression Phase...")
        self._system.advance_time()

        if self._visualize:
            self._system.network.update_plot()

        if self._system.get_success():
            plot_vehicle_gantt_chart(self._system.global_state)

        new_state = self._system.get_state()
        return self._system.get_success(), self._system.get_broken(), adapted, eof_data


    def _reset(self, p_seed):
        self.log(self.C_LOG_TYPE_I, "Resetting scenario...")
        self._system.reset(p_seed=p_seed)

    def _cleanup(self):
        """Called by MLPro after the scenario finishes."""
        self.log(self.C_LOG_TYPE_I, "Cleaning up scenario...")
        if self._visualize:
            input("Simulation finished. Press Enter to close the plot...")
            self._system.network.close_plot()

    def get_latency(self) -> timedelta:
        latency = self._system.get_latency()
        if latency is not None:
            return latency
        else:
            return timedelta(0,0,0,0)