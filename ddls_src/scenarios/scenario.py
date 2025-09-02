# In file: ddls_src/scenarios/scenario.py

import numpy as np
from datetime import timedelta
from mlpro.bf.ml import Scenario
from mlpro.bf.ops import Mode
from ddls_src.core.basics import LogisticsAction
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.actions.action_enums import SimulationAction
from agents.dummy_agent import DummyAgent


class LogisticsScenario(Scenario):
    C_NAME = 'LogisticsScenario'

    def __init__(self, p_mode=Mode.C_MODE_SIM, p_cycle_limit=100, p_visualize: bool = False, p_logging=True,
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
        agent._no_op_idx = self._system.action_map.get((SimulationAction.NO_OPERATION,))
        return agent

    def _run_cycle(self):
        current_state = self._system.get_state()
        system_mask = self._system.get_current_mask()
        action = self._model.compute_action(p_state=current_state, p_action_mask=system_mask)
        action_tuple = self._system._reverse_action_map.get(action.get_sorted_values()[0])
        self.log(self.C_LOG_TYPE_I, f"Cycle {self.get_cycle_id()}: Agent chose action {action_tuple}")
        self._system.simulate_reaction(p_state=current_state, p_action=action)

        # --- NEW: Update visualization at the end of the cycle ---
        if self._visualize:
            self._system.network.update_plot()
        # --------------------------------------------------------

        new_state = self._system.get_state()
        return False, new_state.get_broken(), new_state.get_success(), False

    def _reset(self, p_seed):
        self.log(self.C_LOG_TYPE_I, "Resetting scenario...")
        self._system.reset(p_seed=p_seed)

    def _cleanup(self):
        """Called by MLPro after the scenario finishes."""
        self.log(self.C_LOG_TYPE_I, "Cleaning up scenario...")
        if self._visualize:
            input("Simulation finished. Press Enter to close the plot...")
            self._system.network.close_plot()