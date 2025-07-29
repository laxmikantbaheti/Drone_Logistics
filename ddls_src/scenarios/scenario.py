import numpy as np
from datetime import timedelta
from mlpro.bf.ml import Scenario
from mlpro.bf.ops import Mode
from ddls_src.core.basics import LogisticsAction
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.actions.action_enums import SimulationAction
from agents.dummy_agent import DummyAgent  # <-- Import our new dummy agent


class LogisticsScenario(Scenario):
    """
    A scenario class to run the LogisticsSystem with a DummyAgent.
    It now correctly inherits from mlpro.bf.ml.Scenario and manages a
    system (environment) and a model (agent).
    """
    C_NAME = 'LogisticsScenario'

    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_cycle_limit=100,
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        self._config = p_kwargs.pop('config', {})
        self._system: LogisticsSystem = None
        self._logging = p_logging

        # We now inherit from the ML Scenario, which expects a Model
        super().__init__(p_mode=p_mode,
                         p_cycle_limit=p_cycle_limit,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)

    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        """
        Instantiates the LogisticsSystem (environment) and the DummyAgent (model).
        The parent Scenario class will store the returned model in self._model.
        """
        self.log(self.C_LOG_TYPE_I, "Setting up scenario...")

        # 1. Setup the environment
        self._system = LogisticsSystem(p_id='logsys_001',
                                       p_visualize=p_visualize,
                                       p_logging=p_logging,
                                       config=self._config)

        # 2. Setup the agent
        agent = DummyAgent(p_logging=p_logging)

        # Provide the agent with the necessary spaces and info from the environment
        agent._action_space = self._system.get_action_space()
        agent._no_op_idx = self._system.action_map.get((SimulationAction.NO_OPERATION,))

        # 3. Return the agent to be stored as self._model
        return agent

    def get_latency(self) -> timedelta:
        """
        Returns the latency of the underlying system. This is a required method
        by the MLPro ScenarioBase.
        """
        return self._system.get_latency()

    def _reset(self, p_seed):
        """
        Resets the scenario and the underlying system.
        """
        self.log(self.C_LOG_TYPE_I, "Resetting scenario...")
        self._system.reset(p_seed=p_seed)

    def _run_cycle(self):
        """
        Runs a single cycle of the scenario: get action from model, simulate system reaction.
        """
        # 1. Get the current state and action mask from the system
        current_state = self._system.get_state()
        system_mask = self._system.get_current_mask()

        # 2. Get an action from our model (the DummyAgent)
        action = self._model.compute_action(p_state=current_state, p_action_mask=system_mask)
        action_tuple = self._system._reverse_action_map.get(action.get_sorted_values()[0])
        self.log(self.C_LOG_TYPE_I, f"Cycle {self.get_cycle_id()}: Agent chose action {action_tuple}")

        # 3. Pass the action to the system and simulate one step
        self._system.simulate_reaction(p_state=current_state, p_action=action)

        # 4. Get the new state and check for terminal conditions
        new_state = self._system.get_state()

        return False, new_state.get_broken(), new_state.get_success(), False
