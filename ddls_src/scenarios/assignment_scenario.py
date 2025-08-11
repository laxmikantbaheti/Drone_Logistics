import numpy as np
from datetime import timedelta
from mlpro.bf.ml import Scenario
from mlpro.bf.ops import Mode
from ddls_src.core.basics import LogisticsAction
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.actions.action_enums import SimulationAction
from agents.assignment_only.dummy_agent import AssignmentAgent
from ddls_src.actions.base import SimulationAction, Constraint, OrderAssignableConstraint, VehicleAvailableConstraint, \
    VehicleCapacityConstraint, HubIsActiveConstraint
# Imports for the specific research design
from agents.assignment_only.action_manager import ActionManager as AssignmentActionManager
from agents.assignment_only.agent_action_space import create_agent_action_space
from agents.assignment_only.agent_masker import AgentMasker  # <-- Import the new AgentMasker


class AssignmentScenario(Scenario):
    """
    A scenario class that correctly implements the "assignment-only" research design.
    It now uses the dedicated AgentMasker for this specific experiment.
    """
    C_NAME = 'AssignmentScenario'

    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_cycle_limit=100,
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        self._config = p_kwargs.pop('config', {})
        self._system: LogisticsSystem = None
        self._logging = p_logging
        self._translator_am = None
        self._agent_masker = None  # <-- To hold our new AgentMasker

        super().__init__(p_mode=p_mode,
                         p_cycle_limit=p_cycle_limit,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)

    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        """
        Instantiates the environment, the assignment-only agent, and all related components.
        """
        self.log(self.C_LOG_TYPE_I, "Setting up scenario...")

        # 1. Setup the environment
        self._system = LogisticsSystem(p_id='logsys_001',
                                       p_visualize=p_visualize,
                                       p_logging=p_logging,
                                       config=self._config)

        # 2. Setup the translator ActionManager for this research design
        num_orders = len(self._system.global_state.orders)
        num_vehicles = len(self._system.global_state.trucks) + len(self._system.global_state.drones)
        self._translator_am = AssignmentActionManager(global_state=self._system.global_state,
                                                      supply_chain_manager=self._system.supply_chain_manager,
                                                      num_orders=num_orders,
                                                      num_vehicles=num_vehicles)

        # 3. Setup the agent-specific ActionMasker
        agent_config = {
            'num_orders': num_orders,
            'num_vehicles': num_vehicles,
            'vehicle_map': self._translator_am._vehicle_idx_to_id
        }
        self._agent_masker = AgentMasker(self._system.global_state, self._system.action_map, agent_config)

        # 4. Setup the agent
        agent = AssignmentAgent(p_logging=p_logging)

        # 5. Provide the agent with its simplified action space
        agent._action_space = create_agent_action_space(num_orders, num_vehicles)
        agent._no_op_idx = -1  # NO_OP is not part of the agent's simplified space

        # 6. Return the agent to be stored as self._model
        return agent

    def get_latency(self) -> timedelta:
        """
        Returns the latency of the underlying system.
        """
        if self._system is None: return None
        return self._system.get_latency()

    def _reset(self, p_seed):
        self.log(self.C_LOG_TYPE_I, "Resetting scenario...")
        self._system.reset(p_seed=p_seed)

    def _run_cycle(self):
        """
        Runs a single cycle of the scenario using the translator ActionManager.
        """
        # 1. Get the full system mask from the environment
        system_mask = self._system.get_current_mask()

        # 2. Derive the simplified agent mask using the scenario's dedicated AgentMasker
        agent_mask = self._agent_masker.generate_agent_mask(system_mask)

        # 3. Get a simple integer action from our agent
        action = self._model.compute_action(p_state=self._system.get_state(), p_action_mask=agent_mask)
        agent_action_index = int(action.get_sorted_values()[0])

        # 4. Pass the agent's action to the translator ActionManager
        self._translator_am.process_agent_action(agent_action_index)

        # 5. Call simulate_reaction with a NO_OPERATION action to advance time and trigger automatic logic
        no_op_idx = self._system.action_map.get((SimulationAction.NO_OPERATION,))
        no_op_action = LogisticsAction(p_action_space=self._system.get_action_space(), p_values=[no_op_idx])
        self._system.simulate_reaction(p_state=self._system.get_state(), p_action=no_op_action)

        # 6. Get the new state and check for terminal conditions
        new_state = self._system.get_state()

        return False, new_state.get_broken(), new_state.get_success(), False
