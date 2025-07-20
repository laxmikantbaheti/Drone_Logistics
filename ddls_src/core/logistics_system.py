from typing import Dict, Any, List, Tuple
from datetime import timedelta
import numpy as np

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension

# Local Imports
from ..core.global_state import GlobalState
from ..core.network import Network
from ..managers.action_manager import ActionManager
from ..managers.supply_chain_manager import SupplyChainManager
from ..managers.resource_manager.base import ResourceManager
from ..managers.network_manager import NetworkManager
from ..actions.action_mapping import ACTION_MAP, ACTION_SPACE_SIZE
from ..actions.action_enums import SimulationAction
from ..scenarios.generators.data_loader import DataLoader
from ..scenarios.generators.scenario_generator import ScenarioGenerator
from ..core.logistics_simulation import TimeManager  # Keep TimeManager as a helper


class LogisticsSystem(System):
    """
    The top-level MLPro System that IS the entire logistics simulation engine.
    It manages the state, entities, and managers, and processes actions to advance the simulation.
    """

    C_TYPE = 'Logistics System'
    C_NAME = 'Logistics System'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes the entire LogisticsSystem and all its sub-components.
        """
        self._config = p_kwargs.get('config', {})

        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(seconds=self._config.get("main_timestep_duration", 60.0)))

        # Initialize all core components directly within this system
        self.time_manager = TimeManager(initial_time=self._config.get("initial_time", 0.0))
        self.data_loader = DataLoader(self._config.get("data_loader_config", {}))
        self.action_map = ACTION_MAP
        self.action_space_size = ACTION_SPACE_SIZE

        # Placeholders for components to be initialized by _reset
        self.global_state: GlobalState = None
        self.network: Network = None
        self.scenario_generator: ScenarioGenerator = None
        self.action_manager: ActionManager = None
        self.supply_chain_manager: SupplyChainManager = None
        self.resource_manager: ResourceManager = None
        self.network_manager: NetworkManager = None

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the global state and action spaces for the entire logistics system.
        """
        state_space = MSpace()
        state_space.add_dim(Dimension('total_orders', 'Z', 'Total Orders', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('delivered_orders', 'Z', 'Delivered Orders', p_boundaries=[0, 9999]))

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='global_action',
                                       p_base_set='Z',
                                       p_name_long='Global Flattened Action ID',
                                       p_boundaries=[0, ACTION_SPACE_SIZE - 1]))

        return state_space, action_space

    def _reset(self, p_seed=None):
        """
        Resets the entire simulation to its initial state by re-initializing all components.
        """
        raw_entity_data = self.data_loader.load_initial_simulation_data()
        self.scenario_generator = ScenarioGenerator(raw_entity_data)

        # The ScenarioGenerator needs a global_state reference to instantiate entities correctly
        # This creates a slight chicken-and-egg problem. We solve it by passing a reference
        # to this system, which will have the global_state after it's created.
        initial_entities = self.scenario_generator.build_entities(p_system=self)

        self.global_state = GlobalState(initial_entities)
        self.network = Network(self.global_state)
        self.global_state.network = self.network

        self.supply_chain_manager = SupplyChainManager(p_id='scm', global_state=self.global_state)
        self.resource_manager = ResourceManager(p_id='rm', global_state=self.global_state)
        self.network_manager = NetworkManager(p_id='nm', global_state=self.global_state, network=self.network)

        managers = {
            'supply_chain_manager': self.supply_chain_manager,
            'resource_manager': self.resource_manager,
            'network_manager': self.network_manager
        }

        # ActionMasker needs to be refactored
        action_masker = None  # Placeholder

        self.action_manager = ActionManager(self.global_state, managers, self.action_map, action_masker)

        initial_sim_time = initial_entities.get('initial_time', 0.0)
        self.time_manager.reset_time(new_initial_time=initial_sim_time)
        self.global_state.current_time = initial_sim_time

        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        This is the core of the environment. It takes an action, processes it, advances the
        simulation time by one step, and returns the new state.
        """
        # 1. Process the incoming action from the external agent
        action_index = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()
        action_tuple = self.action_manager._reverse_action_map.get(action_index)

        # For now, we assume a refactored ActionMasker would have been used by the Scenario
        # to ensure the action is valid. We pass a dummy mask.
        dummy_mask = np.ones(self.action_space_size, dtype=bool)
        self.action_manager.execute_action(action_tuple, dummy_mask)

        # 2. Advance the simulation by one main timestep
        timestep_duration = self.get_latency().total_seconds()
        t_step = timedelta(seconds=timestep_duration)
        self.time_manager.advance_time(timestep_duration)
        self.global_state.current_time = self.time_manager.get_current_time()

        # 3. Simulate all active sub-systems
        all_systems = list(self.global_state.trucks.values()) + \
                      list(self.global_state.drones.values()) + \
                      list(self.global_state.micro_hubs.values()) + \
                      [self.supply_chain_manager, self.resource_manager, self.network_manager]

        for system in all_systems:
            system.simulate_reaction(p_state=None, p_action=None, p_t_step=t_step)

        # 4. Update and return the new high-level state
        self._update_state()
        return self._state

    def _update_state(self):
        """
        Updates the high-level state of this system based on the detailed global state.
        """
        if self.global_state:
            orders = self.global_state.get_all_entities("order").values()
            self._state.set_value('total_orders', len(orders))
            self._state.set_value('delivered_orders', sum(1 for o in orders if o.status == 'delivered'))
