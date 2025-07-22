from typing import Dict, Any, List, Tuple
from datetime import timedelta
import numpy as np

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension
from mlpro.bf.events import EventManager, Event

# Local Imports
from ddls_src.core.global_state import GlobalState
from ddls_src.core.network import Network
from ddls_src.managers.action_manager import ActionManager
from ddls_src.managers.supply_chain_manager import SupplyChainManager
from ddls_src.managers.resource_manager.base import ResourceManager
from ddls_src.managers.network_manager import NetworkManager
from ddls_src.actions.action_mapping import ACTION_MAP, ACTION_SPACE_SIZE
from ddls_src.actions.action_enums import SimulationAction
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator
from ddls_src.scenarios.generators.order_generator import OrderGenerator
from ddls_src.actions.state_action_mapper import StateActionMapper
from ddls_src.actions.constraints.base import OrderAssignableConstraint, VehicleAvailableConstraint
from ddls_src.core.logistics_simulation import TimeManager


class LogisticsSystem(System, EventManager):
    """
    The top-level MLPro System that IS the entire logistics simulation engine.
    """

    C_TYPE = 'Logistics System'
    C_NAME = 'Logistics System'
    C_EVENT_NEW_ORDER = 'NEW_ORDER_CREATED'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):

        self._config = p_kwargs.get('config', {})

        System.__init__(self, p_id=p_id,
                        p_name=p_name,
                        p_visualize=p_visualize,
                        p_logging=p_logging,
                        p_mode=System.C_MODE_SIM,
                        p_latency=timedelta(seconds=self._config.get("main_timestep_duration", 60.0)))
        EventManager.__init__(self, p_logging=self.get_log_level())

        self.time_manager = TimeManager(initial_time=self._config.get("initial_time", 0.0))
        self.data_loader = DataLoader(self._config.get("data_loader_config", {}))
        self.action_map = ACTION_MAP.copy()

        self.global_state: GlobalState = None
        self.network: Network = None
        self.scenario_generator: ScenarioGenerator = None
        self.action_manager: ActionManager = None
        self.supply_chain_manager: SupplyChainManager = None
        self.resource_manager: ResourceManager = None
        self.network_manager: NetworkManager = None
        self.order_generator: OrderGenerator = None
        self.state_action_mapper: StateActionMapper = None

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        state_space = MSpace()
        state_space.add_dim(Dimension('total_orders', 'Z', 'Total Orders', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('delivered_orders', 'Z', 'Delivered Orders', p_boundaries=[0, 9999]))

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='global_action',
                                       p_base_set='Z',
                                       p_name_long='Global Flattened Action ID',
                                       p_boundaries=[0, 10000]))

        return state_space, action_space

    def _reset(self, p_seed=None):
        """
        Resets the simulation using a two-phase initialization.
        """
        # --- Phase 1: Create all objects ---
        raw_entity_data = self.data_loader.load_initial_simulation_data()
        self.scenario_generator = ScenarioGenerator(raw_entity_data)
        initial_entities = self.scenario_generator.build_entities()

        self.global_state = GlobalState(initial_entities)
        self.network = Network(self.global_state)

        # --- Phase 2: Inject dependencies ---
        self.global_state.network = self.network
        all_entity_dicts = [
            self.global_state.orders, self.global_state.trucks,
            self.global_state.drones, self.global_state.micro_hubs,
            self.global_state.nodes
        ]
        for entity_dict in all_entity_dicts:
            for entity in entity_dict.values():
                # Inject the global state reference into each entity
                entity.global_state = self.global_state

        # Now that entities have their dependencies, finalize their setup (e.g., adding packages)
        for node in self.global_state.nodes.values():
            if hasattr(node, 'temp_packages'):
                for order_id in node.temp_packages:
                    node.add_package(order_id)
                del node.temp_packages

        # --- Initialize Managers and other components ---
        self.supply_chain_manager = SupplyChainManager(p_id='scm', global_state=self.global_state)
        self.resource_manager = ResourceManager(p_id='rm', global_state=self.global_state)
        self.network_manager = NetworkManager(p_id='nm', global_state=self.global_state, network=self.network)

        self.order_generator = OrderGenerator(self.global_state, self, self._config.get('new_order_config', {}))

        constraints_to_use = [OrderAssignableConstraint(), VehicleAvailableConstraint()]
        self.state_action_mapper = StateActionMapper(self.global_state, self.action_map, constraints_to_use)

        self.register_event_handler(self.C_EVENT_NEW_ORDER, self.state_action_mapper.handle_new_order_event)

        initial_sim_time = initial_entities.get('initial_time', 0.0)
        self.time_manager.reset_time(new_initial_time=initial_sim_time)
        self.global_state.current_time = initial_sim_time

        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        action_index = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()

        timestep_duration = self.get_latency().total_seconds()
        t_step = p_t_step or timedelta(seconds=timestep_duration)
        self.time_manager.advance_time(timestep_duration)
        self.global_state.current_time = self.time_manager.get_current_time()

        self.order_generator.generate(self.global_state.current_time)

        all_systems = list(self.global_state.trucks.values()) + \
                      list(self.global_state.drones.values()) + \
                      list(self.global_state.micro_hubs.values()) + \
                      [self.supply_chain_manager, self.resource_manager, self.network_manager]

        for system in all_systems:
            system.simulate_reaction(p_state=None, p_action=None, p_t_step=t_step)

        self._update_state()
        return self._state

    def get_current_mask(self) -> np.ndarray:
        if self.state_action_mapper:
            return self.state_action_mapper.generate_mask()
        return np.ones(len(self.action_map), dtype=bool)

    def _update_state(self):
        if self.global_state:
            orders = self.global_state.get_all_entities("order").values()
            self._state.set_value('total_orders', len(orders))
            self._state.set_value('delivered_orders', sum(1 for o in orders if o.status == 'delivered'))
