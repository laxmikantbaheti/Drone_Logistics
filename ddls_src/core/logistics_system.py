import random
from typing import Dict, Any, List, Tuple
from datetime import timedelta
import numpy as np
import os

# MLPro Imports
from mlpro.bf.systems import System, State
from mlpro.bf.math import MSpace, Dimension
from mlpro.bf.events import EventManager, Event

# Local Imports
from ddls_src.core.global_state import GlobalState
from ddls_src.core.network import Network
from ddls_src.managers.action_manager import ActionManager
from ddls_src.managers.supply_chain_manager import SupplyChainManager
from ddls_src.managers.resource_manager.base import ResourceManager
from ddls_src.managers.network_manager import NetworkManager
from ddls_src.actions.action_map_generator import generate_action_map
from ddls_src.actions.base import SimulationActions, ActionType
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator
from ddls_src.scenarios.generators.order_generator import OrderGenerator
from ddls_src.core.state_action_mapper import StateActionMapper, ConstraintManager
from ddls_src.core.logistics_simulation import TimeManager
from ddls_src.config.automatic_logic_maps import AUTOMATIC_LOGIC_CONFIG
from ddls_src.core.basics import LogisticsAction
from ddls_src.entities.vehicles.truck import Truck
from ddls_src.entities.vehicles.drone import Drone
from ddls_src.entities import *
from ddls_src.actions.base import SimulationActions, ActionIndex


class LogisticsSystem(System, EventManager):
    """
    The top-level MLPro System that IS the entire logistics simulation engine.
    It now dynamically generates its action map and a permanent mask at runtime.
    """

    C_TYPE = 'Logistics System'
    C_NAME = 'Logistics System'
    C_EVENT_NEW_ORDER = 'NEW_ORDER_CREATED'
    C_MAX_AUTO_ACTIONS_PER_STEP = 20

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

        self.automatic_logic_config = {action: action.is_automatic for action in SimulationActions.get_all_actions()}
        self.time_manager = TimeManager(initial_time=self._config.get("initial_time", 0.0))
        self.data_loader = DataLoader(self._config.get("data_loader_config", {}))

        self.action_map = {}
        self._reverse_action_map = {}
        self.action_space_size = 0
        self._permanent_mask = None  # To hold the scenario's static mask

        self.global_state: GlobalState = None
        self.network: Network = None
        self.scenario_generator: ScenarioGenerator = None
        self.action_manager: ActionManager = None
        self.supply_chain_manager: SupplyChainManager = None
        self.resource_manager: ResourceManager = None
        self.network_manager: NetworkManager = None
        self.order_generator: OrderGenerator = None
        self.state_action_mapper: StateActionMapper = None
        self.constraint_manager:ConstraintManager = None
        self._state = State(self._state_space)
        self.actions = SimulationActions()
        self.action_index = None
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
        raw_entity_data = self.data_loader.load_initial_simulation_data()
        self.scenario_generator = ScenarioGenerator(raw_entity_data)
        self.entities = self.scenario_generator.build_entities()

        self.global_state = GlobalState(initial_entities=self.entities)

        self.action_map, self.action_space_size = self.actions.generate_action_map(self.global_state)
        self.action_index = ActionIndex(self.global_state, self.action_map)
        self._reverse_action_map = {idx: act for act, idx in self.action_map.items()}

        # Create the permanent mask for this scenario
        self._permanent_mask = np.ones(self.action_space_size, dtype=bool)
        for action_tuple, idx in self.action_map.items():
            if not action_tuple[0].active:
                self._permanent_mask[idx] = False

        self.network = Network(self.global_state)
        self.global_state.network = self.network
        self.state_action_mapper = StateActionMapper(self.global_state, self.action_map)
        all_entity_dicts = [
            self.global_state.orders, self.global_state.trucks,
            self.global_state.drones, self.global_state.micro_hubs,
            self.global_state.nodes
        ]
        for entity_dict in all_entity_dicts:
            for entity in entity_dict.values():
                entity.global_state = self.global_state

        self.constraint_manager = ConstraintManager(action_index=self.action_index, action_map=self.action_map)
        self.setup_events()

        self.supply_chain_manager = SupplyChainManager(p_id='scm', global_state=self.global_state,
                                                       p_automatic_logic_config=self.automatic_logic_config)
        self.resource_manager = ResourceManager(p_id='rm', global_state=self.global_state)
        self.network_manager = NetworkManager(p_id='nm', global_state=self.global_state, network=self.network,
                                              p_automatic_logic_config=self.automatic_logic_config)

        for vehicle in list(self.global_state.trucks.values()) + list(self.global_state.drones.values()):
            vehicle.network_manager = self.network_manager

        self.order_generator = OrderGenerator(self.global_state, self, self._config.get('new_order_config', {}))

        self.register_event_handler(self.C_EVENT_NEW_ORDER, self._handle_new_order_request)

        managers = {
            'SupplyChainManager': self.supply_chain_manager,
            'ResourceManager': self.resource_manager,
            'NetworkManager': self.network_manager
        }
        self.action_manager = ActionManager(self.global_state, managers, self.action_map)

        initial_sim_time = self.entities.get('initial_time', 0.0)
        self.time_manager.reset_time(new_initial_time=initial_sim_time)
        self.global_state.current_time = initial_sim_time

        self._update_state()

    def _get_automatic_actions(self) -> List[Tuple]:
        system_mask = self.get_current_mask()
        possible_action_indices = np.where(system_mask)[0]

        automatic_actions = []
        for index in possible_action_indices:
            action_tuple = self._reverse_action_map.get(index)
            if action_tuple:
                action_type = action_tuple[0]
                if self.automatic_logic_config.get(action_type, False):
                    automatic_actions.append(action_tuple)

        return automatic_actions

    def _simulate_reaction(self, p_state: State, p_action: LogisticsAction, p_t_step: timedelta = None) -> State:
        action_values, _ = p_action.get_sorted_values_with_data()
        action_index = int(action_values[0])
        action_tuple = self._reverse_action_map.get(action_index)

        if action_tuple and action_tuple[0] != SimulationActions.NO_OPERATION:
            print(f"  - Executing Agent Action: {action_tuple[0].name}{action_tuple[1:]}")
            self.action_manager.execute_action(action_tuple)

        print("  - Entering Automatic Action Loop...")
        for i in range(self.C_MAX_AUTO_ACTIONS_PER_STEP):
            automatic_actions_to_take = self._get_automatic_actions()

            if not automatic_actions_to_take:
                print(f"  - No more automatic actions possible after {i} iterations. System is stable.")
                break

            auto_action_tuple = automatic_actions_to_take[0]
            print(f"  - Executing Automatic Action: {auto_action_tuple[0].name}{auto_action_tuple[1:]}")
            self.action_manager.execute_action(auto_action_tuple)
        else:
            self.log(self.C_LOG_TYPE_W,
                     f"Automatic action loop reached max iterations ({self.C_MAX_AUTO_ACTIONS_PER_STEP}). Possible action storm.")

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
            dynamic_mask = self.state_action_mapper.generate_masks()
            # Combine the dynamic mask with the permanent scenario mask
            # return np.logical_and(self._permanent_mask, dynamic_mask)
            return dynamic_mask
        return np.ones(len(self.action_map), dtype=bool)

    def _update_state(self):
        if self.global_state:
            state_space = self._state.get_related_set()
            orders = self.global_state.get_all_entities("order").values()
            self._state.set_value(state_space.get_dim_by_name("total_orders").get_id(), len(orders))
            self._state.set_value(state_space.get_dim_by_name("delivered_orders").get_id(),
                                  sum(1 for o in orders if o.status == 'delivered'))

    def _handle_new_order_request(self, p_event_id, p_event_object):
        self.global_state.add_orders(p_orders=p_event_object.get_data()['p_orders'])
        self.state_action_mapper.add_order(p_oredrs= p_event_object.get_data()['p_orders'])

    def get_masks(self):
        return self.state_action_mapper.generate_masks()

    def setup_events(self):
        self.constraint_manager.register_event_handler(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                                                       p_event_handler=self.state_action_mapper.handle_new_masks_event)

        for entities in self.entities.values():
            if isinstance(entities, Dict):
                for entity in entities.values():
                    if isinstance(entity, LogisticEntity):
                        entity.register_event_handler(LogisticEntity.C_EVENT_ENTITY_STATE_CHANGE,
                                                      self.constraint_manager.handle_entity_state_change)


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Validating LogisticsSystem ---")

    script_path = os.path.dirname(os.path.realpath(__file__))
    # Go up two levels from core/ to the project root, then into config
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 300.0,
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {"file_path": config_file_path}
        },
        "new_order_config": {}
    }

    logistics_system = LogisticsSystem(p_id='validation_sys',
                                       p_visualize=False,
                                       p_logging=True,
                                       config=sim_config)

    print("\n--- Running simulation for 3 cycles with NO_OPERATION ---")

    for i in range(100):
        print(f"\n--- Cycle {i + 1} ---")
        masks = logistics_system.get_masks()
        valid_actions = [i for i in range(len(logistics_system.action_map))
                         if masks[i] is True]
        choice = random.choice(valid_actions)
        act = logistics_system._reverse_action_map[choice]
        action = LogisticsAction(p_action_space=logistics_system.get_action_space(),
                                 p_values=[choice], p_data=act)
        logistics_system.simulate_reaction(p_state=None, p_action=action)
        state = logistics_system.get_state()
        print(f"  - Current Time: {logistics_system.time_manager.get_current_time()}s")
        print(f"  - Total Orders: {state.get_value(state.get_related_set().get_dim_by_name('total_orders').get_id())}")
        print(f"  - Delivered Orders: {state.get_value(state.get_related_set().get_dim_by_name('delivered_orders').get_id())}")

    print("\n--- Validation Complete: LogisticsSystem initialized and ran successfully. ---")
