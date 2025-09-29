# In file: ddls_src/core/logistics_system.py

import random
from typing import Dict, Any, List, Tuple
from datetime import timedelta
import numpy as np
import os

# MLPro Imports
from mlpro.bf.systems import System, State
from mlpro.bf.math import MSpace, Dimension
from mlpro.bf.events import EventManager, Event

# from ddls_src.actions.action_map_generator import generate_action_map
# Local Imports
from ddls_src.core.global_state import GlobalState
from ddls_src.core.network import Network
from ddls_src.managers.action_manager import ActionManager
from ddls_src.managers.supply_chain_manager import SupplyChainManager
from ddls_src.managers.resource_manager.base import ResourceManager
from ddls_src.managers.network_manager import NetworkManager
from ddls_src.actions.base import SimulationActions, ActionType, ActionIndex
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator
from ddls_src.scenarios.generators.order_generator import OrderGenerator
from ddls_src.core.state_action_mapper import StateActionMapper, ConstraintManager
from ddls_src.core.logistics_simulation import TimeManager
from ddls_src.core.basics import LogisticsAction
from ddls_src.entities import *


class LogisticsSystem(System, EventManager):
    """
    The top-level MLPro System that IS the entire logistics simulation engine.
    It uses a two-phase cycle: a Decision Phase (action processing) and a
    Progression Phase (time advancement).
    """

    C_TYPE = 'Logistics System'
    C_NAME = 'Logistics System'
    C_EVENT_NEW_ORDER = 'NEW_ORDER_CREATED'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=False,
                 **p_kwargs):

        self._config = p_kwargs.get('config', {})

        System.__init__(self, p_id=p_id,
                        p_name=p_name,
                        p_visualize=p_visualize,
                        p_logging=p_logging,
                        p_mode=System.C_MODE_SIM,
                        p_latency=timedelta(seconds=self._config.get("main_timestep_duration", 60.0)))
        EventManager.__init__(self, p_logging=self.get_log_level())
        self.movement_mode = self._config.get('movement_mode', 'network')
        # Initialize attributes
        self.automatic_logic_config = {}
        self.time_manager = TimeManager(initial_time=self._config.get("initial_time", 0.0))
        self.data_loader = DataLoader(self._config.get("data_loader_config", {}))
        self.action_map: Dict[Tuple, int] = {}
        self._reverse_action_map: Dict[int, Tuple] = {}
        self.action_space_size: int = 0
        self.global_state: GlobalState = None
        self.network: Network = None
        self.action_manager: ActionManager = None
        self.supply_chain_manager: SupplyChainManager = None
        self.resource_manager: ResourceManager = None
        self.network_manager: NetworkManager = None
        self.order_generator: OrderGenerator = None
        self.state_action_mapper: StateActionMapper = None
        self.constraint_manager: ConstraintManager = None
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
                                       p_boundaries=[0, 20000]))
        return state_space, action_space

    def _reset(self, p_seed=None):
        self.automatic_logic_config = {action: action.is_automatic for action in self.actions.get_all_actions()}

        raw_entity_data = self.data_loader.load_initial_simulation_data()
        scenario_generator = ScenarioGenerator(raw_entity_data)
        self.entities = scenario_generator.build_entities(p_logging = self.get_log_level(),
                                                          p_movement_mode = self.movement_mode)

        self.global_state = GlobalState(initial_entities=self.entities, movement_mode=self.movement_mode)

        self.action_map, self.action_space_size = self.actions.generate_action_map(self.global_state)
        self.action_index = ActionIndex(self.global_state, self.action_map)
        self._reverse_action_map = {idx: act for act, idx in self.action_map.items()}

        self.network = Network(self.global_state, self.movement_mode, raw_entity_data['distance_matrix'])
        self.global_state.network = self.network
        self.state_action_mapper = StateActionMapper(self.global_state, self.action_map)

        all_entity_dicts = self.global_state.get_all_entities()
        for entity_dict in all_entity_dicts:
            for entity in entity_dict.values():
                entity.global_state = self.global_state
                # entity.reset()

        self.constraint_manager = ConstraintManager(action_index=self.action_index, reverse_action_map=self._reverse_action_map)
        self.supply_chain_manager = SupplyChainManager(p_id='scm', global_state=self.global_state,
                                                       p_automatic_logic_config=self.automatic_logic_config)
        self.resource_manager = ResourceManager(p_id='rm', global_state=self.global_state)
        self.network_manager = NetworkManager(p_id='nm', global_state=self.global_state, network=self.network,
                                              p_automatic_logic_config=self.automatic_logic_config)
        self.setup_events()


        managers = {'SupplyChainManager': self.supply_chain_manager, 'ResourceManager': self.resource_manager,
                    'NetworkManager': self.network_manager}

        for manager in managers.values():
            manager.system = self

        self.action_manager = ActionManager(self.global_state, managers, self.action_map)

        for vehicle in list(self.global_state.trucks.values()) + list(self.global_state.drones.values()):
            vehicle.network_manager = self.network_manager

        self.order_generator = OrderGenerator(self.global_state, self, self._config.get('new_order_config', {}))
        self.register_event_handler(self.C_EVENT_NEW_ORDER, self._handle_new_order_request)

        initial_sim_time = self.entities.get('initial_time', 0.0)
        self.time_manager.reset_time(new_initial_time=initial_sim_time)
        self.global_state.current_time = initial_sim_time

        self._update_state()

    def get_automatic_actions(self) -> List[Tuple]:
        system_mask = self.get_current_mask()
        possible_action_indices = np.where(system_mask)[0]
        automatic_actions = []
        for index in possible_action_indices:
            action_tuple = self._reverse_action_map.get(index)
            if action_tuple and self.automatic_logic_config.get(action_tuple[0], False):
                automatic_actions.append(action_tuple)
        return automatic_actions

    # def are_automatic_actions_available(self):
    #     automatic_action_to_take = self._get_automatic_actions()

    def run_automatic_action_loop(self):
        i = 0
        while True:
            automatic_actions_to_take = self.get_automatic_actions()
            if not automatic_actions_to_take:
                print(f"Auto-action loop stable after {i} iterations.")
                break
            auto_action_tuple = automatic_actions_to_take[0]
            print(f"  - Auto Action: {auto_action_tuple[0].name}{auto_action_tuple[1:]}")
            self.action_manager.execute_action(auto_action_tuple)
            i += 1
            if i > 20:  # Safety break
                print("Auto-action loop exceeded safety limit of 20 iterations.")
                break

    def process_action(self, p_action: LogisticsAction):
        action_processed = False
        action_values, _ = p_action.get_sorted_values_with_data()
        action_index = int(action_values[0])
        action_tuple = self._reverse_action_map.get(action_index)

        if action_tuple and action_tuple[0] != SimulationActions.NO_OPERATION:
            print(f"  - Agent Action: {action_tuple[0].name}{action_tuple[1:]}")
            action_processed = self.action_manager.execute_action(action_tuple)

        # self.run_automatic_action_loop()
        self._update_state()
        return action_processed


    def advance_time(self, p_t_step: timedelta = None):
        timestep_duration = self.get_latency().total_seconds()
        t_step = p_t_step or timedelta(seconds=timestep_duration)
        self.time_manager.advance_time(timestep_duration)
        self.global_state.current_time = self.time_manager.get_current_time()

        self.log(self.C_LOG_TYPE_I, f"Time advanced to {self.global_state.current_time}s.")
        self.order_generator.generate(self.global_state.current_time)

        all_systems = (list(self.global_state.trucks.values()) +
                       list(self.global_state.drones.values()) +
                       list(self.global_state.micro_hubs.values()) +
                       [self.supply_chain_manager, self.resource_manager, self.network_manager])

        for system in all_systems:
            system.simulate_reaction(p_state=None, p_action=None, p_t_step=t_step)

        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: LogisticsAction) -> State:
        self.process_action(p_action)
        self.advance_time()
        return self._state

    def get_current_mask(self) -> np.ndarray:
        if self.state_action_mapper:
            return self.state_action_mapper.generate_masks()
        return np.ones(len(self.action_map), dtype=bool)

    def get_agent_mask(self) -> np.ndarray:
        system_mask = self.get_current_mask()
        agent_mask = np.zeros(self.action_space_size, dtype=bool)
        for action_tuple, idx in self.action_map.items():
            action_type = action_tuple[0]
            if not self.automatic_logic_config.get(action_type, False) and system_mask[idx]:
                agent_mask[idx] = True
        no_op_idx = self.action_map.get((SimulationActions.NO_OPERATION,))
        if no_op_idx is not None:
            agent_mask[no_op_idx] = True
        return agent_mask

    def _update_state(self):
        if self.global_state:
            state_space = self._state.get_related_set()
            orders = self.global_state.get_all_entities_by_type("order").values()
            self._state.set_value(state_space.get_dim_by_name("total_orders").get_id(), len(orders))
            self._state.set_value(state_space.get_dim_by_name("delivered_orders").get_id(),
                                  sum(1 for o in orders if o.status == 'delivered'))

    def _handle_new_order_request(self, p_event_id, p_event_object):
        orders = p_event_object.get_data()['p_orders']
        # self.global_state.add_orders(p_orders=p_event_object.get_data()['p_orders'])
        self.global_state.add_dynamic_orders(orders)
        # self.state_action_mapper.add_order(p_oredrs=p_event_object.get_data()['p_orders'])
        self.action_map, self.action_space_size = self.actions.generate_action_map(self.global_state)
        self._reverse_action_map = {idx: act for act, idx in self.action_map.items()}
        self.action_index.build_indexes(global_state=self.global_state, action_map=self.action_map)
        self.state_action_mapper.update_action_space(self.action_map)
        self.constraint_manager.update_constraints(self.global_state, self._reverse_action_map)
        self.get_masks()

    def get_masks(self):
        return self.state_action_mapper.generate_masks()

    def setup_events(self):
        self.constraint_manager.register_event_handler(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                                                       p_event_handler=self.state_action_mapper.handle_new_masks_event)
        for entities in self.entities.values():
            if isinstance(entities, Dict):
                for entity in entities.values():
                    if isinstance(entity, LogisticEntity):
                        entity.register_event_handler_for_constraints(LogisticEntity.C_EVENT_ENTITY_STATE_CHANGE,
                                                                      self.constraint_manager.handle_entity_state_change)

        self.supply_chain_manager.register_event_handler(SupplyChainManager.C_EVENT_NEW_ORDER_REQUEST,
                                                         self._handle_new_order_request)


    def get_success(self) -> bool:
        orders = self.global_state.get_orders()
        success = True
        for ords in orders.values():
            success = (ords.get_state_value_by_dim_name(ords.C_DIM_DELIVERY_STATUS[0]) == ords.C_STATUS_DELIVERED) and success
        if success:
            return success
        return success


    def get_broken(self):
        return False


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Validating LogisticsSystem ---")

    script_path = os.path.dirname(os.path.realpath(__file__))
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
                                       p_logging=False,
                                       config=sim_config)

    print("\n--- Running simulation for 20 cycles with Dummy Agent Logic ---")

    no_op_idx = logistics_system.action_map.get((SimulationActions.NO_OPERATION,))

    for i in range(20):
        print(f"\n--- Cycle {i + 1} ---")

        # --- Decision Phase ---
        # A simple agent takes one action per cycle
        agent_mask = logistics_system.get_agent_mask()
        valid_actions = np.where(np.delete(agent_mask, no_op_idx))[0]

        if len(valid_actions) > 0:
            choice = random.choice(valid_actions)
            act_tuple = logistics_system._reverse_action_map.get(choice)
            print(f"Dummy Agent chooses: {act_tuple}")
        else:
            choice = no_op_idx
            print("Dummy Agent chooses: NO_OPERATION")

        action = LogisticsAction(p_action_space=logistics_system.get_action_space(), p_values=[choice])

        logistics_system.process_action(action)

        # --- Progression Phase ---
        logistics_system.advance_time()

        # --- Reporting ---
        state = logistics_system.get_state()
        print(f"  - Current Time: {logistics_system.time_manager.get_current_time()}s")
        state_dim_total = state.get_related_set().get_dim_by_name('total_orders')
        state_dim_delivered = state.get_related_set().get_dim_by_name('delivered_orders')
        print(f"  - Total Orders: {state.get_value(state_dim_total.get_id())}")
        print(f"  - Delivered Orders: {state.get_value(state_dim_delivered.get_id())}")

    print("\n--- Validation Complete: LogisticsSystem initialized and ran successfully. ---")