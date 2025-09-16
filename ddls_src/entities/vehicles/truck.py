from typing import List, Tuple, Any, Dict, Optional
from datetime import timedelta

# Refactored local imports
from .base import Vehicle

# MLPro Imports
from mlpro.bf.systems import State, Action
from mlpro.bf.math import MSpace, Dimension
from ddls_src.actions.action_enums import SimulationAction


# Forward declarations
class GlobalState: pass


class Order: pass


class Node: pass


class Truck(Vehicle):
    """
    Represents a truck vehicle, refactored as a concrete MLPro System.
    It now includes automatic logic for loading and unloading.
    """

    C_TYPE = 'Truck'
    C_NAME = 'Truck'

    def __init__(self,
                 p_id: int,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes a Truck system.
        """
        self.initial_fuel: float = p_kwargs.get('initial_fuel', 100.0)
        self.fuel_consumption_rate: float = p_kwargs.get('fuel_consumption_rate', 0.1)
        self.max_fuel_capacity: float = self.initial_fuel * 1.5
        self.fuel_level: float = self.initial_fuel
        self.global_state: 'GlobalState' = p_kwargs.get('global_state', None)
        self.automatic_logic_config = p_kwargs.get('p_automatic_logic_config', {})

        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        state_space, _ = Vehicle.setup_spaces()
        state_space.add_dim(Dimension('fuel_level', 'R', 'Current Fuel Level', p_boundaries=[0, 9999]))

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='truck_action',
                                       p_base_set='Z',
                                       p_name_long='Truck Action',
                                       p_boundaries=[0, 2]))

        return state_space, action_space

    def _reset(self, p_seed=None):
        super()._reset(p_seed)
        self.fuel_level = self.initial_fuel
        # self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Simulates the truck's state, including automatic loading/unloading logic.
        """
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        # --- Automatic Logic ---
        self._check_and_perform_node_actions()

        # Call parent's simulation for movement
        super()._simulate_reaction(p_state, None, p_t_step)

        # self._update_state()
        return self._state

    def _check_and_perform_node_actions(self):
        """
        Checks for and executes automatic loading or unloading if the truck is idle at a node.
        """
        if self.status != 'idle' or self.current_node_id is None:
            return

        # Check for auto-unloading (delivery)
        if self.automatic_logic_config.get(SimulationAction.UNLOAD_TRUCK_ACTION, False):
            for order_id in self.cargo_manifest:
                order = self.global_state.get_entity("order", order_id)
                if order.customer_node_id == self.current_node_id:
                    print(f"  - AUTOMATIC LOGIC (Truck {self.id}): Unloading Order {order_id} at destination.")
                    self._unload_order(order_id)
                    return  # Only do one action per cycle

        # Check for auto-loading (pickup)
        if self.automatic_logic_config.get(SimulationAction.LOAD_TRUCK_ACTION, False):
            current_node = self.global_state.get_entity("node", self.current_node_id)
            for order_id in current_node.packages_held:
                order = self.global_state.get_entity("order", order_id)
                if order.assigned_vehicle_id == self.id:
                    print(f"  - AUTOMATIC LOGIC (Truck {self.id}): Loading assigned Order {order_id}.")
                    self._load_order(order_id)
                    return  # Only do one action per cycle

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        if self.global_state is None: return False

        # action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()
        action_value = p_action.get_sorted_values()[0]
        action_kwargs = p_action.get_kwargs()

        if action_value == 0:  # GO_TO_NODE
            return super()._process_action(p_action, p_t_step)

        try:
            order_id = action_kwargs['order_id']
            if action_value == 1:  # LOAD_ORDER
                return self._load_order(order_id)
            elif action_value == 2:  # UNLOAD_ORDER
                return self._unload_order(order_id)
        except KeyError:
            return False

        return False

    def update_energy(self, p_time_passed: float):
        fuel_consumed = self.fuel_consumption_rate * abs(p_time_passed)
        self.fuel_level -= fuel_consumed
        self.fuel_level = max(0.0, self.fuel_level)
        # if self.fuel_level <= 0.0:
        #     self.status = "broken_down"

    def _update_state(self):
        # super()._update_state()
        state_space = self._state.get_related_set()
        self._state.set_value(state_space.get_dim_by_name("fuel_level").get_id(), self.fuel_level)

    def _load_order(self, order_id: int) -> bool:
        try:
            order: 'Order' = self.global_state.get_entity("order", order_id)
            if self.current_node_id is None: return False
            current_node: 'Node' = self.global_state.get_entity("node", self.current_node_id)

            if not current_node.is_loadable: return False
            if order_id not in current_node.get_packages(): return False
            if len(self.cargo_manifest) >= self.max_payload_capacity: return False

            current_node.remove_package(order_id)
            self.add_cargo(order_id)
            order.update_status("in_transit")
            self.set_status("loading")
            return True
        except KeyError:
            return False

    def _unload_order(self, order_id: int) -> bool:
        try:
            order: 'Order' = self.global_state.get_entity("order", order_id)
            if self.current_node_id is None: return False
            current_node: 'Node' = self.global_state.get_entity("node", self.current_node_id)

            if not current_node.is_unloadable: return False
            if order_id not in self.get_cargo(): return False

            self.remove_cargo(order_id)
            current_node.add_package(order_id)

            if order.customer_node_id == current_node.id:
                order.update_status("delivered")
                order.delivery_time = self.global_state.current_time
            else:
                order.update_status("at_node")

            self.set_status("unloading")
            return True
        except KeyError:
            return False
