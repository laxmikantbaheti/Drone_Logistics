from typing import List, Tuple, Any, Dict, Optional
from datetime import timedelta

# Refactored local imports
from .base import Vehicle

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


# Forward declarations
class GlobalState: pass


class Order: pass


class Node: pass


class Truck(Vehicle):
    """
    Represents a truck vehicle, refactored as a concrete MLPro System.
    It now processes its own load/unload actions.
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
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)

        self.global_state: 'GlobalState' = p_kwargs.get('global_state')
        if self.global_state is None:
            raise ValueError("Truck requires a reference to GlobalState.")

        self.initial_fuel: float = p_kwargs.get('initial_fuel', 100.0)
        self.fuel_consumption_rate: float = p_kwargs.get('fuel_consumption_rate', 0.1)
        self.max_fuel_capacity: float = self.initial_fuel * 1.5
        self.fuel_level: float = self.initial_fuel

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Extends the Vehicle's state and action spaces with truck-specific dimensions.
        """
        state_space, action_space = Vehicle.setup_spaces()

        state_space.add_dim(Dimension('fuel_level', 'R', 'Current Fuel Level', p_boundaries=[0, 9999]))

        # Redefine action space for truck-specific discrete actions
        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='truck_action',
                                       p_base_set='Z',
                                       p_name_long='Truck Action',
                                       p_boundaries=[0, 2]))
        # 0: GO_TO_NODE (handled by base class)
        # 1: LOAD_ORDER
        # 2: UNLOAD_ORDER

        return state_space, action_space

    def _reset(self, p_seed=None):
        super()._reset(p_seed)
        self.fuel_level = self.initial_fuel
        self._update_state()

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes actions for the truck, including loading and unloading.
        """
        action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()
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
            self.log(self.C_LOG_TYPE_E, "Action requires 'order_id' in kwargs.")
            return False

        return False

    def update_energy(self, p_time_passed: float):
        fuel_consumed = self.fuel_consumption_rate * p_time_passed
        self.fuel_level -= fuel_consumed
        self.fuel_level = max(0.0, self.fuel_level)
        if self.fuel_level <= 0.0:
            self.status = "broken_down"

    def _update_state(self):
        super()._update_state()
        self._state.set_value('fuel_level', self.fuel_level)

    def _load_order(self, order_id: int) -> bool:
        """Internal logic to load an order."""
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
        """Internal logic to unload an order."""
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
