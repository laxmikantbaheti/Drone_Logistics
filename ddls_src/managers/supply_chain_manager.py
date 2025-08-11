from typing import List, Dict, Any, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State
from mlpro.bf.math import MSpace, Dimension

# Local Imports
from ..core.basics import LogisticsAction
from ..actions.base import SimulationAction


# Forward declarations
class GlobalState: pass


class Order: pass


class Truck: pass


class Drone: pass


class MicroHub: pass


class SupplyChainManager(System):
    """
    Manages the lifecycle and assignment of orders as an MLPro System.
    Its action space is now dynamically configured from the action blueprint.
    """

    C_TYPE = 'Supply Chain Manager'
    C_NAME = 'Supply Chain Manager'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes the SupplyChainManager system.
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        self.global_state: 'GlobalState' = p_kwargs.get('global_state')
        self.automatic_logic_config = p_kwargs.get('p_automatic_logic_config', {})
        if self.global_state is None:
            raise ValueError("SupplyChainManager requires a reference to GlobalState.")

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for the SupplyChainManager.
        The action space is dynamically built from the action blueprint.
        """
        state_space = MSpace()
        state_space.add_dim(Dimension('num_orders_total', 'Z', 'Total Orders', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('orders_pending', 'Z', 'Pending Orders', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('orders_in_transit', 'Z', 'In-Transit Orders', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('orders_delivered', 'Z', 'Delivered Orders', p_boundaries=[0, 9999]))

        # Dynamically find all actions handled by this manager
        handler_name = "SupplyChainManager"
        action_ids = [action.id for action in SimulationAction if action.handler == handler_name]

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='scm_action_id',
                                       p_base_set='Z',
                                       p_name_long='Supply Chain Manager Action ID',
                                       p_boundaries=[min(action_ids), max(action_ids)]))

        return state_space, action_space

    def _reset(self, p_seed=None):
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: LogisticsAction, p_t_step: timedelta = None) -> State:
        if p_action is not None:
            self._process_action(p_action)

        self._check_and_assign_orders()

        self._update_state()
        return self._state

    def _check_and_assign_orders(self):
        """
        Scans for pending orders and assigns them to available vehicles if auto-assignment is enabled.
        """
        if not self.automatic_logic_config.get(SimulationAction.ASSIGN_ORDER_TO_TRUCK, False):
            return

        pending_orders = [o for o in self.global_state.orders.values() if o.status == 'pending']
        idle_trucks = [t for t in self.global_state.trucks.values() if t.status == 'idle']

        if not pending_orders or not idle_trucks:
            return

        # Simple logic: assign the first pending order to the first idle truck
        order_to_assign = pending_orders[0]
        truck_to_assign = idle_trucks[0]

        print(
            f"  - AUTOMATIC LOGIC (SupplyChainManager): Assigning Order {order_to_assign.id} to Truck {truck_to_assign.id}")
        order_to_assign.assign_vehicle(truck_to_assign.id)

    def _process_action(self, p_action: LogisticsAction) -> bool:
        """
        Processes a high-level command related to order management.
        """
        action_id = int(p_action.get_sorted_values()[0])
        action_type = SimulationAction._value2member_map_.get(action_id)
        action_kwargs = p_action.data

        try:
            order_id = action_kwargs['order_id']
            order: 'Order' = self.global_state.get_entity("order", order_id)

            if action_type == SimulationAction.ACCEPT_ORDER:
                if order.status == "pending":
                    order.update_status("accepted")
                    return True
                return False

            elif action_type == SimulationAction.CANCEL_ORDER:
                if order.status not in ["delivered", "cancelled"]:
                    if order.assigned_vehicle_id is not None:
                        order.unassign_vehicle()
                    order.update_status("cancelled")
                    return True
                return False

            elif action_type == SimulationAction.ASSIGN_ORDER_TO_TRUCK:
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                if order.status in ["pending", "accepted", "flagged_re_delivery"]:
                    order.assign_vehicle(truck.id)
                    return True
                return False

            elif action_type == SimulationAction.ASSIGN_ORDER_TO_DRONE:
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                if order.status in ["pending", "accepted", "flagged_re_delivery"]:
                    order.assign_vehicle(drone.id)
                    return True
                return False

            elif action_type == SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB:
                hub: 'MicroHub' = self.global_state.get_entity("micro_hub", action_kwargs['micro_hub_id'])
                if order.status in ["pending", "accepted", "flagged_re_delivery"]:
                    order.assign_micro_hub(hub.id)
                    return True
                return False

        except KeyError as e:
            self.log(self.C_LOG_TYPE_E, f"Action parameter missing: {e}")
            return False

        return False

    def _update_state(self):
        """
        Calculates aggregate order statistics and updates the formal state object.
        """
        state_space = self._state.get_related_set()
        orders = self.global_state.get_all_entities("order").values()

        self._state.set_value(state_space.get_dim_by_name("num_orders_total").get_id(), len(orders))
        self._state.set_value(state_space.get_dim_by_name("orders_pending").get_id(),
                              sum(1 for o in orders if o.status == 'pending'))
        self._state.set_value(state_space.get_dim_by_name("orders_in_transit").get_id(),
                              sum(1 for o in orders if o.status == 'in_transit'))
        self._state.set_value(state_space.get_dim_by_name("orders_delivered").get_id(),
                              sum(1 for o in orders if o.status == 'delivered'))
