from typing import List, Dict, Any, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


# Forward declarations
class GlobalState: pass


class Order: pass


class Truck: pass


class Drone: pass


class MicroHub: pass


class SupplyChainManager(System):
    """
    Manages the lifecycle and assignment of orders as an MLPro System.
    It processes high-level strategic commands and orchestrates entities.
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
        if self.global_state is None:
            raise ValueError("SupplyChainManager requires a reference to GlobalState.")

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for the SupplyChainManager.
        """
        state_space = MSpace()
        state_space.add_dim(Dimension('num_orders_total', 'Z', 'Total Orders', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('orders_pending', 'Z', 'Pending Orders', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('orders_in_transit', 'Z', 'In-Transit Orders', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('orders_delivered', 'Z', 'Delivered Orders', p_boundaries=[0, 9999]))

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='scm_action',
                                       p_base_set='Z',
                                       p_name_long='Supply Chain Manager Action',
                                       p_boundaries=[0, 4]))
        # 0: ACCEPT_ORDER, 1: CANCEL_ORDER, 2: ASSIGN_TO_TRUCK, 3: ASSIGN_TO_DRONE, 4: ASSIGN_TO_HUB

        return state_space, action_space

    def _reset(self, p_seed=None):
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        self._update_state()
        return self._state

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a high-level command related to order management.
        """
        action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()
        action_kwargs = p_action.get_kwargs()

        try:
            order_id = action_kwargs['order_id']
            order: 'Order' = self.global_state.get_entity("order", order_id)

            if action_value == 0:  # ACCEPT_ORDER
                if order.status == "pending":
                    order.update_status("accepted")
                    return True
                return False

            elif action_value == 1:  # CANCEL_ORDER
                if order.status not in ["delivered", "cancelled"]:
                    if order.assigned_vehicle_id is not None:
                        order.unassign_vehicle()
                    order.update_status("cancelled")
                    return True
                return False

            elif action_value == 2:  # ASSIGN_TO_TRUCK
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                if order.status in ["pending", "accepted", "flagged_re_delivery"]:
                    order.assign_vehicle(truck.id)
                    return True
                return False

            elif action_value == 3:  # ASSIGN_TO_DRONE
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                if order.status in ["pending", "accepted", "flagged_re_delivery"]:
                    order.assign_vehicle(drone.id)
                    return True
                return False

            elif action_value == 4:  # ASSIGN_TO_HUB
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
        orders = self.global_state.get_all_entities("order").values()

        self._state.set_value('num_orders_total', len(orders))
        self._state.set_value('orders_pending', sum(1 for o in orders if o.status == 'pending'))
        self._state.set_value('orders_in_transit', sum(1 for o in orders if o.status == 'in_transit'))
        self._state.set_value('orders_delivered', sum(1 for o in orders if o.status == 'delivered'))
