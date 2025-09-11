from typing import List, Dict, Any, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State
from mlpro.bf.math import MSpace, Dimension

# Local Imports

from ddls_src.core.basics import LogisticsAction
from ddls_src.actions.base import SimulationActions, ActionType
from ddls_src.core.global_state import GlobalState
from ddls_src.entities import *

# # Forward declarations
# class GlobalState: pass
#
#
# class Order: pass
#
#
# class Truck: pass
#
#
# class Drone: pass
#
#
# class MicroHub: pass


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
        action_ids = [action.id for action in SimulationActions.get_all_actions() if action.handler == handler_name]

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
        if not self.automatic_logic_config.get(SimulationActions.ASSIGN_ORDER_TO_TRUCK, False):
            return

        pending_orders = [o for o in self.global_state.orders.values() if o.status == 'pending']
        idle_trucks = [t for t in self.global_state.trucks.values() if t.status == 'idle']

        if not pending_orders or not idle_trucks:
            return

        order_to_assign = pending_orders[0]
        truck_to_assign = idle_trucks[0]

        print(
            f"  - AUTOMATIC LOGIC (SupplyChainManager): Assigning Order {order_to_assign.get_id()} to Truck {truck_to_assign.get_id()}")
        order_to_assign.assign_vehicle(truck_to_assign.get_id())

    def _process_action(self, p_action: LogisticsAction) -> bool:
        """
        Processes a high-level command related to order management.
        """
        action_id = int(p_action.get_sorted_values()[0])
        action_type = ActionType.get_by_id(action_id)
        action_kwargs = p_action.data

        try:
            if action_type == SimulationActions.CONSOLIDATE_FOR_TRUCK:
                truck_id = action_kwargs['truck_id']
                truck: 'Truck' = self.global_state.get_entity('truck', truck_id)
                if truck and truck.status == 'idle' and len(truck.delivery_orders) > 0:
                    truck.consolidation_confirmed = True
                    self.log(self.C_LOG_TYPE_I, f"Consolidation confirmed for Truck {truck_id}. Starting route.")
                    self.system.network_manager.route_for_assigned_orders(truck_id)
                    return True
                return False

            elif action_type == SimulationActions.CONSOLIDATE_FOR_DRONE:
                drone_id = action_kwargs['drone_id']
                drone: 'Drone' = self.global_state.get_entity('drone', drone_id)
                if drone and drone.status == 'idle' and len(drone.delivery_orders) > 0:
                    drone.consolidation_confirmed = True
                    self.log(self.C_LOG_TYPE_I, f"Consolidation confirmed for Drone {drone_id}. Starting route.")
                    self.system.network_manager.route_for_assigned_orders(drone_id)
                    return True
                return False

            # This block handles all actions that require an order or node_pair.
            if "order" in action_kwargs.keys():
                order_id = action_kwargs['order_id']
                order: 'Order' = self.global_state.get_entity("order", order_id)
            else:
                node_pair = action_kwargs.get("pick_up_drop")
                global_state: GlobalState = self.global_state
                order: Order = global_state.get_order_requests()[node_pair][0]

            if action_type == SimulationActions.ACCEPT_ORDER:
                if order.status == "pending":
                    order.update_status("accepted")
                    return True
                return False

            elif action_type == SimulationActions.CANCEL_ORDER:
                if order.status not in ["delivered", "cancelled"]:
                    if order.assigned_vehicle_id is not None:
                        order.unassign_vehicle()
                    order.update_status("cancelled")
                    return True
                return False

            elif action_type == SimulationActions.ASSIGN_ORDER_TO_TRUCK:
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                assigned = self.assign_order(order, truck)
                return assigned

            elif action_type == SimulationActions.ASSIGN_ORDER_TO_DRONE:
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                assigned = self.assign_order(order, drone)
                return assigned

            elif action_type == SimulationActions.ASSIGN_ORDER_TO_MICRO_HUB:
                hub: 'MicroHub' = self.global_state.get_entity("micro_hub", action_kwargs['micro_hub_id'])
                assigned = self.assign_order(order, hub)
                return assigned

        except KeyError as e:
            self.log(self.C_LOG_TYPE_E, f"Action parameter missing: {e}")
            return False

        return False

    def assign_order(self, p_order:Order, p_entity):
        assigned = False
        if isinstance(p_entity, MicroHub):
            pass
        elif isinstance(p_entity, Truck) or isinstance(p_entity, Drone):
            assigned = p_order.assign_vehicle(p_entity._id)
            assigned = p_entity.assign_orders([p_order]) and assigned
            self.global_state.get_order_requests()[(p_order.get_pickup_node_id(), p_order.get_delivery_node_id())].remove(p_order)
            p_order.raise_state_change_event()
        return assigned


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


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    from pprint import pprint


    # 1. Create Mock Objects for the test
    class MockOrder(System):
        def __init__(self, p_id, status):
            super().__init__(p_id=p_id)
            self.status = status
            self.assigned_vehicle_id = None

        def assign_vehicle(self, vehicle_id):
            self.assigned_vehicle_id = vehicle_id
            self.status = 'assigned'

        def update_status(self, status):
            self.status = status

        @staticmethod
        def setup_spaces(): return None, None


    class MockTruck(System):
        def __init__(self, p_id, status):
            super().__init__(p_id=p_id)
            self.status = status

        @staticmethod
        def setup_spaces(): return None, None


    class MockGlobalState:
        def __init__(self):
            self.orders = {
                0: MockOrder(p_id=0, status='pending'),
                1: MockOrder(p_id=1, status='pending'),
            }
            self.trucks = {
                101: MockTruck(p_id=101, status='idle'),
                102: MockTruck(p_id=102, status='en_route'),
            }
            self.drones = {}
            self.micro_hubs = {}

        def get_entity(self, type, id):
            return getattr(self, type + 's', {}).get(id)

        def get_all_entities(self, type):
            return getattr(self, type + 's', {})


    mock_gs = MockGlobalState()

    print("--- Validating SupplyChainManager ---")

    # 2. Test Manual Action Processing
    print("\n[A] Testing Manual Action Processing...")
    scm_manual = SupplyChainManager(p_id='scm_manual', global_state=mock_gs)
    action_to_process = LogisticsAction(
        p_action_space=scm_manual.get_action_space(),
        p_values=[SimulationActions.ASSIGN_ORDER_TO_TRUCK.id],
        order_id=0,
        truck_id=101
    )
    scm_manual._process_action(action_to_process)
    assert mock_gs.orders[0].status == 'assigned'
    assert mock_gs.orders[0].assigned_vehicle_id == 101
    print("  - PASSED: Correctly assigned Order 0 to Truck 101.")

    # 3. Test Automatic Action Logic
    print("\n[B] Testing Automatic Action Logic...")
    # Reset state for this test
    mock_gs.orders[0].status = 'pending'
    mock_gs.orders[0].assigned_vehicle_id = None

    auto_config = {SimulationActions.ASSIGN_ORDER_TO_TRUCK: True}
    scm_auto = SupplyChainManager(p_id='scm_auto', global_state=mock_gs, p_automatic_logic_config=auto_config)

    # Call simulate_reaction to trigger the automatic logic
    scm_auto._simulate_reaction(p_state=None, p_action=None)

    assert mock_gs.orders[0].status == 'assigned'
    assert mock_gs.orders[0].assigned_vehicle_id == 101
    print("  - PASSED: Correctly auto-assigned pending Order 0 to idle Truck 101.")

    print("\n--- Validation Complete ---")
