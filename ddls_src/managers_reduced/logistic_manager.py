from datetime import timedelta
from ddls_src.actions.base import SimulationActions, ActionType
from ddls_src.actions.conditional_actions import SimulationActions
from ddls_src.core.basics import LogisticsAction
from ddls_src.core.global_state import GlobalState
from ddls_src.entities import *
from ddls_src.entities.order import PseudoOrder
from mlpro.bf.events import Event
from mlpro.bf.math import MSpace, Dimension
# MLPro Imports
from mlpro.bf.systems import System, State
from typing import List, Dict, Any, Optional


# Local Imports


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


class LogisticManager(System):
    """
    Manages the lifecycle and assignment of orders as an MLPro System.
    Its action space is now dynamically configured from the action blueprint.
    """

    C_TYPE = 'Manager'
    C_NAME = 'Logistic'
    C_EVENT_NEW_ORDER_REQUEST = "New order event"

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=False,
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

        self.custom_log = False
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
        handler_name = "Logistic"
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

        # self._check_and_assign_orders()

        self._update_state()
        return self._state

    def _process_action(self, p_action: LogisticsAction) -> bool:
        """
        Processes a high-level command related to order management.
        """
        event_stack = []
        action_id = int(p_action.get_sorted_values()[0])
        action_type = ActionType.get_by_id(action_id)
        action_kwargs = p_action.data
        active_resource = self.global_state.active_resource

        if action_type == SimulationActions.ASSIGN_ORDER_TO_RESOURCE:
            if "pick_up_drop" not in action_kwargs.keys():
                raise ValueError("Assign action provided without the corresponding node pair. Check "
                                 "your action configuration.")

            n_pair_id = action_kwargs["pick_up_drop"]
            n_pair = self.global_state.node_pairs[n_pair_id]
            o = self.global_state.get_order_requests()[n_pair_id][0]
            if active_resource is None:
                raise ValueError("Assign actions must be masked when there is no active resource selected. Please"
                                 "check your masking logic/constraints.")
            if isinstance(active_resource, Truck) or isinstance(active_resource, Drone):
                self.assign_order(o, active_resource)
            event_stack.extend([n_pair, o])

        elif action_type == SimulationActions.SELECT_TRUCK or action_type == SimulationActions.SELECT_DRONE:
            if "truck_id" in action_kwargs.keys():
                vehicle:LogisticEntity = self.global_state.get_vehicle(action_kwargs["truck_id"])
            elif "drone_id" in action_kwargs.keys():
                vehicle = self.global_state.get_vehicle(action_kwargs["drone_id"])
            else:
                raise ValueError("Select truck or select drone action shall have keywords either"
                                 "truck_id or drone_id respectively in the action kwargs.")
            self.global_state.active_resource = vehicle
            vehicle.raise_state_change_event()
            event_stack.extend([vehicle])

        elif action_type == SimulationActions.SELECT_MICROHUB:
            if "micro_hub_id" in action_kwargs.keys():
                mh = self.global_state.get_vehicle(action_kwargs["micro_hub_id"])
            else:
                raise ValueError("Select micro_hub action shall have keywords either"
                                 "micro_hub_id respectively in the action kwargs.")
            self.global_state.active_resource = mh
            mh.raise_state_change_event()
            event_stack.extend([mh])

        elif action_type == SimulationActions.LOAD_TRUCK_ACTION or action_type == SimulationActions.LOAD_DRONE_ACTION:
            if "truck_id" in action_kwargs.keys():
                vehicle: Vehicle = self.global_state.get_vehicle(action_kwargs["truck_id"])
            elif "drone_id" in action_kwargs.keys():
                vehicle = self.global_state.get_vehicle(action_kwargs["drone_id"])
            else:
                raise ValueError("Load truck or load drone action shall have keywords either"
                                 "truck_id or drone_id respectively in the action kwargs.")
            orders_loaded = vehicle.load_orders_at_node()
            event_stack.extend(orders_loaded)
            event_stack.append(vehicle)

        elif action_type == SimulationActions.UNLOAD_TRUCK_ACTION or action_type == SimulationActions.UNLOAD_DRONE_ACTION:
            if "truck_id" in action_kwargs.keys():
                vehicle: Vehicle = self.global_state.get_vehicle(action_kwargs["truck_id"])
            elif "drone_id" in action_kwargs.keys():
                vehicle = self.global_state.get_vehicle(action_kwargs["drone_id"])
            else:
                raise ValueError("Unload truck or unload drone action shall have keywords either"
                                 "truck_id or drone_id respectively in the action kwargs.")
            orders_unloaded = vehicle.unload_orders_at_node()
            event_stack.extend(orders_unloaded)
            event_stack.append(vehicle)

        elif action_type == SimulationActions.CONSOLIDATE:
            if active_resource is None:
                raise ValueError("Consolidate action shall be masked if there is no Active Resource. Please check"
                                 "your masking logic/Constraints.")
            event_stack.extend([self.global_state.active_resource])
            return self.process_consolidate_action()

        else:
            raise ValueError("Invalid action provided to the logistic manager for processing.")

    def process_consolidate_action(self) -> bool:
        active_resource = self.global_state.active_resource
        self.global_state.active_resource = None
        active_resource.raise_state_change_event()
        if active_resource is None:
            raise ValueError("Consolidate actions shall be masked if there is no Active Resource Selected."
                             "Please check your masking logic/constraints.")
        if isinstance(active_resource, Vehicle):
            active_resource.consolidate_route()
            if self.custom_log:
                print(f"Consolidation completed for vehicle {active_resource.get_id()}.")
        return True

    def assign_order(self, p_order: Order, p_entity):
        assigned = True
        if isinstance(p_entity, MicroHub):
            assigned = p_order.assign_micro_hub(p_entity.id) and assigned
            assigned = p_entity.assign_order(p_order) and assigned
            if assigned:
                pseudo_order1, pseudo_order2 = p_order.create_pseudo_orders(p_entity.get_id())
                self.create_order_requests([pseudo_order1, pseudo_order2])
                if self.custom_log:
                    print(f"Order {p_order.get_id()} assigned to micro_hub {p_entity.get_id()}.")
        elif isinstance(p_entity, Truck) or isinstance(p_entity, Drone):
            assigned = p_order.assign_vehicle(p_entity._id, p_entity)
            assigned = p_entity.assign_orders([p_order]) and assigned
            if assigned:
                if self.custom_log:
                    print(f"Order {p_order.get_id()} assigned to vehicle {p_entity.get_id()}")
        p_order.raise_state_change_event()
        p_entity.raise_state_change_event()
        return assigned

    def create_order_requests(self, p_orders: list):
        self._raise_event(p_event_id=self.C_EVENT_NEW_ORDER_REQUEST,
                          p_event_object=Event(p_raising_object=self,
                                               p_orders = p_orders))

    def _update_state(self):
        """
        Calculates aggregate order statistics and updates the formal state object.
        """
        state_space = self._state.get_related_set()
        orders = self.global_state.get_all_entities_by_type("order").values()

        self._state.set_value(state_space.get_dim_by_name("num_orders_total").get_id(), len(orders))
        self._state.set_value(state_space.get_dim_by_name("orders_pending").get_id(),
                              sum(1 for o in orders if o.status == 'pending'))
        self._state.set_value(state_space.get_dim_by_name("orders_in_transit").get_id(),
                              sum(1 for o in orders if o.status == 'in_transit'))
        self._state.set_value(state_space.get_dim_by_name("orders_delivered").get_id(),
                              sum(1 for o in orders if o.status == 'delivered'))


