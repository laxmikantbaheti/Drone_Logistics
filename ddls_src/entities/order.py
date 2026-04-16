from datetime import timedelta
from ddls_src.entities.base import LogisticEntity
from mlpro.bf.events import Event
from mlpro.bf.events import EventManager
from mlpro.bf.math import MSpace, Dimension
# from mlpro.bf import ParamError
# MLPro Imports
from mlpro.bf.systems import System, State, Action
from typing import Optional, List, Any, Dict


class Order(LogisticEntity):
    """
    Represents a customer order as an MLPro System.
    Tracks the lifecycle and assignment of a package for delivery.
    """

    C_TYPE = 'Order'
    C_NAME = 'Order'
    C_STATUS_PLACED = "Placed"
    C_STATUS_ACCEPTED = "Accepted"
    C_STATUS_ASSIGNED = "Assigned"
    C_STATUS_EN_ROUTE = "En Route"
    C_STATUS_DELIVERED = "Delivered"
    C_STATUS_FAILED = "Failed"
    C_STATUS_IN_RELAY = "Order in relay"
    C_VALID_DELIVERY_STATES = [C_STATUS_PLACED,
                               C_STATUS_ACCEPTED,
                               C_STATUS_ASSIGNED,
                               C_STATUS_EN_ROUTE,
                               C_STATUS_DELIVERED,
                               C_STATUS_FAILED,
                               C_STATUS_IN_RELAY]
    C_DIM_DELIVERY_STATUS = ["delivery", "Delivery Status", C_VALID_DELIVERY_STATES]
    C_DIM_PRIORITY = ["pri", "Priority", []]
    C_DIM_PICKUP_NODE = ["p_node", "Pickup Node", []]
    C_DIM_DELIVERY_NODE = ["d_node", "Delivery Node", []]
    C_DIM_ASSIGNED_VEHICLE = ["veh", "Assigned Vehicle", []]
    C_DIM_CURRENT_NODE = ["curn", "Current Node", []]
    C_DIS_DIMS = [C_DIM_DELIVERY_STATUS,
                  C_DIM_PRIORITY,
                  C_DIM_PICKUP_NODE,
                  C_DIM_DELIVERY_NODE,
                  C_DIM_ASSIGNED_VEHICLE,
                  C_DIM_CURRENT_NODE]

    def __init__(self,
                 p_pickup_node_id,
                 p_delivery_node_id,
                 p_id,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=False,
                 **p_kwargs):
        """
        Initializes an Order system.
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        self.successor_orders = []
        self.custom_log = False
        # Order-specific attributes
        self.customer_node_id: int = p_kwargs.get('customer_node_id')
        self.time_received: float = p_kwargs.get('time_received', 0.0)
        self.SLA_deadline: float = p_kwargs.get('SLA_deadline', 0.0)
        self.priority: int = p_kwargs.get('priority', 1)
        self.pickup_node_id = p_pickup_node_id
        self.delivery_node_id = p_delivery_node_id
        # Internal dynamic attributes
        self.status: str = "pending"
        self.assigned_vehicle_id: Optional[int] = None
        self.assigned_micro_hub_id: Optional[int] = None
        self.assigned_micro_hub = None
        self.delivery_time: Optional[float] = None
        # FIX: Make global_state optional during initialization, defaulting to None
        global_state: 'GlobalState' = p_kwargs.get('global_state', None)
        if global_state is not None:
            self.add_global_state(global_state)
        self._state = State(self._state_space)
        self.pseudo_orders: [Order] = []
        self.predecessor_orders = []
        self.mh_assignment_history_ids = []
        self.mh_assignment_history = []
        self.assigned_vehicle = None
        self.carrying_vehicle = None
        if self.global_state is not None:
            self.node_pair = self.global_state.node_pairs[(self.pickup_node_id, self.delivery_node_id)]
        else:
            self.node_pair = None
        self.current_node_id = self.pickup_node_id
        self.location_history = [self.current_node_id]

        # Will be instantiated properly in reset()
        self.state_history = []
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for an Order system.
        """
        state_space = MSpace()
        state_space.add_dim(
            Dimension('w',
                      'R',
                      "Weight"))
        state_space.add_dim(
            Dimension("del_time",
                      "R",
                      "Delivery Window"))

        action_space = MSpace()  # Orders are passive, no actions

        return state_space, action_space

    def log_current_state(self):
        """Captures the current node and time for the history log."""
        current_time = getattr(self.global_state, 'current_time', 0.0) if hasattr(self,
                                                                                  'global_state') and self.global_state else 0.0

        self.state_history.append({
            'time': current_time,
            'status': self.status,
            'actual_node': self.current_node_id,  # This tracks where the order is RIGHT NOW
            'assigned_veh': self.assigned_vehicle_id,
            'carrying_veh': self.carrying_vehicle.get_id() if self.carrying_vehicle else None
        })

    def _reset(self, p_seed=None):
        """
        Resets the order to its initial 'pending' state.
        """
        self.status = "pending"
        self.update_state_value_by_dim_name(self.C_DIM_DELIVERY_STATUS[0], self.C_STATUS_PLACED)
        self.assigned_vehicle_id = None
        self.assigned_micro_hub_id = None
        self.assigned_micro_hub = None
        self.delivery_time = None
        self.pseudo_orders = []
        self._update_state()
        self.current_node_id = self.pickup_node_id
        self.location_history = [self.pickup_node_id]

        # --- NEW: Initialize local history for this specific order ---
        self.state_history = []
        self.log_current_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Synchronizes the formal MLPro state with the order's internal attributes.
        """
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        self._update_state()
        return self._state

    def _update_state(self):
        """
        Helper method to synchronize internal attributes with the formal MLPro state object.
        """
        pass

    # Public methods for managers to call
    def update_status(self, new_status: str):
        self.status = new_status
        if new_status == "delivered":
            pass
        self._update_state()
        self.log_current_state()

    def assign_vehicle(self, vehicle_id: int, vehicle):
        self.assigned_vehicle_id = vehicle_id
        self.assigned_vehicle = vehicle
        self.status = "assigned"
        self.update_state_value_by_dim_name([self.C_DIM_ASSIGNED_VEHICLE[0], self.C_DIM_DELIVERY_STATUS[0]],
                                            [vehicle_id, self.C_STATUS_ASSIGNED])

        self.log_current_state()
        return True

    def assign_micro_hub(self, micro_hub_id: int):
        self.assigned_micro_hub_id = micro_hub_id
        self.assigned_micro_hub = self.global_state.micro_hubs[micro_hub_id]
        self.status = "at_micro_hub"
        self.update_state_value_by_dim_name([self.C_DIM_ASSIGNED_VEHICLE[0], self.C_DIM_DELIVERY_STATUS[0]],
                                            [micro_hub_id, self.C_STATUS_ASSIGNED])

        self._update_state()
        self.log_current_state()
        return True

    # def update_state_value_by_dim_name(self, p_dim_name, p_value):
    #     """Overrides the base method to catch framework-level state/node changes."""
    #     super().update_state_value_by_dim_name(p_dim_name, p_value)
    #
    #     dim_names = p_dim_name if isinstance(p_dim_name, list) else [p_dim_name]
    #
    #     if hasattr(self, 'C_DIM_DELIVERY_STATUS') and self.C_DIM_DELIVERY_STATUS[0] in dim_names:
    #         if isinstance(p_dim_name, list) and isinstance(p_value, list):
    #             idx = p_dim_name.index(self.C_DIM_DELIVERY_STATUS[0])
    #             self.status = str(p_value[idx])
    #         else:
    #             self.status = str(p_value)
    #
    #     # Trigger the smart logger. It will automatically filter out duplicates!
    #     self.log_current_state()

    def unassign_vehicle(self):
        self.assigned_vehicle_id = None
        if self.status in ["assigned", "in_transit"]:
            self.status = "flagged_re_delivery"

        self._update_state()
        self.log_current_state()

    def get_assigned_vehicle_id(self):
        if self.assigned_vehicle_id:
            return self.assigned_vehicle_id

    def set_enroute(self):
        """Triggered when the order is picked up."""
        self.carrying_vehicle = self.assigned_vehicle
        self.assigned_vehicle_id = None
        self.assigned_vehicle = None
        self.update_state_value_by_dim_name(self.C_DIM_DELIVERY_STATUS[0], self.C_STATUS_EN_ROUTE)
        self.status = "En Route"

        # Capture the actual node at the moment of pickup
        self.log_current_state()

    def get_SLA_remaining(self, current_time: float) -> float:
        return self.SLA_deadline - current_time

    def get_pickup_node_id(self):
        return self.pickup_node_id

    def get_delivery_node_id(self):
        return self.delivery_node_id

    def get_global_state(self):
        return self.global_state

    def __repr__(self):
        return f"Order {self.get_id()} - ({self.pickup_node_id},{self.delivery_node_id}) - {self.get_state_value_by_dim_name(self.C_DIM_DELIVERY_STATUS[0])} - {self.assigned_vehicle_id} - {self.assigned_micro_hub_id}"

    def __str__(self):
        return self.__repr__()

    def change_delivery_status(self, status):
        if status not in self.C_VALID_DELIVERY_STATES:
            raise ValueError("Invalid delivery status provided for Order entity.")

        self.update_state_value_by_dim_name(self.C_DIM_DELIVERY_STATUS[0], status)
        self.log_current_state()

    def set_delivered(self):
        self.assigned_vehicle_id = None
        self.assigned_vehicle = None
        self.carrying_vehicle = None
        self.update_state_value_by_dim_name(self.C_DIM_DELIVERY_STATUS[0], self.C_STATUS_DELIVERED)
        self.status = "Delivered"

        self.log_current_state()

        if isinstance(self, PseudoOrder):
            self._raise_event(PseudoOrder.C_EVENT_ORDER_DELIVERED, Event(p_raising_object=self))

    def handle_pseudo_delivery(self, p_event_id, p_event_object):
        delivered = True
        for ordr in self.pseudo_orders:
            if ordr.get_state_value_by_dim_name(self.C_DIM_DELIVERY_STATUS[0]) == self.C_STATUS_DELIVERED:
                delivered = True and delivered
            else:
                delivered = False
        if delivered:
            if self.custom_log:
                print(f"Collaborative order {self.get_id()} is delivered.")
            self.set_delivered()

    def create_pseudo_orders(self, hub_id):
        pseudo_order_1 = PseudoOrder(
            p_id=str(self.get_id()) + "_1",
            p_pickup_node_id=self.get_pickup_node_id(),
            p_delivery_node_id=hub_id,
            global_state=self.global_state,
            p_parent_order=self,
            p_leg=1
        )
        pseudo_order_2 = PseudoOrder(
            p_id=str(self.get_id()) + "_2",
            p_pickup_node_id=hub_id,
            p_delivery_node_id=self.get_delivery_node_id(),
            global_state=self.global_state,
            p_parent_order=self,
            p_leg=2
        )

        self.pseudo_orders.extend([pseudo_order_1, pseudo_order_2])

        # --- GRAPH SURGERY: Transitive Update ---
        pseudo_order_1.predecessor_orders.extend(self.predecessor_orders)
        pseudo_order_1.successor_orders.extend(self.successor_orders)

        pseudo_order_2.predecessor_orders.extend(self.predecessor_orders)
        pseudo_order_2.successor_orders.extend(self.successor_orders)

        pseudo_order_1.successor_orders.append(pseudo_order_2)
        pseudo_order_2.predecessor_orders.append(pseudo_order_1)

        for pred in self.predecessor_orders:
            if pseudo_order_1 not in pred.successor_orders:
                pred.successor_orders.append(pseudo_order_1)
            if pseudo_order_2 not in pred.successor_orders:
                pred.successor_orders.append(pseudo_order_2)

        for succ in self.successor_orders:
            if pseudo_order_1 not in succ.predecessor_orders:
                succ.predecessor_orders.append(pseudo_order_1)
            if pseudo_order_2 not in succ.predecessor_orders:
                succ.predecessor_orders.append(pseudo_order_2)

            if succ.node_pair is not None:
                succ.node_pair.raise_state_change_event()

        self.predecessor_orders = []
        self.successor_orders = []

        return [pseudo_order_1, pseudo_order_2]

    def check_order_precedence(self):
        predecessor_orders: [Order] = self.predecessor_orders
        if not len(predecessor_orders):
            return True

        else:
            precedence_satisfied = True
            for ordr in predecessor_orders:
                if isinstance(ordr, Order):
                    if ordr.get_state_value_by_dim_name(ordr.C_DIM_DELIVERY_STATUS[0]) == ordr.C_STATUS_DELIVERED:
                        precedence_satisfied = True and precedence_satisfied
                    else:
                        precedence_satisfied = False
            return precedence_satisfied

    def register_event_handler_for_constraints(self, p_event_id: str, p_event_handler):
        super().register_event_handler_for_constraints(p_event_id, p_event_handler)
        self.node_pair.register_event_handler_for_constraints(p_event_id, p_event_handler)

    def raise_state_change_event(self):
        super().raise_state_change_event()
        if self.node_pair is not None:
            self.node_pair.raise_state_change_event()

        for successor in self.successor_orders:
            if successor.node_pair is not None:
                successor.node_pair.raise_state_change_event()
            super(Order, successor).raise_state_change_event()

        for predecessor in self.predecessor_orders:
            if predecessor.node_pair is not None:
                predecessor.node_pair.raise_state_change_event()
            super(Order, predecessor).raise_state_change_event()

    def add_global_state(self, global_state):
        self.global_state = global_state
        self.node_pair = global_state.node_pairs[(self.pickup_node_id, self.delivery_node_id)]
        print("Check here")


class PseudoOrder(Order):
    C_TYPE = "Pseudo Order"
    C_EVENT_ORDER_DELIVERED = "Pseudo Order Delivered"

    def __init__(self,
                 p_pickup_node_id,
                 p_delivery_node_id,
                 p_parent_order,
                 p_id,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=False,
                 p_leg=None,
                 **p_kwargs):

        Order.__init__(self,
                       p_pickup_node_id=p_pickup_node_id,
                       p_delivery_node_id=p_delivery_node_id,
                       p_id=p_id,
                       p_name=p_name,
                       p_visualize=p_visualize,
                       p_logging=p_logging,
                       **p_kwargs)
        if p_leg is None:
            raise ParamError("Please provide the number of leg this pseudo order represents.")
        self.p_leg = p_leg
        self.parent_order = p_parent_order

        self.register_event_handler(self.C_EVENT_ORDER_DELIVERED,
                                    self.parent_order.handle_pseudo_delivery)

        self.mh_assignment_history_ids.extend(
            [self.parent_order.assigned_micro_hub_id] + self.parent_order.mh_assignment_history_ids)
        self.mh_assignment_history.extend(
            [self.parent_order.assigned_micro_hub] + self.parent_order.mh_assignment_history)

        self.mh_assignment_history_ids.extend(
            [ordr.assigned_micro_hub_id for ordr in self.parent_order.predecessor_orders if
             ordr.assigned_micro_hub_id is not None])
        self.mh_assignment_history_ids.extend(
            [ordr.assigned_micro_hub for ordr in self.parent_order.predecessor_orders if
             ordr.assigned_micro_hub is not None])

        if self.custom_log:
            print(self, self.mh_assignment_history_ids)

        self.reset()

    def reset(self, p_seed=None) -> None:
        Order.reset(self, p_seed)


class NodePair(LogisticEntity):
    C_TYPE = "Node Pair"
    C_NAME = "Node Pair"
    C_EVENT_ASSIGNABILITY = "Event Order Request Updated"

    def __init__(self, p_pickup_node_id, p_delivery_node_id, p_parent_order=None, p_custom_log=False, **p_kwargs):
        self.p_pickup_node_id = p_pickup_node_id
        self.p_delivery_node_id = p_delivery_node_id
        LogisticEntity.__init__(self, p_id=(self.p_pickup_node_id, self.p_delivery_node_id), p_custom_log=p_custom_log,
                                **p_kwargs)
        self.get_id()

    def __repr__(self):
        return f"({self.p_pickup_node_id}, {self.p_delivery_node_id})"