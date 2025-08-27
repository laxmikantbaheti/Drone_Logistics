from typing import Optional, List, Any, Dict
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension
from ddls_src.entities.base import LogisticEntity
from mlpro.bf.events import Event

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
    C_VALID_DELIVERY_STATES = [C_STATUS_PLACED,
                               C_STATUS_ACCEPTED,
                               C_STATUS_ASSIGNED,
                               C_STATUS_EN_ROUTE,
                               C_STATUS_DELIVERED,
                               C_STATUS_FAILED]
    C_DIM_DELIVERY_STATUS = ["delivery", "Delivery Status", C_VALID_DELIVERY_STATES]
    C_DIM_PRIORITY = ["pri", "Priority", []]
    C_DIM_PICKUP_NODE = ["p_node", "Pickup Node", []]
    C_DIM_DELIVERY_NODE = ["d_node", "Delivery Node", []]
    C_DIM_ASSIGNED_VEHICLE = ["veh", "Assigned Vehicle", []]
    C_DIS_DIMS = [C_DIM_DELIVERY_STATUS,
                  C_DIM_PRIORITY,
                  C_DIM_PICKUP_NODE,
                  C_DIM_DELIVERY_NODE,
                  C_DIM_ASSIGNED_VEHICLE]




    def __init__(self,
                 p_pickup_node_id,
                 p_delivery_node_id,
                 p_id: int,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes an Order system.

        Parameters:
            p_id (int): Unique identifier for the order.
            p_name (str): Name of the order.
            p_visualize (bool): Visualization flag.
            p_logging: Logging level.
            p_kwargs: Additional keyword arguments. Expected keys:
                'customer_node_id': int
                'time_received': float
                'SLA_deadline': float
                'priority': int
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

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
        self.delivery_time: Optional[float] = None
        # FIX: Make global_state optional during initialization, defaulting to None
        self.global_state: 'GlobalState' = p_kwargs.get('global_state', None)
        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for an Order system.
        """
        state_space = MSpace()
        # Status: 0=pending, 1=accepted, 2=assigned, 3=in_transit, 4=at_micro_hub, 5=delivered, 6=cancelled, 7=flagged
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

    def _reset(self, p_seed=None):
        """
        Resets the order to its initial 'pending' state.
        """
        self.status = "pending"
        self.assigned_vehicle_id = None
        self.assigned_micro_hub_id = None
        self.delivery_time = None
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Synchronizes the formal MLPro state with the order's internal attributes.
        """
        # FIX: Only process an action if one is actually passed to the method
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        self._update_state()
        return self._state

    def _update_state(self):
        """
        Helper method to synchronize internal attributes with the formal MLPro state object.
        """
        status_map = {
            "pending": 0, "accepted": 1, "assigned": 2, "in_transit": 3,
            "at_micro_hub": 4, "delivered": 5, "cancelled": 6, "flagged_re_delivery": 7
        }
        self._state.set_value(self._state.get_related_set().get_dim_by_name(self.C_DIM_DELIVERY_STATUS[0]).get_id(),
                              status_map.get(self.status, 0))
        self._state.set_value(self._state.get_related_set().get_dim_by_name(self.C_DIM_PRIORITY[0]).get_id(),
                              self.priority)

    # Public methods for managers to call
    def update_status(self, new_status: str):
        self.status = new_status
        if new_status == "delivered":
            # This should be set with the simulation's current time by the manager
            pass
        self._update_state()

    def assign_vehicle(self, vehicle_id: int):
        self.assigned_vehicle_id = vehicle_id
        self.status = "assigned"
        self.update_state_value_by_dim_name(self.C_DIM_ASSIGNED_VEHICLE[0], vehicle_id)
        return True

    def unassign_vehicle(self):
        self.assigned_vehicle_id = None
        if self.status in ["assigned", "in_transit"]:
            self.status = "flagged_re_delivery"
        self._update_state()

    def assign_micro_hub(self, micro_hub_id: int):
        self.assigned_micro_hub_id = micro_hub_id
        self.status = "at_micro_hub"
        self._update_state()

    def get_SLA_remaining(self, current_time: float) -> float:
        return self.SLA_deadline - current_time

    def get_pickup_node_id(self):
        return self.pickup_node_id

    def get_delivery_node_id(self):
        return self.delivery_node_id

    def get_global_state(self):
        return self.global_state
