from typing import Tuple, Dict, Any
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension
from mlpro.bf.events import Event
from ddls_src.entities.base import LogisticEntity


class Edge(LogisticEntity):
    """
    Represents a connection between two nodes as an MLPro System.
    Edges have dynamic properties like traffic that can be altered via actions.
    """

    C_TYPE = 'Edge'
    C_NAME = 'Edge'
    C_EVENT_ENTITY_STATE_CHANGE = "Entity State Updated"
    C_DIM_ACTIVE = ["ava","Edge Availability",[True, False]]
    C_DIM_LENGTH = ["len","Length",[]]
    C_DIM_TIME_FACTOR = ["xt","Time Factor",[]]
    C_DIS_DIMS = [C_DIM_ACTIVE,
                  C_DIM_LENGTH,
                  C_DIM_TIME_FACTOR]

    def __init__(self,
                 p_id: int,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes an Edge system.

        Parameters:
            p_id (int): Unique identifier for the edge.
            p_name (str): Name of the edge.
            p_visualize (bool): Visualization flag.
            p_logging: Logging level.
            p_kwargs: Additional keyword arguments. Expected keys:
                'start_node_id': int
                'end_node_id': int
                'length': float
                'base_travel_time': float
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        # Edge-specific attributes
        self.start_node_id: int = p_kwargs.get('start_node_id')
        self.end_node_id: int = p_kwargs.get('end_node_id')
        self.length: float = p_kwargs.get('length', 0.0)
        self.base_travel_time: float = p_kwargs.get('base_travel_time', 0.0)

        # Internal dynamic attributes
        self.current_traffic_factor: float = 1.0
        self.is_blocked: bool = False
        self.drone_flight_impact_factor: float = 1.0

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for an Edge system.
        """
        state_space = MSpace()


        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='edge_action',
                                       p_base_set='Z',
                                       p_name_long='Edge Action',
                                       p_boundaries=[0, 5]))
        # 0: Set Traffic Normal (1.0)
        # 1: Set Traffic Medium (1.5)
        # 2: Set Traffic High (2.5)
        # 3: Unblock Edge
        # 4: Block Edge
        # 5: No Action

        return state_space, action_space

    def _reset(self, p_seed=None):
        """
        Resets the edge to its default state.
        """
        self.current_traffic_factor = 1.0
        self.is_blocked = False
        self.drone_flight_impact_factor = 1.0
        self._update_state()

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a discrete flattened action sent to this Edge.
        """
        action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()

        if action_value == 0:
            self.current_traffic_factor = 1.0
        elif action_value == 1:
            self.current_traffic_factor = 1.5
        elif action_value == 2:
            self.current_traffic_factor = 2.5
        elif action_value == 3:
            self.is_blocked = False
        elif action_value == 4:
            self.is_blocked = True
        # Action 5 is a no-op

        self._update_state()
        return True

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Processes an action if provided, and synchronizes the formal MLPro state.
        """
        # FIX: Only process an action if one is actually passed to the method
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        self._update_state()
        return self._state

    def _update_state(self):
        """
        Synchronizes internal attributes with the formal MLPro state object.
        """
        # self._state.set_value('traffic_factor', self.current_traffic_factor)
        # self._state.set_value('is_blocked', 1 if self.is_blocked else 0)
        # self._state.set_value('drone_impact', self.drone_flight_impact_factor)

        self._state.set_value(self._state.get_related_set().get_dim_by_name(self.C_DIM_TIME_FACTOR[0]).get_id(), self.current_traffic_factor)
        self._state.set_value(self._state.get_related_set().get_dim_by_name(self.C_DIM_ACTIVE[0]).get_id(), 1 if self.is_blocked else 0)
        self._raise_event(p_event_id=Edge.C_EVENT_ENTITY_STATE_CHANGE, p_event_object=Event(self))

    # Public methods for getting dynamic travel times
    def get_current_travel_time(self) -> float:
        if self.is_blocked:
            return float('inf')
        return self.base_travel_time * self.current_traffic_factor

    def get_drone_flight_time(self) -> float:
        if self.is_blocked:
            return float('inf')
        return self.base_travel_time * self.drone_flight_impact_factor
