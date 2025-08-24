from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import Set, Dimension, MSpace, ESpace
from datetime import timedelta
from mlpro.bf.events import EventManager, Event

class LogisticEntity(System):

    C_NAME = "Entities"
    C_DIS_DIMS = []
    C_EVENT_ENTITY_STATE_CHANGE = "Entity State Change"
    def __init__(self,
                 p_id: int,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes a Vehicle system.
        """
        super().__init__(p_id=p_id,
                         p_name=self.C_NAME,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))
        self.setup_discrete_spaces()

    def setup_discrete_spaces(self):
        for dim in self.C_DIS_DIMS:
            self._state_space.add_dim(Dimension(dim[0],
                                                  "Z",
                                                  dim[1],
                                                  p_boundaries=[0, len(dim[2])] if len(dim[2]) else []))

    def _reset(self, p_seed=None):
        self.setup_discrete_spaces()

    def register_event_handler(self, p_event_id:str, p_event_handler):
        EventManager.register_event_handler(self, p_event_id, p_event_handler)
        self._raise_event(self.C_EVENT_ENTITY_STATE_CHANGE, Event(self))

    def get_state_value_by_dim_name(self, p_dim_name):
        return self._state.get_value(self._state.get_related_set().get_dim_by_name(p_dim_name).get_id())
