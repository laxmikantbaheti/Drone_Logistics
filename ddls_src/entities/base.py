from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import Set, Dimension, MSpace, ESpace
from datetime import timedelta
from mlpro.bf.events import EventManager, Event

class LogisticEntity(System):

    C_NAME = "Entities"
    C_DIS_DIMS = []
    C_EVENT_ENTITY_STATE_CHANGE = "Entity State Change"
    def __init__(self,
                 p_id,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=System.C_LOG_NOTHING,
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
        self.setup_event_string()
        self.data_storage = {}

    def setup_discrete_spaces(self):
        for dim in self.C_DIS_DIMS:
            self._state_space.add_dim(Dimension(dim[0],
                                                  "Z",
                                                  dim[1],
                                                  p_boundaries=[0, len(dim[2])] if len(dim[2]) else []))

    def _reset(self, p_seed=None):
        self.setup_discrete_spaces()

    def register_event_handler_for_constraints(self, p_event_id:str, p_event_handler):
        EventManager.register_event_handler(self, self.C_EVENT_ENTITY_STATE_CHANGE, p_event_handler)
        self._raise_event(self.C_EVENT_ENTITY_STATE_CHANGE, Event(self))

    def get_state_value_by_dim_name(self, p_dim_name):
        return self._state.get_value(self._state.get_related_set().get_dim_by_name(p_dim_name).get_id())

    def update_state_value_by_dim_name(self, p_dim_name, p_value):
        dim = self.get_state_space().get_dim_by_name(p_dim_name)
        self.log(self.C_LOG_TYPE_S, f"{dim.get_name_long()} updated.")
        print(f"{self.C_NAME}{self.get_id()} - {dim.get_name_long()} updated to {p_value}.")
        self._state.set_value(dim.get_id(), p_value)

    def raise_state_change_event(self):
        self._raise_event(self.C_EVENT_ENTITY_STATE_CHANGE, Event(self))

    def setup_event_string(self):
        self.C_EVENT_ENTITY_STATE_CHANGE = f"{self.C_NAME} - {self._id}: State Change Event"

    def save_data(self, p_key, p_value, p_frame=None):
        if not isinstance(p_value, list):
            p_value = [p_value]
        if p_frame is None:
            if p_key in self.data_storage.keys():
                self.data_storage[p_key].append(p_value)
            else:
                self.data_storage[p_key] = [p_value]

        else:
            if p_frame in self.data_storage.keys():
                if p_key in self.data_storage[p_frame].keys():
                    self.data_storage[p_frame][p_key].append(p_value)
                else:
                    self.data_storage[p_frame][p_key] = [p_value]

            else:
                self.data_storage[p_frame] = {}
                self.data_storage[p_frame][p_key] = [p_value]
