from mlpro.bf.systems.basics import Action
from mlpro.bf.math.basics import Set
from mlpro.bf.streams.basics import TStampType
import numpy as np

class LogisticsAction(Action):

    def __init__( self,
                  p_agent_id = 0,
                  p_action_space : Set = None,
                  p_values: np.ndarray = None,
                  p_tstamp : TStampType = None,
                  **p_kwargs):
        """

        :param p_agent_id:
        :param p_action_space:
        :param p_values:
        :param p_tstamp:
        :param p_kwargs:
        """
        Action.__init__(self,
                        p_action_space=p_action_space,
                        p_values=p_values)
        

        self.data = p_kwargs.copy()

    def get_sorted_values_with_data(self) -> tuple:
        """

        :return:
        """
        return self.get_sorted_values(), self.data

    # def __repr__(self):
    #     return None