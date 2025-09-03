import numpy as np
from mlpro.bf.ml import Model
from ddls_src.core.basics import LogisticsAction
from ddls_src.actions.base import SimulationActions
import random

class DummyAgent(Model):
    """
    A simple, rule-based agent for the logistics simulation.
    Its policy is to always choose the first available valid action.
    """
    C_NAME = 'DummyAgent'

    def __init__(self, p_logging=True):
        self._no_op_idx = -1
        super().__init__(p_ada=False, p_logging=p_logging)

    def compute_action(self, p_state, p_action_mask) -> LogisticsAction:
        """
        Computes an action based on the provided state and action mask.
        """
        valid_actions = np.where(p_action_mask)[0]
        action_to_take_idx = -1

        if len(valid_actions) > 0:
            # action_to_take_idx = valid_actions[0]
            action_to_take_idx = random.choice(valid_actions)
        else:
            # Fallback to NO_OPERATION if no other actions are valid
            # This requires access to the action map, which we'll pass during scenario setup
            action_to_take_idx = self._no_op_idx

        return LogisticsAction(p_action_space=self._action_space, p_values=[action_to_take_idx])

    def _adapt(self, **p_kwargs) -> bool:
        # This is a non-adaptive agent, so we do nothing here.
        return False
