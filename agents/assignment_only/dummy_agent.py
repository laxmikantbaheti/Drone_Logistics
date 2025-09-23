import numpy as np
from mlpro.bf.ml import Model
from ddls_src.core.basics import LogisticsAction
from ddls_src.actions.action_enums import SimulationAction


class AssignmentAgent(Model):
    """
    A simple, rule-based agent for the assignment-only research design.
    Its policy is to always choose the first available valid assignment action.
    """
    C_NAME = 'AssignmentAgent'

    def __init__(self, p_logging=False):
        super().__init__(p_ada=False, p_logging=p_logging)
        self._action_space = None
        self._no_op_idx = -1

    def compute_action(self, p_state, p_action_mask) -> LogisticsAction:
        """
        Computes an assignment action based on the provided action mask.
        """
        valid_actions = np.where(p_action_mask)[0]
        action_to_take_idx = self._no_op_idx

        if len(valid_actions) > 0:
            action_to_take_idx = valid_actions[0]

        return LogisticsAction(p_action_space=self._action_space, p_values=[action_to_take_idx])

    def _adapt(self, **p_kwargs) -> bool:
        # This is a non-adaptive agent.
        return False
