## -------------------------------------------------------------------------------------------------
## -- Project : --MLPro - A Synoptic Framework for Standardized Machine Learning Tasks--
## -- Package : mlpro-logistics
## -- Module  : ddls_simulator.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-24  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-08)

This module provides the base implementations for the vechilc objects in the MLPro Logistics Framework.
"""
from logging import Logger

from mlpro.bf.systems import *
from typing import Tuple, List





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DDLS(MultiSystem):
    """

    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name: str = None,
                 p_id = None,
                 p_range_max=Async.C_RANGE_NONE,
                 p_autorun = Task.C_AUTORUN_NONE,
                 p_class_shared = SystemShared,
                 p_mode=Mode.C_MODE_SIM,
                 p_latency: timedelta = None,
                 p_t_step:timedelta = None,
                 p_fct_strans: FctSTrans = None,
                 p_fct_success: FctSuccess = None,
                 p_fct_broken: FctBroken = None,
                 p_mujoco_file=None,
                 p_frame_skip: int = 1,
                 p_state_mapping=None,
                 p_action_mapping=None,
                 p_camera_conf: tuple = (None, None, None),
                 p_visualize: bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        MultiSystem.__init__(self,
                 p_name = p_name,
                 p_id = p_id,
                 p_range_max = p_range_max,
                 p_autorun = p_autorun,
                 p_class_shared = p_class_shared,
                 p_mode = p_mode,
                 p_latency = p_latency,
                 p_t_step = p_t_step,
                 p_fct_strans = p_fct_strans,
                 p_fct_success = p_fct_success,
                 p_fct_broken = p_fct_broken,
                 p_mujoco_file=p_mujoco_file,
                 p_frame_skip = p_frame_skip,
                 p_state_mapping = p_state_mapping,
                 p_action_mapping = p_action_mapping,
                 p_camera_conf = p_camera_conf,
                 p_visualize = p_visualize,
                 p_logging = p_logging,
                 **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State = None, p_action: Action = None, p_t_step : timedelta = None) -> State:
        """
        Simulates the multisystem, based on the action and time step.

        Parameters
        ----------
        p_state: State
            State of the system.

        p_action: Action.
            Action provided externally for the simulation of the system.

        Returns
        -------
        current_state: State
            The new state of the system after simulation.

        """

        # 1. Register the MultiSystem in the SO, as it is not yet registered, unlike subsystems are
        # registered in the add system call.

        if not self._registered_on_so:
            self._registered_on_so = self.get_so().register_system(p_sys_id=self.get_id(),
                                                                   p_state_space=self.get_state_space(),
                                                                   p_action_space = self.get_action_space())

        # Calculate the greatest possible timestep
        # if self._t_step is None:
        #     ts_list = []
        #     for id in self.get_subsystem_ids():
        #         sys_ts = self.get_subsystem(id)._t_step
        #         if sys_ts is not None:
        #             ts_list.append(self.get_subsystem(id)._t_step)




        # Recommend using Time() instead of using timedelta

        # 2. Get SO
        so = self.get_so()

        # 3. Forward the input action to the corresponding systems
        so._map_values(p_action=p_action)

        # 4. Run the workflow
        self.run(p_action = self.get_so().get_actions(), p_t_step = self._t_step)

        # 5. Return the new state at current timestep
        return so.get_state(p_sys_id=self.get_id())



    @staticmethod
    def setup_spaces():
        pass


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step:timedelta = None) -> State:


        # 1. Check for new order requests from the Supply Chain Management
        #     (Internally in Supply Chain Management) Process the order request

        # 2. Check for action masks
        #     If there are available actions, the particular agent is triggered for taking a decision
        #     Subsequent compute action methods are called
        #     And these actions are then transferred to the system

        # 3. Execute the decision actions on the simulator
        #     The corresponding process action is called on the

        pass


