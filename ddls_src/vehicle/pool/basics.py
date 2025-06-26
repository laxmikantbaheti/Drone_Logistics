## -------------------------------------------------------------------------------------------------
## -- Project : --MLPro - A Synoptic Framework for Standardized Machine Learning Tasks--
## -- Package : mlpro-logistics.vehicle.pool
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-24  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-08)

This module provides the base implementations for pool of the vehicle objects in the MLPro Logistics
Framework.
"""
from logging import Logger

from mlpro.bf.systems import *
from typing import Tuple, List

from mlpro.sl.examples.howto_sl_afct_100_train_and_reload_pytorch_mlp import state_space, action_space
from scipy.sparse import dia_array

from ddls_src.vehicle.basics import Vehicle, VehicleConfiguration





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Drone(Vehicle):


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_id = None,
                 p_name : str =None,
                 p_range_max : int = Async.C_RANGE_NONE,
                 p_autorun = Task.C_AUTORUN_NONE,
                 p_class_shared = None,
                 p_mode = Mode.C_MODE_SIM,
                 p_latency : timedelta = None,
                 p_t_step : timedelta = None,
                 p_vehicle_type = Vehicle.C_VEHICLE_TYPE_DRONE,
                 p_origin_location:Tuple[float, float] = (0, 0),
                 p_current_location:Tuple[float, float] = (0,0),
                 p_vehicle_configuration:VehicleConfiguration = None,
                 p_fct_strans : FctSTrans = None,
                 p_fct_success : FctSuccess = None,
                 p_fct_broken : FctBroken = None,
                 p_mujoco_file = None,
                 p_frame_skip : int = 1,
                 p_state_mapping = None,
                 p_action_mapping = None,
                 p_camera_conf : tuple = (None, None, None),
                 p_visualize : bool = False,
                 p_logging = Log.C_LOG_ALL,
                 **p_kwargs
                 ):

        Vehicle.__init__(self,
                 p_id = p_id,
                 p_name = p_name,
                 p_range_max = p_range_max,
                 p_autorun = p_autorun,
                 p_class_shared = p_class_shared,
                 p_mode = p_mode,
                 p_latency = p_latency,
                 p_t_step = p_t_step,
                 p_vehicle_type = p_vehicle_type,
                 p_origin_location = p_origin_location,
                 p_current_location = p_current_location,
                 p_vehicle_configuration = p_vehicle_configuration,
                 p_fct_strans = p_fct_strans,
                 p_fct_success = p_fct_success,
                 p_fct_broken = p_fct_broken,
                 p_mujoco_file = p_mujoco_file,
                 p_frame_skip = p_frame_skip,
                 p_state_mapping = p_state_mapping,
                 p_action_mapping = p_action_mapping,
                 p_camera_conf = p_camera_conf,
                 p_visualize = p_visualize,
                 p_logging = p_logging,
                 **p_kwargs)


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        """

        :return:
        """
        d_state_space, d_action_space = Vehicle.setup_spaces()

        d_state_space.add_dimension(Dimension(p_name_long='Battery Status', p_name_short='f',
                                              p_base_set=Dimension.C_BASE_SET_R,
                                              p_description='The charge level of the drone'))

        return d_state_space, d_action_space


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step:timedelta = None) -> State:
        """

        :param p_state:
        :param p_action:
        :param p_step:
        :return:
        """

        state_values = p_state.get_values() # [Location, next_node, charge_state]

        # Update the position of the vehicle based on the current assigned edge, direction and speed





