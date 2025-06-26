## -------------------------------------------------------------------------------------------------
## -- Project : --MLPro - A Synoptic Framework for Standardized Machine Learning Tasks--
## -- Package : mlpro-logistics.vehicle
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-05-22  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-08)

This module provides the base implementations for the vechilc objects in the MLPro Logistics Framework.
"""
from logging import Logger

from mlpro.bf.systems import *
from typing import Tuple, List

from pandas import period_range

from ddls_src.geographic.basics import Edge, Node
from shapely import LineString




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VehicleConfiguration(Log, ScientificObject):

    C_VEHICLE_SPEED = "Speed"
    C_VEHICLE_RANGE = "Range"
    C_VEHICLE_CAPACITY = "Capacity"


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_speed = 30,
                 p_range = 50,
                 p_capacity = 10,
                 p_logging = Log.C_LOG_NOTHING):

        self.speed = p_speed
        self.range = p_range
        self.capacity = p_capacity

        Log.__init__(self, p_logging = p_logging)




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Vehicle(System):
    """

    """
    C_NAME = 'Vhicle'

    C_STATE_IDOL = 0
    C_STATE_IN_TRANSIT = 1

    C_ACTION_MOVE = 0
    C_ACTION_ASSIGN_EDGE = 1
    C_ACTION_ASSIGN_ROUTE = 2
    C_ACTION_DELIVER = -1

    C_VEHICLE_TYPE = None
    C_VEHICLE_TYPE_DRONE = 0
    C_VEHICLE_TYPE_TRUCK = 1

    C_LOGISTIC_MODE = 'Process Mode' # To decide whether we want to simulate the vehicle, just for time or also for
                                     # continuous (pseudo-continuous/step-discrete) locations
    C_LOGISTIC_MODE_TIME = 'Time'
    C_LOGISTIC_MODE_LOC = "Location"

    C_EVENT_EDGE_NODE = "Reached terminal ddge node"
    C_FINAL_NODE = "Reached Final Node of the Route"


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
                 p_vehicle_type = C_VEHICLE_TYPE_DRONE,
                 p_origin_location:Tuple[float, float] = (0, 0),
                 p_current_location:Tuple[float, float] = (0,0),
                 p_vehicle_configuration:VehicleConfiguration = None,
                 p_parent_clock = None,
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
                 **p_kwargs):

        """

        :param p_id:
        :param p_name:
        :param p_range_max:
        :param p_autorun:
        :param p_class_shared:
        :param p_mode:
        :param p_latency:
        :param p_t_step:
        :param p_vehicle_type:
        :param p_origin_location:
        :param p_current_location:
        :param p_vehicle_configuration:
        :param p_fct_strans:
        :param p_fct_success:
        :param p_fct_broken:
        :param p_mujoco_file:
        :param p_frame_skip:
        :param p_state_mapping:
        :param p_action_mapping:
        :param p_camera_conf:
        :param p_visualize:
        :param p_logging:
        :param p_kwargs:
        """

        self.edge_history = []
        self.origin_location = p_origin_location
        self.vehicle_type = p_vehicle_type
        self.current_location = p_current_location
        self.config = p_vehicle_configuration

        System.__init__(self,
                        p_id = p_id,
                        p_name =p_name,
                        p_range_max = p_range_max,
                        p_autorun = p_autorun,
                        p_class_shared = p_class_shared,
                        p_mode = p_mode,
                        p_latency = p_latency,
                        p_t_step = p_t_step,
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


        self.current_edge = None
        self.current_assignment = None
        self.current_order = None
        self.current_speed = None
        self.current_movement_vector = None
        self.current_waypoints = []
        self.current_waypoint_index = None
        self.current_route_final_node = None


        self.next_route_node = None
        self.next_trip_node = None
        self.next_edge = None


        self.location_history = []
        self._is_available = True
        self.remaining_range = None

        self.clock = None
        self.next_availability = None


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces() -> (ESpace,ESpace):
        """
        This method set's up the state and action space of the System.

        Returns
        -------
        state_space, action_space

        """
        action_space = ESpace()
        state_space = ESpace()


        state_space.add_dim(Dimension(p_name_long='Location', p_name_short='x,y', p_base_set=Dimension.C_BASE_SET_R,
                                      p_description='Current Co-ordinates of the drone'))
        state_space.add_dim(Dimension(p_name_long="Next Node", p_name_short="n_next",
                                      p_description="The next node to be visited by the drone"))

        action_space.add_dim(Dimension(p_name_long='vehicle instruction', p_name_short='v_i/p',
                                      p_description='Next Action to the Drones'))

        # TODO : setup the ranges of the vehicles based on the capacity
        # Range is to be added later to the state space, when conducting experiments for charge considerations

        return state_space, action_space


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed = None):
        """

        Parameters
        ----------
        p_seed

        Returns
        -------

        """
        state_values = random.randint(0,1)
        state = State(self.get_state_space())
        state.set_values([state_values])
        self._set_state(state)
        # TODO: Setup the logic for handling the remaining range, specific to the vehicle type
        self.current_range = self.setup_range(p_vehicle_config = self.config)


## -------------------------------------------------------------------------------------------------
    def assign_order(self, p_order):
        """

        :param p_order:
        :return:
        """
        self.current_order = p_order


## -------------------------------------------------------------------------------------------------
    def assign_package(self, p_assignment):
        """

        :param p_assignment:
        :return:
        """
        self.current_assignment = p_assignment


## -------------------------------------------------------------------------------------------------
    def update_speed(self, p_speed):
        """

        :param p_speed:
        :return:
        """
        self.current_speed = p_speed


## -------------------------------------------------------------------------------------------------
    def update_current_location(self, p_new_location:Tuple[float, float]):
        """

        :param p_new_location:
        :return:
        """

        self.location_history.append((self.current_location[0], self.current_location[1]))
        self.current_location = p_new_location


## -------------------------------------------------------------------------------------------------
    def get_current_location(self):
        """

        :return:
        """

        return self.current_location


## -------------------------------------------------------------------------------------------------
    def get_current_speed(self):
        """

        :return:
        """
        return self.current_speed


## -------------------------------------------------------------------------------------------------
    def get_current_order(self):
        """

        :return:
        """
        return self.current_order


## -------------------------------------------------------------------------------------------------
    def get_current_assignment(self):
        """

        :return:
        """
        return self.current_location


## -------------------------------------------------------------------------------------------------
    def set_availability(self, p_available:bool):
        """

        :param p_available:
        :return:
        """

        self._is_available = p_available


## -------------------------------------------------------------------------------------------------
    def get_availability(self):
        """

        :return:
        """

        return self._is_available


## -------------------------------------------------------------------------------------------------
    def update_movement_vector(self, p_vector):
        """

        :return:
        """

        self.current_movement_vector = p_vector


## -------------------------------------------------------------------------------------------------
    def set_busy_for_time(self, p_time):
        """

        :param p_time:
        :return:
        """
        # TODO setup the availability update logic in case of time based delivery
        # Logic here
        # ...
        # ...

## -------------------------------------------------------------------------------------------------
    def calculate_remaining_range(self):
        """

        :return:
        """

        # Logic Here
        # ...
        # ...
        return self.remaining_range


## -------------------------------------------------------------------------------------------------
    def setup_range(self, p_vehicle_config:VehicleConfiguration):
        """

        :param p_vehicle_config:
        :return:
        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_logistic_mode(self):
        """

        :return:
        """
        return self.C_LOGISTIC_MODE

## -------------------------------------------------------------------------------------------------
    def get_waypoints(self, p_edge_geometry: LineString,
                      p_speed: float,
                      p_time_step: float  # Assuming "size of steps" is a time duration
                      ) -> list[tuple[float, float]]:
        """

        :param p_edge_geometry:
        :param p_speed:
        :param p_time_step:
        :return:
        """


        if not isinstance(p_edge_geometry, LineString):
            raise TypeError("Input 'line' must be a shapely.geometry.LineString.")

        if p_edge_geometry.is_empty:
            return []

        # Get the start point coordinates. line.interpolate(0) is robust.
        start_point_geom = p_edge_geometry.interpolate(0)
        start_point_coords = (start_point_geom.x, start_point_geom.y)

        line_len = p_edge_geometry.length
        if line_len == 0:  # Line is effectively a single point
            return [start_point_coords]

        # Validate speed and time_step for movement
        if p_speed < 0:
            raise ValueError("Speed cannot be negative.")
        # If speed is positive, time_step must also be positive to define a valid distance_per_step > 0
        if p_speed > 0 and p_time_step <= 0:
            raise ValueError("Time step must be positive if speed is greater than 0.")

        # If no effective movement (speed is 0, or time_step is 0 leading to 0 distance)
        if p_speed == 0 or p_time_step == 0:
            return [start_point_coords]

        distance_per_step = p_speed * p_time_step

        points = []
        current_distance = 0.0

            # Loop to add points at calculated intervals.
            # Shapely's line.interpolate(d) will return the end point if d > line_len.
        while True:
            point_geom = p_edge_geometry.interpolate(current_distance)
            points.append((point_geom.x, point_geom.y))

            # Check if we have processed the point at or beyond the line's length
            # A small tolerance (epsilon) can be used for floating point comparison with line_len
            epsilon = 1e-9  # Small tolerance for float comparisons
            if current_distance >= line_len - epsilon:
                break  # Exit loop once the end point (or beyond) is processed and added

            current_distance += distance_per_step

            # Ensure that if the next step would slightly overshoot due to floating point arithmetic,
            # but is meant to be the end, we effectively treat it as such for the *next* iteration.
            # This is implicitly handled by `current_distance >= line_len - epsilon` check above
            # if current_distance becomes very close to line_len.
            # If current_distance after increment is > line_len, interpolate() will clamp to end.

        return points


## -------------------------------------------------------------------------------------------------
    def move_distance(self):
        """

        :return:
        """
        if len(self.current_waypoints) == 0:
            self.current_waypoints = self.get_waypoints(p_edge_geometry=self.current_edge.get_geometry(),
                                                p_speed=self.get_current_speed(),
                                                p_time_step=self._t_step.total_seconds())
            self.current_waypoint_index = 0

        self.location_history.append(self.current_location)
        self.current_location = self.current_location[self.current_waypoint_index]
        self.current_waypoint_index += 1

        # Check if the vehicle is at a terminal nod
        if self.current_location == self.current_waypoints[-1]:
            self._raise_event(p_event_id=Vehicle.C_EVENT_EDGE_NODE,p_event_object=Event(p_raising_object=self))
            self.edge_history.append(self.current_edge)
            self.current_edge = None

        # Check if we reached the final point of the route
        if self.current_location == self.current_route_final_node:
            self._raise_event(p_event_id=Vehicle.C_EVENT_FINISHED, p_event_object=Event(p_raising_object=self))
            self.current_route = None


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step: timedelta = None) -> State:
        """

        Parameters
        ----------
        p_state
        p_action
        p_step

        Returns
        -------

        """
        action_id, action_value = p_action.get_sorted_values()[0]

        if action_id == Vehicle.C_ACTION_ASSIGN_ROUTE:
            if not isinstance(action_value, List):
                raise Error("Please provide a list of nodes when assigning an edge")
            self.route_nodes = action_value
            self.current_route_final_node = self.route_nodes[-1].get_coords()

        if action_id == Vehicle.C_ACTION_ASSIGN_EDGE:
            if not isinstance(action_value, Edge):
                raise Error("Edge not provided for action \"Assign Edge\"")
            self.current_edge = action_value

        if action_id == Vehicle.C_ACTION_MOVE:
            self.move_distance()

        state_values = list[*self.current_location, self.next_trip_node]

        state = State(ESpace())

        state.set_values(state_values)

        return state


