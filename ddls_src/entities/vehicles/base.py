# In ddls_src/entities/vehicles/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional, Set
from datetime import timedelta

from mlpro.bf.exceptions import ParamError
# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension

from ddls_src.actions.base import SimulationActions
# from ddls_src.actions.action_mapping import truck_id
from ddls_src.actions.base import SimulationActions
from ddls_src.core.basics import LogisticsAction
from ddls_src.entities.base import LogisticEntity
from ddls_src.actions.base import SimulationActions, ActionType
from ddls_src.entities.order import Order


# Forward declaration for NetworkManager
class NetworkManager:
    pass


class Vehicle(LogisticEntity, ABC):
    """
    Abstract base class for all vehicles, refactored as an MLPro System.
    It now explicitly stores the list of nodes in its current route for
    efficient, real-time constraint checking.
    """

    C_TYPE = 'Vehicle'
    C_NAME = '???'

    C_TRIP_STATE_IDLE = "Idle"
    C_TRIP_STATE_EN_ROUTE = "En Route"
    C_TRIP_STATE_HALT = "Halted"
    C_TRIP_STATE_LOADING = "Loading"
    C_TRIP_STATE_UNLOADING = "Unloading"
    C_VALID_TRIP_STATES = [C_TRIP_STATE_IDLE, C_TRIP_STATE_EN_ROUTE, C_TRIP_STATE_HALT]
    C_DIM_TRIP_STATE = ["trip", "Trip Status", C_VALID_TRIP_STATES]
    C_DIM_AVAILABLE = ["ava", "Is Available", [True, False]]
    C_DIM_AT_NODE = ["node_bool", "At Node", [True, False]]
    C_DIM_CURRENT_CARGO = ["cargo", "Current Cargo", []]
    C_DIS_DIMS = [C_DIM_TRIP_STATE, C_DIM_AVAILABLE, C_DIM_AT_NODE, C_DIM_CURRENT_CARGO]

    # C_ACTION_LOAD = [SimulationActions.LOAD_DRONE_ACTION, SimulationActions.LOAD_TRUCK_ACTION]
    # C_ACTION_ROUTE = [SimulationAction.TRUCK_TO_NODE, SimulationActions.DRONE_TO_NODE]
    # C_ACTION_UNLOAD = [SimulationAction.UNLOAD_TRUCK_ACTION, SimulationActions.UNLOAD_DRONE_ACTION]
    # C_ACTION_ASSIGN_ORDER = [SimulationAction.ASSIGN_ORDER_TO_TRUCK, SimulationActions.ASSIGN_ORDER_TO_DRONE]
    # C_ACTION_REROUTE = [SimulationActions.RE_ROUTE_TRUCK_TO_NODE, SimulationActions.RE_ROUTE_DRONE_TO_NODE]
    # C_ACTION_CONSOLIDATION = [SimulationActions.CONSOLIDATE_FOR_TRUCK, SimulationActions.CONSOLIDATE_FOR_DRONE]

    # C_RELATED_ACTIONS = [C_ACTION_ASSIGN_ORDER
    #                      + C_ACTION_LOAD
    #                      + C_ACTION_UNLOAD
    #                      + C_ACTION_ROUTE
    #                      + C_ACTION_REROUTE
    #                      + C_ACTION_CONSOLIDATION]

    C_DATA_FRAME_VEH_TIMELINE = "Vehicle Timeline"
    C_DATA_FRAME_VEH_STATES = "Vehicle Trip States"

    def __init__(self,
                 p_id,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=False,
                 **p_kwargs):
        """
        Initializes a Vehicle system.
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        self.custom_log = False
        # Vehicle-specific attributes
        self.start_node_id: int = p_kwargs.get('start_node_id')
        self.max_payload_capacity: float = p_kwargs.get('max_payload_capacity', 0)
        self.max_speed: float = p_kwargs.get('max_speed', 0)
        self.network_manager: 'NetworkManager' = p_kwargs.get('network_manager')

        # New: Current location coordinates
        self.current_location_coords: Optional[Tuple[float, float]] = None

        # Internal dynamic attributes
        self.status: str = "idle"
        self.current_node_id: Optional[int] = self.start_node_id
        self.cargo_manifest: List[int] = []
        self.current_route: List[int] = []
        self.route_progress: float = 0.0

        # A route is a sequence (list), not a set.
        self.route_nodes: List[int] = []

        # New attributes for matrix-based movement
        if "p_movement_mode" in p_kwargs.keys():
            self.movement_mode = p_kwargs["p_movement_mode"]
        else:
            raise ParamError("Please provide a movement mode value in the simulation config.")
        self.en_route_timer = 0.0  # Timer for matrix-based movement

        self._state = State(self._state_space)
        self.delivery_orders = []
        self.delivery_node_ids = []
        self.pickup_orders = []
        self.pickup_node_ids = []
        self.cargo_stats = {}
        self.reset()
        self.consolidation_confirmed: bool = False

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for a generic Vehicle system.
        """
        state_space = MSpace()

        state_space.add_dim(Dimension('loc x',
                                      "R",
                                      "Current Location X"))

        state_space.add_dim(Dimension('loc y',
                                      "R",
                                      "Current Location Y"))
        action_space = MSpace()
        action_space.add_dim(
            Dimension(p_name_short='target_node', p_base_set='Z', p_name_long='Target Node ID', p_boundaries=[0, 999]))

        return state_space, action_space

    def _reset(self, p_seed=None):
        """
        Resets the vehicle to its initial state at its starting node.
        """
        self.status = "idle"
        # self.update_state_value_by_dim_name(self.C_DIM_AVAILABLE[0], 1)
        # self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
        if self.global_state is not None:
            time = self.global_state.current_time
            self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                           p_frame=self.C_DATA_FRAME_VEH_STATES)
        else:
            self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                           p_frame=self.C_DATA_FRAME_VEH_STATES)
        self.consolidation_confirmed = False
        self.set_current_node_id(self.start_node_id)
        self.cargo_manifest = []
        self.current_route = []
        self.route_progress = 0.0
        self.route_nodes = []  # Clear the route nodes on reset
        self.delivery_orders = []
        self.en_route_timer = 0.0  # Reset timer
        self.cargo_stats = {}

        # New: Reset coordinates to the start node
        # Combined the following into a single update call
        # self.update_state_value_by_dim_name(self.C_DIM_AVAILABLE[0], 1)
        # self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
        if self.global_state and self.start_node_id is not None:
            self.current_location_coords = self.global_state.get_entity('node', self.start_node_id).coords
            self.update_state_value_by_dim_name(p_dim_name=[self.C_DIM_AVAILABLE[0],
                                                            self.C_DIM_TRIP_STATE[0],
                                                            "loc x",
                                                            "loc y"],
                                                p_value=[1,
                                                         self.C_TRIP_STATE_IDLE,
                                                         self.current_location_coords[0],
                                                         self.current_location_coords[1]])

            # self.update_state_value_by_dim_name("loc y", self.current_location_coords[1])
        else:
            self.current_location_coords = (0.0, 0.0)
            # self.update_state_value_by_dim_name(["loc x","loc y"],
            #                                     [self.current_location_coords[0], self.current_location_coords[1]])
            self.update_state_value_by_dim_name(p_dim_name=[self.C_DIM_AVAILABLE[0],
                                                            self.C_DIM_TRIP_STATE[0],
                                                            "loc x",
                                                            "loc y"],
                                                p_value=[1,
                                                         self.C_TRIP_STATE_IDLE,
                                                         self.current_location_coords[0],
                                                         self.current_location_coords[1]])

        if self.global_state is not None:
            time = self.global_state.current_time
            self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                           p_frame=self.C_DATA_FRAME_VEH_STATES)
        else:
            self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                           p_frame=self.C_DATA_FRAME_VEH_STATES)

            # self.update_state_value_by_dim_name("loc x", self.current_location_coords[0])
            # self.update_state_value_by_dim_name("loc y", self.current_location_coords[1])

        # self._update_state()

    def _process_action(self, p_action: LogisticsAction, p_t_step: timedelta = None) -> bool:
        """
        Processes a "go to node" action. Calculates and sets the vehicle's route.
        """
        action_id = int(p_action.get_sorted_values()[0])
        action_type = ActionType.get_by_id(action_id)
        action_kwargs = p_action.data

        if action_type == SimulationActions.LOAD_TRUCK_ACTION:
            truck_id = action_kwargs["truck_id"]
            order_id = action_kwargs["order_id"]
            truck = self.global_state.get_entity("truck", truck_id)
            order: Order = self.global_state.get_entity("order", order_id)
            if order not in truck.pickup_orders:
                raise ValueError(
                    f"The order {order_id} is not assigned to the vehicle {truck_id}. The order is not in the pick up orders.")
            else:
                truck.pickup_orders.remove(order)
                truck.delivery_orders.append(order)
                truck.pickup_node_ids.remove(order.get_pickup_node_id())
                truck.delivery_node_ids.append(order.get_delivery_node_id())
                truck.add_cargo(order)
                order.set_enroute()
                if self.custom_log:
                    print(f"{order_id} is loaded in the truck {truck_id}.")
                self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))
                self.save_data(str(order), self.global_state.current_time, p_frame=self.C_DATA_FRAME_VEH_TIMELINE)
                # input("Press enter")
                return True

        if action_type == SimulationActions.UNLOAD_TRUCK_ACTION:
            truck_id = action_kwargs["truck_id"]
            if not truck_id == self.get_id():
                raise ValueError("Something is wrong, please re-calibrate/check your managers for mapping")
            order_id = action_kwargs["order_id"]
            # self = self.global_state.get_entity("truck", truck_id)
            order = self.global_state.get_entity("order", order_id)
            if order not in self.delivery_orders:

                raise ValueError(
                    "The order is not in the cargo of the vehicle. The order is not in the delivery orders.")
            else:
                self.delivery_orders.remove(order)
                self.remove_cargo(order.get_id())
                order.set_delivered()
                self.delivery_node_ids.remove(order.get_delivery_node_id())
                if self.delivery_node_ids or self.pickup_node_ids:
                    self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)

                # self.log(self.C_LOG_TYPE_I, f"Order {order_id} is unloaded from the truck {truck_id}.")
                if self.custom_log:
                    print(f"Order {order_id} is unloaded from the truck {truck_id}.")
                self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))
                # input("Press Enter")
                self.save_data(str(order), self.global_state.current_time, p_frame=self.C_DATA_FRAME_VEH_TIMELINE)
                # input("Press Enter")
                return True

        if action_type == SimulationActions.LOAD_DRONE_ACTION:
            drone_id = action_kwargs["drone_id"]
            order_id = action_kwargs["order_id"]
            drone = self.global_state.get_entity("drone", drone_id)
            order = self.global_state.get_entity("order", order_id)
            if order not in drone.pickup_orders:
                raise ValueError("The order is not assigned to the vehicle. The order is not in the pick up orders.")
            else:
                drone.pickup_orders.remove(order)
                drone.delivery_orders.append(order)
                drone.pickup_node_ids.remove(order.get_pickup_node_id())
                drone.delivery_node_ids.append(order.get_delivery_node_id())
                drone.add_cargo(order)
                order.set_enroute()
                if self.custom_log:
                    print(f"Order {order_id} is loaded in the Drone {drone_id}.")
                # self.log(self.C_LOG_TYPE_I, f"Order {order_id} is loaded in the Drone {drone_id}.")
                # input("Press enter")
                self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))
                self.save_data(str(order), self.global_state.current_time, p_frame=self.C_DATA_FRAME_VEH_TIMELINE)
                return True

        if action_type == SimulationActions.UNLOAD_DRONE_ACTION:
            drone_id = action_kwargs["drone_id"]
            if drone_id != self.get_id():
                raise ValueError("Please check for the unloading constraints.")
            order_id = action_kwargs["order_id"]
            # drone = self.global_state.get_entity("drone", drone_id)
            order = self.global_state.get_entity("order", order_id)
            if order not in self.delivery_orders:
                raise ValueError(
                    "The order is not in the cargo of the vehicle. The order is not in the delivery orders.")
            else:
                self.delivery_orders.remove(order)
                self.remove_cargo(order.get_id())
                order.set_delivered()
                self.delivery_node_ids.remove(order.get_delivery_node_id())
                if len(self.delivery_orders) or len(self.delivery_orders):
                    self.update_state_value_by_dim_name([self.C_DIM_TRIP_STATE[0], self.C_DIM_CURRENT_CARGO[0]], [self.C_TRIP_STATE_EN_ROUTE, len(self.cargo_manifest)])
                if self.custom_log:
                    print(f"Order {order_id} is unloaded from the drone {drone_id}.")
                # self.log(self.C_LOG_TYPE_I, f"Order {order_id} is unloaded from the drone {drone_id}.")
                self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))
                self.save_data(str(order), self.global_state.current_time, p_frame=self.C_DATA_FRAME_VEH_TIMELINE)
                # input("Press Enter")
                return True

        if action_type == SimulationActions.TRUCK_TO_NODE:
            pass

        if action_type == SimulationActions.DRONE_TO_NODE:
            pass

        if action_type == SimulationActions.DRONE_LAUNCH:
            pass

        if action_type == SimulationActions.DRONE_LAND:
            pass

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Simulates the vehicle's state over a given time step.
        """
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        if self.movement_mode == 'matrix':
            if (self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) in [self.C_TRIP_STATE_EN_ROUTE,
                                                                               self.C_TRIP_STATE_HALT]
                    and self.current_route and len(self.current_route) >= 2):
                self._update_matrix_movement(p_t_step.total_seconds())
        else:  # network mode
            if (self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) == self.C_TRIP_STATE_EN_ROUTE
                    and self.current_route and len(self.current_route) >= 2):
                self._move_along_route(p_t_step.total_seconds())

        return self._state
    #
    # def _update_matrix_movement(self, delta_time: float):
    #     """Handles movement for the 'matrix' mode."""
    #     if self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) == self.C_TRIP_STATE_EN_ROUTE:
    #         # if self.en_route_timer:
    #         #     self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
    #         self.en_route_timer -= delta_time
    #         start_node_id, end_node_id = self.current_route[0], self.current_route[1]
    #         if self.en_route_timer <= 0:
    #             self.set_current_node_id(end_node_id)
    #             self.current_edge = None
    #             if (end_node_id in self.pickup_node_ids) or (end_node_id in self.delivery_node_ids):
    #                 # Combined to a single update state call
    #                 # self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_HALT)
    #                 if self.global_state is not None:
    #                     time = self.global_state.current_time
    #                     self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]),
    #                                           self.get_current_node()], p_frame=self.C_DATA_FRAME_VEH_STATES)
    #                 else:
    #                     self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]),
    #                                        self.get_current_node()], p_frame=self.C_DATA_FRAME_VEH_STATES)
    #                 self.update_state_value_by_dim_name(p_dim_name=[self.C_DIM_AT_NODE[0],
    #                                                                 self.C_DIM_TRIP_STATE[0]],
    #                                                     p_value=[True,
    #                                                              self.C_TRIP_STATE_HALT])
    #
    #                 if end_node_id in self.pickup_node_ids:
    #                     if self.custom_log:
    #                         print(f"\n\n\n\n\nVehicle {self._id} reached pickup node {end_node_id}.\n\n\n\n")
    #                     # self.raise_state_change_event()
    #                 elif end_node_id in self.delivery_node_ids:
    #                     if self.custom_log:
    #                         print(f"\n\n\n\n\nVehicle {self._id} reached delivery node {end_node_id}.\n\n\n\n")
    #                     # self.raise_state_change_event()
    #                     # input("Press Enter to continue")
    #                 return
    #             else:
    #                 self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)
    #             self.current_route.pop(0)
    #             if not self.current_route or len(self.current_route) < 2:
    #                 self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
    #                 if self.global_state is not None:
    #                     time = self.global_state.current_time
    #                     self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]),
    #                                           self.get_current_node()], p_frame=self.C_DATA_FRAME_VEH_STATES)
    #                 else:
    #                     self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]),
    #                                        self.get_current_node()], p_frame=self.C_DATA_FRAME_VEH_STATES)
    #                 self.status = "idle"
    #                 self.route_nodes = []
    #                 # self.raise_state_change_event()
    #             else:
    #                 start_node_id, end_node_id = self.current_route[0], self.current_route[1]
    #                 if self.C_NAME == "Drone":
    #                     network_type = self.global_state.network.C_NETWORK_AIR
    #                 elif self.C_NAME == "Truck":
    #                     network_type = self.global_state.network.C_NETWORK_GROUND
    #                 else:
    #                     raise ValueError("Please provide a valid vehicle type.")
    #                 self.en_route_timer = self.network_manager.network.get_travel_time(start_node_id, end_node_id,
    #                                                                                    network_type=network_type)
    #                 # self.current_node_id = None
    #                 self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
    #                 if self.global_state is not None:
    #                     time = self.global_state.current_time
    #                     self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]),
    #                                           self.get_current_node()], p_frame=self.C_DATA_FRAME_VEH_STATES)
    #                 else:
    #                     self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]),
    #                                        self.get_current_node()], p_frame=self.C_DATA_FRAME_VEH_STATES)
    #                 # self.raise_state_change_event()
    #         else:
    #             self.set_current_node_id(None)
    #
    #     elif self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) in [self.C_TRIP_STATE_HALT]:
    #
    #         if not self.current_route or len(self.current_route) < 2:
    #             self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
    #             if self.global_state is not None:
    #                 time = self.global_state.current_time
    #                 self.save_data(time,
    #                                [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
    #                                p_frame=self.C_DATA_FRAME_VEH_STATES)
    #             else:
    #                 self.save_data(0,
    #                                [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
    #                                p_frame=self.C_DATA_FRAME_VEH_STATES)
    #             self.status = "idle"
    #             self.route_nodes = []
    #             # self.raise_state_change_event()
    #         else:
    #             start_node_id, end_node_id = self.current_route[0], self.current_route[1]
    #             if self.C_NAME == "Drone":
    #                 network_type = self.global_state.network.C_NETWORK_AIR
    #             elif self.C_NAME == "Truck":
    #                 network_type = self.global_state.network.C_NETWORK_GROUND
    #             else:
    #                 raise ValueError("Please provide a valid vehicle type.")
    #             self.en_route_timer = self.network_manager.network.get_travel_time(start_node_id, end_node_id,
    #                                                                                network_type=network_type)
    #             # self.current_node_id = None
    #             self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
    #             if self.global_state is not None:
    #                 time = self.global_state.current_time
    #                 self.save_data(time,
    #                                [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
    #                                p_frame=self.C_DATA_FRAME_VEH_STATES)
    #             else:
    #                 self.save_data(0,
    #                                [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
    #                                p_frame=self.C_DATA_FRAME_VEH_STATES)
    #             # self.raise_state_change_event()
    #
    #         # if self.en_route_timer <= 0:
    #         #     self.current_node_id = self.current_route[1]
    #         #     self.current_route = []
    #         #     self.status = "idle"
    #         #     self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
    #         #     # Update coordinates to be exactly at the new node
    #         #     self._update_location_coords(self.current_node_id, self.current_node_id, 1.0)

    def _update_matrix_movement(self, delta_time: float):
        """Handles movement for the 'matrix' mode."""
        if self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) == self.C_TRIP_STATE_EN_ROUTE:
            self.en_route_timer -= delta_time
            start_node_id, end_node_id = self.current_route[0], self.current_route[1]

            if self.en_route_timer <= 0:
                self.set_current_node_id(end_node_id)
                self.current_edge = None

                if (end_node_id in self.pickup_node_ids) or (end_node_id in self.delivery_node_ids):
                    # VEHICLE ARRIVED. FREEZE AND WAIT FOR AGENT.
                    self.update_state_value_by_dim_name(
                        p_dim_name=[self.C_DIM_AT_NODE[0], self.C_DIM_TRIP_STATE[0]],
                        p_value=[True, self.C_TRIP_STATE_HALT]
                    )

                    if self.global_state is not None:
                        time = self.global_state.current_time
                        self.save_data(time, [self.C_TRIP_STATE_HALT, self.get_current_node()],
                                       p_frame=self.C_DATA_FRAME_VEH_STATES)
                    else:
                        self.save_data(0, [self.C_TRIP_STATE_HALT, self.get_current_node()],
                                       p_frame=self.C_DATA_FRAME_VEH_STATES)

                    if self.custom_log:
                        print(f"\nVehicle {self._id} HALTED at node {end_node_id}. Waiting for Agent.\n")

                    return  # FREEZE: Wait for the RL agent to take action.
                else:
                    # Pass-through node (no pickup/delivery). Update state and pop the node to continue.
                    self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)
                    self.current_route.pop(0)

                    # Continue routing logic if it was a pass-through node
                if not self.current_route or len(self.current_route) < 2:
                    self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
                    if self.global_state is not None:
                        time = self.global_state.current_time
                        self.save_data(time, [self.C_TRIP_STATE_IDLE, self.get_current_node()],
                                       p_frame=self.C_DATA_FRAME_VEH_STATES)
                    else:
                        self.save_data(0, [self.C_TRIP_STATE_IDLE, self.get_current_node()],
                                       p_frame=self.C_DATA_FRAME_VEH_STATES)
                    self.status = "idle"
                    self.route_nodes = []
                else:
                    start_node_id, end_node_id = self.current_route[0], self.current_route[1]
                    network_type = self.global_state.network.C_NETWORK_AIR if self.C_NAME == "Drone" else self.global_state.network.C_NETWORK_GROUND
                    self.en_route_timer = self.network_manager.network.get_travel_time(start_node_id, end_node_id,
                                                                                       network_type=network_type)
                    self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
                    if self.global_state is not None:
                        time = self.global_state.current_time
                        self.save_data(time, [self.C_TRIP_STATE_EN_ROUTE, self.get_current_node()],
                                       p_frame=self.C_DATA_FRAME_VEH_STATES)
                    else:
                        self.save_data(0, [self.C_TRIP_STATE_EN_ROUTE, self.get_current_node()],
                                       p_frame=self.C_DATA_FRAME_VEH_STATES)
            else:
                self.set_current_node_id(None)

        elif self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) in [self.C_TRIP_STATE_HALT]:

            # --- THE FIX: Agent Wait Loop ---
            # If the current node is still in the task lists, the agent hasn't loaded/unloaded yet.
            if (self.current_node_id in self.pickup_node_ids) or (self.current_node_id in self.delivery_node_ids):
                return  # STAY HALTED. Do absolutely nothing until the agent clears the manifest.

            # Agent finished! The node is clear. NOW we can safely pop the node and resume routing.
            if self.current_route:
                self.current_route.pop(0)

            if not self.current_route or len(self.current_route) < 2:
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
                if self.global_state is not None:
                    time = self.global_state.current_time
                    self.save_data(time, [self.C_TRIP_STATE_IDLE, self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)
                else:
                    self.save_data(0, [self.C_TRIP_STATE_IDLE, self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)
                self.status = "idle"
                self.route_nodes = []
            else:
                start_node_id, end_node_id = self.current_route[0], self.current_route[1]
                network_type = self.global_state.network.C_NETWORK_AIR if self.C_NAME == "Drone" else self.global_state.network.C_NETWORK_GROUND
                self.en_route_timer = self.network_manager.network.get_travel_time(start_node_id, end_node_id,
                                                                                   network_type=network_type)
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)

                if self.global_state is not None:
                    time = self.global_state.current_time
                    self.save_data(time, [self.C_TRIP_STATE_EN_ROUTE, self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)
                else:
                    self.save_data(0, [self.C_TRIP_STATE_EN_ROUTE, self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)

    def _move_along_route(self, delta_time: float):
        """
        Internal logic to advance the vehicle along its route.
        """
        start_node_id, end_node_id = self.current_route[0], self.current_route[1]
        edge = self.network_manager.network.get_edge_between_nodes(start_node_id, end_node_id)
        self.current_edge = edge
        if not edge:
            self.status = "idle"
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
            if self.global_state is not None:
                time = self.global_state.current_time
                self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                               p_frame=self.C_DATA_FRAME_VEH_STATES)
            else:
                self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                               p_frame=self.C_DATA_FRAME_VEH_STATES)
            return

        travel_time = edge.get_current_travel_time() if self.C_NAME == 'Truck' else edge.get_drone_flight_time()

        if travel_time <= 0 or travel_time == float('inf'):
            self.status = "idle"
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
            if self.global_state is not None:
                time = self.global_state.current_time
                self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                               p_frame=self.C_DATA_FRAME_VEH_STATES)
            else:
                self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                               p_frame=self.C_DATA_FRAME_VEH_STATES)
            return

        time_needed = (1.0 - self.route_progress) * travel_time
        time_to_move = min(delta_time, time_needed)

        self.route_progress += time_to_move / travel_time

        # New: Update coordinates based on progress
        self._update_location_coords(start_node_id, end_node_id, self.route_progress)

        self.update_energy(-time_to_move)

        if self.route_progress >= 1.0:
            self.set_current_node_id(end_node_id)
            self.current_edge = None
            if (end_node_id in self.pickup_node_ids) or (end_node_id in self.delivery_node_ids):
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_HALT)

                if self.global_state is not None:
                    time = self.global_state.current_time
                    self.save_data(time,
                                   [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)
                else:
                    self.save_data(self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), 0,
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)

                if end_node_id in self.pickup_node_ids:
                    if self.custom_log:
                        print(f"\n\n\n\n\nVehicle {self._id} reached pickup node {end_node_id}.\n\n\n\n")
                    self.raise_state_change_event()
                elif end_node_id in self.delivery_node_ids:
                    if self.custom_log:
                        print(f"\n\n\n\n\nVehicle {self._id} reached delivery node {end_node_id}.\n\n\n\n")
                    self.raise_state_change_event()
                    # input("Press Enter to continue")
            else:
                self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)

            self.current_route.pop(0)
            self.route_progress = 0.0
            if not self.current_route or len(self.current_route) < 2:
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
                if self.global_state is not None:
                    time = self.global_state.current_time
                    self.save_data(time,
                                   [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)
                else:
                    self.save_data(0,
                                   [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)
                self.status = "idle"
                self.route_nodes = []
                self.raise_state_change_event()
            else:
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
                if self.global_state is not None:
                    time = self.global_state.current_time
                    self.save_data(time,
                                   [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)
                else:
                    self.save_data(0,
                                   [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                                   p_frame=self.C_DATA_FRAME_VEH_STATES)
                self.raise_state_change_event()
            # New: Update coordinates to be exactly at the new node
            self._update_location_coords(self.get_current_node(), self.get_current_node(), self.route_progress)
        else:
            self.set_current_node_id(None)

    def _update_location_coords(self, start_node_id, end_node_id, progress):
        """
        Updates the vehicle's location coordinates based on its progress along a route segment.
        """
        start_node = self.global_state.get_entity('node', start_node_id)
        end_node = self.global_state.get_entity('node', end_node_id)

        start_x, start_y = start_node.coords
        end_x, end_y = end_node.coords

        new_x = start_x + (end_x - start_x) * progress
        new_y = start_y + (end_y - start_y) * progress

        self.current_location_coords = (new_x, new_y)
        self.update_state_value_by_dim_name("loc x", new_x)
        self.update_state_value_by_dim_name('loc y', new_y)

    def update_energy(self, p_time_passed: float):
        """
        Abstract method for energy consumption.
        """
        pass

    def _update_state(self):
        """
        Synchronizes internal attributes with the formal MLPro state object.
        """
        pass

    def add_cargo(self, order: int):
        """Adds a package to the vehicle's cargo manifest."""
        if order not in self.cargo_manifest:
            self.cargo_manifest.append(order)
            self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()
            # self.raise_state_change_event()

    def remove_cargo(self, order: int):
        """Removes a package from the vehicle's cargo manifest."""
        if order in self.cargo_manifest:
            self.cargo_manifest.remove(order)
            self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()
            # self.raise_state_change_event()

    def set_route(self, route: List[int]):
        """
        Sets the planned route for the vehicle.
        """
        if not route or len(route) < 2:
            self.log(self.C_LOG_TYPE_W,
                     f"Vehicle {self.get_id()}: Invalid route provided (length < 2). Remaining idle.")
            self.current_route = []
            self.route_nodes = []
            self.status = "idle"
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
            if self.global_state is not None:
                time = self.global_state.current_time
                self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                               p_frame=self.C_DATA_FRAME_VEH_STATES)
            else:
                self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                               p_frame=self.C_DATA_FRAME_VEH_STATES)
            # if self.current_node_id is None and self.route_nodes:
            #     self.set_current_node_id(self.route_nodes[0])
            if self.get_current_node() is None:
                self.set_current_node_id(self.start_node_id)
            # self.raise_state_change_event()
            return

        self.current_route = route
        self.route_nodes = route
        self.status = "en_route"

        if self.movement_mode == 'matrix':
            start_node_id, end_node_id = self.current_route[0], self.current_route[1]
            if self.C_NAME == "Drone":
                network_type = self.global_state.network.C_NETWORK_AIR
            elif self.C_NAME == "Truck":
                network_type = self.global_state.network.C_NETWORK_GROUND
            else:
                raise ValueError("Please provide a valid vehicle type.")
            self.en_route_timer = self.network_manager.network.get_travel_time(start_node_id, end_node_id,
                                                                               network_type=network_type)
            self.set_current_node_id(None)
            # else:
            #     self.status = "idle"  # Or handle error appropriately
        else:  # network mode
            self.route_progress = 0.0
            self.set_current_node_id(None)

        self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
        if self.global_state is not None:
            time = self.global_state.current_time
            self.save_data(time, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                           p_frame=self.C_DATA_FRAME_VEH_STATES)
        else:
            self.save_data(0, [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.get_current_node()],
                           p_frame=self.C_DATA_FRAME_VEH_STATES)
        # self.raise_state_change_event()

    def get_current_location(self):
        return self.get_state_value_by_dim_name("loc x"), self.get_state_value_by_dim_name("loc y")

    def get_delivery_orders(self):
        return self.delivery_orders

    def assign_orders(self, p_orders: list):
        if p_orders:
            for ord in p_orders:
                self.pickup_orders.append(ord)
                self.pickup_node_ids.append(ord.get_pickup_node_id())
                self.raise_state_change_event()  # <-- Added event trigger
            return True

    def unload_order(self, p_order):
        if p_order:
            self.delivery_orders.remove(p_order)
            self.cargo_manifest.remove(p_order)

    def get_current_node(self):
        return self.current_node_id

    def set_current_node_id(self, current_node_id):
        self.current_node_id = current_node_id
        for order in self.get_current_cargo():
            if not isinstance(order, Order):
                raise TypeError("Something is wrong. The assigned orders shall all be of type Order.")
            order.location_history.append(current_node_id)
            order.current_node_id = current_node_id
            if order.current_node_id == order.get_delivery_node_id():
                order.update_state_value_by_dim_name(order.C_DIM_CURRENT_NODE[0], self.current_node_id)

    def get_cargo_capacity(self):
        return self.max_payload_capacity

    def get_current_cargo_size(self):
        return len(self.cargo_manifest)

    def get_current_cargo(self):
        return self.cargo_manifest

    def get_pickup_orders(self):
        return self.pickup_orders

    def __repr__(self):
        return (f"{self.C_NAME} - {self._id} - {self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])} - "
                f"{self.pickup_orders[0].get_id() if len(self.pickup_orders) else 'None'}")

    def check_assignability(self) -> bool:
        status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])
        if status in [self.C_TRIP_STATE_IDLE]:
            return True
        else:
            return False


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Create a Mock Vehicle class for the test
    class MockVehicle(Vehicle):
        def update_energy(self, p_time_passed: float):
            # Dummy implementation for the abstract method
            pass


    print("--- Validating Vehicle Base Class ---")

    # 2. Instantiate the Mock Vehicle
    vehicle = MockVehicle(p_id=101, start_node_id=0)
    print(f"\n[A] Initial State:")
    print(f"  - Status: {vehicle.status}")
    print(f"  - Current Node: {vehicle.current_node_id}")
    print(f"  - Route: {vehicle.current_route}")
    print(f"  - Route Nodes: {vehicle.route_nodes}")

    assert vehicle.status == 'idle'
    assert vehicle.route_nodes == []

    # 3. Test the set_route method with a valid route
    new_route = [0, 5, 12, 5]  # A route that revisits a node
    print(f"\n[B] Setting a new route: {new_route}")
    vehicle.set_route(new_route)

    print("\n[C] State after setting route:")
    print(f"  - Status: {vehicle.status}")
    print(f"  - Current Node: {vehicle.current_node_id}")
    print(f"  - Route: {vehicle.current_route}")
    print(f"  - Route Nodes: {vehicle.route_nodes}")

    assert vehicle.status == 'en_route'
    assert vehicle.current_node_id is None
    assert vehicle.route_nodes == [0, 5, 12, 5]
    print("\n  - PASSED: State updated correctly for a valid route.")

    # 4. Test with an invalid route
    invalid_route = [15]
    print(f"\n[D] Setting an invalid route: {invalid_route}")
    vehicle.set_route(invalid_route)

    print("\n[E] State after setting invalid route:")
    print(f"  - Status: {vehicle.status}")
    print(f"  - Current Node: {vehicle.current_node_id}")
    print(f"  - Route: {vehicle.current_route}")
    print(f"  - Route Nodes: {vehicle.route_nodes}")

    assert vehicle.status == 'idle'
    assert vehicle.route_nodes == []
    # FIX: Add assertion to check that the current_node_id is restored
    assert vehicle.current_node_id == 0
    print("\n  - PASSED: State correctly reverted to idle for an invalid route.")

    print("\n--- Validation Complete ---")