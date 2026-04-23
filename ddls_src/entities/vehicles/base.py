# In ddls_src/entities/vehicles/base.py
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import timedelta
from ddls_src.actions.base import SimulationActions
from ddls_src.actions.base import SimulationActions, ActionType
from ddls_src.core.basics import LogisticsAction
from ddls_src.entities.base import LogisticEntity
from ddls_src.entities.order import Order
from mlpro.bf.exceptions import ParamError
from mlpro.bf.math import MSpace, Dimension
# MLPro Imports
from mlpro.bf.systems import System, State, Action
from typing import List, Tuple, Any, Dict, Optional, Set
from ddls_src.entities.vehicles.sequencer import HeuristicSequencer
from ddls_src.entities.vehicles.sequencer import HeuristicSequencer2


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

        self.current_sequence_index = 0
        self.planned_order_sequence = []
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
        self.cargo_manifest: List = []
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
        # --- NEW: Staging Area for the Batch Sequencer ---
        self.staged_pickup_orders = defaultdict()
        self.staged_delivery_orders = defaultdict()
        self.staged_pickup_leg2_orders = defaultdict()
        self.staged_delivery_leg2_orders = defaultdict()

        self.planned_node_sequence = []
        self.consolidation_confirmed = False
        self.debug_planned_node_sequence = []
        self.debug_planned_order_sequence = []

        # Attach the interchangeable sequencer (assuming you create this file next)
        try:
            # from ddls_src.entities.vehicles.sequencers import AI4DroneHeuristicSequencer
            self.sequencer = p_kwargs.get('sequencer', HeuristicSequencer2())
        except ImportError:
            self.log(self.C_LOG_TYPE_E,
                     "Could not import HeuristicSequencer. Please ensure sequencers.py exists.")
            self.sequencer = None

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

    def log_current_state(self):
        """Helper to append the current state to the vehicle's localized history."""
        if hasattr(self, 'global_state') and self.global_state is not None:
            current_time = getattr(self.global_state, 'current_time', 0.0)
            status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])
            battery = getattr(self, 'battery_level', getattr(self, 'fuel_level', None))

            # Extract Order IDs
            pickup_list = [str(order) for order in self.pickup_orders] if self.pickup_orders else []
            delivery_list = [str(order) for order in self.delivery_orders] if self.delivery_orders else []

            # --- MODIFICATION: Append to self.state_history instead of DataManager ---
            self.state_history.append({
                'time': current_time,
                'vehicle_id': self.get_id(),
                'node_id': self.get_current_node(),
                'status': status,'num_pickup_tasks': len(pickup_list),
                'pickup_orders': str(pickup_list),
                'num_delivery_tasks': len(delivery_list),
                'delivery_orders': str(delivery_list),
                'battery': battery
            })

    def _reset(self, p_seed=None):
        """
        Resets the vehicle to its initial state at its starting node.
        """
        self.status = "idle"
        self.consolidation_confirmed = False
        self.set_current_node_id(self.start_node_id)
        self.cargo_manifest = []
        self.current_route = []
        self.route_progress = 0.0
        self.route_nodes = []
        self.delivery_orders = []
        self.en_route_timer = 0.0
        self.cargo_stats = {}
        self.en_route_timer = 0.0  # Timer for matrix-based movement
        self.current_leg_duration = 0.0  # NEW: Tracks the total time of the current matrix leg

        # --- NEW: Initialize local history for this specific vehicle ---
        self.state_history = []

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
        else:
            self.current_location_coords = (0.0, 0.0)
            self.update_state_value_by_dim_name(p_dim_name=[self.C_DIM_AVAILABLE[0],
                                                            self.C_DIM_TRIP_STATE[0],
                                                            "loc x",
                                                            "loc y"],
                                                p_value=[1,
                                                         self.C_TRIP_STATE_IDLE,
                                                         self.current_location_coords[0],
                                                         self.current_location_coords[1]])

        # --- Staging Area for the Agent ---
        self.staged_pickup_orders = defaultdict()
        self.staged_delivery_orders = defaultdict()
        self.staged_pickup_leg2_orders = defaultdict()
        self.staged_delivery_leg2_orders = defaultdict()

        # The finalized sequence generated after CONSOLIDATE
        self.planned_node_sequence: List[int] = []
        self.planned_order_sequence: List[Order] = []

        # Initialize the strategy
        # from ddls_src.entities.vehicles.sequencers import HeuristicSequencer
        self.sequencer.reset()
        # Log initial state
        self.log_current_state()

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
            elif str(truck.current_node_id) != str(order.get_pickup_node_id()):
                raise ValueError(
                    f"Location mismatch! Truck {truck_id} cannot load order {order_id} at node {truck.current_node_id}. Order is at {order.get_pickup_node_id()}.")
            else:
                # truck.pickup_orders.remove(order)
                # truck.delivery_orders.append(order)
                # truck.pickup_node_ids.remove(order.get_pickup_node_id())
                # truck.delivery_node_ids.append(order.get_delivery_node_id())
                truck.add_cargo(order)
                order.set_enroute()
                # self._evaluate_route_state()
                if self.custom_log:
                    print(f"{order_id} is loaded in the truck {truck_id}.")
                self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))

                # NEW CENTRALIZED LOGGER
                if hasattr(self, 'global_state') and self.global_state:
                    self.global_state.data_manager.log_order_event(
                        current_time=self.global_state.current_time,
                        order_id=order.get_id(),
                        event_type='Loaded',
                        vehicle_id=self.get_id()
                    )
                return True

        if action_type == SimulationActions.UNLOAD_TRUCK_ACTION:
            truck_id = action_kwargs["truck_id"]
            if not truck_id == self.get_id():
                raise ValueError("Something is wrong, please re-calibrate/check your managers for mapping")
            order_id = action_kwargs["order_id"]
            order = self.global_state.get_entity("order", order_id)
            if order not in self.delivery_orders:
                raise ValueError(
                    "The order is not in the cargo of the vehicle. The order is not in the delivery orders.")
            elif str(self.current_node_id) != str(order.get_delivery_node_id()):
                raise ValueError(
                    f"Location mismatch! Truck {truck_id} cannot unload order {order_id} at node {self.current_node_id}. Destination is {order.get_delivery_node_id()}.")
            else:
                # self.delivery_orders.remove(order)
                self.remove_cargo(order.get_id())
                order.set_delivered()
                # self._evaluate_route_state()
                # self.delivery_node_ids.remove(order.get_delivery_node_id())
                # if self.delivery_node_ids or self.pickup_node_ids:
                #     self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)

                if self.custom_log:
                    print(f"Order {order_id} is unloaded from the truck {truck_id}.")
                self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))

                # NEW CENTRALIZED LOGGER
                if hasattr(self, 'global_state') and self.global_state:
                    self.global_state.data_manager.log_order_event(
                        current_time=self.global_state.current_time,
                        order_id=order.get_id(),
                        event_type='Unloaded',
                        vehicle_id=self.get_id()
                    )
                return True

        if action_type == SimulationActions.LOAD_DRONE_ACTION:
            drone_id = action_kwargs["drone_id"]
            order_id = action_kwargs["order_id"]
            drone = self.global_state.get_entity("drone", drone_id)
            order = self.global_state.get_entity("order", order_id)
            if order not in drone.pickup_orders:
                raise ValueError("The order is not assigned to the vehicle. The order is not in the pick up orders.")
            elif str(drone.current_node_id) != str(order.get_pickup_node_id()):
                raise ValueError(
                    f"Location mismatch! Drone {drone_id} cannot load order {order_id} at node {drone.current_node_id}. Order is at {order.get_pickup_node_id()}.")
            else:
                # drone.pickup_orders.remove(order)
                # drone.delivery_orders.append(order)
                # drone.pickup_node_ids.remove(order.get_pickup_node_id())
                # drone.delivery_node_ids.append(order.get_delivery_node_id())
                drone.add_cargo(order)
                order.set_enroute()
                # self._evaluate_route_state()
                if self.custom_log:
                    print(f"Order {order_id} is loaded in the Drone {drone_id}.")
                self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))

                # NEW CENTRALIZED LOGGER
                if hasattr(self, 'global_state') and self.global_state:
                    self.global_state.data_manager.log_order_event(
                        current_time=self.global_state.current_time,
                        order_id=order.get_id(),
                        event_type='Loaded',
                        vehicle_id=self.get_id()
                    )
                return True

        if action_type == SimulationActions.UNLOAD_DRONE_ACTION:
            drone_id = action_kwargs["drone_id"]
            if drone_id != self.get_id():
                raise ValueError("Please check for the unloading constraints.")
            order_id = action_kwargs["order_id"]
            order = self.global_state.get_entity("order", order_id)
            if order not in self.delivery_orders:
                raise ValueError(
                    "The order is not in the cargo of the vehicle. The order is not in the delivery orders.")
            elif str(self.current_node_id) != str(order.get_delivery_node_id()):
                raise ValueError(
                    f"Location mismatch! Drone {drone_id} cannot unload order {order_id} at node {self.current_node_id}. Destination is {order.get_delivery_node_id()}.")
            else:
                # self.delivery_orders.remove(order)
                self.remove_cargo(order.get_id())
                order.set_delivered()
                # self._evaluate_route_state()
                # self.delivery_node_ids.remove(order.get_delivery_node_id())
                # if len(self.delivery_orders) or len(self.delivery_orders):
                #     self.update_state_value_by_dim_name([self.C_DIM_TRIP_STATE[0], self.C_DIM_CURRENT_CARGO[0]],
                #                                         [self.C_TRIP_STATE_EN_ROUTE, len(self.cargo_manifest)])
                if self.custom_log:
                    print(f"Order {order_id} is unloaded from the drone {drone_id}.")
                self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))

                # NEW CENTRALIZED LOGGER
                if hasattr(self, 'global_state') and self.global_state:
                    self.global_state.data_manager.log_order_event(
                        current_time=self.global_state.current_time,
                        order_id=order.get_id(),
                        event_type='Unloaded',
                        vehicle_id=self.get_id()
                    )
                return True

        if action_type == SimulationActions.TRUCK_TO_NODE:
            pass

        if action_type == SimulationActions.DRONE_TO_NODE:
            pass

        if action_type == SimulationActions.DRONE_LAUNCH:
            pass

        if action_type == SimulationActions.DRONE_LAND:
            pass

    # def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
    #     """
    #     Simulates the vehicle's state over a given time step.
    #     """
    #     if p_action is not None:
    #         self._process_action(p_action, p_t_step)
    #
    #     if self.movement_mode == 'matrix':
    #         if (self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) in [self.C_TRIP_STATE_EN_ROUTE,
    #                                                                            self.C_TRIP_STATE_HALT]
    #                 and self.current_route and len(self.current_route) >= 2):
    #             self._update_matrix_movement(p_t_step.total_seconds())
    #     else:  # network mode
    #         if (self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) == self.C_TRIP_STATE_EN_ROUTE
    #                 and self.current_route and len(self.current_route) >= 2):
    #             self._move_along_route(p_t_step.total_seconds())
    #
    #     return self._state

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Simulates the vehicle's state over a given time step.
        """
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        if self.movement_mode == 'matrix':
            # --- 1. NAVIGATION (The Brain) ---
            # Check if we are IDLE and if our tasks are clear to start moving again.
            self._evaluate_route_state()

            # --- 2. PHYSICS (The Engine) ---
            # Move the vehicle, drain battery, and handle arrival.
            self._update_matrix_movement(p_t_step.total_seconds())

        else:  # network mode
            # Kept consistent with MLPro state tracking
            current_status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])
            if (current_status == self.C_TRIP_STATE_EN_ROUTE
                    and self.current_route and len(self.current_route) >= 2):
                self._move_along_route(p_t_step.total_seconds())

        return self._state

    # def _update_matrix_movement_old(self, delta_time: float):
    #     """Handles movement for the 'matrix' mode."""
    #     if self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) == self.C_TRIP_STATE_EN_ROUTE:
    #         self.en_route_timer -= delta_time
    #         start_node_id, end_node_id = self.current_route[0], self.current_route[1]
    #
    #         if self.en_route_timer <= 0:
    #             self.set_current_node_id(end_node_id)
    #             self.current_edge = None
    #
    #             if (end_node_id in self.pickup_node_ids) or (end_node_id in self.delivery_node_ids):
    #                 # VEHICLE ARRIVED. FREEZE AND WAIT FOR AGENT.
    #                 self.update_state_value_by_dim_name(
    #                     p_dim_name=[self.C_DIM_AT_NODE[0], self.C_DIM_TRIP_STATE[0]],
    #                     p_value=[True, self.C_TRIP_STATE_HALT]
    #                 )
    #
    #                 self.log_current_state()
    #
    #                 if self.custom_log:
    #                     print(f"\nVehicle {self._id} HALTED at node {end_node_id}. Waiting for Agent.\n")
    #
    #                 return  # FREEZE: Wait for the RL agent to take action.
    #             else:
    #                 # Pass-through node (no pickup/delivery). Update state and pop the node to continue.
    #                 self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)
    #                 self.current_route.pop(0)
    #
    #             # Continue routing logic if it was a pass-through node
    #             if not self.current_route or len(self.current_route) < 2:
    #                 self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
    #                 self.status = "idle"
    #                 self.route_nodes = []
    #                 self.log_current_state()
    #             else:
    #                 start_node_id, end_node_id = self.current_route[0], self.current_route[1]
    #                 network_type = self.global_state.network.C_NETWORK_AIR if self.C_NAME == "Drone" else self.global_state.network.C_NETWORK_GROUND
    #                 self.en_route_timer = self.network_manager.network.get_travel_time(start_node_id, end_node_id,
    #                                                                                    network_type=network_type)
    #                 self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
    #                 self.log_current_state()
    #         else:
    #             self.set_current_node_id(None)
    #
    #     elif self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) in [self.C_TRIP_STATE_HALT]:
    #         # --- THE FIX: Agent Wait Loop ---
    #         # If the current node is still in the task lists, the agent hasn't loaded/unloaded yet.
    #         if (self.current_node_id in self.pickup_node_ids) or (self.current_node_id in self.delivery_node_ids):
    #             return  # STAY HALTED. Do absolutely nothing until the agent clears the manifest.
    #
    #         # Agent finished! The node is clear. NOW we can safely pop the node and resume routing.
    #         if self.current_route:
    #             self.current_route.pop(0)
    #
    #         if not self.current_route or len(self.current_route) < 2:
    #             self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
    #             self.status = "idle"
    #             self.route_nodes = []
    #             self.log_current_state()
    #         else:
    #             start_node_id, end_node_id = self.current_route[0], self.current_route[1]
    #             network_type = self.global_state.network.C_NETWORK_AIR if self.C_NAME == "Drone" else self.global_state.network.C_NETWORK_GROUND
    #             self.en_route_timer = self.network_manager.network.get_travel_time(start_node_id, end_node_id,
    #                                                                                network_type=network_type)
    #             self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
    #             self.log_current_state()
    def _update_matrix_movement(self, delta_time: float):
        current_status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])

        if current_status != self.C_TRIP_STATE_EN_ROUTE or not self.planned_node_sequence:
            return

        time_to_move = min(delta_time, self.en_route_timer)
        self.en_route_timer -= time_to_move
        self.update_energy(-time_to_move)

        if self.en_route_timer <= 0:
            destination_node = self.planned_node_sequence[self.current_sequence_index]

            self.set_current_node_id(destination_node)
            self.current_edge = None

            self._update_location_coords(destination_node, destination_node, 1.0)

            self.update_state_value_by_dim_name(
                p_dim_name=[self.C_DIM_AT_NODE[0], self.C_DIM_TRIP_STATE[0]],
                p_value=[True, self.C_TRIP_STATE_HALT]
            )
            self.log_current_state()

            # --- ADD THIS LINE ---
            if self.custom_log:
                print(f"Vehicle {self._id} reached node {destination_node}.")

        else:
            self.set_current_node_id(None)

    def _move_along_route(self, delta_time: float):
        """
        Internal logic to advance the vehicle along its route in network mode.
        """
        current_status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])

        # FIXED: Checks planned_node_sequence
        if current_status != self.C_TRIP_STATE_EN_ROUTE or not self.planned_node_sequence:
            return

        start_node_id = self.get_current_node_id()
        end_node_id = self.planned_node_sequence[0]

        if start_node_id is None:
            return self._update_matrix_movement(delta_time)

        edge = self.network_manager.network.get_edge_between_nodes(start_node_id, end_node_id)
        self.current_edge = edge
        if not edge:
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_HALT)
            return

        travel_time = edge.get_current_travel_time() if self.C_NAME == 'Truck' else edge.get_drone_flight_time()

        if travel_time <= 0 or travel_time == float('inf'):
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_HALT)
            return

        time_needed = (1.0 - self.route_progress) * travel_time
        time_to_move = min(delta_time, time_needed)

        self.route_progress += time_to_move / travel_time
        self._update_location_coords(start_node_id, end_node_id, self.route_progress)
        self.update_energy(-time_to_move)

        if self.route_progress >= 1.0:
            self.set_current_node_id(end_node_id)
            self.current_edge = None
            self.route_progress = 0.0
            self._update_location_coords(end_node_id, end_node_id, 1.0)

            # FIXED: Halt upon arrival, let the brain take over
            self.update_state_value_by_dim_name(
                p_dim_name=[self.C_DIM_AT_NODE[0], self.C_DIM_TRIP_STATE[0]],
                p_value=[True, self.C_TRIP_STATE_HALT]
            )
            self.log_current_state()
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

    # def add_cargo(self, order: int):
    #     """Adds a package to the vehicle's cargo manifest."""
    #     if order not in self.cargo_manifest:
    #         self.cargo_manifest.append(order)
    #         self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()
    #
    # def remove_cargo(self, order: int):
    #     """Removes a package from the vehicle's cargo manifest."""
    #     if order in self.cargo_manifest:
    #         self.cargo_manifest.remove(order)
    #         self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()

    # def add_cargo(self, order: 'Order'):
    #     """Adds a package to the manifest and safely mutates the MLPro state."""
    #     # Sanity check to prevent physics-breaking glitches
    #     if len(self.cargo_manifest) >= self.max_payload_capacity:
    #         raise ValueError(f"Vehicle {self.get_id()} physically exceeded payload capacity!")
    #
    #     if order not in self.cargo_manifest:
    #         self.cargo_manifest.append(order)
    #         self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()
    #
    #         # --- CENTRALIZED STATE MUTATION ---
    #         self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], self.get_current_cargo_size())

    def add_cargo(self, order: 'Order'):
        """Adds a package to the manifest, updates state, and crosses off the sequence clipboard."""
        if self.get_current_cargo_size() + order.size > self.max_payload_capacity:
            raise ValueError(f"Vehicle {self.get_id()} physically exceeded payload capacity! "
                             f"Current: {self.get_current_cargo_size()}, Adding: {order.size}, Max: {self.max_payload_capacity}")

        if order not in self.cargo_manifest:
            self.cargo_manifest.append(order)
            self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()

            self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], self.get_current_cargo_size())

            # --- THE NEW CLIPBOARD LOGIC ---
            # If we are actively running a mission, check the current sequence step
            # if self.consolidation_confirmed and self.current_sequence_index in self.planned_order_sequence:
            pickup_orders, delivery_orders = self.planned_order_sequence[self.current_sequence_index]
            if order in pickup_orders:
                pickup_orders.remove(order)  # Cross it off!
                if self.custom_log:
                    print(
                        f"[Vehicle {self.get_id()}] Picked up {order.get_id()} - crossed off Sequence {self.current_sequence_index}")
            self.pickup_orders.remove(order)
            self.delivery_orders.append(order)

        self._evaluate_route_state()

    # def remove_cargo(self, order_id: int):
    #     """Removes a package from the manifest by ID and safely mutates the MLPro state."""
    #     # Find the order by ID since the manifest holds Order objects
    #     order_to_remove = next((o for o in self.cargo_manifest if o.get_id() == order_id), None)
    #
    #     if order_to_remove:
    #         self.cargo_manifest.remove(order_to_remove)
    #         self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()
    #
    #         # --- CENTRALIZED STATE MUTATION ---
    #         self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], self.get_current_cargo_size())

    def remove_cargo(self, order_id: int):
        """Removes a package from the manifest, updates state, and crosses off the sequence clipboard."""
        order_to_remove = next((o for o in self.cargo_manifest if o.get_id() == order_id), None)

        if order_to_remove:
            self.cargo_manifest.remove(order_to_remove)
            self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()

            self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], self.get_current_cargo_size())

            # --- THE NEW CLIPBOARD LOGIC ---
            # if self.consolidation_confirmed and self.current_sequence_index in self.planned_order_sequence:
            pickup_orders, delivery_orders = self.planned_order_sequence[self.current_sequence_index]
            if order_to_remove in delivery_orders:
                delivery_orders.remove(order_to_remove)  # Cross it off!
                if self.custom_log:
                    print(
                        f"[Vehicle {self.get_id()}] Dropped off {order_to_remove.get_id()} - crossed off Sequence {self.current_sequence_index}")

            self.delivery_orders.remove(order_to_remove)
            # print("Debugging")
        self._evaluate_route_state()

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

            self.log_current_state()

            if self.get_current_node() is None:
                self.set_current_node_id(self.start_node_id)
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
        else:  # network mode
            self.route_progress = 0.0
            self.set_current_node_id(None)

        self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
        self.log_current_state()

    def get_current_location(self):
        return self.get_state_value_by_dim_name("loc x"), self.get_state_value_by_dim_name("loc y")

    def get_delivery_orders(self):
        return self.delivery_orders

    # def assign_orders(self, p_orders: list):
    #     if p_orders:
    #         for ord in p_orders:
    #             self.pickup_orders.append(ord)
    #             self.pickup_node_ids.append(ord.get_pickup_node_id())
    #             self.raise_state_change_event()
    #         return True

    # def assign_orders(self, p_orders: list):
    #     """
    #     Receives a list of orders from the Supply Chain Manager.
    #     Maintains backward compatibility by populating old lists while simultaneously
    #     staging orders for the new Batch Sequencer.
    #     """
    #     if p_orders:
    #         from ddls_src.entities.order import PseudoOrder  # Ensure this is imported safely
    #
    #         if self.get_remaining_capacity() < len(p_orders):
    #             self.log(self.C_LOG_TYPE_W,
    #                      f"Vehicle {self.get_id()} REJECTED assignment. "
    #                      f"Attempted: {len(p_orders)}, Remaining Capacity: {self.get_remaining_capacity()}")
    #             raise ValueError("Vehicle Overloaded. Please check capacity management and/or constraint management. "
    #                              "Agent is taking impossible actions.")
    #
    #         for order in p_orders:
    #
    #             # --- BACKWARD COMPATIBILITY BLOCK ---
    #             self.pickup_orders.append(order)
    #             self.pickup_node_ids.append(order.get_pickup_node_id())
    #
    #             if hasattr(self, 'delivery_orders'):
    #                 self.delivery_orders.append(order)
    #             if hasattr(self, 'delivery_node_ids'):
    #                 self.delivery_node_ids.append(order.get_delivery_node_id())
    #             # ------------------------------------
    #
    #             # --- NEW STAGING ARCHITECTURE ---
    #             if isinstance(order, PseudoOrder) and getattr(order, 'predecessor', None) is not None:
    #                 # It is a Leg 2 PseudoOrder
    #                 self.staged_pickup_leg2_orders.append(order.get_pickup_node_id())
    #                 self.staged_delivery_leg2_orders.append(order.get_delivery_node_id())
    #                 self.log(self.C_LOG_TYPE_I,
    #                          f"Staged Leg 2 Order: Pickup {order.get_pickup_node_id()} -> Delivery {order.get_delivery_node_id()}")
    #             else:
    #                 # It is a Normal Order OR a Leg 1 PseudoOrder
    #                 self.staged_pickup_orders.append(order.get_pickup_node_id())
    #                 self.staged_delivery_orders.append(order.get_delivery_node_id())
    #                 self.log(self.C_LOG_TYPE_I,
    #                          f"Staged Normal/Leg 1 Order: Pickup {order.get_pickup_node_id()} -> Delivery {order.get_delivery_node_id()}")
    #
    #         self.raise_state_change_event()
    #         return True
    #
    #     return False

    def assign_orders(self, p_orders: list):
        """
        Receives a list of orders from the Supply Chain Manager.
        Maintains backward compatibility by populating old lists while simultaneously
        staging orders for the new Route Sequencer.
        """
        if p_orders:
            from ddls_src.entities.order import PseudoOrder  # Ensure this is imported safely

            if self.get_remaining_capacity() < len(p_orders):
                self.log(self.C_LOG_TYPE_W,
                         f"Vehicle {self.get_id()} REJECTED assignment. "
                         f"Attempted: {len(p_orders)}, Remaining Capacity: {self.get_remaining_capacity()}")
                raise ValueError("Vehicle Overloaded. Please check capacity management and/or constraint management. "
                                 "Agent is taking impossible actions.")

            for order in p_orders:
                pickup_node_id = order.get_pickup_node_id()
                delivery_node_id = order.get_delivery_node_id()
                # --- BACKWARD COMPATIBILITY BLOCK ---
                self.pickup_orders.append(order)
                self.pickup_node_ids.append(order.get_pickup_node_id())

                # if hasattr(self, 'delivery_orders'):
                #     self.delivery_orders.append(order)
                # if hasattr(self, 'delivery_node_ids'):
                #     self.delivery_node_ids.append(order.get_delivery_node_id())
                # ------------------------------------

                # --- NEW STAGING ARCHITECTURE (Node:Order Dictionary Grouping) ---
                if isinstance(order, PseudoOrder) and len(order.predecessor_orders):
                    # It is a Leg 2 PseudoOrder
                    if pickup_node_id in self.staged_pickup_leg2_orders.keys():
                        self.staged_pickup_leg2_orders[pickup_node_id].append(order)
                    else:
                        self.staged_pickup_leg2_orders[pickup_node_id] = [order]
                    if delivery_node_id in self.staged_delivery_leg2_orders:
                        self.staged_delivery_leg2_orders[delivery_node_id].append(order)
                    else:
                        self.staged_delivery_leg2_orders[delivery_node_id] = [order]
                    if self.custom_log:
                             print(f"Staged Leg 2 Order at Pickup {order.get_pickup_node_id()} -> Delivery {order.get_delivery_node_id()}")
                else:
                    # It is a Normal Order OR a Leg 1 PseudoOrder
                    if pickup_node_id in self.staged_pickup_orders:
                        self.staged_pickup_orders[pickup_node_id].append(order)
                    else:
                        self.staged_pickup_orders[pickup_node_id] = [order]
                    if delivery_node_id in self.staged_delivery_orders:
                        self.staged_delivery_orders[order.get_delivery_node_id()].append(order)
                    else:
                        self.staged_delivery_orders[delivery_node_id] = [order]
                    if self.custom_log:
                             print(f"Staged Normal/Leg 1 Order at Pickup {order.get_pickup_node_id()} -> Delivery {order.get_delivery_node_id()}")

            self.raise_state_change_event()
            return True

        return False

    def consolidate_route(self):
        """
        Triggers the Route Sequencer to process all staged orders into a strict timeline,
        clears the staging area, and updates the MLPro state to trigger constraints.
        """
        if self.sequencer is None:
            self.log(self.C_LOG_TYPE_E, "Cannot consolidate: No Route Sequencer assigned to this vehicle.")
            return

        # Guard clause: Don't consolidate if there's nothing staged
        if not (self.staged_pickup_orders or self.staged_pickup_leg2_orders):
            self.log(self.C_LOG_TYPE_I, "Consolidate called, but no new orders are staged.")
            return

            # Inside consolidate_route()

        # 1. Ask the Sequencer to generate the strict timeline
        self.planned_node_sequence, self.planned_order_sequence = self.sequencer.generate_sequence(
            current_node=self.current_node_id,
            pickup_orders=self.staged_pickup_orders,
            delivery_orders=self.staged_delivery_orders,
            pickup_leg2_orders=self.staged_pickup_leg2_orders,
            delivery_leg2_orders=self.staged_delivery_leg2_orders
        )

        self.debug_planned_node_sequence = self.planned_node_sequence.copy()
        self.debug_planned_order_sequence = self.planned_order_sequence.copy()

        # 2. Clear the staging area for the next operational cycle
        self.staged_pickup_orders.clear()
        self.staged_delivery_orders.clear()
        self.staged_pickup_leg2_orders.clear()
        self.staged_delivery_leg2_orders.clear()
        self.current_sequence_index = 0

        # 3. Lock in the mission flag
        self.consolidation_confirmed = True
        self.log(self.C_LOG_TYPE_I, f"Route Consolidated. Strict Sequence: {self.planned_node_sequence}")

        # --- THE CRITICAL FIX: STATE MUTATION ---
        # This shifts the truck out of IDLE and into HALT.
        # This inherently calls `self.raise_state_change_event()`, which wakes up
        # your Constraint Manager to apply the action masks!
        self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_HALT)

        self.log_current_state()
        # self._evaluate_route_state()
    # def _evaluate_route_state(self):
    #     """
    #     Navigation Logic: Evaluates the planned_node_sequence, waits for the Scenario
    #     actions to clear the cargo manifests, and then drives to the next target.
    #     """
    #     current_status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])
    #
    #     # Only evaluate navigation if we are waiting at a node (IDLE or HALT)
    #     if current_status in [self.C_TRIP_STATE_IDLE, self.C_TRIP_STATE_HALT]:
    #
    #         # Guard Clause 1: Sequence is completely empty (Mission Complete)
    #         if not self.planned_node_sequence:
    #             if current_status != self.C_TRIP_STATE_IDLE:
    #                 self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
    #                 self.log_current_state()
    #             return
    #
    #         # Do we still have cargo to load/unload here?
    #         # If yes, stay parked and let the Scenario actions do their work!
    #         if (self.current_node_id in self.pickup_node_ids) or (self.current_node_id in self.delivery_node_ids):
    #             return
    #
    #         # TODO: Add a custom halt check method here
    #         if self._evaluate_halt():
    #             return
    #
    #             # If the manifests for this node are clear, pop it off the sequence!
    #         if self.planned_node_sequence and self.planned_node_sequence[0] == self.current_node_id:
    #             self.planned_node_sequence.pop(0)
    #
    #             # If popping that node emptied the sequence, go IDLE and wait for SCM
    #             if not self.planned_node_sequence:
    #                 self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
    #                 self.consolidation_confirmed = False  # Unlock the mission flag!
    #                 self.log_current_state()
    #                 return
    #
    #         # --- STARTING THE NEXT LEG ---
    #         next_target_node = self.planned_node_sequence[0]
    #
    #         if self.C_NAME == "Drone":
    #             network_type = self.global_state.network.C_NETWORK_AIR
    #         else:
    #             network_type = self.global_state.network.C_NETWORK_GROUND
    #
    #         self.current_leg_duration = self.network_manager.network.get_travel_time(
    #             self.current_node_id,
    #             next_target_node,
    #             network_type=network_type
    #         )
    #
    #         self.en_route_timer = self.current_leg_duration
    #
    #         # Transition state to moving
    #         self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
    #         self.set_current_node_id(None)
    #         self.log_current_state()

    def _evaluate_route_state(self):
        """
        The Brain: Evaluates the sequence clipboard using active list mutation.
        """
        current_status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])

        if current_status != self.C_TRIP_STATE_HALT or not self.consolidation_confirmed:
            return

        # 1. Mission Complete Check (Physical Ground Truth)
        if not self.pickup_orders and not self.delivery_orders:
            self.consolidation_confirmed = False
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
            if self.custom_log:
                print(f"[Vehicle {self.get_id()}] Mission Complete. Vehicle is now IDLE.")
            return

        if self.current_sequence_index not in self.planned_node_sequence:
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
            self.consolidation_confirmed = False
            return

        target_node = self.planned_node_sequence[self.current_sequence_index]

        # 2. Are we at the target node?
        if self.current_node_id == target_node:

            # --- THE MASSIVE SIMPLIFICATION ---
            # Just check if the clipboard for this step is empty!
            orders_left = self.planned_order_sequence.get(self.current_sequence_index, [])

            pickup_orders, delivery_orders = self.planned_order_sequence[self.current_sequence_index]

            all_picked_up = True
            for o in pickup_orders:
                if not isinstance(o, Order):
                    raise TypeError("Order must be of type Order")
                if o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) in [o.C_STATUS_EN_ROUTE]:
                    all_picked_up = True or all_picked_up
                elif o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) not in [o.C_STATUS_EN_ROUTE]:
                    all_picked_up = False
                if not all_picked_up:
                    break

            all_delivered = True
            for o in delivery_orders:
                if not isinstance(o, Order):
                    raise TypeError("Order must be of type Order")
                if o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) in [o.C_STATUS_DELIVERED]:
                    all_delivered = True or all_delivered
                elif o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) not in [o.C_STATUS_DELIVERED]:
                    all_delivered = False
                if not all_delivered:
                    break

            if (not all_picked_up) or (not all_delivered):
                # The list isn't empty yet. Wait for add_cargo/remove_cargo to finish crossing things off.
                return

                # Turn the page!
            self.current_sequence_index += 1

            # Re-evaluate mission completion
            if not self.pickup_orders and not self.delivery_orders:
                self.consolidation_confirmed = False
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
                return

            if self.current_sequence_index not in self.planned_node_sequence:
                return

            target_node = self.planned_node_sequence[self.current_sequence_index]

        # 3. START THE ENGINE
        distance = self.global_state.network.calculate_distance(self.current_node_id, target_node)
        self.en_route_timer = distance / self.get_speed()

        self.set_current_node_id(None)
        self.update_state_value_by_dim_name(
            p_dim_name=[self.C_DIM_AT_NODE[0], self.C_DIM_TRIP_STATE[0]],
            p_value=[False, self.C_TRIP_STATE_EN_ROUTE]
        )

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
        return sum(order.size for order in self.cargo_manifest)

    def get_committed_cargo_size(self) -> int:
        """
        Calculates the total cargo currently in the vehicle PLUS the cargo
        it is promised to pick up (but hasn't yet).
        Note: Assumes 1 order = 1 unit of capacity.
        """
        current_size = self.get_current_cargo_size()
        pending_size = sum(order.size for order in self.pickup_orders)
        return current_size+pending_size

    def get_remaining_capacity(self):
        """
        Calculates exactly how many more orders the vehicle can safely commit to.
        """
        return self.max_payload_capacity - self.get_committed_cargo_size()

    def get_current_cargo(self):
        return self.cargo_manifest

    def get_pickup_orders(self):
        return self.pickup_orders

    def __repr__(self):
        return (f"{self.C_NAME} - {self._id} - {self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])} - "
                f"{self.pickup_orders[0].get_id() if len(self.pickup_orders) else 'None'} - {len(self.get_current_cargo())} - {self.current_node_id}")

    def check_assignability(self) -> bool:
        status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])
        if status in [self.C_TRIP_STATE_IDLE]:
            return True
        else:
            return False

    def get_speed(self):
        return 1

    # def _evaluate_route_state(self):
    #     """
    #     Navigation Logic: Evaluates the route and transitions from IDLE to EN_ROUTE
    #     if all tasks at the current node are cleared by the Event Cascade.
    #     """
    #     current_status = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])
    #
    #     # Only evaluate navigation if we are sitting idle
    #     if current_status == self.C_TRIP_STATE_IDLE:
    #
    #         # Guard Clause: Wait for the Event Cascade to handle loading/unloading
    #         if (self.current_node_id in self.pickup_node_ids) or (self.current_node_id in self.delivery_node_ids):
    #             return
    #
    #         # If tasks are clear (or it was a pass-through node), prepare the next leg
    #         if self.current_route:
    #             self.current_route.pop(0)
    #
    #         if len(self.current_route) >= 2:
    #             new_start, new_end = self.current_route[0], self.current_route[1]
    #             network_type = self.global_state.network.C_NETWORK_AIR if self.C_NAME == "Drone" else self.global_state.network.C_NETWORK_GROUND
    #
    #             # Fetch travel time and set timers
    #             self.current_leg_duration = self.network_manager.network.get_travel_time(new_start, new_end,
    #                                                                                      network_type=network_type)
    #             self.en_route_timer = self.current_leg_duration
    #
    #             # Transition back to moving
    #             self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
    #             self.set_current_node_id(None)
    #             self.log_current_state()
    #         else:
    #             self.route_nodes = []


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
    assert vehicle.current_node_id == 0
    print("\n  - PASSED: State correctly reverted to idle for an invalid route.")

    print("\n--- Validation Complete ---")