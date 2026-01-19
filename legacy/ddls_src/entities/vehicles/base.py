# In ddls_src/entities/vehicles/base.py
from abc import ABC
from typing import List, Tuple, Optional
from datetime import timedelta

from mlpro.bf.exceptions import ParamError
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension

from ddls_src.actions.action_enums import SimulationAction
from ddls_src.actions.base import SimulationActions, ActionType
from ddls_src.core.basics import LogisticsAction
from ddls_src.entities.base import LogisticEntity
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

    C_ACTION_LOAD = [SimulationActions.LOAD_DRONE_ACTION, SimulationActions.LOAD_TRUCK_ACTION]
    C_ACTION_ROUTE = [SimulationAction.TRUCK_TO_NODE, SimulationActions.DRONE_TO_NODE]
    C_ACTION_UNLOAD = [SimulationAction.UNLOAD_TRUCK_ACTION, SimulationActions.UNLOAD_DRONE_ACTION]
    C_ACTION_ASSIGN_ORDER = [SimulationAction.ASSIGN_ORDER_TO_TRUCK, SimulationActions.ASSIGN_ORDER_TO_DRONE]
    C_ACTION_REROUTE = [SimulationActions.RE_ROUTE_TRUCK_TO_NODE, SimulationActions.RE_ROUTE_DRONE_TO_NODE]
    C_ACTION_CONSOLIDATION = [SimulationActions.CONSOLIDATE_FOR_TRUCK, SimulationActions.CONSOLIDATE_FOR_DRONE]

    C_RELATED_ACTIONS = [C_ACTION_ASSIGN_ORDER
                         + C_ACTION_LOAD
                         + C_ACTION_UNLOAD
                         + C_ACTION_ROUTE
                         + C_ACTION_REROUTE
                         + C_ACTION_CONSOLIDATION]

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

        # Movement mode
        if "p_movement_mode" in p_kwargs.keys():
            self.movement_mode = p_kwargs["p_movement_mode"]
        else:
            raise ParamError("Please provide a movement mode value in the simulation config.")

        # Matrix-based movement attributes
        self.en_route_timer: float = 0.0  # travel time remaining on current leg

        self._state = State(self._state_space)

        self.delivery_orders: List[Order] = []
        self.delivery_node_ids: List[int] = []
        self.pickup_orders: List[Order] = []
        self.pickup_node_ids: List[int] = []

        self.cargo_stats = {}
        self.reset()
        self.consolidation_confirmed: bool = False

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for a generic Vehicle system.
        """
        state_space = MSpace()

        state_space.add_dim(Dimension('loc x', "R", "Current Location X"))
        state_space.add_dim(Dimension('loc y', "R", "Current Location Y"))

        action_space = MSpace()
        action_space.add_dim(
            Dimension(p_name_short='target_node',
                      p_base_set='Z',
                      p_name_long='Target Node ID',
                      p_boundaries=[0, 999]))

        return state_space, action_space

    # --------------------------
    # Helpers (added/fixed)
    # --------------------------
    def _trip_time(self):
        return self.global_state.current_time if self.global_state is not None else 0

    def _log_trip_state(self):
        self.save_data(
            self._trip_time(),
            [self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]), self.current_node_id],
            p_frame=self.C_DATA_FRAME_VEH_STATES
        )

    def _get_network_type(self):
        if self.global_state is None:
            raise RuntimeError("global_state is required to compute travel times.")
        if self.C_NAME == "Drone":
            return self.global_state.network.C_NETWORK_AIR
        if self.C_NAME == "Truck":
            return self.global_state.network.C_NETWORK_GROUND
        raise ValueError(f"Unknown vehicle type: {self.C_NAME}")

    def _has_work_at_node(self, node_id: Optional[int]) -> bool:
        """
        HALT should remain active while there is pickup/delivery work at this node.
        The contract you described: work is 'dealt with' when LOAD/UNLOAD actions
        remove this node from pickup_node_ids/delivery_node_ids.
        """
        if node_id is None:
            return False
        return (node_id in self.pickup_node_ids) or (node_id in self.delivery_node_ids)

    def _start_next_leg_matrix(self) -> bool:
        """
        Starts the next matrix-mode leg if current_route has >= 2 nodes.
        Returns True if EN_ROUTE started, False if went IDLE.
        """
        if not self.current_route or len(self.current_route) < 2:
            # No next hop -> IDLE at current node (if known)
            self.status = "idle"
            self.current_edge = None
            self.en_route_timer = 0.0
            self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
            self.route_nodes = []
            self._log_trip_state()
            return False

        start_node_id, end_node_id = self.current_route[0], self.current_route[1]
        network_type = self._get_network_type()

        self.en_route_timer = float(
            self.network_manager.network.get_travel_time(start_node_id, end_node_id, network_type=network_type)
        )

        # In-transit invariants
        self.current_node_id = None
        self.current_edge = (start_node_id, end_node_id)
        self.status = "en_route"
        self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], False)
        self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
        self._log_trip_state()
        return True

    def _reset(self, p_seed=None):
        """
        Resets the vehicle to its initial state at its starting node.
        """
        self.status = "idle"
        self.consolidation_confirmed = False
        self.current_node_id = self.start_node_id
        self.cargo_manifest = []
        self.current_route = []
        self.route_progress = 0.0
        self.route_nodes = []
        self.delivery_orders = []
        self.en_route_timer = 0.0
        self.cargo_stats = {}

        if self.global_state and self.start_node_id is not None:
            self.current_location_coords = self.global_state.get_entity('node', self.start_node_id).coords
            self.update_state_value_by_dim_name(
                p_dim_name=[self.C_DIM_AVAILABLE[0], self.C_DIM_TRIP_STATE[0], "loc x", "loc y", self.C_DIM_AT_NODE[0]],
                p_value=[1, self.C_TRIP_STATE_IDLE, self.current_location_coords[0], self.current_location_coords[1], True]
            )
        else:
            self.current_location_coords = (0.0, 0.0)
            self.update_state_value_by_dim_name(
                p_dim_name=[self.C_DIM_AVAILABLE[0], self.C_DIM_TRIP_STATE[0], "loc x", "loc y", self.C_DIM_AT_NODE[0]],
                p_value=[1, self.C_TRIP_STATE_IDLE, self.current_location_coords[0], self.current_location_coords[1], True]
            )

        self._log_trip_state()

    def _process_action(self, p_action: LogisticsAction, p_t_step: timedelta = None) -> bool:
        """
        Processes actions.
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
                    f"The order {order_id} is not assigned to the vehicle {truck_id}. "
                    f"The order is not in the pick up orders."
                )

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
            return True

        if action_type == SimulationActions.UNLOAD_TRUCK_ACTION:
            truck_id = action_kwargs["truck_id"]
            order_id = action_kwargs["order_id"]
            truck = self.global_state.get_entity("truck", truck_id)
            order = self.global_state.get_entity("order", order_id)

            if order not in truck.delivery_orders:
                raise ValueError("The order is not in the cargo of the vehicle. The order is not in the delivery orders.")

            truck.delivery_orders.remove(order)
            truck.remove_cargo(order)
            order.set_delivered()
            truck.delivery_node_ids.remove(order.get_delivery_node_id())

            if self.custom_log:
                print(f"Order {order_id} is unloaded from the truck {truck_id}.")

            self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))
            self.save_data(str(order), self.global_state.current_time, p_frame=self.C_DATA_FRAME_VEH_TIMELINE)
            return True

        if action_type == SimulationActions.LOAD_DRONE_ACTION:
            drone_id = action_kwargs["drone_id"]
            order_id = action_kwargs["order_id"]
            drone = self.global_state.get_entity("drone", drone_id)
            order = self.global_state.get_entity("order", order_id)

            if order not in drone.pickup_orders:
                raise ValueError("The order is not assigned to the vehicle. The order is not in the pick up orders.")

            drone.pickup_orders.remove(order)
            drone.delivery_orders.append(order)
            drone.pickup_node_ids.remove(order.get_pickup_node_id())
            drone.delivery_node_ids.append(order.get_delivery_node_id())
            drone.add_cargo(order)
            order.set_enroute()

            if self.custom_log:
                print(f"Order {order_id} is loaded in the Drone {drone_id}.")

            self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))
            self.save_data(str(order), self.global_state.current_time, p_frame=self.C_DATA_FRAME_VEH_TIMELINE)
            return True

        if action_type == SimulationActions.UNLOAD_DRONE_ACTION:
            drone_id = action_kwargs["drone_id"]
            order_id = action_kwargs["order_id"]
            drone = self.global_state.get_entity("drone", drone_id)
            order = self.global_state.get_entity("order", order_id)

            if order not in drone.delivery_orders:
                raise ValueError("The order is not in the cargo of the vehicle. The order is not in the delivery orders.")

            drone.delivery_orders.remove(order)
            drone.remove_cargo(order)
            order.set_delivered()
            drone.delivery_node_ids.remove(order.get_delivery_node_id())

            if self.custom_log:
                print(f"Order {order_id} is unloaded from the drone {drone_id}.")

            self.update_state_value_by_dim_name(self.C_DIM_CURRENT_CARGO[0], len(self.cargo_manifest))
            self.save_data(str(order), self.global_state.current_time, p_frame=self.C_DATA_FRAME_VEH_TIMELINE)
            return True

        if action_type == SimulationActions.TRUCK_TO_NODE:
            pass
        if action_type == SimulationActions.DRONE_TO_NODE:
            pass
        if action_type == SimulationActions.DRONE_LAUNCH:
            pass
        if action_type == SimulationActions.DRONE_LAND:
            pass

        return False

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Simulates the vehicle's state over a given time step.
        """
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        if p_t_step is None:
            return self._state

        if self.movement_mode == 'matrix':
            # IMPORTANT: let HALT process even if route is short/empty
            if self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) in [self.C_TRIP_STATE_EN_ROUTE, self.C_TRIP_STATE_HALT]:
                self._update_matrix_movement(p_t_step.total_seconds())
        else:
            if (self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0]) == self.C_TRIP_STATE_EN_ROUTE
                    and self.current_route and len(self.current_route) >= 2):
                self._move_along_route(p_t_step.total_seconds())

        return self._state

    # --------------------------
    # FIXED: matrix movement with condition-based HALT
    # --------------------------
    def _update_matrix_movement(self, delta_time: float):
        """Handles movement for the 'matrix' mode (fixed; HALT waits until node work is done)."""
        dt_left = float(delta_time)

        while dt_left > 0:
            trip_state = self.get_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0])

            # HALT: stay until no pickup/delivery work remains at this node
            if trip_state == self.C_TRIP_STATE_HALT:
                self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)

                if self._has_work_at_node(self.current_node_id):
                    return  # still work to do here

                # Work cleared -> depart if possible
                self._start_next_leg_matrix()
                return

            # EN_ROUTE: progress along edge
            if trip_state == self.C_TRIP_STATE_EN_ROUTE:
                if not self.current_route or len(self.current_route) < 2:
                    # Invalid route while EN_ROUTE -> recover safely
                    self.status = "idle"
                    self.current_edge = None
                    self.en_route_timer = 0.0
                    self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)
                    self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
                    self._log_trip_state()
                    return

                if self.en_route_timer is None:
                    self.en_route_timer = 0.0

                start_node_id, end_node_id = self.current_route[0], self.current_route[1]

                # Consume travel time
                step = min(dt_left, max(self.en_route_timer, 0.0))
                self.en_route_timer -= step
                dt_left -= step

                if self.en_route_timer > 0:
                    # Still traveling
                    self.current_node_id = None
                    self.current_edge = (start_node_id, end_node_id)
                    self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], False)
                    return

                # ARRIVAL
                self.current_node_id = end_node_id
                self.current_edge = None
                self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)

                # CRITICAL FIX: ALWAYS advance route head on arrival (even at pickup/delivery)
                self.current_route.pop(0)

                # If work exists here -> HALT
                if self._has_work_at_node(end_node_id):
                    self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_HALT)
                    self._log_trip_state()

                    if self.custom_log:
                        kind = "pickup" if end_node_id in self.pickup_node_ids else "delivery"
                        print(f"\nVehicle {self._id} reached {kind} node {end_node_id} (HALT until cleared)\n")

                    return

                # No work here -> continue or finish
                if not self.current_route or len(self.current_route) < 2:
                    self.status = "idle"
                    self.route_nodes = []
                    self.en_route_timer = 0.0
                    self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
                    self._log_trip_state()
                    return

                # Start next leg and keep consuming dt_left if any remains
                self._start_next_leg_matrix()
                continue

            # IDLE or other states: nothing to do
            return

    # --------------------------
    # Existing network-mode movement (unchanged)
    # --------------------------
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
            self._log_trip_state()
            return

        travel_time = edge.get_current_travel_time() if self.C_NAME == 'Truck' else edge.get_drone_flight_time()

        if travel_time <= 0 or travel_time == float('inf'):
            self.status = "idle"
            self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
            self._log_trip_state()
            return

        time_needed = (1.0 - self.route_progress) * travel_time
        time_to_move = min(delta_time, time_needed)

        self.route_progress += time_to_move / travel_time

        # Update coordinates based on progress
        self._update_location_coords(start_node_id, end_node_id, self.route_progress)

        self.update_energy(-time_to_move)

        if self.route_progress >= 1.0:
            self.current_node_id = end_node_id
            self.current_edge = None

            if (end_node_id in self.pickup_node_ids) or (end_node_id in self.delivery_node_ids):
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_HALT)
                self._log_trip_state()
            else:
                self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)

            self.current_route.pop(0)
            self.route_progress = 0.0

            if not self.current_route or len(self.current_route) < 2:
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_IDLE)
                self.status = "idle"
                self.route_nodes = []
                self._log_trip_state()
                self.raise_state_change_event()
            else:
                self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
                self._log_trip_state()
                self.raise_state_change_event()

            self._update_location_coords(self.current_node_id, self.current_node_id, self.route_progress)
        else:
            self.current_node_id = None

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

    def remove_cargo(self, order: int):
        """Removes a package from the vehicle's cargo manifest."""
        if order in self.cargo_manifest:
            self.cargo_manifest.remove(order)
            self.cargo_stats[self.global_state.current_time] = self.get_current_cargo_size()

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

            # Restore a node if needed
            if self.current_node_id is None:
                self.current_node_id = self.start_node_id

            self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], True)
            self._log_trip_state()
            return

        self.current_route = route
        self.route_nodes = route

        if self.movement_mode == 'matrix':
            # Start first leg using the unified helper (sets AT_NODE False, trip state, timer, logging)
            self._start_next_leg_matrix()
            return

        # network mode
        self.status = "en_route"
        self.route_progress = 0.0
        self.current_node_id = None
        self.update_state_value_by_dim_name(self.C_DIM_AT_NODE[0], False)
        self.update_state_value_by_dim_name(self.C_DIM_TRIP_STATE[0], self.C_TRIP_STATE_EN_ROUTE)
        self._log_trip_state()

    def get_current_location(self):
        return self.get_state_value_by_dim_name("loc x"), self.get_state_value_by_dim_name("loc y")

    def get_delivery_orders(self):
        return self.delivery_orders

    def assign_orders(self, p_orders: list):
        if p_orders:
            for ord in p_orders:
                self.pickup_orders.append(ord)
                self.pickup_node_ids.append(ord.get_pickup_node_id())
                self.raise_state_change_event()
            return True
        return False

    def unload_order(self, p_order):
        if p_order:
            self.delivery_orders.remove(p_order)

    def get_current_node(self):
        return self.current_node_id

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
        return status in [self.C_TRIP_STATE_IDLE]


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Create a Mock Vehicle class for the test
    class MockVehicle(Vehicle):
        C_NAME = "Truck"  # needed for network type selection in matrix mode

        def update_energy(self, p_time_passed: float):
            pass


    print("--- Validating Vehicle Base Class ---")

    # NOTE: This validation block won't fully run without a real global_state and network_manager.
    # It's kept here as in your original file, but movement requires those dependencies.

    vehicle = MockVehicle(p_id=101, start_node_id=0, p_movement_mode="matrix", network_manager=None)
    print(f"\n[A] Initial State:")
    print(f"  - Status: {vehicle.status}")
    print(f"  - Current Node: {vehicle.current_node_id}")
    print(f"  - Route: {vehicle.current_route}")
    print(f"  - Route Nodes: {vehicle.route_nodes}")

    assert vehicle.status == 'idle'
    assert vehicle.route_nodes == []

    new_route = [0, 5, 12, 5]
    print(f"\n[B] Setting a new route: {new_route}")
    vehicle.set_route(new_route)

    print("\n[C] State after setting route:")
    print(f"  - Status: {vehicle.status}")
    print(f"  - Current Node: {vehicle.current_node_id}")
    print(f"  - Route: {vehicle.current_route}")
    print(f"  - Route Nodes: {vehicle.route_nodes}")

    # In matrix mode, set_route will attempt to start travel time using network_manager+global_state.
    # If those aren't set, it'll raise at runtime when starting the next leg.

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

    print("\n--- Validation Complete ---")
 