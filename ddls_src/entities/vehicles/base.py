from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


# Forward declaration for NetworkManager
class NetworkManager:
    pass


class Vehicle(System, ABC):
    """
    Abstract base class for all vehicles, refactored as an MLPro System.
    Defines common state, actions, and the core movement simulation logic.
    """

    C_TYPE = 'Vehicle'
    C_NAME = 'Vehicle'

    def __init__(self,
                 p_id: int,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes a Vehicle system.

        Parameters:
            p_id (int): Unique identifier for the vehicle.
            p_name (str): Name of the vehicle.
            p_visualize (bool): Visualization flag.
            p_logging: Logging level.
            p_kwargs: Additional keyword arguments. Expected keys:
                'start_node_id': int
                'max_payload_capacity': float
                'max_speed': float
                'network_manager': NetworkManager
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        # Vehicle-specific attributes
        self.start_node_id: int = p_kwargs.get('start_node_id')
        self.max_payload_capacity: float = p_kwargs.get('max_payload_capacity', 0)
        self.max_speed: float = p_kwargs.get('max_speed', 0)
        self.network_manager: 'NetworkManager' = p_kwargs.get('network_manager')

        # Internal dynamic attributes
        self.status: str = "idle"
        self.current_node_id: Optional[int] = self.start_node_id
        self.current_location_coords: Tuple[float, float] = (0.0, 0.0)
        self.cargo_manifest: List[int] = []
        self.current_route: List[int] = []
        self.route_progress: float = 0.0

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for a generic Vehicle system.
        """
        state_space = MSpace()
        # Status: 0=idle, 1=en_route, 2=loading, 3=unloading, 4=charging, 5=maintenance, 6=broken_down
        state_space.add_dim(Dimension('status', 'Z', 'Vehicle Status', p_boundaries=[0, 6]))
        state_space.add_dim(
            Dimension('current_node_id', 'Z', 'Current Node ID (-1 for en-route)', p_boundaries=[-1, 999]))
        state_space.add_dim(Dimension('cargo_count', 'Z', 'Number of packages in cargo', p_boundaries=[0, 99]))

        action_space = MSpace()
        action_space.add_dim(
            Dimension(p_name_short='target_node', p_base_set='Z', p_name_long='Target Node ID', p_boundaries=[0, 999]))

        return state_space, action_space

    def _reset(self, p_seed=None):
        """
        Resets the vehicle to its initial state at its starting node.
        """
        self.status = "idle"
        self.current_node_id = self.start_node_id
        self.cargo_manifest = []
        self.current_route = []
        self.route_progress = 0.0
        # In a full implementation, coords would be set from the start node's coords
        # self.current_location_coords = self.network_manager.global_state.get_entity("node", self.start_node_id).coords
        self._update_state()

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a "go to node" action. Calculates and sets the vehicle's route.
        """
        if self.status not in ["idle", "loading", "unloading"]:
            self.log(self.C_LOG_TYPE_W,
                     f'Vehicle {self.id} is busy (status: {self.status}) and cannot start a new route.')
            return False

        target_node_id = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()

        if self.current_node_id is None:
            self.log(self.C_LOG_TYPE_E, f'Vehicle {self.id} is en-route and cannot start a new route this way.')
            return False

        path = self.network_manager.network.calculate_shortest_path(self.current_node_id, target_node_id,
                                                                    self.C_NAME.lower())

        if not path:
            self.log(self.C_LOG_TYPE_W, f'No path found from {self.current_node_id} to {target_node_id}.')
            return False

        self.set_route(path)
        return True

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Simulates the vehicle's movement over a given time step.
        """
        if self.status == "en_route" and self.current_route and len(self.current_route) >= 2:
            self._move_along_route(p_t_step.total_seconds())

        self._update_state()
        return self._state

    def _move_along_route(self, delta_time: float):
        """
        Internal logic to advance the vehicle along its route.
        """
        # This logic is adapted from the original move_along_route method
        start_node_id, end_node_id = self.current_route[0], self.current_route[1]
        edge = self.network_manager.network.get_edge_between_nodes(start_node_id, end_node_id)

        if not edge:
            self.status = "idle"
            return

        travel_time = edge.get_current_travel_time() if self.C_NAME == 'Truck' else edge.get_drone_flight_time()

        if travel_time <= 0 or travel_time == float('inf'):
            self.status = "idle"
            return

        time_needed = (1.0 - self.route_progress) * travel_time
        time_to_move = min(delta_time, time_needed)

        self.route_progress += time_to_move / travel_time

        # Update energy (to be implemented by child classes)
        self.update_energy(-time_to_move)

        if self.route_progress >= 1.0:
            self.current_node_id = end_node_id
            self.current_route.pop(0)
            self.route_progress = 0.0
            if not self.current_route or len(self.current_route) < 2:
                self.status = "idle"
            # Here you would also update coordinates to the node's exact location
        else:
            self.current_node_id = None  # En-route
            # Here you would interpolate coordinates based on route_progress

    @abstractmethod
    def update_energy(self, p_time_passed: float):
        """
        Abstract method for energy consumption.
        """
        pass

    def _update_state(self):
        """
        Synchronizes internal attributes with the formal MLPro state object.
        """
        status_map = {"idle": 0, "en_route": 1, "loading": 2, "unloading": 3, "charging": 4, "maintenance": 5,
                      "broken_down": 6}
        self._state.set_value('status', status_map.get(self.status, 0))
        self._state.set_value('current_node_id', self.current_node_id if self.current_node_id is not None else -1)
        self._state.set_value('cargo_count', len(self.cargo_manifest))

    # Public methods for managers to call
    def set_route(self, route: List[int]):
        self.current_route = route
        self.status = "en_route"
        self.route_progress = 0.0
        self.current_node_id = None
        self._update_state()
