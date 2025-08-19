from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional, Set
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
    It now explicitly stores the list of nodes in its current route for
    efficient, real-time constraint checking.
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
        self.cargo_manifest: List[int] = []
        self.current_route: List[int] = []
        self.route_progress: float = 0.0

        # A route is a sequence (list), not a set.
        self.route_nodes: List[int] = []

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for a generic Vehicle system.
        """
        state_space = MSpace()
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
        self.route_nodes = []  # Clear the route nodes on reset
        self._update_state()

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a "go to node" action. Calculates and sets the vehicle's route.
        """
        if self.status not in ["idle", "loading", "unloading"]:
            self.log(self.C_LOG_TYPE_W,
                     f'Vehicle {self.get_id()} is busy (status: {self.status}) and cannot start a new route.')
            return False

        target_node_id = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()

        if self.current_node_id is None:
            self.log(self.C_LOG_TYPE_E, f'Vehicle {self.get_id()} is en-route and cannot start a new route this way.')
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
        Simulates the vehicle's state over a given time step.
        """
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        if self.status == "en_route" and self.current_route and len(self.current_route) >= 2:
            self._move_along_route(p_t_step.total_seconds())

        self._update_state()
        return self._state

    def _move_along_route(self, delta_time: float):
        """
        Internal logic to advance the vehicle along its route.
        """
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

        self.update_energy(-time_to_move)

        if self.route_progress >= 1.0:
            self.current_node_id = end_node_id
            self.current_route.pop(0)
            self.route_progress = 0.0
            if not self.current_route or len(self.current_route) < 2:
                self.status = "idle"
                self.route_nodes = []  # Clear route nodes upon completion
        else:
            self.current_node_id = None

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
        state_space = self._state.get_related_set()
        status_map = {"idle": 0, "en_route": 1, "loading": 2, "unloading": 3, "charging": 4, "maintenance": 5,
                      "broken_down": 6}
        self._state.set_value(state_space.get_dim_by_name("status").get_id(), status_map.get(self.status, 0))
        self._state.set_value(state_space.get_dim_by_name("current_node_id").get_id(),
                              self.current_node_id if self.current_node_id is not None else -1)
        self._state.set_value(state_space.get_dim_by_name("cargo_count").get_id(), len(self.cargo_manifest))

    def set_route(self, route: List[int]):
        """
        Sets the planned route for the vehicle and updates the route_nodes list.
        """
        if not route or len(route) < 2:
            self.log(self.C_LOG_TYPE_W,
                     f"Vehicle {self.get_id()}: Invalid route provided (length < 2). Remaining idle.")
            self.current_route = []
            self.route_nodes = []
            self.status = "idle"
            # FIX: An idle vehicle must be at a node. Restore its last known location.
            # In this context, the last known valid location is the start of the previous route.
            if self.current_node_id is None and self.route_nodes:
                self.current_node_id = self.route_nodes[0]
            elif self.current_node_id is None:
                self.current_node_id = self.start_node_id
            return

        self.current_route = route
        self.route_nodes = route
        self.status = "en_route"
        self.route_progress = 0.0
        self.current_node_id = None
        self._update_state()


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
