from typing import List, Dict, Any, Tuple, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State
from mlpro.bf.math import MSpace, Dimension

# Local Imports
from ddls_src.actions.base import SimulationActions, ActionType
from ddls_src.core.basics import LogisticsAction


# Forward declarations
class GlobalState: pass


class Network: pass


class Truck: pass


class Drone: pass


class Order: pass


class NetworkManager(System):
    """
    Manages all network operations, including vehicle routing, as an MLPro System.
    It now includes automatic, rule-based logic for routing vehicles.
    """

    C_TYPE = 'Network Manager'
    C_NAME = 'Network Manager'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes the NetworkManager system.
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        self.global_state: 'GlobalState' = p_kwargs.get('global_state')
        self.network: 'Network' = p_kwargs.get('network')
        self.automatic_logic_config = p_kwargs.get('p_automatic_logic_config', {})

        if self.global_state is None or self.network is None:
            raise ValueError("NetworkManager requires references to GlobalState and Network.")

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for the NetworkManager.
        """
        state_space = MSpace()
        state_space.add_dim(Dimension('total_nodes', 'Z', 'Total Nodes', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('total_edges', 'Z', 'Total Edges', p_boundaries=[0, 9999]))
        state_space.add_dim(Dimension('blocked_edges', 'Z', 'Blocked Edges', p_boundaries=[0, 9999]))

        # Dynamically find all actions handled by this manager
        handler_name = "NetworkManager"
        action_ids = [action.id for action in SimulationActions.get_all_actions() if action.handler == handler_name]

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='nm_action_id',
                                       p_base_set='Z',
                                       p_name_long='Network Manager Action ID',
                                       p_boundaries=[min(action_ids), max(action_ids)]))

        return state_space, action_space

    def _reset(self, p_seed=None):
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: LogisticsAction, p_t_step: timedelta = None) -> State:
        """
        Updates the manager's state and triggers automatic routing logic if enabled.
        """
        if p_action is not None:
            self._process_action(p_action)

        # --- Automatic Logic ---
        self._check_and_route_vehicles()

        self._update_state()
        return self._state

    def _check_and_route_vehicles(self):
        """
        Scans for newly assigned, idle vehicles and routes them if auto-routing is enabled.
        """
        if not self.automatic_logic_config.get(SimulationActions.TRUCK_TO_NODE, False):
            return

        for order in self.global_state.orders.values():
            if order.status == 'assigned' and hasattr(order,
                                                      'assigned_vehicle_id') and order.assigned_vehicle_id is not None:
                vehicle_id = order.assigned_vehicle_id

                try:
                    vehicle = self.global_state.get_entity("truck",
                                                           vehicle_id) if vehicle_id in self.global_state.trucks else self.global_state.get_entity(
                        "drone", vehicle_id)

                    if vehicle.status == 'idle':
                        self.route_vehicle_for_order(vehicle.get_id(), order.get_id())
                        # We typically only route one vehicle per cycle to avoid action storms
                        break
                except KeyError:
                    continue

    def route_vehicle_for_order(self, vehicle_id: int, order_id: int):
        """
        Calculates a multi-stop route for a vehicle to pick up and deliver an order.
        """
        try:
            order = self.global_state.get_entity("order", order_id)
            vehicle = self.global_state.get_entity("truck",
                                                   vehicle_id) if vehicle_id in self.global_state.trucks else self.global_state.get_entity(
                "drone", vehicle_id)
            vehicle_type_str = 'truck' if vehicle_id in self.global_state.trucks else 'drone'

            pickup_node_id = None
            for node in self.global_state.nodes.values():
                if order_id in node.packages_held:
                    pickup_node_id = node.id
                    break

            if pickup_node_id is None and order_id not in vehicle.cargo_manifest:
                return

            destination_node_id = order.customer_node_id

            if vehicle.current_node_id == pickup_node_id:
                path = self.network.calculate_shortest_path(vehicle.current_node_id, destination_node_id,
                                                            vehicle_type_str)
                if path and len(path) > 1:
                    print(
                        f"  - AUTOMATIC LOGIC (NetworkManager): Vehicle {vehicle_id} is at pickup. Routing directly to destination via {path}.")
                    vehicle.set_route(path)
                return

            path_to_pickup = self.network.calculate_shortest_path(vehicle.current_node_id, pickup_node_id,
                                                                  vehicle_type_str)
            if not path_to_pickup: return

            path_to_destination = self.network.calculate_shortest_path(pickup_node_id, destination_node_id,
                                                                       vehicle_type_str)
            if not path_to_destination: return

            full_path = path_to_pickup + path_to_destination[1:]

            print(
                f"  - AUTOMATIC LOGIC (NetworkManager): Routing Vehicle {vehicle_id} on path {full_path} for Order {order_id}")
            vehicle.set_route(full_path)

        except KeyError:
            pass

    def _process_action(self, p_action: LogisticsAction) -> bool:
        """
        Processes a command by calculating a route and setting it on a vehicle.
        """
        action_id = int(p_action.get_sorted_values()[0])
        action_type = ActionType.get_by_id(action_id)
        action_kwargs = p_action.data

        try:
            if action_type in [SimulationActions.TRUCK_TO_NODE, SimulationActions.RE_ROUTE_TRUCK_TO_NODE]:
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                destination_node_id = action_kwargs['destination_node_id']

                # Pathfinding
                path = self.network.calculate_shortest_path(
                    start_node_id=truck.current_node_id,
                    end_node_id=destination_node_id,
                    vehicle_type='truck'
                )

                if path:
                    truck.set_route(path)
                    return True
                return False

            elif action_type in [SimulationActions.DRONE_LAUNCH, SimulationActions.DRONE_TO_NODE,
                                 SimulationActions.DRONE_TO_CHARGING_STATION]:
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                # Simplified for this example; a real implementation would determine the destination
                destination_node_id = drone.start_node_id
                if 'destination_node_id' in action_kwargs:
                    destination_node_id = action_kwargs['destination_node_id']
                elif 'station_id' in action_kwargs:
                    destination_node_id = action_kwargs['station_id']

                path = self.network.calculate_shortest_path(
                    start_node_id=drone.current_node_id,
                    end_node_id=destination_node_id,
                    vehicle_type='drone'
                )

                if path:
                    drone.set_route(path)
                    return True
                return False

        except (KeyError, AttributeError) as e:
            self.log(self.C_LOG_TYPE_E, f"Action parameter missing or invalid: {e}")
            return False

        return False

    def _update_state(self):
        state_space = self._state.get_related_set()
        nodes = self.global_state.get_all_entities("node").values()
        edges = self.global_state.get_all_entities("edge").values()

        self._state.set_value(state_space.get_dim_by_name("total_nodes").get_id(), len(nodes))
        self._state.set_value(state_space.get_dim_by_name("total_edges").get_id(), len(edges))
        self._state.set_value(state_space.get_dim_by_name("blocked_edges").get_id(),
                              sum(1 for e in edges if e.is_blocked))

    def _simulate_reaction(self, p_state: State, p_action: LogisticsAction, p_t_step: timedelta = None) -> State:
        """
        Updates the manager's state and triggers automatic routing logic if enabled.
        """
        if p_action is not None:
            self._process_action(p_action)

        # The _check_and_route_vehicles method has been removed, as routing is now
        # explicitly triggered by the 'consolidate' action in the SupplyChainManager.

        self._update_state()
        return self._state

    def route_for_assigned_orders(self, vehicle_id: int):
        """
        Calculates a multi-stop tour for a vehicle based on all assigned orders.
        The route includes both pickup and delivery locations.
        """
        try:
            if vehicle_id in self.global_state.trucks:
                vehicle = self.global_state.get_entity("truck", vehicle_id)
            elif vehicle_id in self.global_state.drones:
                vehicle = self.global_state.get_entity("drone", vehicle_id)
            else:
                raise KeyError

            vehicle_type_str = vehicle.C_NAME.lower()

            current_node = vehicle.current_node_id
            full_path = [current_node]
            visited_nodes = {current_node}

            # Gather all pickup and delivery locations for assigned orders
            all_stops = []
            for order in vehicle.delivery_orders:
                # order = self.global_state.get_entity("order", order_id)
                if order:
                    # A consolidated trip needs to visit pickup locations for orders not yet in cargo
                    if order.get_pickup_node_id() not in visited_nodes:
                        all_stops.append(order.get_pickup_node_id())
                        visited_nodes.add(order.get_pickup_node_id())
                    # And deliver to the customer node
                    if order.get_delivery_node_id() not in visited_nodes:
                        all_stops.append(order.get_delivery_node_id())
                        visited_nodes.add(order.get_delivery_node_id())

            # For simplicity, we'll visit the stops in the order they were gathered.
            # A more complex algorithm (e.g., TSP) could be used here.
            for next_stop in all_stops:
                path_to_next_stop = self.network.calculate_shortest_path(current_node, next_stop, vehicle_type_str)
                if path_to_next_stop and len(path_to_next_stop) > 1:
                    full_path.extend(path_to_next_stop[1:])
                    current_node = next_stop
                else:
                    self.log(self.C_LOG_TYPE_W, f"Could not find a path from {current_node} to {next_stop} for vehicle {vehicle_id}.")

            if len(full_path) > 1:
                self.log(self.C_LOG_TYPE_I, f"Routing consolidated Vehicle {vehicle_id} on path: {full_path}.")
                vehicle.set_route(full_path)
            else:
                self.log(self.C_LOG_TYPE_W, f"Could not create a valid route for consolidated Vehicle {vehicle_id}.")
                vehicle.set_route([])

        except KeyError:
            self.log(self.C_LOG_TYPE_E, f"Entity not found for consolidated routing: {vehicle_id}")
        return True


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
if __name__ == '__main__':
    from pprint import pprint


    # 1. Create Mock Objects for the test
    class MockVehicle(System):
        def __init__(self, p_id):
            super().__init__(p_id=p_id)
            self.last_action_received = None

        def get_action_space(self):
            space = MSpace()
            space.add_dim(Dimension(p_name_short="mock_dim"))
            return space

        def process_action(self, p_action):
            self.last_action_received = p_action
            print(
                f"  - MockVehicle '{self.get_id()}' received action with ID {p_action.get_sorted_values()[0]} and data {p_action.data}")
            return True

        @staticmethod
        def setup_spaces(): return None, None


    class MockGlobalState:
        def __init__(self):
            self.trucks = {101: MockVehicle(p_id=101)}
            self.drones = {201: MockVehicle(p_id=201)}

        def get_entity(self, type, id):
            return getattr(self, type + 's', {}).get(id)

        def get_all_entities(self, type):
            return getattr(self, type + 's', {})


    class MockNetwork:
        pass


    mock_gs = MockGlobalState()
    mock_network = MockNetwork()

    print("--- Validating NetworkManager ---")

    # 2. Instantiate NetworkManager
    nm = NetworkManager(p_id='nm_test', global_state=mock_gs, network=mock_network)

    # 3. Test dispatching a truck-related action
    print("\n[A] Testing dispatch to Truck...")
    truck_action = LogisticsAction(
        p_action_space=nm.get_action_space(),
        p_values=[SimulationActions.TRUCK_TO_NODE.id],
        truck_id=101,
        destination_node_id=5
    )
    nm._process_action(truck_action)
    assert mock_gs.trucks[101].last_action_received is not None
    assert mock_gs.trucks[101].last_action_received.get_sorted_values()[0] == SimulationActions.TRUCK_TO_NODE.id
    print("  - PASSED: Correctly dispatched to Truck.")

    # 4. Test dispatching a drone-related action
    print("\n[B] Testing dispatch to Drone...")
    drone_action = LogisticsAction(
        p_action_space=nm.get_action_space(),
        p_values=[SimulationActions.DRONE_TO_NODE.id],
        drone_id=201,
        destination_node_id=10
    )
    nm._process_action(drone_action)
    assert mock_gs.drones[201].last_action_received is not None
    assert mock_gs.drones[201].last_action_received.get_sorted_values()[0] == SimulationActions.DRONE_TO_NODE.id
    print("  - PASSED: Correctly dispatched to Drone.")

    print("\n--- Validation Complete ---")
