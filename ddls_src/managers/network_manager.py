from typing import List, Dict, Any, Tuple, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State
from mlpro.bf.math import MSpace, Dimension

# Local Imports
from ..actions.base import SimulationAction
from ..core.basics import LogisticsAction


# Forward declarations
class GlobalState: pass


class Network: pass


class Truck: pass


class Drone: pass


class Order: pass


class NetworkManager(System):
    """
    Manages all network operations, including vehicle routing, as an MLPro System.
    It now dynamically configures its action space and dispatches actions based
    on the central action blueprint.
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
        action_ids = [action.id for action in SimulationAction if action.handler == handler_name]

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

        self._check_and_route_vehicles()

        self._update_state()
        return self._state

    def _check_and_route_vehicles(self):
        """
        Scans for newly assigned, idle vehicles and routes them if auto-routing is enabled.
        """
        if not self.automatic_logic_config.get(SimulationAction.TRUCK_TO_NODE, False):
            return

        for order in self.global_state.orders.values():
            if order.status == 'assigned' and order.assigned_vehicle_id is not None:
                vehicle_id = order.assigned_vehicle_id

                try:
                    vehicle = self.global_state.get_entity("truck",
                                                           vehicle_id) if vehicle_id in self.global_state.trucks else self.global_state.get_entity(
                        "drone", vehicle_id)

                    if vehicle.status == 'idle':
                        self.route_vehicle_for_order(vehicle.id, order.id)
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
                if path:
                    vehicle.set_route(path)
                return

            path_to_pickup = self.network.calculate_shortest_path(vehicle.current_node_id, pickup_node_id,
                                                                  vehicle_type_str)
            if not path_to_pickup: return

            path_to_destination = self.network.calculate_shortest_path(pickup_node_id, destination_node_id,
                                                                       vehicle_type_str)
            if not path_to_destination: return

            full_path = path_to_pickup + path_to_destination[1:]

            vehicle.set_route(full_path)

        except KeyError:
            pass

    def _process_action(self, p_action: LogisticsAction) -> bool:
        """
        Processes a command by creating a specific action for a vehicle and dispatching it.
        """
        action_id = int(p_action.get_sorted_values()[0])
        action_type = SimulationAction._value2member_map_.get(action_id)
        action_kwargs = p_action.data

        try:
            if action_type in [SimulationAction.TRUCK_TO_NODE, SimulationAction.RE_ROUTE_TRUCK_TO_NODE]:
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                target_node = action_kwargs['destination_node_id']
                truck_action = LogisticsAction(p_action_space=truck.get_action_space(), p_values=[action_id],
                                               **action_kwargs)
                return truck.process_action(truck_action)

            elif action_type in [SimulationAction.LAUNCH_DRONE, SimulationAction.DRONE_TO_NODE,
                                 SimulationAction.DRONE_TO_CHARGING_STATION]:
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                drone_action = LogisticsAction(p_action_space=drone.get_action_space(), p_values=[action_id],
                                               **action_kwargs)
                return drone.process_action(drone_action)

        except KeyError as e:
            self.log(self.C_LOG_TYPE_E, f"Action parameter missing: {e}")
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
