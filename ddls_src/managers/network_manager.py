from typing import List, Dict, Any, Tuple, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


# Forward declarations
class GlobalState: pass


class Network: pass


class Truck: pass


class Drone: pass


class Order: pass


class NetworkManager(System):
    """
    Manages all network operations, including vehicle routing, as an MLPro System.
    It dispatches high-level navigation commands to the appropriate vehicle systems.
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

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='nm_action',
                                       p_base_set='Z',
                                       p_name_long='Network Manager Action',
                                       p_boundaries=[0, 3]))
        # 0: TRUCK_TO_NODE, 1: REROUTE_TRUCK, 2: LAUNCH_DRONE, 3: DRONE_TO_CHARGING

        return state_space, action_space

    def _reset(self, p_seed=None):
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        self._update_state()
        return self._state

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a command by creating a specific action for a vehicle and dispatching it.
        """
        action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()
        action_kwargs = p_action.get_kwargs()

        try:
            if action_value == 0:  # TRUCK_TO_NODE
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                target_node = action_kwargs['destination_node_id']
                # Create a GO_TO_NODE action for the truck
                truck_action = Action(p_action_space=truck.get_action_space(), p_values=[0], target_node=target_node)
                return truck.process_action(truck_action)

            elif action_value == 1:  # REROUTE_TRUCK (same as TRUCK_TO_NODE for now)
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                target_node = action_kwargs['new_destination_node_id']
                truck_action = Action(p_action_space=truck.get_action_space(), p_values=[0], target_node=target_node)
                return truck.process_action(truck_action)

            elif action_value == 2:  # LAUNCH_DRONE
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                order: 'Order' = self.global_state.get_entity("order", action_kwargs['order_id'])
                if action_kwargs['order_id'] not in drone.cargo_manifest: return False
                target_node = order.customer_node_id
                drone_action = Action(p_action_space=drone.get_action_space(), p_values=[0], target_node=target_node)
                return drone.process_action(drone_action)

            elif action_value == 3:  # DRONE_TO_CHARGING
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                target_node = action_kwargs['charging_station_id']
                drone_action = Action(p_action_space=drone.get_action_space(), p_values=[0], target_node=target_node)
                return drone.process_action(drone_action)

        except KeyError as e:
            self.log(self.C_LOG_TYPE_E, f"Action parameter missing: {e}")
            return False

        return False

    def _update_state(self):
        """
        Calculates aggregate network statistics and updates the formal state object.
        """
        nodes = self.global_state.get_all_entities("node").values()
        edges = self.global_state.get_all_entities("edge").values()

        self._state.set_value('total_nodes', len(nodes))
        self._state.set_value('total_edges', len(edges))
        self._state.set_value('blocked_edges', sum(1 for e in edges if e.is_blocked))
