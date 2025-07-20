from typing import List, Dict, Any, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


# Forward declarations for GlobalState and refactored entities
class GlobalState: pass


class Truck: pass


class Drone: pass


class Node: pass


class Order: pass


class FleetManager(System):
    """
    Manages vehicle fleet operations by dispatching actions to the appropriate vehicle systems.
    It acts as a mid-level manager in the action processing chain.
    """

    C_TYPE = 'Fleet Manager'
    C_NAME = 'Fleet Manager'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes the FleetManager system.
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        self.global_state: 'GlobalState' = p_kwargs.get('global_state')
        if self.global_state is None:
            raise ValueError("FleetManager requires a reference to GlobalState.")

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for the FleetManager.
        """
        state_space = MSpace()
        state_space.add_dim(Dimension('num_trucks', 'Z', 'Total Trucks', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('num_drones', 'Z', 'Total Drones', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('trucks_idle', 'Z', 'Idle Trucks', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('drones_idle', 'Z', 'Idle Drones', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('trucks_en_route', 'Z', 'Trucks En Route', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('drones_en_route', 'Z', 'Drones En Route', p_boundaries=[0, 999]))

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='fm_action',
                                       p_base_set='Z',
                                       p_name_long='Fleet Manager Action',
                                       p_boundaries=[0, 3]))
        # 0: LOAD_TRUCK, 1: UNLOAD_TRUCK, 2: LOAD_DRONE, 3: UNLOAD_DRONE

        return state_space, action_space

    def _reset(self, p_seed=None):
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        self._update_state()
        return self._state

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a command by dispatching it to the correct vehicle entity.
        """
        action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()
        action_kwargs = p_action.get_kwargs()

        try:
            if action_value == 0:  # LOAD_TRUCK
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                truck_action = Action(p_action_space=truck.get_action_space(), p_values=[1],
                                      **action_kwargs)  # 1 = LOAD_ORDER
                return truck.process_action(truck_action)

            elif action_value == 1:  # UNLOAD_TRUCK
                truck: 'Truck' = self.global_state.get_entity("truck", action_kwargs['truck_id'])
                truck_action = Action(p_action_space=truck.get_action_space(), p_values=[2],
                                      **action_kwargs)  # 2 = UNLOAD_ORDER
                return truck.process_action(truck_action)

            elif action_value == 2:  # LOAD_DRONE
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                drone_action = Action(p_action_space=drone.get_action_space(), p_values=[1],
                                      **action_kwargs)  # 1 = LOAD_ORDER
                return drone.process_action(drone_action)

            elif action_value == 3:  # UNLOAD_DRONE
                drone: 'Drone' = self.global_state.get_entity("drone", action_kwargs['drone_id'])
                drone_action = Action(p_action_space=drone.get_action_space(), p_values=[2],
                                      **action_kwargs)  # 2 = UNLOAD_ORDER
                return drone.process_action(drone_action)

        except KeyError as e:
            self.log(self.C_LOG_TYPE_E, f"Action parameter missing: {e}")
            return False

        return False

    def _update_state(self):
        """
        Calculates aggregate fleet statistics and updates the formal MLPro state object.
        """
        trucks = self.global_state.get_all_entities("truck").values()
        drones = self.global_state.get_all_entities("drone").values()

        self._state.set_value('num_trucks', len(trucks))
        self._state.set_value('num_drones', len(drones))
        self._state.set_value('trucks_idle', sum(1 for t in trucks if t.status == 'idle'))
        self._state.set_value('drones_idle', sum(1 for d in drones if d.status == 'idle'))
        self._state.set_value('trucks_en_route', sum(1 for t in trucks if t.status == 'en_route'))
        self._state.set_value('drones_en_route', sum(1 for d in drones if d.status == 'en_route'))
