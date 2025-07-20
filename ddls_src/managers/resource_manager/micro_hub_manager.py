from typing import List, Dict, Any, Optional
from datetime import timedelta

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


# Forward declarations
class GlobalState: pass


class MicroHub: pass


class Drone: pass


class MicroHubsManager(System):
    """
    Manages micro-hub operations by dispatching actions to the appropriate hub systems.
    It acts as a mid-level manager in the action processing chain.
    """

    C_TYPE = 'Micro-Hubs Manager'
    C_NAME = 'Micro-Hubs Manager'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes the MicroHubsManager system.
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        self.global_state: 'GlobalState' = p_kwargs.get('global_state')
        if self.global_state is None:
            raise ValueError("MicroHubsManager requires a reference to GlobalState.")

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for the MicroHubsManager.
        """
        state_space = MSpace()
        state_space.add_dim(Dimension('num_micro_hubs', 'Z', 'Total Micro-Hubs', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('active_hubs', 'Z', 'Active Micro-Hubs', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('total_charging_slots', 'Z', 'Total Charging Slots', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('occupied_slots', 'Z', 'Occupied Charging Slots', p_boundaries=[0, 999]))

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='mhm_action',
                                       p_base_set='Z',
                                       p_name_long='Micro-Hub Manager Action',
                                       p_boundaries=[0, 2]))
        # 0: ACTIVATE_HUB, 1: DEACTIVATE_HUB, 2: ADD_TO_CHARGING_QUEUE

        return state_space, action_space

    def _reset(self, p_seed=None):
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        self._update_state()
        return self._state

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a command by dispatching it to the correct MicroHub entity or handling it directly.
        """
        action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()
        action_kwargs = p_action.get_kwargs()

        try:
            hub_id = action_kwargs['micro_hub_id']
            hub: 'MicroHub' = self.global_state.get_entity("micro_hub", hub_id)

            if action_value == 0:  # ACTIVATE_HUB
                hub_action = Action(p_action_space=hub.get_action_space(), p_values=[1])  # 1 = Activate
                return hub.process_action(hub_action)

            elif action_value == 1:  # DEACTIVATE_HUB
                hub_action = Action(p_action_space=hub.get_action_space(), p_values=[0])  # 0 = Deactivate
                return hub.process_action(hub_action)

            elif action_value == 2:  # ADD_TO_CHARGING_QUEUE
                drone_id = action_kwargs['drone_id']
                return self._add_to_charging_queue(hub, drone_id)

        except KeyError as e:
            self.log(self.C_LOG_TYPE_E, f"Action parameter missing: {e}")
            return False

        return False

    def _update_state(self):
        """
        Calculates aggregate micro-hub statistics and updates the formal MLPro state object.
        """
        hubs = self.global_state.get_all_entities("micro_hub").values()
        self._state.set_value('num_micro_hubs', len(hubs))
        self._state.set_value('active_hubs', sum(1 for h in hubs if h.operational_status == 'active'))
        self._state.set_value('total_charging_slots', sum(h.num_charging_slots for h in hubs))
        self._state.set_value('occupied_slots',
                              sum(len(h.charging_slots) - len(h.get_available_charging_slots()) for h in hubs))

    def _add_to_charging_queue(self, p_hub: 'MicroHub', p_drone_id: int) -> bool:
        """
        Internal logic to assign a drone to a charging slot at a specific hub.
        This logic stays in the manager as it coordinates between a Hub and a Drone.
        """
        try:
            drone: 'Drone' = self.global_state.get_entity("drone", p_drone_id)

            if p_hub.operational_status != "active": return False
            if drone.current_node_id != p_hub.id: return False

            available_slots = p_hub.get_available_charging_slots()
            if not available_slots: return False

            slot_to_assign = available_slots[0]
            if p_hub.assign_charging_slot(slot_to_assign, p_drone_id):
                drone.set_status("charging")
                return True
            return False
        except KeyError:
            return False
