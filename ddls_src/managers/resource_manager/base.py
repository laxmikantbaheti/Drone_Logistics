from typing import List, Dict, Any, Optional
from datetime import timedelta

# Refactored local imports
from ddls_src.managers.resource_manager.fleet_manager import FleetManager
from ddls_src.managers.resource_manager.micro_hub_manager import MicroHubsManager
from ddls_src.core.basics import LogisticsAction
from ddls_src.actions.base import SimulationAction

# MLPro Imports
from mlpro.bf.systems import System, State
from mlpro.bf.math import MSpace, Dimension


# Forward declaration for GlobalState to avoid circular dependency
class GlobalState:
    pass


class ResourceManager(System):
    """
    Manages various resources within the simulation. It now dynamically configures
    its action space and dispatches actions based on the central action blueprint.
    """

    C_TYPE = 'Resource Manager'
    C_NAME = 'Resource Manager'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes the ResourceManager system.
        """
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))

        self.global_state: 'GlobalState' = p_kwargs.get('global_state')
        if self.global_state is None:
            raise ValueError("ResourceManager requires a reference to GlobalState.")

        # Instantiate sub-managers
        self.fleet_manager = FleetManager(p_id=self.get_id() + '.fleet', global_state=self.global_state)
        self.micro_hubs_manager = MicroHubsManager(p_id=self.get_id() + '.micro_hubs', global_state=self.global_state)

        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for the ResourceManager.
        """
        state_space = MSpace()
        state_space.add_dim(Dimension('num_vehicles', 'Z', 'Total Vehicles', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('num_hubs', 'Z', 'Total Hubs', p_boundaries=[0, 999]))
        state_space.add_dim(Dimension('vehicles_in_maintenance', 'Z', 'Vehicles in Maintenance', p_boundaries=[0, 999]))

        # Dynamically find all actions handled by this manager
        handler_name = "ResourceManager"
        action_ids = [action.id for action in SimulationAction if action.handler == handler_name]

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='rm_action_id',
                                       p_base_set='Z',
                                       p_name_long='Resource Manager Action ID',
                                       p_boundaries=[min(action_ids), max(action_ids)]))

        return state_space, action_space

    def _reset(self, p_seed=None):
        self.fleet_manager.reset(p_seed)
        self.micro_hubs_manager.reset(p_seed)
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: LogisticsAction, p_t_step: timedelta = None) -> State:
        """
        Simulates the reaction of sub-managers and updates its own aggregate state.
        """
        if p_action is not None:
            self._process_action(p_action)

        self.fleet_manager.simulate_reaction(p_state=None, p_action=None, p_t_step=p_t_step)
        self.micro_hubs_manager.simulate_reaction(p_state=None, p_action=None, p_t_step=p_t_step)
        self._update_state()
        return self._state

    def _process_action(self, p_action: LogisticsAction) -> bool:
        """
        Processes a command by dispatching it to the correct sub-manager.
        """
        action_id = int(p_action.get_sorted_values()[0])
        action_type = SimulationAction._value2member_map_.get(action_id)
        action_kwargs = p_action.data

        # This logic could be made more robust by reading sub-handler info from the blueprint
        # For now, we assume a simple mapping based on the action type name

        # Actions for the FleetManager
        if "TRUCK" in action_type.name or "DRONE" in action_type.name:
            fm_action = LogisticsAction(p_action_space=self.fleet_manager.get_action_space(), p_values=[action_id],
                                        **action_kwargs)
            return self.fleet_manager.process_action(fm_action)

        # Actions for the MicroHubsManager
        elif "HUB" in action_type.name:
            mhm_action = LogisticsAction(p_action_space=self.micro_hubs_manager.get_action_space(),
                                         p_values=[action_id], **action_kwargs)
            return self.micro_hubs_manager.process_action(mhm_action)

        return False

    def _update_state(self):
        """
        Calculates aggregate resource statistics and updates the formal state object.
        """
        state_space = self._state.get_related_set()
        trucks = self.global_state.get_all_entities("truck").values()
        drones = self.global_state.get_all_entities("drone").values()
        hubs = self.global_state.get_all_entities("micro_hub").values()

        self._state.set_value(state_space.get_dim_by_name("num_vehicles").get_id(), len(trucks) + len(drones))
        self._state.set_value(state_space.get_dim_by_name("num_hubs").get_id(), len(hubs))
        self._state.set_value(state_space.get_dim_by_name("vehicles_in_maintenance").get_id(),
                              sum(1 for v in trucks if v.status == 'maintenance') +
                              sum(1 for v in drones if v.status == 'maintenance'))
