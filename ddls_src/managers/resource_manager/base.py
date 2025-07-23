from typing import List, Dict, Any, Optional
from datetime import timedelta

# Refactored local imports
from .fleet_manager import FleetManager
from .micro_hub_manager import MicroHubsManager

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


# Forward declarations
class GlobalState: pass


class ResourceManager(System):
    """
    Top-level manager for all physical resources (vehicles and micro-hubs).
    As an MLPro System, it aggregates state data from its sub-managers and
    dispatches high-level resource commands down the chain.
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

        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='rm_action',
                                       p_base_set='Z',
                                       p_name_long='Resource Manager Action',
                                       p_boundaries=[0, 5]))
        # 0: LOAD_TRUCK, 1: UNLOAD_TRUCK, 2: LOAD_DRONE, 3: UNLOAD_DRONE (to FleetManager)
        # 4: ACTIVATE_HUB, 5: DEACTIVATE_HUB (to MicroHubsManager)

        return state_space, action_space

    def _reset(self, p_seed=None):
        self.fleet_manager.reset(p_seed)
        self.micro_hubs_manager.reset(p_seed)
        self._update_state()

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Simulates the reaction of sub-managers and updates its own aggregate state.
        """
        # The simulation of individual vehicles/hubs is handled at their level.
        # This manager's simulation step is to update its own aggregate state.
        self.fleet_manager.simulate_reaction(p_state=None, p_action=None, p_t_step=p_t_step)
        self.micro_hubs_manager.simulate_reaction(p_state=None, p_action=None, p_t_step=p_t_step)
        self._update_state()
        return self._state

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a command by dispatching it to the correct sub-manager.
        """
        action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()
        action_kwargs = p_action.get_kwargs()

        # Actions 0-3 are for the FleetManager
        if 0 <= action_value <= 3:
            fm_action = Action(p_action_space=self.fleet_manager.get_action_space(), p_values=[action_value],
                               **action_kwargs)
            return self.fleet_manager.process_action(fm_action)

        # Actions 4-5 are for the MicroHubsManager
        elif 4 <= action_value <= 5:
            # Map RM action value to MHM action value (4->0, 5->1)
            mhm_action_value = action_value - 4
            mhm_action = Action(p_action_space=self.micro_hubs_manager.get_action_space(), p_values=[mhm_action_value],
                                **action_kwargs)
            return self.micro_hubs_manager.process_action(mhm_action)

        return False

    def _update_state(self):
        """
        Calculates aggregate resource statistics and updates the formal state object.
        """
        trucks = self.global_state.get_all_entities("truck").values()
        drones = self.global_state.get_all_entities("drone").values()
        hubs = self.global_state.get_all_entities("micro_hub").values()

        self._state.set_value(self._state.get_related_set().get_dim_by_name('num_vehicles').get_id(),
                              len(trucks) + len(drones))
        self._state.set_value(self._state.get_related_set().get_dim_by_name('num_hubs').get_id(),
                              len(hubs))
        self._state.set_value(self._state.get_related_set().get_dim_by_name('vehicles_in_maintenance').get_id(),
                              sum(1 for v in trucks if v.status == 'maintenance') + sum(
            1 for v in drones if v.status == 'maintenance'))
