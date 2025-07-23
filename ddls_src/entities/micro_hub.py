from typing import List, Tuple, Any, Dict, Optional
from datetime import timedelta

# Refactored local imports
from .node import Node

# MLPro Imports
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


class MicroHub(Node):
    """
    Represents a micro-hub, a specialized Node that acts as an MLPro System.
    It can activate/deactivate, provide charging slots, and hold packages.
    It now defines its own action space to handle its state changes directly.
    """

    C_TYPE = 'Micro-Hub'
    C_NAME = 'Micro-Hub'

    def __init__(self,
                 p_id: int,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes a MicroHub system.

        Parameters:
            p_id (int): Unique identifier for the micro-hub.
            p_name (str): Name of the micro-hub.
            p_visualize (bool): Visualization flag.
            p_logging: Logging level.
            p_kwargs: Additional keyword arguments. Expected keys:
                'coords': Tuple[float, float]
                'num_charging_slots': int
        """
        p_kwargs['is_loadable'] = True
        p_kwargs['is_unloadable'] = True
        p_kwargs['is_charging_station'] = True

        self.num_charging_slots = p_kwargs.get('num_charging_slots', 0)
        self.charging_slots: Dict[int, Optional[int]] = {i: None for i in range(self.num_charging_slots)}

        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


        self.is_blocked_for_launches: bool = False
        self.is_blocked_for_recoveries: bool = False
        self.is_package_transfer_unavailable: bool = False
        # FIX: Make global_state optional during initialization, defaulting to None
        self.global_state: 'GlobalState' = p_kwargs.get('global_state', None)
        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Extends the Node's state space and defines the MicroHub's flattened discrete action space.
        """
        state_space, _ = Node.setup_spaces()

        state_space.add_dim(
            Dimension('op_status', 'Z', 'Operational Status (0=inactive, 1=active)', p_boundaries=[0, 1]))
        state_space.add_dim(Dimension('occupied_slots', 'Z', 'Occupied Charging Slots', p_boundaries=[0, 99]))
        state_space.add_dim(
            Dimension('blocked_launches', 'Z', 'Blocked for Launches (0=no, 1=yes)', p_boundaries=[0, 1]))
        state_space.add_dim(
            Dimension('blocked_recoveries', 'Z', 'Blocked for Recoveries (0=no, 1=yes)', p_boundaries=[0, 1]))

        # Define a single-dimension, discrete action space for the MicroHub
        action_space = MSpace()
        action_space.add_dim(Dimension(p_name_short='hub_action',
                                       p_base_set='Z',
                                       p_name_long='Hub Action',
                                       p_boundaries=[0, 5]))
        # 0: Deactivate, 1: Activate
        # 2: Unblock Launches, 3: Block Launches
        # 4: Unblock Recoveries, 5: Block Recoveries

        return state_space, action_space

    def _reset(self, p_seed=None):
        """
        Resets the micro-hub to its initial state.
        """
        super()._reset(p_seed)
        self.operational_status = "inactive"
        self.charging_slots = {i: None for i in range(self.num_charging_slots)}
        self.is_blocked_for_launches = False
        self.is_blocked_for_recoveries = False
        self.is_package_transfer_unavailable = False
        self._update_state()

    def _process_action(self, p_action: Action, p_t_step: timedelta = None) -> bool:
        """
        Processes a discrete flattened action sent directly to this MicroHub.
        """
        action_value = p_action.get_elem(self._action_space.get_dim_ids()[0]).get_value()

        if action_value == 0:
            self.operational_status = 'inactive'
        elif action_value == 1:
            self.operational_status = 'active'
        elif action_value == 2:
            self.is_blocked_for_launches = False
        elif action_value == 3:
            self.is_blocked_for_launches = True
        elif action_value == 4:
            self.is_blocked_for_recoveries = False
        elif action_value == 5:
            self.is_blocked_for_recoveries = True

        self._update_state()
        return True

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:

        """
              Processes an action if provided, and synchronizes the formal MLPro state.
        """
        # FIX: Only process an action if one is actually passed to the method
        if p_action is not None:
            self._process_action(p_action, p_t_step)

        self._update_state()
        return self._state

    def _update_state(self):
        """
        Helper method to synchronize all internal attributes with the formal MLPro state object.
        """
        super()._update_state()
        self._state.set_value(self._state.get_related_set().get_dim_by_name('op_status').get_id(),
                              1 if self.operational_status == 'active' else 0)
        self._state.set_value(self._state.get_related_set().get_dim_by_name('occupied_slots').get_id(),
                              len(self.charging_slots) - len(self.get_available_charging_slots()))
        self._state.set_value(self._state.get_related_set().get_dim_by_name('blocked_launches').get_id(),
                              1 if self.is_blocked_for_launches else 0)
        self._state.set_value(self._state.get_related_set().get_dim_by_name('blocked_recoveries').get_id(),
                              1 if self.is_blocked_for_recoveries else 0)

    # Business logic for charging slots remains, as this is managed internally
    # based on other actions (like a drone requesting a charge).
    def assign_charging_slot(self, slot_id: int, drone_id: int) -> bool:
        if slot_id in self.charging_slots and self.charging_slots[slot_id] is None:
            self.charging_slots[slot_id] = drone_id
            self._update_state()
            return True
        return False

    def release_charging_slot(self, slot_id: int) -> bool:
        if slot_id in self.charging_slots and self.charging_slots[slot_id] is not None:
            self.charging_slots[slot_id] = None
            self._update_state()
            return True
        return False

    def get_available_charging_slots(self) -> List[int]:
        return [slot_id for slot_id, drone_id in self.charging_slots.items() if drone_id is None]
