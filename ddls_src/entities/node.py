from typing import List, Tuple, Any, Dict
from datetime import timedelta

# MLPro Imports - Assuming mlpro is in the python path
from mlpro.bf.systems import System, State, Action
from mlpro.bf.math import MSpace, Dimension


class Node(System):
    """
    Represents a location in the simulation network as an MLPro System.
    Nodes can be various types like depots, customer locations, or junctions.
    Inherits from the MLPro System class to standardize its behavior.
    """

    C_TYPE = 'Node'
    C_NAME = 'Node'

    def __init__(self,
                 p_id: int,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=True,
                 **p_kwargs):
        """
        Initializes a Node system.

        Parameters:
            p_id (int): Unique identifier for the node.
            p_name (str): Name of the node.
            p_visualize (bool): Visualization flag.
            p_logging: Logging level.
            p_kwargs: Additional keyword arguments for node properties. Expected keys:
                'coords': Tuple[float, float]
                'is_loadable': bool
                'is_unloadable': bool
                'is_charging_station': bool
        """
        # Call the parent System's constructor
        super().__init__(p_id=p_id,
                         p_name=p_name,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         p_mode=System.C_MODE_SIM,
                         p_latency=timedelta(0, 0, 0))  # Nodes react instantly

        # Store node-specific attributes
        self.coords: Tuple[float, float] = p_kwargs.get('coords', (0.0, 0.0))
        self.packages_held: List[int] = []
        self.is_loadable: bool = p_kwargs.get('is_loadable', False)
        self.is_unloadable: bool = p_kwargs.get('is_unloadable', False)
        self.is_charging_station: bool = p_kwargs.get('is_charging_station', False)

        # Initialize the formal state object
        self._state = State(self._state_space)
        self.reset()

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for a Node system.
        A node's state is simply the number of packages it holds.
        A node is passive, so its action space is empty.
        """
        state_space = MSpace()
        state_space.add_dim(Dimension('num_packages', 'Z', 'Number of packages', p_boundaries=[0, 9999]))

        action_space = MSpace()

        return state_space, action_space

    def _reset(self, p_seed=None):
        """
        Resets the node's internal state (clears held packages) and updates the formal state object.
        """
        self.packages_held = []
        self._state.set_value(self._state.get_related_set().get_dim_by_name("num_packages").get_id(), 0)

    def _simulate_reaction(self, p_state: State, p_action: Action, p_t_step: timedelta = None) -> State:
        """
        Simulates the node's reaction. As nodes are passive, this method primarily ensures
        the formal MLPro state object is synchronized with the node's internal attributes.
        """
        # The node's state is changed by external managers calling add/remove_package.
        # This method ensures the formal _state object reflects those changes.
        self._update_state()
        return self._state

    def add_package(self, order_id: int):
        """
        Adds a package to the node and updates its state. Called by external managers.
        """
        if order_id not in self.packages_held:
            self.packages_held.append(order_id)
        self._update_state()

    def remove_package(self, order_id: int):
        """
        Removes a package from the node and updates its state. Called by external managers.
        """
        if order_id in self.packages_held:
            self.packages_held.remove(order_id)
        self._update_state()

    def _update_state(self):
        """Helper method to synchronize the internal list of packages with the formal MLPro state object."""
        self._state.set_value(self._state.get_related_set().get_dim_by_name("num_packages").get_id(),
                              len(self.packages_held))

    def get_packages(self) -> List[int]:
        """
        Returns a copy of the list of order IDs currently held at this node.
        """
        return list(self.packages_held)
