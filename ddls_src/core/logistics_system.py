# In file: ddls_src/core/logistics_system.py

# Import the 'random' module for generating random numbers, used in the validation block.
import random
# Import typing hints for better code readability and static analysis.
from typing import Dict, Any, List, Tuple
# Import 'timedelta' for representing differences in time.
from datetime import timedelta
# Import NumPy for numerical operations, especially for handling arrays like action masks.
import numpy as np
# Import the 'os' module for interacting with the operating system, used for path manipulation.
import os

# MLPro Imports
# Import base classes from MLPro for creating systems and states.
from mlpro.bf.systems import System, State
# Import mathematical space and dimension classes from MLPro for defining state and action spaces.
from mlpro.bf.math import MSpace, Dimension
# Import event management classes from MLPro for handling events within the system.
from mlpro.bf.events import EventManager, Event

# from ddls_src.actions.action_map_generator import generate_action_map
# Local Imports
# Import core components of the logistics simulation.
from ddls_src.core.global_state import GlobalState
from ddls_src.core.network import Network
# Import manager classes that handle different aspects of the simulation logic.
from ddls_src.managers.action_manager import ActionManager
from ddls_src.managers.supply_chain_manager import SupplyChainManager
from ddls_src.managers.resource_manager.base import ResourceManager
from ddls_src.managers.network_manager import NetworkManager
# Import action-related classes.
from ddls_src.actions.base import SimulationActions, ActionType, ActionIndex
# Import scenario and data generation utilities.
from ddls_src.scenarios.generators.data_loader import DataLoader
from ddls_src.scenarios.generators.scenario_generator import ScenarioGenerator
from ddls_src.scenarios.generators.order_generator import OrderGenerator
# Import mapping and constraint management classes.
from ddls_src.core.state_action_mapper import StateActionMapper, ConstraintManager
# Import simulation time management and action representation.
from ddls_src.core.logistics_simulation import TimeManager
from ddls_src.core.basics import LogisticsAction
# Import all entity classes (e.g., Truck, Drone, Hub).
from ddls_src.entities import *


class LogisticsSystem(System, EventManager):
    """
    The top-level MLPro System that IS the entire logistics simulation engine.
    It uses a two-phase cycle: a Decision Phase (action processing) and a
    Progression Phase (time advancement).
    """

    # Class constant defining the type of the system.
    C_TYPE = 'Logistics System'
    # Class constant defining the name of the system.
    C_NAME = 'Logistics System'
    # Class constant for the event triggered when a new order is created.
    C_EVENT_NEW_ORDER = 'NEW_ORDER_CREATED'

    def __init__(self,
                 p_id=None,
                 p_name: str = '',
                 p_visualize: bool = False,
                 p_logging=False,
                 **p_kwargs):
        """
        Initializes the LogisticsSystem.

        Parameters:
            p_id: Unique ID of the system.
            p_name (str): Name of the system.
            p_visualize (bool): Flag to enable/disable visualization.
            p_logging: Flag or level for logging.
            **p_kwargs: Additional keyword arguments, expected to contain 'config'.
        """
        # Retrieve the configuration dictionary from keyword arguments.
        self._config = p_kwargs.get('config', {})

        # Initialize the parent System class from MLPro.
        System.__init__(self, p_id=p_id,
                        p_name=p_name,
                        p_visualize=p_visualize,
                        p_logging=p_logging,
                        p_mode=System.C_MODE_SIM,  # Set the system mode to simulation.
                        # Set the simulation latency (timestep duration) from the config.
                        p_latency=timedelta(seconds=self._config.get("main_timestep_duration", 60.0)))
        # Initialize the parent EventManager class from MLPro.
        EventManager.__init__(self, p_logging=self.get_log_level())
        # Determine the movement mode (e.g., 'network' or continuous space) from the config.
        self.movement_mode = self._config.get('movement_mode', 'network')
        # Initialize attributes to be configured later.
        self.automatic_logic_config = {}
        # Initialize the TimeManager with the initial time from the config.
        self.time_manager = TimeManager(initial_time=self._config.get("initial_time", 0.0))
        # Initialize the DataLoader to load initial simulation data.
        self.data_loader = DataLoader(self._config.get("data_loader_config", {}))
        # Dictionary to map action tuples to integer indices.
        self.action_map: Dict[Tuple, int] = {}
        # Dictionary to map integer indices back to action tuples.
        self._reverse_action_map: Dict[int, Tuple] = {}
        # Total size of the action space.
        self.action_space_size: int = 0
        # The central state repository for the entire simulation.
        self.global_state: GlobalState = None
        # The network graph representing locations and distances.
        self.network: Network = None
        # Manages the execution of actions.
        self.action_manager: ActionManager = None
        # Manages orders, products, and supply chain logic.
        self.supply_chain_manager: SupplyChainManager = None
        # Manages resources like vehicles and hubs.
        self.resource_manager: ResourceManager = None
        # Manages vehicle movements and network-related logic.
        self.network_manager: NetworkManager = None
        # Generates new orders during the simulation.
        self.order_generator: OrderGenerator = None
        # Maps the current global state to valid actions.
        self.state_action_mapper: StateActionMapper = None
        # Manages constraints on actions.
        self.constraint_manager: ConstraintManager = None
        # The MLPro State object representing the system's state for the reinforcement learning agent.
        self._state = State(self._state_space)
        # An object representing all possible simulation actions.
        self.actions = SimulationActions()
        # An object for indexing actions.
        self.action_index = None
        # Call the reset method to perform the main setup.
        self.reset()

    # --------------------------------------------------------------------------------------------------

    @staticmethod
    def setup_spaces():
        """
        Defines the state and action spaces for the MLPro environment.
        This is a static method as the space structure is fixed for the system.

        Returns:
            A tuple containing the state space and action space (MSpace objects).
        """
        # Create the state space.
        state_space = MSpace()
        # Add a dimension for the total number of orders.
        state_space.add_dim(Dimension('total_orders', 'Z', 'Total Orders', p_boundaries=[0, 9999]))
        # Add a dimension for the number of successfully delivered orders.
        state_space.add_dim(Dimension('delivered_orders', 'Z', 'Delivered Orders', p_boundaries=[0, 9999]))

        # Create the action space.
        action_space = MSpace()
        # Add a single dimension for the action space, which is a flattened integer ID for a specific action tuple.
        action_space.add_dim(Dimension(p_name_short='global_action',
                                       p_base_set='Z',
                                       p_name_long='Global Flattened Action ID',
                                       p_boundaries=[0, 20000]))
        # Return both spaces.
        return state_space, action_space

    # --------------------------------------------------------------------------------------------------

    def _reset(self, p_seed=None):
        """
        Resets the entire simulation environment to its initial state.

        Parameters:
            p_seed: A seed for random number generators to ensure reproducibility.
        """
        # Configure which actions are to be handled automatically by the system's internal logic.
        self.automatic_logic_config = {action: action.is_automatic for action in self.actions.get_all_actions()}

        # Load the initial simulation data (e.g., from a JSON file).
        raw_entity_data = self.data_loader.load_initial_simulation_data()
        # Use a ScenarioGenerator to create entity objects from the raw data.
        scenario_generator = ScenarioGenerator(raw_entity_data)
        self.entities = scenario_generator.build_entities(p_logging=self.get_log_level(),
                                                          p_movement_mode=self.movement_mode)

        # Initialize the GlobalState, which holds all entities and simulation state.
        self.global_state = GlobalState(initial_entities=self.entities, movement_mode=self.movement_mode)

        # Generate the mapping from action tuples to integer IDs based on the initial global state.
        self.action_map, self.action_space_size = self.actions.generate_action_map(self.global_state)
        # Create an ActionIndex for efficient lookup of actions.
        self.action_index = ActionIndex(self.global_state, self.action_map)
        # Create the reverse mapping from integer IDs back to action tuples.
        self._reverse_action_map = {idx: act for act, idx in self.action_map.items()}

        # Initialize the network graph using the distance matrix from the loaded data.
        self.network = Network(self.global_state, self.movement_mode, raw_entity_data['distance_matrix'])
        # Link the network to the global state.
        self.global_state.network = self.network
        # Initialize the mapper that determines valid actions based on the state.
        self.state_action_mapper = StateActionMapper(self.global_state, self.action_map)

        # Ensure all entities have a reference to the global state.
        all_entity_dicts = self.global_state.get_all_entities()
        for entity_dict in all_entity_dicts:
            for entity in entity_dict.values():
                entity.global_state = self.global_state
                # entity.reset()

        # Initialize the ConstraintManager which is responsible for tracking action constraints.
        self.constraint_manager = ConstraintManager(action_index=self.action_index,
                                                    reverse_action_map=self._reverse_action_map)
        # Initialize the SupplyChainManager.
        self.supply_chain_manager = SupplyChainManager(p_id='scm', global_state=self.global_state,
                                                       p_automatic_logic_config=self.automatic_logic_config)
        # Initialize the ResourceManager.
        self.resource_manager = ResourceManager(p_id='rm', global_state=self.global_state)
        # Initialize the NetworkManager.
        self.network_manager = NetworkManager(p_id='nm', global_state=self.global_state, network=self.network,
                                              p_automatic_logic_config=self.automatic_logic_config)
        # Set up the event handling system.
        self.setup_events()

        # Create a dictionary of managers for easy access.
        managers = {'SupplyChainManager': self.supply_chain_manager, 'ResourceManager': self.resource_manager,
                    'NetworkManager': self.network_manager}

        # Give each manager a reference to the parent system.
        for manager in managers.values():
            manager.system = self

        # Initialize the ActionManager, passing it the managers it needs to orchestrate.
        self.action_manager = ActionManager(self.global_state, managers, self.action_map)

        # Give each vehicle a reference to the NetworkManager for movement.
        for vehicle in list(self.global_state.trucks.values()) + list(self.global_state.drones.values()):
            vehicle.network_manager = self.network_manager

        # Initialize the OrderGenerator to create new orders over time.
        self.order_generator = OrderGenerator(self.global_state, self, self._config.get('new_order_config', {}))
        # Register an event handler for when new orders are created.
        self.register_event_handler(self.C_EVENT_NEW_ORDER, self._handle_new_order_request)

        # Set the initial simulation time.
        initial_sim_time = self.entities.get('initial_time', 0.0)
        self.time_manager.reset_time(new_initial_time=initial_sim_time)
        self.global_state.current_time = initial_sim_time

        # Perform an initial update of the MLPro state object.
        self._update_state()

    # --------------------------------------------------------------------------------------------------

    def get_automatic_actions(self) -> List[Tuple]:
        """
        Identifies all valid actions that are configured to be executed automatically.

        Returns:
            A list of action tuples that are both valid and automatic.
        """
        # Get the mask of all currently valid actions in the system.
        system_mask = self.get_current_mask()
        # Find the indices where the mask is True (i.e., the valid actions).
        possible_action_indices = np.where(system_mask)[0]
        # Initialize a list to store automatic actions.
        automatic_actions = []
        # Iterate through the indices of all possible actions.
        for index in possible_action_indices:
            # Convert the action index back to its tuple representation.
            action_tuple = self._reverse_action_map.get(index)
            # Check if the action is configured as automatic.
            if action_tuple and self.automatic_logic_config.get(action_tuple[0], False):
                # If it is, add it to the list.
                automatic_actions.append(action_tuple)
        # Return the list of valid automatic actions.
        return automatic_actions

    # --------------------------------------------------------------------------------------------------

    # def are_automatic_actions_available(self):
    #     automatic_action_to_take = self._get_automatic_actions()

    # --------------------------------------------------------------------------------------------------

    def run_automatic_action_loop(self):
        """
        Executes a loop that processes all available automatic actions until none are left.
        This ensures that the system state is stable before handing control back to the agent.
        """
        # Counter for loop iterations.
        i = 0
        # Loop indefinitely until no automatic actions are available.
        while True:
            # Get the list of currently available automatic actions.
            automatic_actions_to_take = self.get_automatic_actions()
            # If the list is empty, the system is stable.
            if not automatic_actions_to_take:
                print(f"Auto-action loop stable after {i} iterations.")
                break
            # Select the first available automatic action to execute.
            auto_action_tuple = automatic_actions_to_take[0]
            print(f"  - Auto Action: {auto_action_tuple[0].name}{auto_action_tuple[1:]}")
            # Execute the action using the ActionManager.
            self.action_manager.execute_action(auto_action_tuple)
            # Increment the counter.
            i += 1
            # A safety break to prevent infinite loops.
            if i > 20:
                print("Auto-action loop exceeded safety limit of 20 iterations.")
                break

    # --------------------------------------------------------------------------------------------------

    def process_action(self, p_action: LogisticsAction):
        """
        Processes an action submitted by an external agent.

        Parameters:
            p_action (LogisticsAction): The action to be processed.

        Returns:
            bool: True if an action was successfully processed, False otherwise.
        """
        # Flag to track if the action was processed.
        action_processed = False
        # Extract the sorted values from the MLPro action object.
        action_values, _ = p_action.get_sorted_values_with_data()
        # The first value is the integer index of the action.
        action_index = int(action_values[0])
        # Convert the index back to the action tuple.
        action_tuple = self._reverse_action_map.get(action_index)

        # Check if the action is valid and is not the "No Operation" action.
        if action_tuple and action_tuple[0] != SimulationActions.NO_OPERATION:
            print(f"  - Agent Action: {action_tuple[0].name}{action_tuple[1:]}")
            # Execute the action via the ActionManager.
            action_processed = self.action_manager.execute_action(action_tuple)

        # self.run_automatic_action_loop()
        # Update the MLPro state object after the action.
        self._update_state()
        # Return whether an action was processed.
        return action_processed

    # --------------------------------------------------------------------------------------------------

    def advance_time(self, p_t_step: timedelta = None):
        """
        Advances the simulation time by one timestep.

        Parameters:
            p_t_step (timedelta): The amount of time to advance. If None, uses the system's default latency.
        """
        # Get the duration of a single timestep from the system's latency.
        timestep_duration = self.get_latency().total_seconds()
        # Use the provided t_step or the default duration.
        t_step = p_t_step or timedelta(seconds=timestep_duration)
        # Advance the time in the TimeManager.
        self.time_manager.advance_time(timestep_duration)
        # Update the current time in the GlobalState.
        self.global_state.current_time = self.time_manager.get_current_time()

        # Log the time advancement.
        self.log(self.C_LOG_TYPE_I, f"Time advanced to {self.global_state.current_time}s.")
        # Trigger the OrderGenerator to see if any new orders should be created at this time.
        self.order_generator.generate(self.global_state.current_time)

        # Collect all entities and managers that need to be updated with the time progression.
        all_systems = (list(self.global_state.trucks.values()) +
                       list(self.global_state.drones.values()) +
                       list(self.global_state.micro_hubs.values()) +
                       [self.supply_chain_manager, self.resource_manager, self.network_manager])

        # Call the simulate_reaction method on each component to process time-based events (e.g., vehicle movement).
        for system in all_systems:
            system.simulate_reaction(p_state=None, p_action=None, p_t_step=t_step)

        # Update the MLPro state object after time has advanced.
        self._update_state()

    # --------------------------------------------------------------------------------------------------

    def _simulate_reaction(self, p_state: State, p_action: LogisticsAction) -> State:
        """
        The core simulation step method required by MLPro's System class.
        It processes an agent's action and then advances simulation time.

        Parameters:
            p_state (State): The current state (not used directly here as state is managed internally).
            p_action (LogisticsAction): The action to be taken by the agent.

        Returns:
            State: The new state of the system after the step.
        """
        # First, process the agent's action.
        self.process_action(p_action)
        # Then, advance the simulation time by one step.
        self.advance_time()
        # Return the updated MLPro state object.
        return self._state

    # --------------------------------------------------------------------------------------------------

    def get_current_mask(self) -> np.ndarray:
        """
        Generates a boolean mask indicating all currently valid actions in the system.

        Returns:
            np.ndarray: A boolean array where True indicates a valid action.
        """
        # If the StateActionMapper is initialized, use it to generate the mask.
        if self.state_action_mapper:
            return self.state_action_mapper.generate_masks()
        # As a fallback, return a mask of all ones (all actions considered possible).
        return np.ones(len(self.action_map), dtype=bool)

    # --------------------------------------------------------------------------------------------------

    def get_agent_mask(self) -> np.ndarray:
        """
        Generates a boolean mask for actions available to the agent (excluding automatic actions).

        Returns:
            np.ndarray: A boolean array where True indicates a valid, non-automatic action.
        """
        # Get the complete mask of all valid actions.
        system_mask = self.get_current_mask()
        # Create a new mask of zeros, which will be populated with agent-available actions.
        agent_mask = np.zeros(self.action_space_size, dtype=bool)
        # Iterate through all defined actions.
        for action_tuple, idx in self.action_map.items():
            # Get the type of the action (e.g., MOVE, LOAD).
            action_type = action_tuple[0]
            # Check if the action is not automatic AND is currently valid according to the system mask.
            if not self.automatic_logic_config.get(action_type, False) and system_mask[idx]:
                # If so, mark it as available for the agent.
                agent_mask[idx] = True
        # Ensure the NO_OPERATION action is always available to the agent.
        no_op_idx = self.action_map.get((SimulationActions.NO_OPERATION,))
        if no_op_idx is not None:
            agent_mask[no_op_idx] = True
        # Return the final agent-specific mask.
        return agent_mask

    # --------------------------------------------------------------------------------------------------

    def _update_state(self):
        """
        Updates the MLPro state object with current data from the GlobalState.
        This is what the agent observes.
        """
        # Check if the global state exists.
        if self.global_state:
            # Get the state space definition.
            state_space = self._state.get_related_set()
            # Get all current order entities.
            orders = self.global_state.get_all_entities_by_type("order").values()
            # Update the 'total_orders' dimension.
            self._state.set_value(state_space.get_dim_by_name("total_orders").get_id(), len(orders))
            # Update the 'delivered_orders' dimension by counting orders with 'delivered' status.
            self._state.set_value(state_space.get_dim_by_name("delivered_orders").get_id(),
                                  sum(1 for o in orders if o.status == 'delivered'))

    # --------------------------------------------------------------------------------------------------

    def _handle_new_order_request(self, p_event_id, p_event_object):
        """
        Event handler for when new orders are dynamically added to the simulation.

        Parameters:
            p_event_id: The ID of the event being handled.
            p_event_object: The event object containing data (the new orders).
        """
        # Extract the new orders from the event data.
        orders = p_event_object.get_data()['p_orders']
        # self.global_state.add_orders(p_orders=p_event_object.get_data()['p_orders'])
        # Add the new orders to the global state.
        self.global_state.add_dynamic_orders(orders)
        # self.state_action_mapper.add_order(p_oredrs=p_event_object.get_data()['p_orders'])
        # Regenerate the action map to include actions related to the new orders.
        self.action_map, self.action_space_size = self.actions.generate_action_map(self.global_state)
        # Update the reverse action map.
        self._reverse_action_map = {idx: act for act, idx in self.action_map.items()}
        # Update the action index with the new action map.
        self.action_index.update_indexes(global_state=self.global_state, action_map=self.action_map)
        # Link the updated action index to the constraint manager.
        self.constraint_manager.action_index = self.action_index
        # Update the state-action mapper with the new action space.
        self.state_action_mapper.update_action_space(self.action_map)
        # Update constraints to reflect the new state.
        self.constraint_manager.update_constraints(self.global_state, self._reverse_action_map)
        # Re-generate the masks to account for the new state and actions.
        self.get_masks()

    # --------------------------------------------------------------------------------------------------

    def get_masks(self):
        """
        A convenience method to get the current action masks.
        """
        # Calls the state-action mapper to generate the masks.
        return self.state_action_mapper.generate_masks()

    # --------------------------------------------------------------------------------------------------

    def setup_events(self):
        """
        Sets up the event-driven communication between different components of the system.
        """
        # When the ConstraintManager updates a mask, it triggers an event that the StateActionMapper handles.
        self.constraint_manager.register_event_handler(p_event_id=ConstraintManager.C_EVENT_MASK_UPDATED,
                                                       p_event_handler=self.state_action_mapper.handle_new_masks_event)
        # Iterate through all entities to set up their event handlers.
        for entities in self.entities.values():
            if isinstance(entities, Dict):
                for entity in entities.values():
                    # If an entity's state changes (e.g., a truck becomes full), it triggers an event.
                    if isinstance(entity, LogisticEntity):
                        # This event is handled by the ConstraintManager to update relevant action constraints.
                        entity.register_event_handler_for_constraints(LogisticEntity.C_EVENT_ENTITY_STATE_CHANGE,
                                                                      self.constraint_manager.handle_entity_state_change)

        # When the SupplyChainManager requests a new order, it triggers an event.
        self.supply_chain_manager.register_event_handler(SupplyChainManager.C_EVENT_NEW_ORDER_REQUEST,
                                                         # This event is handled by the system's own method to add the order.
                                                         self._handle_new_order_request)

    # --------------------------------------------------------------------------------------------------

    def get_success(self) -> bool:
        """
        Checks if the simulation has reached a success condition (e.g., all orders delivered).

        Returns:
            bool: True if the success condition is met, False otherwise.
        """
        # Get all current orders.
        orders = self.global_state.get_orders()
        # Assume success is true initially.
        success = True
        # Check each order's status.
        for ords in orders.values():
            # The overall success is only true if every single order is delivered.
            success = (ords.get_state_value_by_dim_name(
                ords.C_DIM_DELIVERY_STATUS[0]) == ords.C_STATUS_DELIVERED) and success
        # If any order is not delivered, success will be false.
        if success:
            return success
        return success

    # --------------------------------------------------------------------------------------------------

    def get_broken(self):
        """
        Checks if the simulation has reached a broken/failure state.
        Currently, this is not implemented and always returns False.

        Returns:
            bool: Always False.
        """
        return False


# -------------------------------------------------------------------------
# -- Validation Block
# -------------------------------------------------------------------------
# This block is executed only when the script is run directly.
if __name__ == "__main__":
    # Print a header for the validation process.
    print("--- Validating LogisticsSystem ---")

    # Define the path to the configuration file relative to this script's location.
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'initial_entity_data.json')
    # Normalize the path to be compatible with the operating system.
    config_file_path = os.path.normpath(config_file_path)

    # Define the simulation configuration dictionary.
    sim_config = {
        "initial_time": 0.0,
        "main_timestep_duration": 300.0,  # Each simulation step represents 300 seconds (5 minutes).
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {"file_path": config_file_path}
        },
        "new_order_config": {}  # Configuration for dynamic order generation.
    }

    # Instantiate the LogisticsSystem with the defined configuration.
    logistics_system = LogisticsSystem(p_id='validation_sys',
                                       p_visualize=False,
                                       p_logging=False,
                                       config=sim_config)

    # Print a header for the simulation run.
    print("\n--- Running simulation for 20 cycles with Dummy Agent Logic ---")

    # Get the index for the "NO_OPERATION" action.
    no_op_idx = logistics_system.action_map.get((SimulationActions.NO_OPERATION,))

    # Run the simulation for 20 cycles.
    for i in range(20):
        print(f"\n--- Cycle {i + 1} ---")

        # --- Decision Phase ---
        # A simple dummy agent that takes one random valid action per cycle.
        # Get the mask of actions available to the agent.
        agent_mask = logistics_system.get_agent_mask()
        # Find all valid actions, excluding the NO_OPERATION action for selection purposes.
        valid_actions = np.where(np.delete(agent_mask, no_op_idx))[0]

        # If there are valid actions to take...
        if len(valid_actions) > 0:
            # ...choose one randomly.
            choice = random.choice(valid_actions)
            # Get the tuple representation of the chosen action for logging.
            act_tuple = logistics_system._reverse_action_map.get(choice)
            print(f"Dummy Agent chooses: {act_tuple}")
        # Otherwise, if no actions are available...
        else:
            # ...choose NO_OPERATION.
            choice = no_op_idx
            print("Dummy Agent chooses: NO_OPERATION")

        # Create an MLPro action object with the chosen action index.
        action = LogisticsAction(p_action_space=logistics_system.get_action_space(), p_values=[choice])

        # Process the chosen action in the simulation.
        logistics_system.process_action(action)

        # --- Progression Phase ---
        # Advance the simulation time by one step.
        logistics_system.advance_time()

        # --- Reporting ---
        # Get the current state of the system for reporting.
        state = logistics_system.get_state()
        print(f"  - Current Time: {logistics_system.time_manager.get_current_time()}s")
        # Get the state dimensions for total and delivered orders.
        state_dim_total = state.get_related_set().get_dim_by_name('total_orders')
        state_dim_delivered = state.get_related_set().get_dim_by_name('delivered_orders')
        # Print the current values from the state object.
        print(f"  - Total Orders: {state.get_value(state_dim_total.get_id())}")
        print(f"  - Delivered Orders: {state.get_value(state_dim_delivered.get_id())}")

    # Print a final message indicating the validation script completed successfully.
    print("\n--- Validation Complete: LogisticsSystem initialized and ran successfully. ---")