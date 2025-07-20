import numpy as np
from typing import Dict, Any, List, Tuple, Callable

# Import core components
from .global_state import GlobalState
from .network import Network


# Assuming TimeManager is from the base framework, we won't explicitly import it here
# but will assume it's available or managed externally. For the purpose of this code,
# we'll keep the placeholder for TimeManager for consistency with the plan's attributes.
import heapq  # For implementing the scheduled_events as a priority queue
from typing import List, Dict, Any, Tuple


class TimeManager:
    """
    Manages the simulation clock and schedules/triggers exogenous events.
    Events are stored in a min-heap (priority queue) ordered by their scheduled time.
    """

    def __init__(self, initial_time: float = 0.0):
        """
        Initializes the simulation clock and the event scheduler.

        Args:
            initial_time (float): The starting time of the simulation. Defaults to 0.0.
        """
        self.current_time: float = initial_time
        # scheduled_events is a min-heap (priority queue) of tuples:
        # (event_time: float, event_id: int, event_data: dict)
        # event_id is used to break ties for events scheduled at the exact same time,
        # ensuring consistent ordering and preventing comparison issues if event_data is complex.
        self.scheduled_events: List[Tuple[float, int, Dict[str, Any]]] = []
        self._event_id_counter: int = 0  # Unique ID generator for events

        print(f"TimeManager initialized at time: {self.current_time}")

    def advance_time(self, delta_time: float) -> None:
        """
        Increments the current_time of the simulation.

        Args:
            delta_time (float): The amount of time to advance the clock by.
        """
        if delta_time < 0:
            raise ValueError("delta_time cannot be negative.")
        self.current_time += delta_time
        # print(f"TimeManager: Advanced time by {delta_time}. Current time: {self.current_time}")

    def get_current_time(self) -> float:
        """
        Returns the current simulation time.

        Returns:
            float: The current time of the simulation.
        """
        return self.current_time

    def schedule_event(self, event_time: float, event_data: Dict[str, Any]) -> None:
        """
        Adds an event to the scheduled events queue.
        Events will be processed when the simulation time reaches or surpasses their event_time.

        Args:
            event_time (float): The time at which the event should occur.
            event_data (dict): A dictionary containing all relevant data for the event.
        """
        if event_time < self.current_time:
            print(
                f"Warning: Scheduling event at {event_time} which is in the past (current time: {self.current_time}).")
            # You might want to raise an error or handle this differently based on simulation needs
            # For now, we'll allow it but warn.

        self._event_id_counter += 1  # Increment to ensure unique ID
        heapq.heappush(self.scheduled_events, (event_time, self._event_id_counter, event_data))
        # print(f"TimeManager: Scheduled event at {event_time} with data: {event_data}")

    def get_due_events(self) -> List[Dict[str, Any]]:
        """
        Returns and removes all events whose scheduled event_time is less than or equal
        to the current_time.

        Returns:
            List[dict]: A list of event data dictionaries that are due.
        """
        due_events = []
        while self.scheduled_events and self.scheduled_events[0][0] <= self.current_time:
            # Pop the smallest item (event with the earliest time)
            event_time, _, event_data = heapq.heappop(self.scheduled_events)
            due_events.append(event_data)
            # print(f"TimeManager: Processed due event at {event_time} with data: {event_data}")
        return due_events

    def reset_time(self, new_initial_time: float = 0.0) -> None:
        """
        Resets the simulation clock to a new initial time and clears all scheduled events.
        This is useful for starting a new simulation episode.

        Args:
            new_initial_time (float): The time to reset the clock to. Defaults to 0.0.
        """
        self.current_time = new_initial_time
        self.scheduled_events = []  # Clear the priority queue
        self._event_id_counter = 0  # Reset event ID counter
        print(f"TimeManager: Reset to initial time: {self.current_time}. All scheduled events cleared.")


# Import manager classes
from ..managers.action_manager import ActionManager
from ..managers.action_masking.action_masker import ActionMasker
from ..managers.supply_chain_manager import SupplyChainManager
from ..managers.resource_manager.base import ResourceManager  # Assuming resource_manager/base.py for ResourceManager
from ..managers.resource_manager.fleet_manager import FleetManager  # Sub-manager, instantiated by ResourceManager
from ..managers.resource_manager.micro_hub_manager import \
    MicroHubsManager  # Sub-manager, instantiated by ResourceManager
from ..managers.network_manager import NetworkManager

# Import Action Enums and Action Map
from ..actions.action_enums import SimulationAction
from ..actions.action_mapping import ACTION_MAP, ACTION_SPACE_SIZE

# Import DataLoader and ScenarioGenerator (UPDATED IMPORT PATH)
from ..scenarios.generators.data_loader import DataLoader
from ..scenarios.generators.scenario_generator import ScenarioGenerator  # UPDATED: Corrected import path and class name

# NO_OPERATION is now part of the SimulationAction enum
NO_OPERATION = SimulationAction.NO_OPERATION


class LogisticsSimulation:
    """
    The central orchestrator of the simulation logic, independent of any specific RL environment interface.
    It manages the two-phase timestep (decision loop and progression).
    """

    def __init__(self, config: dict):
        """
        Initializes all managers and sub-components based on configuration.

        Args:
            config (dict): A dictionary holding simulation configuration parameters.
                           Expected keys might include initial_state_config, action_space_size, etc.
        """
        self.config = config

        # Initialize core components (these will be instances of classes defined later)
        self.global_state: GlobalState = None
        self.action_manager: ActionManager = None
        self.action_masker: ActionMasker = None
        # TimeManager is assumed to be from base framework, but keeping attribute for consistency
        self.time_manager: TimeManager = TimeManager(initial_time=self.config.get("initial_time", 0.0))
        self.network: Network = None  # The graph structure

        # Exogenous event generators will be populated based on config
        self.exogenous_event_generators: List[Any] = []  # List of event generator instances

        # Action map and space size are now imported from action_mapping.py
        self.action_map: Dict[Tuple, int] = ACTION_MAP
        self.action_space_size: int = ACTION_SPACE_SIZE

        # DataLoader instance
        self.data_loader: DataLoader = DataLoader(self.config.get("data_loader_config", {}))
        # ScenarioGenerator instance (NEW)
        self.scenario_generator: ScenarioGenerator = None  # Initialized in initialize_simulation

        # Internal flags for simulation control
        self._decision_loop_active: bool = False
        self._last_action_executed_valid: bool = True  # Default to True, assumes first action is valid until proven otherwise

        print("LogisticsSimulation __init__ completed. Call initialize_simulation() to set up components.")

    def initialize_simulation(self) -> None:
        """
        Resets the simulation to an initial state, populating GlobalState and resetting TimeManager.
        This method should be called before starting a new simulation run.
        """
        print("LogisticsSimulation: Initializing simulation components...")

        # 1. Load initial raw entity data using DataLoader
        raw_entity_data = self.data_loader.load_initial_simulation_data()

        # 2. Use ScenarioGenerator to instantiate entities (UPDATED CLASS NAME)
        self.scenario_generator = ScenarioGenerator(raw_entity_data)
        initial_entities = self.scenario_generator.build_entities()

        # 3. Initialize GlobalState with instantiated entities
        # GlobalState's __init__ now takes a dict of instantiated entity objects
        self.global_state = GlobalState(initial_entities)

        # 4. Initialize Network (which uses GlobalState's nodes and edges)
        self.network = Network(self.global_state)
        # Ensure GlobalState also has a reference to the Network
        self.global_state.network = self.network

        # 5. Initialize all specialized managers
        # SupplyChainManager needs GlobalState
        self.supply_chain_manager = SupplyChainManager(self.global_state)

        # ResourceManager needs GlobalState and internally initializes FleetManager and MicroHubsManager
        self.resource_manager = ResourceManager(self.global_state)

        # NetworkManager needs GlobalState and Network
        self.network_manager = NetworkManager(self.global_state, self.network)

        # 6. Initialize ActionMasker and ActionManager
        # ActionMasker needs reference to self (LogisticsSimulation), action_space_size, and action_map
        self.action_masker = ActionMasker(self, self.action_space_size, self.action_map)

        # ActionManager needs GlobalState, a dictionary of managers, action_map, and ActionMasker
        # Pass the instantiated managers to ActionManager
        managers_for_action_manager = {
            'supply_chain_manager': self.supply_chain_manager,
            'resource_manager': self.resource_manager,
            'network_manager': self.network_manager
        }
        self.action_manager = ActionManager(
            self.global_state,
            managers_for_action_manager,
            self.action_map,
            self.action_masker
        )

        # 7. Reset TimeManager (assuming it has a reset_time method)
        # Use initial_time from the loaded entities if available, otherwise from config
        initial_sim_time = initial_entities.get('initial_time', self.config.get("initial_time", 0.0))
        self.time_manager.reset_time(new_initial_time=initial_sim_time)
        self.global_state.current_time = initial_sim_time  # Ensure GlobalState's time is also set

        # 8. Initialize exogenous event generators (structurally for now)
        # This would typically involve iterating through a list of generator configs
        # and instantiating them, passing necessary references like global_state.
        # For example:
        # from ..events.new_order_generator import NewOrderGenerator
        # self.exogenous_event_generators = [
        #     NewOrderGenerator(self.global_state, self.config.get("new_order_config", {}))
        # ]
        self.exogenous_event_generators = []  # Placeholder: Add actual generators here

        # Reset internal flags for simulation control
        self._decision_loop_active = False
        self._last_action_executed_valid = True

        print("LogisticsSimulation: All core components and managers initialized successfully.")

    def start_decision_loop(self) -> None:
        """
        Initiates the agent's decision loop for the current main timestep.
        Sets _decision_loop_active to True.
        """
        self._decision_loop_active = True
        # print(f"Decision loop started at time: {self.time_manager.get_current_time()}")

    def process_agent_micro_action(self, action_index: int) -> bool:
        """
        Processes a single action from the agent within the ongoing decision loop,
        performing micro-updates and re-evaluating the mask.

        Args:
            action_index (int): The flattened integer index of the action chosen by the agent.

        Returns:
            bool: True if the action was valid and executed, False otherwise.
                  If action_index corresponds to NO_OPERATION, it sets _decision_loop_active to False.
        """
        if not (0 <= action_index < self.action_space_size):
            print(f"LogisticsSimulation: Error - Action index {action_index} is out of bounds.")
            self._last_action_executed_valid = False
            return False

        action_tuple_from_index = self.action_manager._reverse_action_map.get(action_index)

        if action_tuple_from_index is None:
            print(f"LogisticsSimulation: Error - Action index {action_index} does not map to a valid action tuple.")
            self._last_action_executed_valid = False
            return False

        # If NO_OPERATION is chosen, end the decision loop
        if action_tuple_from_index == (NO_OPERATION,):  # NO_OPERATION is a tuple (SimulationAction.NO_OPERATION,)
            self._decision_loop_active = False
            self._last_action_executed_valid = True  # NO_OPERATION is always considered valid for loop termination
            return True

        # Get the current mask to validate the action
        current_mask = self.action_masker.generate_mask()

        # Attempt to execute the action via ActionManager
        action_executed = self.action_manager.execute_action(action_tuple_from_index, current_mask)
        self._last_action_executed_valid = action_executed

        return action_executed

    def advance_main_timestep(self) -> dict:
        """
        Advances the main simulation clock by one full timestep duration,
        processing continuous dynamics and exogenous events, returning raw simulation outcomes.
        This method is called after the decision loop for the current main timestep is complete.

        Returns:
            dict: Raw simulation outcomes (e.g., {'events': [...], 'state_changes': {...}}).
        """
        if self._decision_loop_active:
            raise RuntimeError("Cannot advance main timestep while decision loop is active. "
                               "Agent must choose NO_OPERATION to end decision phase.")

        timestep_duration = self.config.get("main_timestep_duration", 60.0)  # e.g., 60 seconds (1 minute)
        self.time_manager.advance_time(timestep_duration)
        self.global_state.current_time = self.time_manager.get_current_time()  # Update GlobalState's time

        raw_outcomes = {
            'completed_deliveries': [],
            'new_orders': [],
            'vehicle_breakdowns': [],
            'traffic_updates': [],
            'state_changes': {}  # General dictionary for broader state changes
        }

        # 1. Continuous Dynamics: Update vehicles based on their current routes and speeds
        # This relies on the NetworkManager to handle vehicle movement logic
        for truck_id, truck in self.global_state.trucks.items():
            if truck.status == 'en_route' or truck.status == 'loading' or truck.status == 'unloading':  # Allow movement while loading/unloading if it's part of the process
                truck.move_along_route(timestep_duration, self.network_manager)
                # Fuel consumption is handled within truck.move_along_route
        for drone_id, drone in self.global_state.drones.items():
            if drone.status == 'en_route' or drone.status == 'loading' or drone.status == 'unloading':
                drone.move_along_route(timestep_duration, self.network_manager)
                # Battery drain is handled within drone.move_along_route
            elif drone.status == 'charging':
                drone.charge_battery(timestep_duration)  # Charge if status is charging
                # Check if fully charged and change status to idle
                if drone.battery_level >= drone.max_battery_capacity:
                    drone.set_status("idle")
                    print(f"Drone {drone_id} fully charged and is now idle.")

        # 2. Process Exogenous Events:
        # Query TimeManager for events scheduled during this timestep.
        # This assumes TimeManager is integrated and provides events.
        due_events = self.time_manager.get_due_events()
        for event_data in due_events:
            # Process scheduled events based on event_data['type']
            # Example:
            # if event_data['type'] == 'delivery_completion':
            #     raw_outcomes['completed_deliveries'].append(event_data['order_id'])
            #     # Update order status in GlobalState if not already done by FleetManager
            # elif event_data['type'] == 'vehicle_breakdown':
            #     raw_outcomes['vehicle_breakdowns'].append(event_data['vehicle_id'])
            #     # Flag vehicle for maintenance in ResourceManager
            pass  # Implement actual event processing logic here

        # Iterate through exogenous event generators (if implemented)
        for event_generator in self.exogenous_event_generators:
            # Each generator would have a method to check/generate events for the current time
            # For example:
            # new_orders = event_generator.generate_new_orders(self.global_state.current_time, self.global_state)
            # raw_outcomes['new_orders'].extend(new_orders)
            # breakdowns = event_generator.check_for_breakdowns(self.global_state.current_time, self.global_state)
            # raw_outcomes['vehicle_breakdowns'].extend(breakdowns)
            pass  # Implement actual calls once generators are defined

        # Post-timestep checks (e.g., SLA breaches, delivery completions)
        # This would update statuses in GlobalState and populate raw_outcomes
        for order_id, order in self.global_state.orders.items():
            if order.status != "delivered" and order.status != "cancelled":
                if order.get_SLA_remaining(self.global_state.current_time) < 0:
                    # Penalize for SLA breach
                    raw_outcomes['state_changes']['SLA_breach'] = raw_outcomes['state_changes'].get('SLA_breach',
                                                                                                    []) + [order_id]
                    # Optionally, change order status to 'SLA_breached'
                    # order.update_status("SLA_breached")

        return raw_outcomes

    def get_current_mask(self) -> np.ndarray:
        """
        Public method to get the current action mask from ActionMasker.
        """
        if self.action_masker is None:
            raise RuntimeError("ActionMasker not initialized. Call initialize_simulation() first.")
        return self.action_masker.generate_mask()

    def get_current_global_state(self) -> GlobalState:
        """
        Public method to expose the current GlobalState for external components (like a Gym wrapper).
        """
        if self.global_state is None:
            raise RuntimeError("GlobalState not initialized. Call initialize_simulation() first.")
        return self.global_state

    def is_decision_loop_active(self) -> bool:
        """
        Returns the status of the decision loop.
        """
        return self._decision_loop_active

    def get_last_action_validity(self) -> bool:
        """
        Returns whether the last process_agent_micro_action call resulted in a valid execution.
        """
        return self._last_action_executed_valid

