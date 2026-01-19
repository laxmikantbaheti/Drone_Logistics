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



