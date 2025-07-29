from typing import List, Dict, Any, Optional
import random

# MLPro Imports
from mlpro.bf.events import Event


# Forward Declarations
class GlobalState: pass


class Order: pass


class LogisticsSystem: pass


class OrderGenerator:
    """
    Generates new orders dynamically during the simulation and signals their
    creation by raising events through the main LogisticsSystem.
    """

    def __init__(self, global_state: 'GlobalState', logistics_system: 'LogisticsSystem', config: Dict[str, Any]):
        self.global_state = global_state
        self.logistics_system = logistics_system  # Reference to the event manager
        self.config = config
        self._next_order_id = self._get_initial_max_order_id() + 1

        self._arrival_schedule = config.get('arrival_schedule', {})

    def _get_initial_max_order_id(self) -> int:
        """Finds the highest order ID at the start of the simulation."""
        if not self.global_state.orders:
            return -1
        return max(self.global_state.orders.keys())

    def generate(self, current_time: float):
        """
        Checks the current time against the arrival schedule and generates new orders,
        raising an event for each one. Does not return the orders.
        """
        time_key = str(current_time)
        if time_key in self._arrival_schedule:
            num_new_orders = self._arrival_schedule.pop(time_key)

            print(f"\n  >>> DYNAMIC EVENT: {num_new_orders} new order(s) arriving at time {current_time}...")

            customer_nodes = [n.id for n in self.global_state.nodes.values() if n.type_of_node == 'customer']
            if not customer_nodes:
                return

            for _ in range(num_new_orders):
                new_order_data = {
                    'p_id': self._next_order_id,
                    'global_state': self.global_state,
                    'customer_node_id': random.choice(customer_nodes),
                    'time_received': current_time,
                    'SLA_deadline': current_time + random.uniform(1800, 7200),
                    'priority': random.randint(1, 3)
                }

                from ...entities.order import Order
                new_order = Order(**new_order_data)

                # The generator's only job is to raise the event.
                # The system that listens to the event will be responsible for adding the order to the state.
                event = Event(p_raising_object=self, order=new_order)
                self.logistics_system._raise_event(self.logistics_system.C_EVENT_NEW_ORDER, event)

                self._next_order_id += 1
