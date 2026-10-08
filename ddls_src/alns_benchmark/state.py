import copy
from typing import List, Set, Optional
from alns import State

class PriorityPlanState(State):
    def __init__(
        self,
        order_priority: List[int],
        truck_only_orders: Optional[Set[int]] = None,
        unassigned: Optional[Set[int]] = None
    ):
        self.order_priority: List[int] = list(order_priority)
        self.truck_only_orders: Set[int] = set(truck_only_orders) if truck_only_orders else set()
        self.unassigned: Set[int] = set(unassigned) if unassigned else set()
        self._cost: float = float("inf")
        self.metrics: dict = {}

    def copy(self) -> "PriorityPlanState":
        new_state = PriorityPlanState(
            order_priority=copy.deepcopy(self.order_priority),
            truck_only_orders=copy.deepcopy(self.truck_only_orders),
            unassigned=copy.deepcopy(self.unassigned)
        )
        new_state._cost = self._cost
        new_state.metrics = copy.deepcopy(self.metrics)
        return new_state

    @property
    def cost(self) -> float:
        return self._cost