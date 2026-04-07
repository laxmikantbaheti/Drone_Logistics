from abc import ABC, abstractmethod
from typing import List


class RouteSequencer(ABC):
    """
    Abstract base class for batch-processing order assignments into a strict node sequence.
    """

    @abstractmethod
    def generate_sequence(self,
                          current_node: int,
                          pickup_orders: List[int],  # Assuming these contain Order IDs or Node IDs
                          delivery_orders: List[int],
                          pickup_leg2_orders: List[int],
                          delivery_leg2_orders: List[int]) -> List[int]:
        """
        Takes the categorized staging lists from the vehicle upon the CONSOLIDATE action
        and returns a strict, chronologically ordered sequence of node IDs to visit.
        """
        pass

    def reset(self):
        pass


class HeuristicSequencer(RouteSequencer):
    def generate_sequence(self, current_node, pickup_orders, delivery_orders, pickup_leg2_orders, delivery_leg2_orders):
        sequence = []

        # 1. Normal Orders (Pickups first, then Deliveries)
        # We filter out the current node so the vehicle doesn't "travel" to where it already is
        normal_pickups = [node for node in pickup_orders if node != current_node]
        sequence.extend(normal_pickups)
        sequence.extend(delivery_orders)

        # 2. Leg 2 Pseudo Orders (Appended to the absolute end)
        # Assuming we need to pick them up, then deliver them
        for p_node, d_node in zip(pickup_leg2_orders, delivery_leg2_orders):
            if not sequence or sequence[-1] != p_node:
                sequence.append(p_node)
            sequence.append(d_node)

        return sequence

