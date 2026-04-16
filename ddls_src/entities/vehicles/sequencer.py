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


class HeuristicSequencer2(RouteSequencer):
    """
    Generates a strict, indexed timeline for a vehicle, grouping consecutive
    tasks at the same node into a single operational step.

    Example Output:
        node_sequence = {0: Node_A, 1: Node_B, 2: Node_A}
        order_sequence = {0: [Order_1, Order_2], 1: [Order_3], 2: [Order_4]}
    """

    def generate_sequence(self, current_node, pickup_orders: dict, delivery_orders: dict, pickup_leg2_orders: dict,
                          delivery_leg2_orders: dict):
        # Step 1: Flatten everything into a strict logical timeline
        flat_nodes = []
        flat_orders = []

        def extract_phase(order_dict):
            for node_id, orders in order_dict.items():
                for order in orders:
                    flat_nodes.append(node_id)
                    flat_orders.append(order)

        # Build the timeline prioritizing pickups -> deliveries -> leg 2
        extract_phase(pickup_orders)
        extract_phase(delivery_orders)
        extract_phase(pickup_leg2_orders)
        extract_phase(delivery_leg2_orders)

        node_sequence = {}
        order_sequence = {}

        # Guard: If no orders were assigned, return empty dicts
        if not flat_nodes:
            return node_sequence, order_sequence

        # Step 2: Group consecutive identical nodes into index keys
        current_index = 0
        node_sequence[current_index] = flat_nodes[0]
        order_sequence[current_index] = [flat_orders[0]]

        for i in range(1, len(flat_nodes)):
            node = flat_nodes[i]
            order = flat_orders[i]

            # If the node is the exact same as the current step, just add the order to the list
            if node == node_sequence[current_index]:
                order_sequence[current_index].append(order)
            else:
                # The node changed! Advance the index and create a new sequence step
                current_index += 1
                node_sequence[current_index] = node
                order_sequence[current_index] = [order]

        return node_sequence, order_sequence

