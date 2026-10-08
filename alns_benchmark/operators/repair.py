import numpy.random as rnd
from alns_benchmark.state import PriorityPlanState
from alns_benchmark.evaluator import EnvironmentEvaluator

def repair_greedy(
    state: PriorityPlanState,
    rng: rnd.Generator,
    evaluator: EnvironmentEvaluator,
    instance_path: str
) -> PriorityPlanState:
    """
    Reinserts unassigned orders into sampled priority indices that minimize simulation cost.
    """
    new_state = state.copy()
    orders_to_insert = list(new_state.unassigned)
    rng.shuffle(orders_to_insert)

    for order in orders_to_insert:
        best_cost = float("inf")
        best_pos = 0

        # Sample position candidates to preserve evaluation speed
        n = len(new_state.order_priority)
        test_positions = {0, n // 2, n}
        if n > 4:
            test_positions.update(rng.integers(0, n + 1, size=min(3, n)).tolist())

        for pos in test_positions:
            trial = new_state.copy()
            trial.order_priority.insert(pos, order)
            cost = evaluator.evaluate(trial, instance_path)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos

        new_state.order_priority.insert(best_pos, order)

    new_state.unassigned.clear()
    evaluator.evaluate(new_state, instance_path)
    return new_state