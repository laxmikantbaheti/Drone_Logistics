import numpy.random as rnd
from alns_benchmark.state import PriorityPlanState

def destroy_random(state: PriorityPlanState, rng: rnd.Generator) -> PriorityPlanState:
    """Removes 20-30% of orders to explore priority re-ordering."""
    new_state = state.copy()
    if len(new_state.order_priority) <= 3:
        return new_state

    fraction = rng.uniform(0.2, 0.3)
    k = max(1, int(len(new_state.order_priority) * fraction))
    removed = set(rng.choice(new_state.order_priority, size=k, replace=False))

    new_state.order_priority = [o for o in new_state.order_priority if o not in removed]
    new_state.unassigned.update(removed)
    return new_state

def destroy_toggle_truck_mode(state: PriorityPlanState, rng: rnd.Generator) -> PriorityPlanState:
    """
    Shifts orders between drone eligibility and direct-truck delivery,
    reallocating microhub drone capacity.
    """
    new_state = state.copy()
    if not new_state.order_priority:
        return new_state

    # Pick 1-2 orders to toggle
    targets = rng.choice(new_state.order_priority, size=min(2, len(new_state.order_priority)), replace=False)
    for target in targets:
        if target in new_state.truck_only_orders:
            new_state.truck_only_orders.remove(target)
        else:
            new_state.truck_only_orders.add(target)
    return new_state