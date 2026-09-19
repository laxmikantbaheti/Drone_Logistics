import time
import argparse
import inspect
import numpy.random as rnd
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

from rl_ext.env import LogisticsEnv
from alns_benchmark.state import PriorityPlanState
from alns_benchmark.evaluator import EnvironmentEvaluator
from alns_benchmark.operators.destroy import destroy_random, destroy_toggle_truck_mode
from alns_benchmark.operators.repair import repair_greedy
from alns_benchmark.config import ALNSConfig


def run_benchmark(instance_path: str, seed: int = 42):
    print("=" * 60)
    print(f"Running ALNS Benchmark on: {instance_path}")
    print("=" * 60)

    env = LogisticsEnv()
    evaluator = EnvironmentEvaluator(env)

    # Initialize environment once to extract customer order IDs[cite: 1]
    try:
        obs, info = env.reset(options={"instance_file": instance_path})
    except TypeError:
        obs, info = env.reset()

    if hasattr(env.unwrapped, "supply_chain_manager"):
        order_ids = [o.id for o in env.unwrapped.supply_chain_manager.get_all_orders()][cite: 1]
    else:
        order_ids = list(range(1, env.action_space.n))

    # Evaluate initial natural priority sequence
    init_state = PriorityPlanState(order_priority=order_ids)
    start_init_eval = time.time()
    init_cost = evaluator.evaluate(init_state, instance_path)
    print(f"Initial Feasible Cost: {init_cost:.2f} (Computed in {time.time() - start_init_eval:.2f}s)")

    # Setup ALNS
    rng = rnd.default_rng(seed)
    alns = ALNS(rng)

    # Register operators FIRST
    alns.add_destroy_operator(destroy_random)
    alns.add_destroy_operator(destroy_toggle_truck_mode)
    alns.add_repair_operator(lambda s, r: repair_greedy(s, r, evaluator, instance_path))

    # Dynamic RouletteWheel configuration based on installed signature
    num_destroy = len(alns.destroy_operators)
    num_repair = len(alns.repair_operators)
    rw_params = inspect.signature(RouletteWheel.__init__).parameters

    rw_kwargs = {"scores": ALNSConfig.SCORES, "decay": ALNSConfig.DECAY}
    if "num_destroy" in rw_params:
        rw_kwargs["num_destroy"] = num_destroy
    if "num_repair" in rw_params:
        rw_kwargs["num_repair"] = num_repair
    if "seg_length" in rw_params:
        rw_kwargs["seg_length"] = ALNSConfig.UPDATE_INTERVAL

    select = RouletteWheel(**rw_kwargs)

    # Acceptance and dynamic stopping criteria
    node_count = len(order_ids)
    if node_count <= 25:
        max_iters = ALNSConfig.ITERATIONS_SMALL
    elif node_count <= 45:
        max_iters = ALNSConfig.ITERATIONS_MEDIUM
    else:
        max_iters = ALNSConfig.ITERATIONS_LARGE

    accept = SimulatedAnnealing(
        start_temperature=ALNSConfig.START_TEMPERATURE,
        end_temperature=ALNSConfig.END_TEMPERATURE,
        step=ALNSConfig.STEP_DECAY,
    )
    stop = MaxIterations(max_iters)

    # Run optimization
    start_time = time.time()
    result = alns.iterate(init_state, select, accept, stop)
    total_time = time.time() - start_time

    best_state = result.best_state
    improvement = ((init_cost - best_state.cost) / init_cost) * 100 if init_cost != 0 else 0.0

    print("-" * 60)
    print(f"Optimization finished in: {total_time:.2f}s")
    print(f"Best ALNS Objective Cost: {best_state.cost:.2f}")
    print(f"Improvement over baseline: {improvement:.2f}%")
    print(f"Solution Metrics: {best_state.metrics}")
    print("-" * 60)

    return best_state, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ALNS benchmark on VRP-D instances.")
    parser.add_argument(
        "--instance",
        type=str,
        default="ddls_src/scenarios/vrp_d_instances/VRP-D/A-n32-k5.vrp",
        help="Path to .vrp benchmark instance",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_benchmark(args.instance, args.seed)