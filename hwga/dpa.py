import os
import argparse
import random
import copy
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from rl_ext.training.base import Training


# ----------------------------------------------------------------------
# 1. Environment Harness
# ----------------------------------------------------------------------
class _EnvLoader(Training):
    def train(self, *args, **kwargs):
        pass


def make_env(vrp_instance_path: str, sim_config: dict, instance_name: str):
    loader = _EnvLoader(
        config_path=vrp_instance_path,
        sim_config=sim_config,
        instance_name=instance_name,
        save_models=False,
        save_episode_data=False,
    )
    return loader.env


# ----------------------------------------------------------------------
# 2. Action Decoder & Multi-Modal Network Extractor
# ----------------------------------------------------------------------
class ActionFeatureExtractor:
    """
    Decodes the composite action space:
    1. Nodes [0 .. num_nodes-1]: Depots, microhubs, customers.
    2. Drone Routes [num_nodes .. num_nodes + (num_hubs * num_nodes) - 1]:
       Dispatches drone from a microhub to a node.
    """

    def __init__(self, env, num_nodes: int = 32, num_microhubs: int = 6):
        self.env = env
        self.num_nodes = num_nodes
        self.num_microhubs = max(1, num_microhubs)

        self.land_matrix, self.air_matrix = self._extract_matrices()

        self.max_land = (
            float(np.max(self.land_matrix))
            if self.land_matrix is not None and np.max(self.land_matrix) > 0
            else 1.0
        )
        self.max_air = (
            float(np.max(self.air_matrix))
            if self.air_matrix is not None and np.max(self.air_matrix) > 0
            else 1.0
        )

    def _ensure_numpy_matrix(self, mat: Any) -> Optional[np.ndarray]:
        """
        Converts extracted matrix data (which might be a dict, nested dict,
        or wrapped structure) into a standard 2D NumPy float array.
        """
        if mat is None:
            return None

        if isinstance(mat, np.ndarray):
            return mat.astype(np.float32)

        if isinstance(mat, dict):
            for inner_key in ["matrix", "data", "distances", "distance_matrix", "values", "array"]:
                if inner_key in mat:
                    sub_val = mat[inner_key]
                    if isinstance(sub_val, np.ndarray):
                        return sub_val.astype(np.float32)
                    elif isinstance(sub_val, list):
                        return np.array(sub_val, dtype=np.float32)

            sample_val = next(iter(mat.values())) if len(mat) > 0 else None
            if isinstance(sample_val, dict):
                all_keys = set(mat.keys())
                for sub_dict in mat.values():
                    all_keys.update(sub_dict.keys())
                n = max(int(k) for k in all_keys) + 1 if all_keys else len(mat)
                arr = np.zeros((n, n), dtype=np.float32)
                for i, row in mat.items():
                    for j, dist in row.items():
                        arr[int(i), int(j)] = float(dist)
                return arr

            sample_key = next(iter(mat.keys())) if len(mat) > 0 else None
            if isinstance(sample_key, tuple):
                nodes = set()
                for i, j in mat.keys():
                    nodes.add(int(i))
                    nodes.add(int(j))
                n = max(nodes) + 1 if nodes else 0
                arr = np.zeros((n, n), dtype=np.float32)
                for (i, j), dist in mat.items():
                    arr[int(i), int(j)] = float(dist)
                return arr

            try:
                return np.array(list(mat.values()), dtype=np.float32)
            except Exception:
                pass

        if isinstance(mat, list):
            return np.array(mat, dtype=np.float32)

        return None

    def _extract_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts land and air matrices directly from the environment or data generator."""
        unwrapped = getattr(self.env, "unwrapped", self.env)

        targets = [unwrapped]
        for attr in ["data_loader", "generator", "scenario", "config"]:
            if hasattr(unwrapped, attr):
                obj = getattr(unwrapped, attr)
                targets.append(obj)
                if hasattr(obj, "generator"):
                    targets.append(getattr(obj, "generator"))

        raw_land, raw_air = None, None

        for t in targets:
            try:
                if raw_land is None and hasattr(t, "_system") and hasattr(t._system, "network"):
                    raw_land = getattr(t._system.network, "land_distance_matrix", None)
                if raw_air is None and hasattr(t, "_system") and hasattr(t._system, "network"):
                    raw_air = getattr(t._system.network, "air_distance_matrix", None)
            except Exception:
                pass

        land_mat = self._ensure_numpy_matrix(raw_land)
        air_mat = self._ensure_numpy_matrix(raw_air)

        return land_mat, air_mat

    def decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Maps an integer action to its operational context and target node."""
        if action_idx < self.num_nodes:
            return {
                "type": "node_visit",
                "origin": None,
                "target_node": action_idx,
                "is_drone": False,
                "is_hub": (action_idx < self.num_microhubs),
                "is_depot": (action_idx == 0),
            }
        else:
            offset = action_idx - self.num_nodes
            hub_idx = offset // self.num_nodes
            target_node = offset % self.num_nodes
            return {
                "type": "drone_route",
                "origin": hub_idx,
                "target_node": target_node,
                "is_drone": True,
                "is_hub": False,
                "is_depot": False,
            }


# ----------------------------------------------------------------------
# 3. Dynamic Perturbation Optimizer (DPO)
# ----------------------------------------------------------------------
class DynamicPerturbationOptimizer:
    """
    Adaptive Priority Search over Node Action Space.
    1. Operates directly on continuous customer/node priorities.
    2. Completely avoids mask collisions (100% compliant with env.action_masks()).
    3. Uses dynamic adaptive perturbation to break out of reward plateaus.
    """

    def __init__(
        self,
        env,
        num_nodes: int = 32,
        num_microhubs: int = 6,
        pop_size: int = 24,
        generations: int = 150,
        base_mutation_rate: float = 0.20,
        elite_ratio: float = 0.15,
        patience_threshold: int = 8,
        seed: int = 42,
    ):
        self.env = env
        self.num_nodes = num_nodes
        self.num_microhubs = max(1, num_microhubs)
        self.extractor = ActionFeatureExtractor(env, num_nodes, num_microhubs)

        self.pop_size = pop_size
        self.generations = generations
        self.base_mutation_rate = base_mutation_rate
        self.elite_count = max(2, int(pop_size * elite_ratio))
        self.patience_threshold = patience_threshold
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Chromosome represents priority keys [0.0, 1.0] for every node
        self.chromosome_dim = self.num_nodes
        self.population = np.random.uniform(0.0, 1.0, size=(self.pop_size, self.chromosome_dim))

    def evaluate_chromosome(self, priorities: np.ndarray) -> float:
        """
        Rolls out an entire episode.
        At each step, picks the valid action whose target node has the highest priority score.
        Always feasible: zero penalties, zero crashes, zero heuristic collisions.
        """
        obs, info = self.env.reset(seed=self.seed)
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            mask = self.env.action_masks()
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) == 0:
                break

            # Map all valid actions to their target destination nodes
            target_nodes = [
                self.extractor.decode_action(a)["target_node"]
                for a in valid_indices
            ]

            # Extract priority scores for valid target nodes
            action_priorities = priorities[target_nodes]

            # Select the valid action with the maximum priority
            best_action = valid_indices[np.argmax(action_priorities)]

            step_res = self.env.step(best_action)
            if len(step_res) == 5:
                obs, reward, done, truncated, info = step_res
            else:
                obs, reward, done, info = step_res
                truncated = False

            total_reward += reward

        return float(total_reward)

    def optimize(self) -> Tuple[np.ndarray, float]:
        print("\n--- STARTING DYNAMIC PERTURBATION OPTIMIZER (DPO) ---")
        print(f"Population: {self.pop_size} | Generations: {self.generations} | Nodes: {self.num_nodes}")
        print(f"Land Matrix: {'DETECTED' if self.extractor.land_matrix is not None else 'UNAVAILABLE'}")
        print(f"Air Matrix : {'DETECTED' if self.extractor.air_matrix is not None else 'UNAVAILABLE'}\n")

        start_time = time.time()
        best_overall_genome = None
        best_overall_fitness = -float("inf")
        stagnation_counter = 0

        for gen in range(1, self.generations + 1):
            gen_start = time.time()

            # 1. Evaluate population
            fitness_scores = np.array([self.evaluate_chromosome(ind) for ind in self.population])

            # Rank descending (highest reward first)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            self.population = self.population[sorted_indices]
            fitness_scores = fitness_scores[sorted_indices]

            gen_best_fit = fitness_scores[0]

            # Track stagnation and adapt perturbation strength
            if gen_best_fit > best_overall_fitness:
                best_overall_fitness = gen_best_fit
                best_overall_genome = copy.deepcopy(self.population[0])
                stagnation_counter = 0
                status_tag = "[IMPROVED]"
            else:
                stagnation_counter += 1
                status_tag = f"[STAGNANT {stagnation_counter}/{self.patience_threshold}]"

            # Dynamic perturbation scaling
            if stagnation_counter >= self.patience_threshold:
                perturbation_sigma = 0.35
                mutation_prob = 0.50
                status_tag = "[*HEAVY PERTURBATION*]"
                stagnation_counter = 0
            else:
                perturbation_sigma = 0.10 + (0.02 * stagnation_counter)
                mutation_prob = self.base_mutation_rate

            # 2. Build Next Generation
            new_population = []

            # (A) Elitism: retain top individuals
            for i in range(self.elite_count):
                new_population.append(copy.deepcopy(self.population[i]))

            # (B) Inject fresh random individuals when stagnating
            inject_count = 2 if status_tag == "[*HEAVY PERTURBATION*]" else 1
            for _ in range(inject_count):
                new_population.append(np.random.uniform(0.0, 1.0, size=self.chromosome_dim))

            # (C) Biased Crossover & Perturbation Mutation
            elite_pool = self.population[:self.elite_count]
            non_elite_pool = self.population[self.elite_count:]

            while len(new_population) < self.pop_size:
                p_elite = elite_pool[random.randint(0, len(elite_pool) - 1)]
                p_other = non_elite_pool[random.randint(0, len(non_elite_pool) - 1)]

                # 70% allele bias toward superior elite parent
                crossover_mask = np.random.uniform(0.0, 1.0, size=self.chromosome_dim) < 0.70
                child = np.where(crossover_mask, p_elite, p_other)

                # Dynamic Gaussian jitter
                if random.random() < mutation_prob:
                    noise = np.random.normal(0.0, perturbation_sigma, size=self.chromosome_dim)
                    child = np.clip(child + noise, 0.0, 1.0)

                new_population.append(child)

            self.population = np.array(new_population)
            gen_time = time.time() - gen_start

            print(
                f"Gen {gen:03d}/{self.generations:03d} {status_tag:22s} | "
                f"Best: {best_overall_fitness:8.4f} | "
                f"Gen Best: {gen_best_fit:8.4f} | "
                f"Avg: {np.mean(fitness_scores):8.4f} | "
                f"Time: {gen_time:4.1f}s"
            )

        total_time = time.time() - start_time
        print(f"\n--- OPTIMIZATION COMPLETE ---")
        print(f"Optimal Cumulative Return: {best_overall_fitness:.4f}")
        print(f"Total Search Time: {total_time:.2f}s")

        return best_overall_genome, best_overall_fitness


# ----------------------------------------------------------------------
# 4. CLI Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    vrp_instance_path = os.path.normpath(
        os.path.join(script_path, "..", "ddls_src", "scenarios", "vrp_d_instances", "VRP-D", "A-n32-k5")
    )
    instance_name = os.path.splitext(os.path.basename(vrp_instance_path))[0].replace("-", "_")

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            "generator_type": "f2evrpd",
            "generator_config": {
                "instance_path": vrp_instance_path,
                "num_drones": 6,
                "num_microhubs": 6,
                "bbox": (0, 0, 100, 100),
                "std_dev_scale": 4.0,
                "drone_capacity_ratio": 0.2,
                "truck_speed": 1.0,
                "drone_speed": 1.5,
                "seed": 42,
            },
        },
    }

    parser = argparse.ArgumentParser(description="Dynamic Perturbation Optimizer (DPO)")
    parser.add_argument("--pop", type=int, default=24, help="Population size")
    parser.add_argument("--gens", type=int, default=300, help="Generations")
    parser.add_argument("--nodes", type=int, default=32, help="Instance node count")
    parser.add_argument("--hubs", type=int, default=6, help="Microhub count")
    parser.add_argument("--patience", type=int, default=8, help="Generations before scaling perturbation")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    env = make_env(vrp_instance_path, sim_config, instance_name)

    optimizer = DynamicPerturbationOptimizer(
        env=env,
        num_nodes=args.nodes,
        num_microhubs=args.hubs,
        pop_size=args.pop,
        generations=args.gens,
        patience_threshold=args.patience,
        seed=args.seed,
    )

    best_genome, best_fitness = optimizer.optimize()