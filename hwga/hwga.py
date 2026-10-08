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
# 2. Action Decoder & Multi-Modal Feature Extractor
# ----------------------------------------------------------------------
class ActionFeatureExtractor:
    """
    Decodes the composite action space:
    1. Nodes [0 .. num_nodes-1]: Depots, microhubs, customers.
    2. Drone Routes [num_nodes .. num_nodes + (num_hubs * num_nodes) - 1]:
       Dispatches drone from a microhub to a node.
    """

    def __init__(self, env, num_nodes: int = 60, num_microhubs: int = 6):
        self.env = env
        self.num_nodes = num_nodes
        self.num_microhubs = max(1, num_microhubs)

        self.land_matrix, self.air_matrix = self._extract_matrices()

        # Cache matrix maxima to avoid redundant np.max calls inside step loops
        self.max_land = float(np.max(self.land_matrix)) if self.land_matrix is not None and np.max(
            self.land_matrix) > 0 else 1.0
        self.max_air = float(np.max(self.air_matrix)) if self.air_matrix is not None and np.max(
            self.air_matrix) > 0 else 1.0

    def _ensure_numpy_matrix(self, mat: Any) -> Optional[np.ndarray]:
        """
        Converts extracted matrix data (which might be a dict, nested dict,
        or wrapped structure) into a standard 2D NumPy float array.
        """
        if mat is None:
            return None

        # 1. Direct NumPy array
        if isinstance(mat, np.ndarray):
            return mat.astype(np.float32)

        # 2. Dictionary-wrapped matrix
        if isinstance(mat, dict):
            # Check for inner array keys
            for inner_key in ["matrix", "data", "distances", "distance_matrix", "values", "array"]:
                if inner_key in mat:
                    sub_val = mat[inner_key]
                    if isinstance(sub_val, np.ndarray):
                        return sub_val.astype(np.float32)
                    elif isinstance(sub_val, list):
                        return np.array(sub_val, dtype=np.float32)

            # Check if nested dict: dict[i][j] = distance
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

            # Check if tuple keys: dict[(i, j)] = distance
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

            # Generic dict-of-lists or array values
            try:
                return np.array(list(mat.values()), dtype=np.float32)
            except Exception:
                pass

        # 3. Standard list of lists
        if isinstance(mat, list):
            return np.array(mat, dtype=np.float32)

        return None

    def _extract_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts land and air matrices directly from the environment or data generator."""
        unwrapped = getattr(self.env, "unwrapped", self.env)

        # Search targets across hierarchy
        targets = [unwrapped]
        for attr in ["data_loader", "generator", "scenario", "config"]:
            if hasattr(unwrapped, attr):
                obj = getattr(unwrapped, attr)
                targets.append(obj)
                if hasattr(obj, "generator"):
                    targets.append(getattr(obj, "generator"))

        raw_land, raw_air = None, None

        for t in targets:
            if raw_land is None:
                raw_land = t._system.network.land_distance_matrix
            if raw_air is None:
                raw_air = t._system.network.air_distance_matrix

        land_mat = self._ensure_numpy_matrix(raw_land)
        air_mat = self._ensure_numpy_matrix(raw_air)

        return land_mat, air_mat

    def decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Maps an integer action to its operational context."""
        if action_idx < self.num_nodes:
            return {
                "type": "node_visit",
                "origin": None,
                "target_node": action_idx,
                "is_drone": False,
                "is_hub": (action_idx < self.num_microhubs),
                "is_depot": (action_idx == 0)
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
                "is_depot": False
            }

    def compute_action_features(self, action_idx: int, current_loc: int) -> np.ndarray:
        """
        Extracts 6 normalized heuristic features:
        [norm_land_dist, norm_air_dist, detour_saving_bonus, is_drone, is_hub, is_depot]
        """
        info = self.decode_action(action_idx)
        target = info["target_node"]

        # 1. Land distance
        land_dist = 1.0
        if self.land_matrix is not None:
            from_loc = current_loc if not info["is_drone"] else info["origin"]
            if from_loc < self.land_matrix.shape[0] and target < self.land_matrix.shape[1]:
                land_dist = float(self.land_matrix[from_loc, target]) / self.max_land

        # 2. Air distance
        air_dist = 1.0
        if self.air_matrix is not None:
            from_loc = info["origin"] if info["is_drone"] else current_loc
            if from_loc < self.air_matrix.shape[0] and target < self.air_matrix.shape[1]:
                air_dist = float(self.air_matrix[from_loc, target]) / self.max_air

        # 3. Detour bonus (how much distance the drone bypasses vs land travel)
        detour_bonus = max(0.0, land_dist - air_dist) if info["is_drone"] else 0.0

        return np.array([
            land_dist,  # Negative weight preferred (closer = better)
            air_dist if info["is_drone"] else 0.0,  # Flight distance
            detour_bonus,  # Positive weight preferred (incentivize drone bypass)
            1.0 if info["is_drone"] else 0.0,  # Drone action bias
            1.0 if info["is_hub"] else 0.0,  # Microhub visit bias
            1.0 if info["is_depot"] else 0.0  # Depot return bias
        ], dtype=np.float32)


# ----------------------------------------------------------------------
# 3. Heuristic-Weight Genetic Algorithm (HWGA)
# ----------------------------------------------------------------------
class HWGAOptimizer:
    """
    Optimizes a 6-parameter linear dispatch policy via Genetic Algorithm.
    Never uses neural networks; 100% compliant with action masks.
    """

    def __init__(
            self,
            env,
            num_nodes: int = 60,
            num_microhubs: int = 6,
            pop_size: int = 24,
            generations: int = 30,
            mutation_rate: float = 0.20,
            elite_ratio: float = 0.15,
            seed: int = 42
    ):
        self.env = env
        self.extractor = ActionFeatureExtractor(env, num_nodes, num_microhubs)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = max(2, int(pop_size * elite_ratio))
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.dim = 6
        self.population = np.random.uniform(-1.0, 1.0, size=(self.pop_size, self.dim))

        # Intuitive baseline seed: penalize distances, reward drone detour bypass
        self.population[0] = np.array([-1.0, -0.8, 1.2, 0.3, -0.2, -0.5])

    def evaluate_chromosome(self, weights: np.ndarray) -> float:
        """
        Executes a single episode rollout using the chromosome as a linear ranker.
        Always picks strictly from valid_indices: zero infeasible penalties.
        """
        obs, info = self.env.reset(seed=self.seed)
        done = False
        truncated = False
        total_reward = 0.0
        current_loc = 0

        while not (done or truncated):
            mask = self.env.action_masks()
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) == 0:
                break

            # Fast matrix projection of candidate features
            feature_batch = np.array([
                self.extractor.compute_action_features(a, current_loc)
                for a in valid_indices
            ])

            scores = feature_batch @ weights
            best_action = valid_indices[np.argmax(scores)]

            # Track location for next ground hop
            action_meta = self.extractor.decode_action(best_action)
            current_loc = action_meta["target_node"]

            step_res = self.env.step(best_action)
            if len(step_res) == 5:
                obs, reward, done, truncated, info = step_res
            else:
                obs, reward, done, info = step_res
                truncated = False

            total_reward += reward

        return float(total_reward)

    def optimize(self) -> Tuple[np.ndarray, float]:
        print("--- STARTING HWGA OPTIMIZER ---")
        print(f"Population: {self.pop_size} | Generations: {self.generations} | Gene Dimension: {self.dim}")
        print(
            f"Land Matrix: {'DETECTED (Numpy Array)' if self.extractor.land_matrix is not None else 'UNAVAILABLE (Unit Scaling)'}")
        print(
            f"Air Matrix : {'DETECTED (Numpy Array)' if self.extractor.air_matrix is not None else 'UNAVAILABLE (Unit Scaling)'}\n")

        start_time = time.time()
        best_overall_weights = None
        best_overall_reward = -float("inf")

        for gen in range(1, self.generations + 1):
            gen_start = time.time()

            # Sequential evaluation of individuals
            fitness_scores = np.array([self.evaluate_chromosome(ind) for ind in self.population])

            # Rank population descending
            sorted_indices = np.argsort(fitness_scores)[::-1]
            self.population = self.population[sorted_indices]
            fitness_scores = fitness_scores[sorted_indices]

            gen_best_fit = fitness_scores[0]
            if gen_best_fit > best_overall_reward:
                best_overall_reward = gen_best_fit
                best_overall_weights = copy.deepcopy(self.population[0])

            # Elitism: retain top individuals
            new_population = [copy.deepcopy(self.population[i]) for i in range(self.elite_count)]

            # Tournament selection and uniform blending crossover
            while len(new_population) < self.pop_size:
                p1_idx = min(random.sample(range(self.pop_size), 3))
                p2_idx = min(random.sample(range(self.pop_size), 3))
                parent1, parent2 = self.population[p1_idx], self.population[p2_idx]

                alpha = np.random.uniform(0.0, 1.0, size=self.dim)
                child = alpha * parent1 + (1.0 - alpha) * parent2

                # Gaussian mutation
                if random.random() < self.mutation_rate:
                    mutation_noise = np.random.normal(0.0, 0.25, size=self.dim)
                    child += mutation_noise

                new_population.append(np.clip(child, -3.0, 3.0))

            self.population = np.array(new_population)
            gen_time = time.time() - gen_start

            print(
                f"Gen {gen:02d}/{self.generations:02d} | "
                f"Best Fit: {best_overall_reward:8.2f} | "
                f"Gen Best: {gen_best_fit:8.2f} | "
                f"Gen Avg: {np.mean(fitness_scores):8.2f} | "
                f"Time: {gen_time:4.1f}s"
            )

        total_time = time.time() - start_time
        print("\n--- HWGA OPTIMIZATION COMPLETED ---")
        print(f"Optimal Cumulative Return: {best_overall_reward:.2f}")
        print(f"Optimized Heuristic Weights: {np.round(best_overall_weights, 3)}")
        print(f"Total Computation Duration: {total_time:.2f}s")

        return best_overall_weights, best_overall_reward


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
            }
        },
    }

    parser = argparse.ArgumentParser(description="Heuristic-Weight GA (HWGA)")
    parser.add_argument("--pop", type=int, default=24, help="Population size")
    parser.add_argument("--gens", type=int, default=300, help="Generations")
    parser.add_argument("--nodes", type=int, default=32, help="Instance node count")
    parser.add_argument("--hubs", type=int, default=6, help="Microhub count")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    env = make_env(vrp_instance_path, sim_config, instance_name)

    optimizer = HWGAOptimizer(
        env=env,
        num_nodes=args.nodes,
        num_microhubs=args.hubs,
        pop_size=args.pop,
        generations=args.gens,
        seed=args.seed
    )

    best_weights, best_reward = optimizer.optimize()