import os
import argparse
import random
import math
import copy
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from rl_ext.training.base import Training


# ----------------------------------------------------------------------
# 1. Environment Construction Harness
# ----------------------------------------------------------------------
class _EnvLoader(Training):
    """
    Minimal concrete subclass to bypass the abstract method check
    and extract self.env initialized via the base Training harness.
    """
    C_RANDOM = False
    C_NAME = "ALNS"
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
# 2. Reward-Driven ALNS Optimizer Class
# ----------------------------------------------------------------------
class RewardDrivenALNSOptimizer:
    """
    ALNS metaheuristic optimizing action sequences strictly through
    environment cumulative return, handling dynamic action masks natively.
    """
    def __init__(
        self,
        env,
        iterations: int = 150,
        init_temp: float = 200.0,
        cooling_rate: float = 0.97,
        reheat_patience: int = 20,
        segment_size: int = 30,
        seed: int = 42,
    ):
        self.env = env
        self.iterations = iterations
        self.init_temp = init_temp
        self.temperature = init_temp
        self.cooling_rate = cooling_rate
        self.reheat_patience = reheat_patience
        self.segment_size = segment_size
        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)

        # 3 Destroy, 3 Repair operators
        self.destroy_operators = [
            self.destroy_random,
            self.destroy_worst_segments,
            self.destroy_related_cluster,
        ]
        self.repair_operators = [
            self.repair_reward_regret2,
            self.repair_reward_greedy,
            self.repair_random_insertion,
        ]

        self.d_weights = [1.0] * len(self.destroy_operators)
        self.r_weights = [1.0] * len(self.repair_operators)
        self.d_scores = [0.0] * len(self.destroy_operators)
        self.r_scores = [0.0] * len(self.repair_operators)

        # Adaptive score increments
        self.SIGMA_1 = 35.0  # New global best
        self.SIGMA_2 = 15.0  # Improved current
        self.SIGMA_3 = 5.0   # Accepted worsening move

        self.penalty_per_infeasible = 50.0  # Auto-calibrated on startup

    def _fallback_valid_action(self, valid_indices: np.ndarray) -> int:
        """
        Picks a fallback action when the candidate target action is masked out.
        Prefers non-zero actions to avoid deadlocks or premature return-to-depots.
        """
        if len(valid_indices) == 1:
            return int(valid_indices[0])

        non_zero = valid_indices[valid_indices != 0]
        if len(non_zero) > 0:
            return int(non_zero[0])

        return int(valid_indices[0])

    def evaluate_sequence(self, action_sequence: list) -> Tuple[float, list, dict, int, float]:
        """
        Rolls out the action sequence against the environment.
        Fitness is purely: Total Cumulative Environment Reward - Infeasible Action Penalties.
        """
        obs, info = self.env.reset(seed=self.seed)
        done = False
        truncated = False
        total_reward = 0.0
        infeasible_repairs = 0
        actual_executed_actions = []

        for target_action in action_sequence:
            if done or truncated:
                break

            mask = self.env.action_masks()

            if target_action < len(mask) and mask[target_action] == 1:
                chosen = target_action
            else:
                infeasible_repairs += 1
                valid_indices = np.where(mask == 1)[0]
                if len(valid_indices) == 0:
                    break
                chosen = self._fallback_valid_action(valid_indices)

            step_result = self.env.step(chosen)

            # Accommodates Gym 4-tuple and Gymnasium 5-tuple returns
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                truncated = False

            actual_executed_actions.append(chosen)
            total_reward += reward

        fitness = total_reward - (infeasible_repairs * self.penalty_per_infeasible)
        return fitness, actual_executed_actions, info, infeasible_repairs, total_reward

    def generate_initial_solution(self, max_steps: int = 60) -> list:
        """Generates a feasible baseline sequence using masked exploration."""
        obs, info = self.env.reset(seed=self.seed)
        sequence = []
        done = False
        truncated = False

        while not (done or truncated) and len(sequence) < max_steps:
            mask = self.env.action_masks()
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) == 0:
                break
            chosen = self._fallback_valid_action(valid_actions)
            step_result = self.env.step(chosen)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                truncated = False

            sequence.append(chosen)

        return sequence

    # --- Destroy Operators ---
    def destroy_random(self, sequence: list, remove_ratio: float = 0.25):
        if len(sequence) <= 3:
            return sequence.copy(), []
        num_to_remove = max(1, int(len(sequence) * remove_ratio))
        indices_to_remove = set(random.sample(range(len(sequence)), num_to_remove))

        destroyed = [node for idx, node in enumerate(sequence) if idx not in indices_to_remove]
        removed_nodes = [sequence[idx] for idx in indices_to_remove]
        return destroyed, removed_nodes

    def destroy_worst_segments(self, sequence: list, remove_ratio: float = 0.25):
        if len(sequence) <= 4:
            return self.destroy_random(sequence, remove_ratio)

        seg_len = max(2, int(len(sequence) * remove_ratio))
        start_idx = random.randint(0, len(sequence) - seg_len)

        removed_nodes = sequence[start_idx : start_idx + seg_len]
        destroyed = sequence[:start_idx] + sequence[start_idx + seg_len :]
        return destroyed, removed_nodes

    def destroy_related_cluster(self, sequence: list, remove_ratio: float = 0.25):
        if len(sequence) <= 4:
            return self.destroy_random(sequence, remove_ratio)

        num_to_remove = max(2, int(len(sequence) * remove_ratio))
        pivot_idx = random.randint(0, len(sequence) - 1)
        half_window = num_to_remove // 2
        start_idx = max(0, pivot_idx - half_window)
        end_idx = min(len(sequence), start_idx + num_to_remove)

        removed_nodes = sequence[start_idx:end_idx]
        destroyed = sequence[:start_idx] + sequence[end_idx:]
        return destroyed, removed_nodes

    # --- Reward-Based Repair Operators ---
    def repair_reward_greedy(self, destroyed_seq: list, removed_nodes: list):
        """Inserts removed nodes into the position that yields the highest cumulative reward."""
        current_seq = copy.deepcopy(destroyed_seq)

        for node in removed_nodes:
            best_pos = len(current_seq)
            best_score = -float("inf")

            candidate_positions = list(range(len(current_seq) + 1))
            # Sample up to 4 positions to prevent simulation bottleneck
            if len(candidate_positions) > 4:
                candidate_positions = random.sample(candidate_positions, 4)

            for pos in candidate_positions:
                test_seq = current_seq[:pos] + [node] + current_seq[pos:]
                score, _, _, _, _ = self.evaluate_sequence(test_seq)
                if score > best_score:
                    best_score = score
                    best_pos = pos

            current_seq.insert(best_pos, node)
        return current_seq

    def repair_reward_regret2(self, destroyed_seq: list, removed_nodes: list):
        """Computes regret directly on reward differences between top 2 candidate positions."""
        current_seq = copy.deepcopy(destroyed_seq)
        unplaced_nodes = list(removed_nodes)

        while unplaced_nodes:
            best_node_idx = 0
            max_regret = -float("inf")
            best_pos_for_chosen_node = 0

            for n_idx, node in enumerate(unplaced_nodes):
                scores = []
                candidate_positions = list(range(len(current_seq) + 1))
                if len(candidate_positions) > 4:
                    candidate_positions = random.sample(candidate_positions, 4)

                for pos in candidate_positions:
                    test_seq = current_seq[:pos] + [node] + current_seq[pos:]
                    score, _, _, _, _ = self.evaluate_sequence(test_seq)
                    scores.append((score, pos))

                scores.sort(key=lambda x: x[0], reverse=True)
                regret = (scores[0][0] - scores[1][0]) if len(scores) >= 2 else 0.0

                if regret > max_regret:
                    max_regret = regret
                    best_node_idx = n_idx
                    best_pos_for_chosen_node = scores[0][1]

            node_to_insert = unplaced_nodes.pop(best_node_idx)
            current_seq.insert(best_pos_for_chosen_node, node_to_insert)

        return current_seq

    def repair_random_insertion(self, destroyed_seq: list, removed_nodes: list):
        current_seq = copy.deepcopy(destroyed_seq)
        for node in removed_nodes:
            pos = random.randint(0, len(current_seq))
            current_seq.insert(pos, node)
        return current_seq

    # --- Search Loop ---
    def optimize(self):
        print("Calibrating reward scale and baseline sequence...")
        curr_sol = self.generate_initial_solution(max_steps=60)
        curr_fitness, _, _, init_repairs, init_raw_reward = self.evaluate_sequence(curr_sol)

        # Scale penalty automatically to 15% of the baseline reward magnitude
        self.penalty_per_infeasible = max(5.0, abs(init_raw_reward) * 0.15)

        best_sol = copy.deepcopy(curr_sol)
        best_fitness = curr_fitness

        print(f"\n--- STARTING REWARD-DRIVEN ALNS ---")
        print(f"Iterations: {self.iterations} | Init Temp: {self.temperature:.1f} | Calibrated Mask Penalty: {self.penalty_per_infeasible:.2f}")
        print(f"Baseline Reward: {init_raw_reward:.2f} | Baseline Fitness: {best_fitness:.2f} | Actions: {len(best_sol)}\n")

        start_time = time.time()
        stagnation_counter = 0

        for it in range(1, self.iterations + 1):
            d_idx = random.choices(range(len(self.destroy_operators)), weights=self.d_weights, k=1)[0]
            r_idx = random.choices(range(len(self.repair_operators)), weights=self.r_weights, k=1)[0]

            destroyed, removed = self.destroy_operators[d_idx](curr_sol)
            candidate_sol = self.repair_operators[r_idx](destroyed, removed)

            cand_fitness, _, _, cand_repairs, cand_raw = self.evaluate_sequence(candidate_sol)
            accepted = False

            if cand_fitness > best_fitness:
                best_sol = copy.deepcopy(candidate_sol)
                best_fitness = cand_fitness
                curr_sol = copy.deepcopy(candidate_sol)
                curr_fitness = cand_fitness
                self.d_scores[d_idx] += self.SIGMA_1
                self.r_scores[r_idx] += self.SIGMA_1
                accepted = True
                stagnation_counter = 0
                tag = "[*NEW BEST*]"
            elif cand_fitness > curr_fitness:
                curr_sol = copy.deepcopy(candidate_sol)
                curr_fitness = cand_fitness
                self.d_scores[d_idx] += self.SIGMA_2
                self.r_scores[r_idx] += self.SIGMA_2
                accepted = True
                stagnation_counter += 1
                tag = "[IMPROVED]"
            else:
                stagnation_counter += 1
                delta = cand_fitness - curr_fitness
                accept_prob = math.exp(delta / max(self.temperature, 1e-4))
                if random.random() < accept_prob:
                    curr_sol = copy.deepcopy(candidate_sol)
                    curr_fitness = cand_fitness
                    self.d_scores[d_idx] += self.SIGMA_3
                    self.r_scores[r_idx] += self.SIGMA_3
                    accepted = True
                    tag = "[ACCEPTED]"
                else:
                    tag = "[REJECTED]"

            self.temperature *= self.cooling_rate

            # Temperature reheating on stagnation
            if stagnation_counter >= self.reheat_patience:
                self.temperature = max(self.temperature, self.init_temp * 0.5)
                curr_sol, deep_removed = self.destroy_random(curr_sol, remove_ratio=0.40)
                curr_sol = self.repair_random_insertion(curr_sol, deep_removed)
                curr_fitness, _, _, _, _ = self.evaluate_sequence(curr_sol)
                stagnation_counter = 0
                tag = "[!REHEAT!]"

            # Segment-based operator weight updates
            if it % self.segment_size == 0:
                decay = 0.8
                for i in range(len(self.d_weights)):
                    self.d_weights[i] = decay * self.d_weights[i] + (1 - decay) * max(0.1, self.d_scores[i])
                    self.d_scores[i] = 0.0
                for j in range(len(self.r_weights)):
                    self.r_weights[j] = decay * self.r_weights[j] + (1 - decay) * max(0.1, self.r_scores[j])
                    self.r_scores[j] = 0.0

            elapsed_time = time.time() - start_time
            if it % 5 == 0 or accepted:
                print(
                    f"Iter {it:03d}/{self.iterations} {tag:12s} | "
                    f"Best Fit: {best_fitness:7.2f} (Raw: {cand_raw:7.2f}) | "
                    f"Violations: {cand_repairs:2d} | "
                    f"Temp: {self.temperature:6.2f} | "
                    f"Elapsed: {elapsed_time:.1f}s"
                )

        total_elapsed = time.time() - start_time
        final_fit, final_executed, final_info, final_repairs, final_raw = self.evaluate_sequence(best_sol)

        print("\n--- OPTIMIZATION SUMMARY ---")
        print(f"Optimal Cumulative Reward : {final_raw:.2f}")
        print(f"Optimal Fitness (Penalized): {final_fit:.2f}")
        print(f"Mask Corrections Incurred : {final_repairs}")
        print(f"Executed Steps Count       : {len(final_executed)}")
        print(f"Total Computation Time     : {total_elapsed:.2f}s")
        return best_sol, final_raw


# ----------------------------------------------------------------------
# 3. Execution Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    # Standard VRP scenario setup
    vrp_instance_path = os.path.join(
        script_path,
        "..",
        "ddls_src",
        "scenarios",
        "vrp_d_instances",
        "VRP-D",
        "A-n32-k5"
    )
    vrp_instance_path = os.path.normpath(vrp_instance_path)
    instance_name = os.path.splitext(os.path.basename(vrp_instance_path))[0].replace("-", "_")

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            "generator_type": "f2evrpd",
            "generator_config": {
                "instance_path": vrp_instance_path,
                "num_drones": 0,
                "num_microhubs": 0,
                "bbox": (0, 0, 100, 100),
                "std_dev_scale": 4.0,
                "drone_capacity_ratio": 0.2,
                "truck_speed": 1.0,
                "drone_speed": 1.0,
                "seed": 42,
            }
        },
    }

    parser = argparse.ArgumentParser(description="Pure Reward ALNS Optimizer")
    parser.add_argument("--iterations", type=int, default=150, help="ALNS search iterations")
    parser.add_argument("--temp", type=float, default=200.0, help="Initial simulated annealing temperature")
    parser.add_argument("--cooling", type=float, default=0.97, help="Cooling schedule factor")
    parser.add_argument("--patience", type=int, default=20, help="Stagnation iterations before reheat")
    parser.add_argument("--eval_seed", type=int, default=42, help="Deterministic evaluation seed")
    args = parser.parse_args()

    env = make_env(
        vrp_instance_path=vrp_instance_path,
        sim_config=sim_config,
        instance_name=instance_name
    )

    optimizer = RewardDrivenALNSOptimizer(
        env=env,
        iterations=args.iterations,
        init_temp=args.temp,
        cooling_rate=args.cooling,
        reheat_patience=args.patience,
        seed=args.eval_seed
    )

    best_sequence, best_reward = optimizer.optimize()