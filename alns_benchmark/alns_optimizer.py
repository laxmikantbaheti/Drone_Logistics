import os
import argparse
import random
import math
import copy
import time
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
# 2. ALNS Optimizer Class
# ----------------------------------------------------------------------
class ALNSOptimizer:
    def __init__(
        self,
        env,
        iterations: int = 50,
        init_temp: float = 150.0,
        cooling_rate: float = 0.96,
        seed: int = 42,
    ):
        self.env = env
        self.iterations = iterations
        self.temperature = init_temp
        self.cooling_rate = cooling_rate
        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Operators and rolling weights
        self.destroy_operators = [self.destroy_random, self.destroy_worst_segments]
        self.repair_operators = [self.repair_greedy_insertion, self.repair_random_insertion]

        self.d_weights = [1.0] * len(self.destroy_operators)
        self.r_weights = [1.0] * len(self.repair_operators)
        self.d_scores = [0.0] * len(self.destroy_operators)
        self.r_scores = [0.0] * len(self.repair_operators)

        # Performance score adjustments
        self.SIGMA_1 = 30.0  # Found new global best
        self.SIGMA_2 = 15.0  # Improved upon current solution
        self.SIGMA_3 = 5.0   # Accepted worsening move

    # --- Evaluation ---
    def evaluate_sequence(self, action_sequence: list):
        """
        Rolls out an action sequence step-by-step against env.action_masks().
        Falls back to a valid action if a mask constraint is breached.
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
                    break  # Deadlock reached
                chosen = int(valid_indices[0])

            obs, reward, done, truncated, info = self.env.step(chosen)
            actual_executed_actions.append(chosen)
            total_reward += reward

        fitness = total_reward - (infeasible_repairs * 50.0)
        return fitness, actual_executed_actions, info

    def generate_initial_solution(self, max_steps: int = 50):
        """Builds an initial feasible route using greedy valid mask sampling."""
        obs, info = self.env.reset(seed=self.seed)
        sequence = []
        done = False
        truncated = False

        while not (done or truncated) and len(sequence) < max_steps:
            mask = self.env.action_masks()
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) == 0:
                break
            chosen = int(random.choice(valid_actions))
            obs, reward, done, truncated, info = self.env.step(chosen)
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

    # --- Repair Operators ---
    def repair_greedy_insertion(self, destroyed_seq: list, removed_nodes: list):
        current_seq = copy.deepcopy(destroyed_seq)

        for node in removed_nodes:
            best_pos = len(current_seq)
            best_score = -float("inf")

            candidate_positions = list(range(len(current_seq) + 1))
            if len(candidate_positions) > 8:
                candidate_positions = random.sample(candidate_positions, 8)

            for pos in candidate_positions:
                test_seq = current_seq[:pos] + [node] + current_seq[pos:]
                score, _, _ = self.evaluate_sequence(test_seq)
                if score > best_score:
                    best_score = score
                    best_pos = pos

            current_seq.insert(best_pos, node)

        return current_seq

    def repair_random_insertion(self, destroyed_seq: list, removed_nodes: list):
        current_seq = copy.deepcopy(destroyed_seq)
        for node in removed_nodes:
            pos = random.randint(0, len(current_seq))
            current_seq.insert(pos, node)
        return current_seq

    # --- Search Loop ---
    def optimize(self):
        print("Generating initial route via valid mask rollout...")
        curr_sol = self.generate_initial_solution(max_steps=50)
        curr_fitness, _, _ = self.evaluate_sequence(curr_sol)

        best_sol = copy.deepcopy(curr_sol)
        best_fitness = curr_fitness

        print(f"\n--- STARTING ALNS (Iterations: {self.iterations}, Initial Temp: {self.temperature}) ---")
        print(f"Initial Fitness: {best_fitness:.2f} | Sequence Length: {len(best_sol)}\n")

        start_time = time.time()

        for it in range(1, self.iterations + 1):
            d_idx = random.choices(range(len(self.destroy_operators)), weights=self.d_weights, k=1)[0]
            r_idx = random.choices(range(len(self.repair_operators)), weights=self.r_weights, k=1)[0]

            d_op = self.destroy_operators[d_idx]
            r_op = self.repair_operators[r_idx]

            destroyed, removed = d_op(curr_sol)
            candidate_sol = r_op(destroyed, removed)

            cand_fitness, _, _ = self.evaluate_sequence(candidate_sol)
            accepted = False

            if cand_fitness > best_fitness:
                best_sol = copy.deepcopy(candidate_sol)
                best_fitness = cand_fitness
                curr_sol = copy.deepcopy(candidate_sol)
                curr_fitness = cand_fitness
                self.d_scores[d_idx] += self.SIGMA_1
                self.r_scores[r_idx] += self.SIGMA_1
                accepted = True
                tag = "[*NEW BEST*]"
            elif cand_fitness > curr_fitness:
                curr_sol = copy.deepcopy(candidate_sol)
                curr_fitness = cand_fitness
                self.d_scores[d_idx] += self.SIGMA_2
                self.r_scores[r_idx] += self.SIGMA_2
                accepted = True
                tag = "[IMPROVED]"
            else:
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

            # Cooling schedule & weight updates
            self.temperature *= self.cooling_rate
            decay = 0.8
            self.d_weights[d_idx] = decay * self.d_weights[d_idx] + (1 - decay) * max(0.1, self.d_scores[d_idx])
            self.r_weights[r_idx] = decay * self.r_weights[r_idx] + (1 - decay) * max(0.1, self.r_scores[r_idx])

            elapsed_time = time.time() - start_time

            if it % 5 == 0 or accepted:
                print(
                    f"Iter {it:03d}/{self.iterations} {tag} | "
                    f"Best: {best_fitness:9.2f} | "
                    f"Current: {curr_fitness:9.2f} | "
                    f"Temp: {self.temperature:6.2f} | "
                    f"Elapsed: {elapsed_time:.1f}s"
                )

        total_elapsed = time.time() - start_time
        print("\n--- ALNS COMPLETE ---")
        print(f"Optimal Fitness Found: {best_fitness:.2f}")
        print(f"Total Execution Time: {total_elapsed:.2f}s")
        return best_sol, best_fitness


# ----------------------------------------------------------------------
# 3. Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

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

    parser = argparse.ArgumentParser(description="ALNS Optimizer Module")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--temp", type=float, default=300.0)
    parser.add_argument("--cooling", type=float, default=0.96)
    parser.add_argument("--eval_seed", type=int, default=42)
    args = parser.parse_args()

    env = make_env(
        vrp_instance_path=vrp_instance_path,
        sim_config=sim_config,
        instance_name=instance_name
    )

    optimizer = ALNSOptimizer(
        env=env,
        iterations=args.iterations,
        init_temp=args.temp,
        cooling_rate=args.cooling,
        seed=args.eval_seed
    )

    best_sequence, best_fitness = optimizer.optimize()