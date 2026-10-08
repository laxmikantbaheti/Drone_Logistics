import os
import argparse
import copy
import random
import numpy as np
import pandas as pd
from rl_ext.training.base import Training


# ----------------------------------------------------------------------
# 1. Base Environment Factory / Resolver
# ----------------------------------------------------------------------
class _EnvLoader(Training):
    """
    Minimal concrete subclass to bypass the abstract method check
    and extract self.env initialized via the base Training harness.
    """
    def train(self, *args, **kwargs):
        pass


def make_env(vrp_instance_path: str, sim_config: dict, instance_name: str):
    """
    Creates and returns the RL environment initialized with the identical
    sim_config dictionary used in your MaskablePPO setup.
    """
    loader = _EnvLoader(
        config_path=vrp_instance_path,
        sim_config=sim_config,
        instance_name=instance_name,
        save_models=False,
        save_episode_data=False,
    )
    return loader.env

# ----------------------------------------------------------------------
# 2. Heuristic Action-Mask Compliant Rollout Evaluator
# ----------------------------------------------------------------------
def evaluate_sequence(env, action_sequence: list, seed: int = 42):
    """
    Executes a high-level action/node sequence in the environment while
    strictly respecting action_masks at each time step.
    """
    obs, info = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0.0
    step_idx = 0
    infeasible_attempts = 0

    for target_action in action_sequence:
        if done or truncated:
            break

        mask = env.action_masks()

        # Capacity & Transition Mask Validation
        if mask[target_action] == 1:
            chosen_action = target_action
        else:
            # Mask violation (e.g. capacity exhausted, node invalid for current vehicle state)
            infeasible_attempts += 1
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) == 0:
                break  # Deadlock
            # Fallback: Pick first valid action (or return-to-depot action)
            chosen_action = int(valid_indices[0])

        obs, reward, done, truncated, info = env.step(chosen_action)
        total_reward += reward
        step_idx += 1

    # Fitness: reward (negative cost) minus penalty for mask breaches
    fitness = total_reward
    return fitness, info


# ----------------------------------------------------------------------
# 3. Main Runner with sim_config
# ----------------------------------------------------------------------
if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    # VRP-D Instance Path
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

    # Complete Simulation Configuration matching MaskablePPO setup
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

    parser = argparse.ArgumentParser(description="Heuristic Evaluation with sim_config")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval_seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n--- INITIALIZING HEURISTIC RUNNER ---")
    print(f"Instance Name : {instance_name}")
    print(f"Instance Path : {vrp_instance_path}")
    print(f"Generator Type: {sim_config['data_loader_config']['generator_type']}\n")

    # Build environment using sim_config
    env = make_env(
        vrp_instance_path=vrp_instance_path,
        sim_config=sim_config,
        instance_name=instance_name
    )

    # Example baseline evaluation
    obs, info = env.reset(seed=args.eval_seed)
    num_actions = env.action_space.n if hasattr(env.action_space, "n") else len(env.action_masks())

    # Generate test candidate sequence
    sample_candidate = [random.randint(0, num_actions - 1) for _ in range(50)]

    score, run_info = evaluate_sequence(env, sample_candidate, seed=args.eval_seed)
    print(f"Evaluation Complete | Fitness Score: {score:.2f}")