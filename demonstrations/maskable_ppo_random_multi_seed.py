import os
import random
import argparse
from typing import Dict, Any, Optional
import numpy as np
import torch
import pandas as pd
from sb3_contrib import MaskablePPO
from rl_ext.training.base import Training


def set_global_seeds(seed: int):
    """
    Sets deterministic seeds across all random number generators:
    Python stdlib, NumPy, PyTorch (CPU and CUDA), and Python hash seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_large_instance_rl_sim_config(num_nodes: int = 60, seed: int = 42) -> Dict[str, Any]:
    """
    Constructs the simulation configuration dictionary for RL training using
    DistanceMatrixDataGenerator on a large-scale network (>= 50 nodes).
    Propagates the exact seed through all layers.
    """
    return {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "seed": seed,
        "p_seed": seed,
        "data_loader_config": {
            "generator_type": "distance_matrix",
            "generator_config": {
                "seed": seed,
                "base_scale_factor": 10,
                "num_nodes": max(30, num_nodes),  # Strictly >= 50 nodes
                "area_x_range": (0.0, 200.0),
                "area_y_range": (0.0, 200.0),
                "scaling_factors": {
                    "nodes": 6.0,
                    "depots": 0.3,        # ~3 depots
                    "customers": 4.5,     # ~45 customers
                    "micro_hubs": 0.6,    # ~6 micro-hubs (and 6 drones)
                    "trucks": 0.5,        # ~5 trucks
                    "initial_orders": 3.5 # ~35 orders
                },
                "truck_payload_range": [8, 16],
                "drone_payload_range": [1, 3],
                "truck_speed_range": [40.0, 80.0],
                "drone_speed_range": [25.0, 50.0],
                "initial_fuel_range": [100.0, 200.0],
                "initial_battery_range": [0.85, 1.0],
                "sla_min_hours": 1.5,
                "sla_max_hours": 6.0,
                "priority_distribution": {1: 0.6, 2: 0.3, 3: 0.1},
                "truck_fuel_consumption_rate": 0.08,
                "drone_battery_drain_rate_flying": 0.004,
                "drone_battery_drain_rate_idle": 0.0008,
                "drone_battery_charge_rate": 0.02,
                "drone_eligible_order_ratio": 0.45
            }
        }
    }


class RandomInstanceMaskablePPOTraining(Training):
    """
    Maskable PPO Training pipeline configured for procedural,
    in-memory distance-matrix-based logistics instances with complete seed reproducibility.
    """
    C_NAME = "MaskablePPO_RandomInstance"
    C_RANDOM = True

    def __init__(
        self,
        sim_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        model_seed: int = 42,
        save_models: bool = True,
        save_episode_data: bool = True,
        save_summary: bool = True,
        **kwargs
    ):
        self.model_seed = model_seed

        # Keep sim config generation deterministic based on its own config
        if sim_config is not None:
            self.sim_config = sim_config
            super().__init__(
                sim_config=sim_config,
                config_path=config_path,
                save_models=save_models,
                save_episode_data=save_episode_data,
                save_summary=save_summary,
                **kwargs
            )
            self.sim_config = sim_config
        else:
            super().__init__(
                config_path=config_path,
                save_models=save_models,
                save_episode_data=save_episode_data,
                save_summary=save_summary,
                **kwargs
            )

    def train(self, total_timesteps: int = 500000):
        # Apply model-specific seed to PyTorch / SB3 policy weights & exploration
        set_global_seeds(self.model_seed)

        # Tensorboard directory setup
        tb_log = os.path.join(self.run_dir, "tb_logs") if self.save_summary else None

        custom_policy_kwargs = dict(
            net_arch=dict(pi=[64, 128, 64], vf=[64, 128, 64])
        )

        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            policy_kwargs=custom_policy_kwargs,
            learning_rate=1e-4,
            n_steps=1024,
            n_epochs=15,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.04,
            tensorboard_log=tb_log,
            device="cuda",
            seed=self.model_seed
        )

        print(f"\n--- RL TRAINING SESSION STARTING: {self.name} ---")
        print(f"Model Seed: {self.model_seed}")
        print(f"Root Directory: {self.project_root}")
        print(f"Run Directory: {self.run_dir}")
        print(f"Total Timesteps: {total_timesteps}\n")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.get_episode_callback(),
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\nInterrupt detected. Cleaning up and finalizing logs...")

        # Run-wide Aggregation of Report Metrics
        if hasattr(self, "all_episodes_kpis") and self.all_episodes_kpis:
            df = pd.DataFrame(self.all_episodes_kpis)
            self.save_custom_file(f"{self.name}_seed_{self.model_seed}_averages.json", df.mean().to_dict())

        if self.save_models and hasattr(self, "model"):
            model_save_path = os.path.join(self.run_dir, f"final_{self.name}_model_seed_{self.model_seed}")
            self.model.save(model_save_path)
            print(f"[Model Checkpoint] Saved trained model to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Maskable PPO across 5 model seeds on a fixed dataset config instance")
    parser.add_argument("--nodes", type=int, default=60, help="Number of nodes to generate (>= 50)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total RL training timesteps per seed")
    args = parser.parse_args()

    # 1. Dataset / scenario config remains strictly default (seed=42)
    fixed_sim_config = get_large_instance_rl_sim_config(num_nodes=args.nodes)

    # 2. Define the 5 separate model training seeds
    model_seeds = [42, 101, 202, 303, 404]

    print("==================================================")
    print("Running 5 Model Seeds on Identical Instance/Config")
    print(f"Instance Seed: {fixed_sim_config['seed']}")
    print(f"Model Seeds: {model_seeds}")
    print("==================================================\n")

    for idx, m_seed in enumerate(model_seeds, start=1):
        print(f"\n>>> [Run {idx}/5] Training with Model Seed: {m_seed} <<<")

        trainer = RandomInstanceMaskablePPOTraining(
            sim_config=fixed_sim_config,
            model_seed=m_seed,
            save_models=True,
            save_episode_data=True,
            save_summary=True
        )
        trainer.train(total_timesteps=args.timesteps)