import os
import argparse
import pandas as pd
from sb3_contrib import MaskablePPO
from rl_ext.training.base import Training


class MaskablePPOTraining(Training):
    """
    Top-layer Maskable PPO Training implementation.
    """
    name = "MaskablePPODynamicInst"

    def train(self, total_timesteps: int = 500000):
        # Tensorboard path setup
        tb_log = os.path.join(self.run_dir, "tb_logs") if self.save_summary else None

        custom_policy_kwargs = dict(
            net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64])
        )

        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            policy_kwargs=custom_policy_kwargs,
            learning_rate=1e-4,
            n_steps=4096,
            n_epochs=15,
            batch_size=256,
            gamma=0.99,
            ent_coef=0.04,
            tensorboard_log=tb_log,
            device="cuda"
        )

        print(f"\n--- SESSION STARTING: {self.name} ---")
        print(f"Root: {self.project_root}")
        print(f"Saving to: {self.run_dir}\n")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.get_episode_callback(),
            progress_bar=True
        )

        # Run-wide Aggregation of Report Metrics
        if self.all_episodes_kpis:
            df = pd.DataFrame(self.all_episodes_kpis)
            self.save_custom_file(f"{self.name}_run_averages.json", df.mean(numeric_only=True).to_dict())

        if self.save_models:
            self.model.save(os.path.join(self.run_dir, f"final_{self.name}_model"))


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    vrp_instance_path = os.path.join(
        script_path,
        "..",
        "..",
        "ddls_src",
        "scenarios",
        "vrp_d_instances",
        "VRP-D",
        "A-n33-k6"
    )
    vrp_instance_path = os.path.normpath(vrp_instance_path)

    # Append _dynamic to separate these logs/models from static runs
    base_instance_name = os.path.splitext(os.path.basename(vrp_instance_path))[0].replace("-", "_")
    instance_name = f"{base_instance_name}_dynamic"

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            # IMPORTANT: Map "dynamic_vrpd" to DynamicDemandPerturbationGenerator in your factory
            "generator_type": "dynamic_vrpd",
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

                # Dynamic generator specific configuration
                "demand_variance": 0.25
            }
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ddls_src/config/large_instance.json")
    parser.add_argument("--timesteps", type=int, default=5000000)
    args = parser.parse_args()

    trainer = MaskablePPOTraining(
        config_path=vrp_instance_path,
        save_models=True,
        save_episode_data=True,
        sim_config=sim_config,
        instance_name=instance_name,
    )

    # Utilize the parsed arguments
    trainer.train(total_timesteps=args.timesteps)