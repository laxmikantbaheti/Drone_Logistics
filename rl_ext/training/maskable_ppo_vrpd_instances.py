import os
import argparse
import pandas as pd
from sb3_contrib import MaskablePPO
from rl_ext.training.base import Training


class MaskablePPOTraining(Training):
    """
    Top-layer Maskable PPO Training implementation.
    """
    name = "MaskablePPO"

    def train(self, total_timesteps: int = 500000):
        # Tensorboard path setup
        tb_log = os.path.join(self.run_dir, "tb_logs") if self.save_summary else None

        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=tb_log
        )

        print(f"\n--- SESSION STARTING: {self.name} ---")
        print(f"Root: {self.project_root}")
        print(f"Saving to: {self.run_dir}\n")


        self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.get_episode_callback(),
                progress_bar=True
            )
        # except KeyboardInterrupt:
        #     print("\nInterrupt detected. Cleaning up and finalizing logs...")

        # Run-wide Aggregation of Report Metrics
        if self.all_episodes_kpis:
            df = pd.DataFrame(self.all_episodes_kpis)
            self.save_custom_file(f"{self.name}_run_averages.json", df.mean(numeric_only=True).to_dict())

        if self.save_models:
            self.model.save(os.path.join(self.run_dir, f"final_{self.name}_model"))


if __name__ == "__main__":


    script_path = os.path.dirname(os.path.realpath(__file__))
    # Point to the new matrix-specific data file
    # config_file_path = os.path.join(script_path, '..', 'config', 'large_instance.json')
    # config_file_path = os.path.normpath(config_file_path)
    # Change this to match where your VRP-D instances are stored
    vrp_instance_path = os.path.join(
        script_path,
        "..",
        "..",
        "ddls_src",
        "scenarios",
        "vrp_d_instances",
        "VRP-D",
        "A-n32-k5.vrp"
    )
    vrp_instance_path = os.path.normpath(vrp_instance_path)
    instance_name = os.path.splitext(os.path.basename(vrp_instance_path))[0].replace("-", "_")
    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            # IMPORTANT: must match your generator factory registry name
            "generator_type": "vrpd",
            "generator_config": {
                "instance_path": vrp_instance_path,

                # Keep these custom for your delivery model
                "num_drones": 0,
                "num_microhubs": 0,
                "bbox": (0, 0, 100, 100),
                "std_dev_scale": 4.0,

                # Capacity-derived configs inside generator
                "drone_capacity_ratio": 0.2,

                # Speeds (if your sim uses them)
                "truck_speed": 1.0,
                "drone_speed": 1.0,

                # Optional
                "seed": 42,

                # If you want to override trucks instead of using -k# from filename:
                # "num_trucks": 5,
            }
        },
    }
    parser = argparse.ArgumentParser()
    # Path is relative to Drone_Logistics/
    parser.add_argument("--config", type=str, default="ddls_src/config/large_instance.json")
    parser.add_argument("--timesteps", type=int, default=5000000)
    args = parser.parse_args()

    # Pass command line arguments to the Training context
    trainer = MaskablePPOTraining(
        config_path=vrp_instance_path,
        save_models=True,
        save_episode_data=True,
        sim_config=sim_config,
        instance_name=instance_name,
    )
    trainer.train(total_timesteps=300000)