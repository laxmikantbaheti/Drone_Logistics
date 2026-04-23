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
            n_steps=200,
            verbose=1,
            tensorboard_log=tb_log,
            stats_window_size=1
        )

        print(f"\n--- SESSION STARTING: {self.name} ---")
        print(f"Root: {self.project_root}")
        print(f"Saving to: {self.run_dir}\n")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.get_episode_callback(),
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\nInterrupt detected. Cleaning up and finalizing logs...")

        # Run-wide Aggregation of Report Metrics
        if self.all_episodes_kpis:
            df = pd.DataFrame(self.all_episodes_kpis)
            self.save_custom_file(f"{self.name}_run_averages.json", df.mean().to_dict())

        if self.save_models:
            self.model.save(os.path.join(self.run_dir, f"final_{self.name}_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path is relative to Drone_Logistics/
    parser.add_argument("--config", type=str, default="ddls_src/config/large_instance.json")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    # Pass command line arguments to the Training context
    trainer = MaskablePPOTraining(
        config_path=args.config,
        save_models=True,
        save_episode_data=True
    )
    trainer.train(total_timesteps=args.timesteps)