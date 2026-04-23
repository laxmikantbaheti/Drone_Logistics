import os
import json
import argparse
import pandas as pd
from datetime import datetime
from sb3_contrib import MaskablePPO

# Local Project Imports
from rl_ext.training.base import Training


class MaskablePPOTraining(Training):
    """
    Top-layer Training logic for Maskable PPO.
    Integrates constraints, custom toggles, and framework-level reporting.
    """
    # --- MODIFICATION: Identity defined as a class attribute ---
    name = "MaskablePPO"

    def train(self, total_timesteps: int = 500000):
        """
        Executes the training loop and performs run-wide data aggregation.
        """
        # 1. Initialize the Maskable PPO Model
        # It automatically interfaces with LogisticsEnv.action_masks()
        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            gamma=0.99,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            # --- MODIFICATION: Conditional TensorBoard logging ---
            tensorboard_log=os.path.join(self.run_dir, "tb_logs") if self.save_summary else None
        )

        print(f"\n" + "=" * 60)
        print(f"RUN STARTED: {self.name}")
        print(f"Directory: {self.run_dir}")
        print(f"=" * 60 + "\n")

        try:
            # 2. Start Learning
            # The callback handles per-episode data and framework reports.py logic
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.get_episode_callback(),
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted. Finalizing logs...")

        # --- MODIFICATION: Run-Wide Report Aggregation ---
        # Combines numerical KPIs from every episode's export_simulation_reports()
        if self.all_episodes_kpis:
            df = pd.DataFrame(self.all_episodes_kpis)

            # Calculate Averages for the whole run
            run_averages = df.mean().to_dict()

            # Save the "Grand Total" Report
            summary_report = {
                "experiment_name": self.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_episodes_sampled": len(self.all_episodes_kpis),
                "run_wide_averages": run_averages
            }

            # Use the Base Training API to save the final meta-report
            self.save_custom_file(f"{self.name}_final_run_summary.json", summary_report)
            print(f"-> Successfully aggregated KPIs for {len(self.all_episodes_kpis)} episodes.")

        # --- MODIFICATION: Respect Model Save Toggle ---
        if self.save_models:
            final_path = os.path.join(self.run_dir, f"final_{self.name}_model")
            self.model.save(final_path)
            print(f"Final model saved to: {final_path}")


# ---------------------------------------------------------
# TOP MOST EXECUTION LAYER
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4Drone: Maskable PPO Training Layer")

    # Fundamental Args
    parser.add_argument("--config", type=str, default="ddls_src/config/large_instance.json", help="Path to config")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")

    # --- MODIFICATION: Data Toggle Arguments ---
    # These control the flags in the Training base class
    parser.add_argument("--no_models", action="store_false", dest="save_models", help="Disable model saving")
    parser.add_argument("--no_ep_data", action="store_false", dest="save_episode_data",
                        help="Disable individual episode reports")
    parser.add_argument("--no_summary", action="store_false", dest="save_summary", help="Disable aggregate CSV logging")

    args = parser.parse_args()

    # --- MODIFICATION: Initialize the Trainer (Context) ---
    # The base class Training.__init__ handles directory creation and config cloning
    trainer = MaskablePPOTraining(
        config_path=args.config,
        save_models=args.save_models,
        save_episode_data=args.save_episode_data,
        save_summary=args.save_summary
    )

    # --- MODIFICATION: Trigger Training (Strategy) ---
    trainer.train(total_timesteps=args.timesteps)