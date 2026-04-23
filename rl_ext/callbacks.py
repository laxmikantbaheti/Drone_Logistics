import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# --- MODIFICATION: Import your specific report function ---
from ddls_src.functions.reports import export_simulation_reports


class SaveOnEpisodeCallback(BaseCallback):
    def __init__(self, trainer_instance, verbose: int = 0):
        super(SaveOnEpisodeCallback, self).__init__(verbose)
        self.trainer = trainer_instance
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.episode_count += 1

            # Access simulation state from the environment
            system = self.training_env.envs[0].unwrapped._system
            global_state = system.global_state

            # --- MODIFICATION: Generate reports for the current episode ---
            # Define filepath inside the episode subfolder
            base_filename = f"{self.trainer.name}_ep_{self.episode_count}"
            report_path = os.path.join(self.trainer.episodes_dir, base_filename)

            # Call your function (using JSON for easy internal parsing + CSV for artifacts)
            nodes, orders, timeline = export_simulation_reports(
                global_state=global_state,
                output_format='csv',  # Generates the 3 CSV files you want
                base_filepath=report_path
            )

            # --- MODIFICATION: Extract and aggregate numerical KPIs ---
            if orders:
                # Calculate simple KPIs from this episode's order records
                durations = [o['Duration (s)'] for o in orders if o['Status'] == 'Delivered']
                avg_duration = np.mean(durations) if durations else 0
                total_delivered = sum(1 for o in orders if o['Status'] == 'Delivered')

                ep_kpis = {
                    "episode": self.episode_count,
                    "avg_delivery_duration": avg_duration,
                    "total_delivered": total_delivered,
                    "total_orders": len(orders),
                    "success_rate": total_delivered / len(orders) if orders else 0
                }

                # Push to trainer for run-wide aggregation
                self.trainer.all_episodes_kpis.append(ep_kpis)

                # Also log these specific KPIs to the parent training_summary.csv
                if self.trainer.save_summary:
                    self.trainer.log_summary({
                        "step": self.num_timesteps,
                        "reward": float(self.locals["rewards"][0]),
                        **ep_kpis
                    })

        return True