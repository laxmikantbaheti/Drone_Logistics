import os
import json
import csv
import shutil
import pandas as pd  # Used for aggregation
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from rl_ext.env import LogisticsEnv


class Training(ABC):
    name = "BaseTraining"

    def __init__(
            self,
            config_path: str,
            base_results_dir: str = "results",
            save_models: bool = True,
            save_episode_data: bool = True,
            save_summary: bool = True,
            save_metadata: bool = True
    ):
        self.save_models = save_models
        self.save_episode_data = save_episode_data
        self.save_summary = save_summary
        self.save_metadata = save_metadata

        # --- MODIFICATION: Store numerical data from reports for aggregation ---
        self.all_episodes_kpis = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.name}_{timestamp}"
        self.run_dir = os.path.join(base_results_dir, run_id)

        if self.save_episode_data or self.save_models:
            self.episodes_dir = os.path.join(self.run_dir, "episodes")
            os.makedirs(self.episodes_dir, exist_ok=True)

        if self.save_metadata:
            self.metadata_dir = os.path.join(self.run_dir, "metadata")
            os.makedirs(self.metadata_dir, exist_ok=True)

        self.summary_csv = os.path.join(self.run_dir, f"{self.name}_summary.csv")

        with open(config_path, 'r') as f:
            self.sim_config = json.load(f)

        if self.save_metadata:
            os.makedirs(self.run_dir, exist_ok=True)
            shutil.copy(config_path, os.path.join(self.run_dir, f"{self.name}_config_snapshot.json"))

        self.env = LogisticsEnv(sim_config=self.sim_config)

    def log_summary(self, metrics: Dict[str, Any]):
        if not self.save_summary: return
        file_exists = os.path.isfile(self.summary_csv)
        with open(self.summary_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(metrics)

    @abstractmethod
    def train(self, total_timesteps: int):
        pass

    def get_episode_callback(self):
        from ddls_src.rl_extension.callbacks import SaveOnEpisodeCallback
        return SaveOnEpisodeCallback(trainer_instance=self)