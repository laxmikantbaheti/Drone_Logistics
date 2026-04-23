import os
import json
import csv
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from rl_ext.env import LogisticsEnv


class Training(ABC):
    name = "BaseTraining"

    def __init__(self, config_path: str, **kwargs):
        # 1. Resolve Project Root (Drone_Logistics)
        self.project_root = Path(__file__).resolve().parents[2]

        # 2. Resolve Absolute Config Path
        clean_path = config_path.replace("Drone_Logistics/", "").replace("\\", "/")
        self.config_full_path = (self.project_root / clean_path).resolve()

        if not self.config_full_path.exists():
            raise FileNotFoundError(f"Config not found at: {self.config_full_path}")

        # 3. Movement Config Wrapper
        self.sim_wrapper_config = {
            "movement_mode": "matrix",
            "initial_time": 0.0,
            "main_timestep_duration": 1.0,
            "data_loader_config": {
                "generator_type": "json_file",
                "generator_config": {"file_path": str(self.config_full_path)}
            }
        }

        self.run_id_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.name}_{self.run_id_timestamp}"

        self.save_models = kwargs.get('save_models', True)
        self.save_episode_data = kwargs.get('save_episode_data', True)
        self.save_summary = kwargs.get('save_summary', True)
        self.save_metadata = kwargs.get('save_metadata', True)

        self.run_dir = self.project_root / "results" / run_id
        self.all_episodes_kpis = []

        if self.save_episode_data or self.save_models:
            self.episodes_dir = self.run_dir / "episodes"
            os.makedirs(self.episodes_dir, exist_ok=True)

        if self.save_metadata:
            self.metadata_dir = self.run_dir / "metadata"
            os.makedirs(self.metadata_dir, exist_ok=True)
            shutil.copy(self.config_full_path, self.run_dir / f"{self.name}_data.json")

        self.summary_csv = self.run_dir / f"{self.name}_summary.csv"
        self.episode_log_csv = self.run_dir / f"{self.name}_episode_log.csv"

        self.env = LogisticsEnv(sim_config=self.sim_wrapper_config)

    def log_episode(self, metrics: Dict[str, Any]):
        if not self.save_episode_data: return
        file_exists = os.path.isfile(self.episode_log_csv)
        with open(self.episode_log_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(metrics)

    def log_summary(self, metrics: Dict[str, Any]):
        if not self.save_summary: return
        file_exists = os.path.isfile(self.summary_csv)
        with open(self.summary_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(metrics)

    def save_custom_file(self, filename, data, format='json', sub_dir=None):
        if not self.save_metadata: return
        base = sub_dir if sub_dir else self.metadata_dir
        path = os.path.join(base, filename)
        with open(path, 'w') as f:
            if format == 'json':
                json.dump(data, f, indent=4)
            else:
                f.write(str(data))

    @abstractmethod
    def train(self, total_timesteps: int):
        pass

    def get_episode_callback(self):
        from rl_ext.callbacks import SaveOnEpisodeCallback
        return SaveOnEpisodeCallback(trainer_instance=self)