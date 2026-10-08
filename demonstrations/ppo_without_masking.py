import os
import sys
import warnings
import argparse
import pandas as pd
import numpy as np

# Suppress gym unmaintained/deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO
from rl_ext.training.base import Training


# ----------------------------------------------------------------------
# Action Mask Wrapper for Standard (Unmasked) PPO
# ----------------------------------------------------------------------
class InvalidActionMaskPenaltyWrapper(gym.Wrapper):
    """
    Wraps the simulation environment for standard PPO.
    Immediately terminates the episode and applies a penalty whenever
    an invalid action (according to env.action_masks()) is selected.
    Supplies all required callback KPI keys ('reward', 'makespan', 'terminal_logs').
    """

    def __init__(self, env, invalid_penalty: float = -1.0, terminate_on_invalid: bool = True):
        super().__init__(env)
        self.invalid_penalty = invalid_penalty
        self.terminate_on_invalid = terminate_on_invalid
        self.last_obs = None
        self.last_info = {}

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple):
            self.last_obs = res[0]
            self.last_info = res[1] if len(res) > 1 and isinstance(res[1], dict) else {}
            return res
        else:
            self.last_obs = res
            self.last_info = {}
            return res, {}

    def step(self, action: int):
        # Extract action mask from the base environment
        mask = self.env.action_masks()

        # Check if the chosen action is invalid
        if mask[action] == 0:
            reward = float(self.invalid_penalty)
            terminated = self.terminate_on_invalid
            truncated = False

            # Inherit last known simulation KPIs to avoid KeyError in callbacks
            info = dict(self.last_info) if isinstance(self.last_info, dict) else {}
            info["is_invalid"] = True
            info["selected_action"] = int(action)
            info["reward"] = reward

            if "makespan" not in info:
                info["makespan"] = 0.0

            # Provide synthetic terminal logs expected by rl_ext/callbacks.py
            if "terminal_logs" not in info or terminated:
                info["terminal_logs"] = {
                    "status": "terminated_due_to_invalid_action",
                    "invalid_action": int(action),
                    "reason": "action_mask_violation",
                    "total_distance": info.get("total_distance", 0.0),
                    "delivered_orders": info.get("delivered_orders", 0),
                    "unserved_orders": info.get("unserved_orders", 0),
                }

            obs = self.last_obs
            if obs is None:
                obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

            return obs, reward, terminated, truncated, info

        # If valid, proceed normally through the simulator
        step_res = self.env.step(action)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, info = step_res
        else:
            obs, reward, terminated, info = step_res
            truncated = False

        if info is None or not isinstance(info, dict):
            info = {}

        self.last_obs = obs
        self.last_info = dict(info)

        info["is_invalid"] = False
        info["reward"] = float(reward)
        if "makespan" not in info:
            info["makespan"] = 0.0

        # Ensure terminal_logs exists if the episode finished naturally
        if (terminated or truncated) and "terminal_logs" not in info:
            info["terminal_logs"] = {
                "status": "completed",
                "makespan": info.get("makespan", 0.0),
            }

        return obs, reward, terminated, truncated, info


# ----------------------------------------------------------------------
# Standard PPO Training Class
# ----------------------------------------------------------------------
class PPOTraining(Training):
    """
    Top-layer Standard PPO Training implementation without action masking.
    Terminates episodes immediately upon sampling an illegal action.
    """
    name = "StandardPPO"

    def train(self, total_timesteps: int = 500000):
        # Tensorboard path setup
        tb_log = os.path.join(self.run_dir, "tb_logs") if self.save_summary else None

        # Wrap self.env to guard invalid actions and feed required callback info
        self.env = InvalidActionMaskPenaltyWrapper(
            self.env,
            invalid_penalty=-1.0,
            terminate_on_invalid=True
        )

        custom_policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )

        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            policy_kwargs=custom_policy_kwargs,
            learning_rate=1e-4,
            n_steps=1024,
            n_epochs=15,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.02,
            tensorboard_log=tb_log,
            device="cuda"
        )

        print(f"\n--- SESSION STARTING: {self.name} ---")
        print(f"Termination on invalid action: ENABLED")
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ddls_src/config/large_instance.json")
    parser.add_argument("--timesteps", type=int, default=5000000)
    args = parser.parse_args()

    trainer = PPOTraining(
        config_path=vrp_instance_path,
        save_models=True,
        save_episode_data=True,
        sim_config=sim_config,
        instance_name=instance_name,
    )
    trainer.train(total_timesteps=args.timesteps)