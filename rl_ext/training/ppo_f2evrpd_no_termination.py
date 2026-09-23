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
# Action Mask Wrapper for Standard (Unmasked) PPO - Option 1
# ----------------------------------------------------------------------
class InvalidActionMaskPenaltyWrapper(gym.Wrapper):
    """
    Wraps the simulation environment for standard PPO (Option 1).
    Penalizes invalid actions without immediately terminating the episode,
    allowing the agent to try again from the current state.
    Includes a maximum consecutive invalid threshold to prevent infinite loops.
    """

    def __init__(
        self,
        env,
        invalid_penalty: float = -1.0,
        terminate_on_invalid: bool = False,
        max_consecutive_invalid: int = 50,
    ):
        super().__init__(env)
        self.invalid_penalty = invalid_penalty
        self.terminate_on_invalid = terminate_on_invalid
        self.max_consecutive_invalid = max_consecutive_invalid
        self.consecutive_invalid_count = 0
        self.last_obs = None
        self.last_info = {}

    def reset(self, **kwargs):
        self.consecutive_invalid_count = 0
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
        mask = self.env.action_masks()

        # Check if the chosen action is invalid
        if mask[action] == 0:
            self.consecutive_invalid_count += 1
            reward = float(self.invalid_penalty)

            # Option 1: Do not terminate immediately unless exceeding consecutive failure threshold
            hit_limit = self.consecutive_invalid_count >= self.max_consecutive_invalid
            terminated = self.terminate_on_invalid or hit_limit
            truncated = False

            info = dict(self.last_info) if isinstance(self.last_info, dict) else {}
            info["is_invalid"] = True
            info["selected_action"] = int(action)
            info["consecutive_invalids"] = self.consecutive_invalid_count
            info["reward"] = reward

            if "makespan" not in info:
                info["makespan"] = 0.0

            # Generate synthetic terminal logs if terminated due to threshold or config
            if terminated:
                info["terminal_logs"] = {
                    "status": "terminated_due_to_excessive_invalid_actions" if hit_limit else "terminated_invalid",
                    "invalid_action": int(action),
                    "consecutive_invalids": self.consecutive_invalid_count,
                    "reason": "action_mask_violation",
                    "total_distance": info.get("total_distance", 0.0),
                    "delivered_orders": info.get("delivered_orders", 0),
                    "unserved_orders": info.get("unserved_orders", 0),
                }

            obs = self.last_obs
            if obs is None:
                obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

            return obs, reward, terminated, truncated, info

        # If action is valid, reset consecutive invalid counter and proceed
        self.consecutive_invalid_count = 0
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
    Top-layer Standard PPO Training implementation without action masking (Option 1).
    Penalizes illegal actions and allows retries from the same state.
    """
    name = "StandardPPO_Option1"

    def train(self, total_timesteps: int = 500000):
        tb_log = os.path.join(self.run_dir, "tb_logs") if self.save_summary else None

        # Option 1: terminate_on_invalid=False
        self.env = InvalidActionMaskPenaltyWrapper(
            self.env,
            invalid_penalty=-1.0,
            terminate_on_invalid=False,
            max_consecutive_invalid=50,
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
            device="cuda",
        )

        print(f"\n--- SESSION STARTING: {self.name} ---")
        print("Termination on invalid action: DISABLED (Option 1: Retry with -1.0 Penalty)")
        print(f"Root: {self.project_root}")
        print(f"Saving to: {self.run_dir}\n")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.get_episode_callback(),
            progress_bar=True,
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
        "A-n32-k5",
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
            },
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