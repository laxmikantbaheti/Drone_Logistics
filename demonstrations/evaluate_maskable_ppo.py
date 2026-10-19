import os
import argparse
from sb3_contrib import MaskablePPO
from rl_ext.training.base import Training

# --- CODE MODIFICATION NOTICE ---
# Added compatibility shim at the top of evaluate_maskable_ppo.py to map
# NumPy 2.x's internal 'numpy._core' namespace to NumPy 1.x's 'numpy.core'
# during cloudpickle deserialization.
import sys
import types
import numpy as np

# Create shims for numpy._core so models saved under NumPy 2.x load cleanly in NumPy 1.x
try:
    import numpy.core as np_core
    import numpy.core.numeric as np_core_numeric

    if "numpy._core" not in sys.modules:
        core_mod = types.ModuleType("numpy._core")
        core_mod.numeric = np_core_numeric
        core_mod.multiarray = getattr(np_core, "multiarray", None)

        sys.modules["numpy._core"] = core_mod
        sys.modules["numpy._core.numeric"] = np_core_numeric
        if hasattr(np_core, "multiarray"):
            sys.modules["numpy._core.multiarray"] = np_core.multiarray
except Exception as e:
    print(f"[Warning] Could not set up NumPy core compatibility shim: {e}")


# ------------------------------------------------------------------
class MaskablePPOEvaluation(Training):
    """
    Evaluation class for loading and testing pre-trained MaskablePPO models.
    Inherits setup, path resolution, and LogisticsEnv initialization from Training.
    """
    name = "MaskablePPOEvaluation"
    C_RANDOM = True

    def train(self, total_timesteps: int = 0):
        # The base class requires train() to be implemented since it's abstract.
        # For evaluation, we override/use this to run the evaluation loop instead.
        pass

    def evaluate(self, model_path: str, num_episodes: int = 5):
        print(f"\n--- LOADING MODEL: {model_path} ---")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load the pre-trained models and attach it to the environment initialized by Training
        model = MaskablePPO.load(model_path, env=self.env, device="cuda")
        print("Model loaded successfully onto device: cuda\n")

        episode_rewards = []

        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            total_reward = 0.0

            print(f"Starting Evaluation Episode {ep + 1}/{num_episodes}")

            while not (done or truncated):
                # Retrieve action masks if supported by the environment wrapper
                action_masks = None
                if hasattr(self.env, "action_masks"):
                    action_masks = self.env.action_masks()
                elif hasattr(self.env.unwrapped, "action_masks"):
                    action_masks = self.env.unwrapped.action_masks()

                # Predict action using the loaded models with action masking & determinism enabled
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)
            print(f"Episode {ep + 1} Finished | Total Reward: {total_reward:.2f} | Total Distance: {info['Truck Distance']+info['Drone Distance']}\n")

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print("--- EVALUATION SUMMARY ---")
        print(f"Evaluated Episodes: {num_episodes}")
        print(f"Average Reward: {avg_reward:.2f}")


if __name__ == "__main__":
    # Current folder of this file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path from this script's directory as the default
    default_rel_model_path = os.path.join("..","models", "final_MaskablePPO_A_n32_k5_model.zip")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_rel_model_path,
        help="Path to the saved .zip models file (relative to this script or absolute)."
    )
    parser.add_argument("--episodes", type=int, default=1, help="No. of episodes you want to run the evaluation for.")
    parser.add_argument("--seed", type=int, default=42, help="The initialization seed")
    parser.add_argument("--numnodes", type=int, default=60, help="The initialization seed")
    args = parser.parse_args()

    # Resolve relative models path against this script's directory
    if not os.path.isabs(args.model_path):
        resolved_model_path = os.path.normpath(os.path.join(script_dir, args.model_path))
    else:
        resolved_model_path = os.path.normpath(args.model_path)

    seed = args.seed
    num_nodes = args.numnodes

    script_path = os.path.dirname(os.path.realpath(__file__))

    vrp_instance_path = os.path.join(
        script_path,
        # "..",
        # "..",
        "..",
        "ddls_src",
        "scenarios",
        "vrp_d_instances",
        "VRP-D",
        "A-n32-k5"
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

    evaluator = MaskablePPOEvaluation(
        save_models=False,
        config_path=None,
        save_episode_data=False,
        save_summary=False,
        save_metadata=False,
        sim_config=sim_config,
        instance_name=f"random_eval_{num_nodes}",
        eval=True
    )

    evaluator.evaluate(model_path=resolved_model_path, num_episodes=args.episodes)