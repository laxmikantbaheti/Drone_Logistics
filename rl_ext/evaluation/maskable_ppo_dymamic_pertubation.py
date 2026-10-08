import os
import argparse
from sb3_contrib import MaskablePPO
from rl_ext.training.base import Training


class MaskablePPOEvaluation(Training):
    """
    Evaluation class for loading and testing pre-trained MaskablePPO models.
    Inherits setup, path resolution, and LogisticsEnv initialization from Training.
    """
    name = "MaskablePPOEvaluation"

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
            print(f"Episode {ep + 1} Finished | Total Reward: {total_reward:.2f}\n")

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print("--- EVALUATION SUMMARY ---")
        print(f"Evaluated Episodes: {num_episodes}")
        print(f"Average Reward: {avg_reward:.2f}")


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

    base_instance_name = os.path.splitext(os.path.basename(vrp_instance_path))[0].replace("-", "_")
    instance_name = f"{base_instance_name}_evaluation"

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
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
                "seed": 100,  # Evaluation seed
                "demand_variance": 0.25
            }
        },
    }
    abs_model_path = "D:\\03_Development\\Drone_Logistics\\results\\MaskablePPODynamicInst_A_n33_k6_dynamic_20260927_203843\\final_MaskablePPODynamicInst_A_n33_k6_dynamic_model.zip"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=abs_model_path, help="Path to the saved .zip models file.")

    parser.add_argument("--episodes", type=int, default=1, help="No. of episodes you want to run the evalution for.")
    args = parser.parse_args()

    evaluator = MaskablePPOEvaluation(
        config_path=vrp_instance_path,
        save_models=False,
        save_episode_data=False,
        save_summary=False,
        save_metadata=False,
        sim_config=sim_config,
        instance_name=instance_name,
        eval = True
    )

    evaluator.evaluate(model_path=args.model_path, num_episodes=args.episodes)