import os
import numpy as np
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

# Import your custom Gymnasium environment
from ddls_src.rl_interface_old2.environment import LogisticsEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Extracts the action mask from the environment to prevent illegal moves.
    """
    mask = env.unwrapped.sim.get_agent_mask()
    return np.array(mask, dtype=bool)


class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Custom callback that stops training when a specified number of episodes is reached.
    """

    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.n_episodes = 0

    def _on_step(self) -> bool:
        # 'dones' is a boolean array provided by SB3 indicating if the episode just finished
        if self.locals.get("dones") is not None:
            self.n_episodes += sum(self.locals["dones"])

            if self.n_episodes >= self.max_episodes:
                if self.verbose > 0:
                    print(f"\n--- Target of {self.max_episodes} episodes reached! Stopping training. ---")
                return False  # Returning False instantly stops the model.learn() loop
        return True


def main():
    print("======================================================")
    print("=== Initiating DDLS Reinforcement Learning Agent   ===")
    print("======================================================")

    os.makedirs("models", exist_ok=True)

    # 1. Define the simulation configuration EXACTLY as your runner script does
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_path, '..', 'config', 'large_instance.json')
    config_file_path = os.path.normpath(config_file_path)

    sim_config = {
        "movement_mode": "matrix",
        "initial_time": 0.0,
        "main_timestep_duration": 1.0,
        "data_loader_config": {
            "generator_type": "json_file",
            "generator_config": {"file_path": config_file_path}
        }
    }

    # 2. Instantiate the Base Environment
    print("Loading Logistics Environment and JSON Matrices...")
    raw_env = LogisticsEnv(
        config=sim_config,
        p_visualize=False,
        p_logging=False,
        custom_log=False  # Set to false to avoid console spam during high-speed training
    )

    # 3. Wrap the Environment with SB3's ActionMasker
    env = ActionMasker(raw_env, mask_fn)

    print("Building Maskable PPO Neural Network...")

    # 4. Initialize the Algorithm
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99,
        seed=42
    )

    # 5. Setup Callbacks
    # Saves a backup every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='ddls_agent'
    )

    # Stops training after exactly 500 complete scenarios
    episode_callback = StopTrainingOnMaxEpisodes(max_episodes=500, verbose=1)

    # Combine callbacks into a list
    callback_list = CallbackList([checkpoint_callback, episode_callback])

    print("Starting Training Pipeline! (Press Ctrl+C to stop at any time)")

    # 6. Train the Agent
    # We set total_timesteps to a massive number so the episode callback becomes the true limit
    model.learn(
        total_timesteps=int(1e10),
        callback=callback_list
    )

    # 7. Save the final trained model
    model.save("models/ddls_logistics_agent_final")
    print("\nTraining complete! Final model saved to /models.")


if __name__ == "__main__":
    main()