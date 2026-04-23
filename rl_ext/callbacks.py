import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Core Project Imports
from ddls_src.functions.reports import export_simulation_reports


class SaveOnEpisodeCallback(BaseCallback):
    def __init__(self, trainer_instance, verbose: int = 0):
        super(SaveOnEpisodeCallback, self).__init__(verbose)
        self.trainer = trainer_instance
        self.episode_count = 0
        self.reward_history = []

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.episode_count += 1
            env = self.training_env.envs[0].unwrapped
            system = env._system
            logger = system.global_state.event_logger

            info = self.locals["infos"][0]
            reward = float(info["reward"])
            makespan = float(info["makespan"])
            self.reward_history.append(reward)

            # Record to TensorBoard
            self.logger.record("results/reward", reward)
            self.logger.record("results/makespan", makespan)
            self.logger.dump(step=self.num_timesteps)

            # Export reports to episode subfolder
            report_name = f"{self.trainer.name}_ep_{self.episode_count}"
            report_path = os.path.join(self.trainer.episodes_dir, report_name)
            logger.export_reports(base_filepath=report_path)

            # --- MODIFICATION: Determine the Termination Status (Plain Text) ---
            if info.get("is_success"):
                status_text = "SUCCESS"
            elif info.get("is_broken"):
                status_text = "BROKEN"
            else:
                status_text = "FINISHED"

            makespan = info.get("current_time", 0.0)
            delivered = info.get("delivered_count", 0)

            ep_kpis = {
                "episode": self.episode_count,
                "reward": reward,
                "makespan": makespan,
                "status": status_text,
                "delivered": delivered
            }

            # --- MODIFICATION: Plain Text Console Print Card ---
            print(f"\n" + "-" * 50, flush=True)
            print(f"   EPISODE {self.episode_count} COMPLETED", flush=True)
            print(f"   RESULT:    {status_text}", flush=True)
            print(f"   REWARD:    {reward}", flush=True)
            print(f"   MAKESPAN:  {makespan}", flush=True)
            print(f"   DELIVERED: {delivered} items", flush=True)
            print("-" * 50 + "\n", flush=True)

            self.trainer.all_episodes_kpis.append(ep_kpis)
            self.trainer.log_episode(ep_kpis)

            if self.trainer.save_summary and self.episode_count % 10 == 0:
                self.trainer.log_summary({
                    "step": self.num_timesteps,
                    "avg_reward_10ep": np.mean(self.reward_history[-10:])
                })

        return True