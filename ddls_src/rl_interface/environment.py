import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# Import your core framework classes
from ddls_src.core.logistics_system import LogisticsSystem
from ddls_src.core.basics import LogisticsAction
from ddls_src.rl_interface.state_extractor import StateExtractor


class LogisticsEnv(gym.Env):
    """
    Standard Gymnasium Environment wrapping the DDLS LogisticsSystem.
    Implements a Semi-Markov Decision Process (SMDP) with frozen-time cascading.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, config=None, p_visualize=False, p_logging=False, custom_log=False):
        super().__init__()

        self.config = config if config is not None else {}
        self.visualize = p_visualize

        # 1. Instantiate the Logistics Framework
        self.sim = LogisticsSystem(
            p_id='logsys_001',
            p_visualize=p_visualize,
            p_logging=p_logging,
            config=self.config,
            custom_log=custom_log
        )

        # 2. Setup the State Extractor
        self.state_extractor = StateExtractor(max_vehicles=11, max_order_slots=20)

        # 3. Define the Observation Space
        self.observation_space = spaces.Box(
            low=-1.0,
            high=np.inf,
            shape=(self.state_extractor.obs_size,),
            dtype=np.float32
        )

        # 4. Define the Action Space
        num_agent_actions = len(self.sim.agent_to_system_map)
        self.action_space = spaces.Discrete(num_agent_actions)

        # Identify the NO_OP index
        system_no_op_idx = list(self.sim.action_map.values())[-1]
        self.agent_no_op_idx = self.sim.agent_to_system_map.index(system_no_op_idx)

        self.current_step = 0
        self.max_steps = 10000

        # Episode Tracking Variables
        self.episode_counter = 0
        self.episode_start_time = time.time()

    def _get_patched_mask(self) -> np.ndarray:
        """
        Retrieves the mask from the simulation. If all actions are masked (meaning
        there are no valid assignments right now), it forcefully unmasks the
        NO_OPERATION action so the agent can legally advance time.
        """
        agent_mask = self.sim.get_agent_mask()
        if not np.any(agent_mask):
            agent_mask[self.agent_no_op_idx] = True
        return agent_mask

    def reset(self, seed=None, options=None):
        """
        Resets the simulation and prepares the first valid observation.
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Start the real-time stopwatch
        self.episode_counter += 1
        self.episode_start_time = time.time()

        # Reset the underlying physics engine
        self.sim.reset(p_seed=seed if seed else 42)

        if self.visualize:
            self.sim.network.setup_visualization()
            self.sim.network.update_plot()

        # VERY IMPORTANT: Settle the system before the agent takes its first look
        self.sim.run_automatic_action_loop()

        obs = self.state_extractor.extract_observation(self.sim.global_state)
        return obs, self._get_info()

    def step(self, action: int):
        """
        The core RL loop. Time only advances if the agent chooses NO_OP.
        """
        self.current_step += 1
        reward = 0.0  # Sparse reward: 0 until episode ends

        # 1. Guard against invalid actions (using the patched mask!)
        agent_mask = self._get_patched_mask()
        if not agent_mask[action]:
            obs = self.state_extractor.extract_observation(self.sim.global_state)
            return obs, -100000.0, False, False, self._get_info()

            # 2. The Agent Yields (Advance Time)
        if action == self.agent_no_op_idx:
            self.sim.advance_time()
            self.sim.run_automatic_action_loop()

        # 3. The Agent Acts (Frozen Time Cascade)
        else:
            sys_action_id = self.sim.agent_to_system_map[action]
            action_obj = LogisticsAction(p_action_space=self.sim._action_space, p_values=[sys_action_id])
            self.sim.process_action(action_obj)
            self.sim.run_automatic_action_loop()

        if self.visualize:
            self.sim.network.update_plot()

        # 4. Extract Next State
        obs = self.state_extractor.extract_observation(self.sim.global_state)

        # 5. Check Terminations (Relying strictly on your simulator physics now)
        is_success = self.sim.get_success()
        is_broken = self.sim.get_broken()

        terminated = is_success or is_broken
        truncated = self.current_step >= self.max_steps

        # 6. Episode Termination & Logging
        if terminated or truncated:
            # Stop the real-time clock
            real_time_taken = time.time() - self.episode_start_time

            # Calculate the Simulation Makespan
            current_time = getattr(self.sim.global_state, 'current_time', 0.0)
            initial_time = getattr(self.sim.global_state, 'initial_time', 0.0)
            makespan = current_time - initial_time

            # Apply Rewards and Print Telemetry
            if is_success:
                reward = -float(makespan)
                print(
                    f"\n✅ Episode {self.episode_counter} SUCCESS | Makespan: {makespan:.2f} units | Real Time: {real_time_taken:.3f} sec")
                self._handle_success_reports()
            else:
                reward = -100000.0
                fail_reason = "Broken/Timeout"
                print(
                    f"\n❌ Episode {self.episode_counter} FAILED ({fail_reason}) | Makespan at failure: {makespan:.2f} | Real Time: {real_time_taken:.3f} sec")

        return obs, reward, terminated, truncated, self._get_info()

    def _get_info(self):
        """
        Passes the action mask directly to the RL Algorithm.
        """
        # Ensure the RL agent sees the NO_OP option when everything else is blocked
        agent_mask = self._get_patched_mask()
        return {
            "action_mask": np.array(agent_mask, dtype=np.int8)
        }

    def _handle_success_reports(self):
        """
        Generates Gantt charts and event logs upon success.
        """
        if hasattr(self.sim.global_state, 'event_logger'):
            self.sim.global_state.event_logger.export_reports(base_filepath='scenario_report')

            try:
                from ddls_src.functions.plotting import SimulationPlotter
                plotter = SimulationPlotter(base_filepath='scenario_report')
                plotter.generate_plot('cargo_gantt', save_to_disk=False)
                plotter.generate_plot('state_timeline', save_to_disk=False)
            except ImportError:
                pass