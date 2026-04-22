from abc import ABC, abstractmethod
import numpy as np
from ddls_src.core.logistics_system import LogisticsSystem


class BaseRewards(ABC):
    """Abstract base class for reward calculation variations."""

    @abstractmethod
    def get_reward(self, system: LogisticsSystem, action_idx: int) -> float:
        """Calculates the reward based on the transition result."""
        pass

    @abstractmethod
    def reset(self, system: LogisticsSystem):
        """Resets internal counters for a new episode."""
        pass


class DefaultRewards(BaseRewards):
    """
    Standard reward variant combining success bonuses, failure penalties,
    and progress-based incentives for deliveries.
    """

    def __init__(self,
                 success_reward: float = 1000.0,
                 failure_penalty: float = -500.0,
                 delivery_bonus: float = 100.0,
                 step_penalty: float = -1.0):
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.delivery_bonus = delivery_bonus
        self.step_penalty = step_penalty

        # Track delivered count to provide intermediate rewards
        self._last_delivered_count = 0

    def reset(self, system: LogisticsSystem):
        """Initializes the delivery counter based on the starting state."""
        orders = system.global_state.orders.values()
        self._last_delivered_count = sum(1 for o in orders if o.status == "Delivered")

    def get_reward(self, system: LogisticsSystem, action_idx: int) -> float:
        # 1. Check Terminal Conditions
        if system.get_success():
            return self.success_reward

        if system.get_broken():
            return self.failure_penalty

        reward = 0.0

        # 2. Calculate Progress Reward (New Deliveries)
        # We count orders with status "Delivered" in the GlobalState
        orders = system.global_state.orders.values()
        current_delivered = sum(1 for o in orders if o.status == "Delivered")

        if current_delivered > self._last_delivered_count:
            # Reward for each order delivered during this macro-step
            new_deliveries = current_delivered - self._last_delivered_count
            reward += new_deliveries * self.delivery_bonus
            self._last_delivered_count = current_delivered

        # 3. Add a constant step penalty to encourage speed
        reward += self.step_penalty

        return reward