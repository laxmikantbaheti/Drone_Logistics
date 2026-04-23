from abc import ABC, abstractmethod
from ddls_src.core.logistics_system import LogisticsSystem


class BaseRewards(ABC):
    @abstractmethod
    def get_reward(self, system: LogisticsSystem) -> float:
        pass

    @abstractmethod
    def reset(self, system: LogisticsSystem):
        pass


class DefaultRewards(BaseRewards):
    def __init__(self, delivery_bonus: float = 100.0, step_penalty: float = -1.0):
        self.delivery_bonus = delivery_bonus
        self.step_penalty = step_penalty
        self._last_delivered_count = 0

    def reset(self, system: LogisticsSystem):
        # Count delivered orders at reset using class attributes
        orders = system.global_state.orders.values()
        self._last_delivered_count = sum(
            1 for o in orders if o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) == o.C_STATUS_DELIVERED
        )

    def get_reward(self, system: LogisticsSystem) -> float:
        # 1. Success and Failure checks
        if system.get_success():
            return 1000.0
        if system.get_broken():
            return -500.0

        # 2. Progress-based reward
        orders = system.global_state.orders.values()
        current_delivered = sum(
            1 for o in orders if o.get_state_value_by_dim_name(o.C_DIM_DELIVERY_STATUS[0]) == o.C_STATUS_DELIVERED
        )

        # Reward for new deliveries since the last macro-step
        reward = (current_delivered - self._last_delivered_count) * self.delivery_bonus
        self._last_delivered_count = current_delivered

        # Apply standard step penalty
        return reward + self.step_penalty