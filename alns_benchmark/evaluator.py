import numpy as np
from alns_benchmark.state import PriorityPlanState


class EnvironmentEvaluator:
    def __init__(self, env):
        self.env = env

    def evaluate(self, state: PriorityPlanState, instance_path: str) -> float:
        """
        Executes a deterministic rollout of state on DDLSEnv.
        """
        # Reset with instance
        if hasattr(self.env, "reset"):
            obs, info = self.env.reset(options={"instance_file": instance_path})
        else:
            obs, info = self.env.reset()

        terminated = truncated = False
        total_cost = 0.0

        # Local queue to dispatch sequentially
        queue = list(state.order_priority)
        served_orders = set()

        while not (terminated or truncated):
            # 1. Retrieve dynamic action mask[cite: 1]
            if hasattr(self.env, "action_masks"):
                mask = self.env.action_masks()
            elif hasattr(self.env.unwrapped, "action_masker"):
                mask = self.env.unwrapped.action_masker.get_mask()[cite: 1]
            else:
                mask = info.get("action_mask", np.ones(self.env.action_space.n))

            legal_actions = np.where(mask == 1)[0]
            if len(legal_actions) == 0:
                break

            # 2. Determine current entity/phase from sequencer or info[cite: 1]
            active_vehicle = getattr(self.env.unwrapped, "current_vehicle", None)
            is_drone_phase = False
            if active_vehicle and hasattr(active_vehicle, "type"):
                is_drone_phase = "drone" in str(active_vehicle.type).lower()
            elif "active_entity" in info:
                is_drone_phase = ("drone" in info["active_entity"].lower() or
                                  "microhub" in info["active_entity"].lower())

            # 3. Match next order from priority list
            selected_action = None
            for order_id in queue:
                # If currently filling microhub drone, skip truck-reserved orders
                if is_drone_phase and (order_id in state.truck_only_orders):
                    continue

                if order_id in legal_actions:
                    selected_action = order_id
                    queue.remove(order_id)
                    served_orders.add(order_id)
                    break

            # 4. Fallback when no customer order is feasible (e.g. capacity full,
            #    delivering pseudo-order to microhub, or finishing route)
            if selected_action is None:
                # Action 0 is typically Depot/End-of-phase
                if 0 in legal_actions:
                    selected_action = 0
                else:
                    selected_action = legal_actions[0]

            # 5. Advance discrete-event simulation[cite: 1]
            obs, reward, terminated, truncated, info = self.env.step(selected_action)
            step_cost = info.get("operational_cost", info.get("cost", -reward))
            total_cost += step_cost

        # Penalty for unserved orders (maintains feasibility boundary)
        unserved = len(state.order_priority) - len(served_orders)
        if unserved > 0:
            total_cost += unserved * 10000.0

        state._cost = total_cost
        state.metrics = {
            "served": len(served_orders),
            "unserved": unserved,
            "makespan": info.get("makespan", 0.0),
            "total_distance": info.get("total_distance", 0.0)
        }
        return total_cost