import os
import argparse
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from rl_ext.training.base import Training


# ----------------------------------------------------------------------
# 1. Environment Harness
# ----------------------------------------------------------------------
class _EnvLoader(Training):
    def train(self, *args, **kwargs):
        pass


def make_env(vrp_instance_path: str, sim_config: dict, instance_name: str):
    loader = _EnvLoader(
        config_path=vrp_instance_path,
        sim_config=sim_config,
        instance_name=instance_name,
        save_models=False,
        save_episode_data=False,
    )
    return loader.env


# ----------------------------------------------------------------------
# 2. Native System Action Resolver & Distance Extractor
# ----------------------------------------------------------------------
class SystemActionResolver:
    """
    Interfaces directly with the environment's `agent_to_system_map` and
    `reverse_action_map` to accurately extract action semantics, node destinations,
    and road distances.
    """

    def __init__(self, env, depot_idx: int = 0):
        self.env = env
        self.depot_idx = depot_idx
        self.land_matrix = self._extract_land_matrix()

    def _ensure_numpy_matrix(self, mat: Any) -> Optional[np.ndarray]:
        if mat is None:
            return None
        if isinstance(mat, np.ndarray):
            return mat.astype(np.float32)
        if isinstance(mat, dict):
            for inner_key in ["matrix", "data", "distances", "distance_matrix", "values", "array"]:
                if inner_key in mat:
                    sub_val = mat[inner_key]
                    if isinstance(sub_val, np.ndarray):
                        return sub_val.astype(np.float32)
                    elif isinstance(sub_val, list):
                        return np.array(sub_val, dtype=np.float32)

            sample_val = next(iter(mat.values())) if len(mat) > 0 else None
            if isinstance(sample_val, dict):
                all_keys = set(mat.keys())
                for sub_dict in mat.values():
                    all_keys.update(sub_dict.keys())
                n = max(int(k) for k in all_keys) + 1 if all_keys else len(mat)
                arr = np.zeros((n, n), dtype=np.float32)
                for i, row in mat.items():
                    for j, dist in row.items():
                        arr[int(i), int(j)] = float(dist)
                return arr

            try:
                return np.array(list(mat.values()), dtype=np.float32)
            except Exception:
                pass

        if isinstance(mat, list):
            return np.array(mat, dtype=np.float32)
        return None

    def _extract_land_matrix(self) -> Optional[np.ndarray]:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        targets = [unwrapped]
        for attr in ["data_loader", "generator", "scenario", "config"]:
            if hasattr(unwrapped, attr):
                obj = getattr(unwrapped, attr)
                targets.append(obj)
                if hasattr(obj, "generator"):
                    targets.append(getattr(obj, "generator"))

        raw_land = None
        for t in targets:
            try:
                if raw_land is None and hasattr(t, "_system") and hasattr(t._system, "network"):
                    raw_land = getattr(t._system.network, "land_distance_matrix", None)
            except Exception:
                pass

        return self._ensure_numpy_matrix(raw_land)

    def get_system(self):
        return getattr(self.env.unwrapped, "_system", None)

    def get_active_vehicle_location(self) -> int:
        system = self.get_system()
        if system and hasattr(system, "global_state"):
            active_veh = getattr(system.global_state, "active_vehicle", None)
            if active_veh and hasattr(active_veh, "current_node_id"):
                return int(active_veh.current_node_id)
        return self.depot_idx

    def decode_agent_action(self, agent_action_idx: int) -> Dict[str, Any]:
        """
        Uses system.agent_to_system_map and system.reverse_action_map to inspect
        the actual ActionType and associated entity parameters.
        """
        system = self.get_system()
        if system is None:
            return {"action_type": "UNKNOWN", "target_node": agent_action_idx, "is_movement": True}

        # Step 1: Agent index -> System index
        agent_to_sys = getattr(system, "agent_to_system_map", None)
        if agent_to_sys is not None:
            if isinstance(agent_to_sys, dict):
                sys_idx = agent_to_sys.get(agent_action_idx, agent_action_idx)
            else:
                sys_idx = agent_to_sys[agent_action_idx]
        else:
            sys_idx = agent_action_idx

        # Step 2: System index -> Action blueprint tuple
        reverse_map = getattr(system, "reverse_action_map", None)
        if reverse_map is None:
            action_map = getattr(system, "action_map", None)
            if action_map and isinstance(action_map, dict):
                reverse_map = {v: k for k, v in action_map.items()}

        action_tuple = None
        if reverse_map is not None:
            if isinstance(reverse_map, dict):
                action_tuple = reverse_map.get(sys_idx, None)
            elif isinstance(reverse_map, list) and sys_idx < len(reverse_map):
                action_tuple = reverse_map[sys_idx]

        if not action_tuple or not isinstance(action_tuple, tuple):
            return {"action_type": "RAW", "target_node": agent_action_idx, "is_movement": True, "is_depot": False}

        action_type = action_tuple[0]
        action_name = getattr(action_type, "name", str(action_type))
        params = action_tuple[1:]

        target_node = None
        is_movement = False
        is_depot = False

        # Semantic resolution based on SimulationActions blueprints
        if action_name == "ASSIGN_ORDER_TO_RESOURCE" and len(params) > 0:
            pick_up_drop = params[0]
            if isinstance(pick_up_drop, (tuple, list)) and len(pick_up_drop) >= 2:
                # Target destination is the delivery drop node
                target_node = int(pick_up_drop[1])
                is_movement = True
                is_depot = (target_node == self.depot_idx)

        elif "TO_NODE" in action_name or "ROUTER" in action_name:
            for p in params:
                if isinstance(p, (int, np.integer)):
                    target_node = int(p)
                    is_movement = True
                    is_depot = (target_node == self.depot_idx)
                    break

        elif action_name in ["LOAD_TRUCK_ACTION", "UNLOAD_TRUCK_ACTION", "CONSOLIDATE_FOR_VEHICLE", "NO_OPERATION"]:
            # Administrative actions execute locally without travel
            is_movement = False
            target_node = None

        return {
            "action_name": action_name,
            "target_node": target_node,
            "is_movement": is_movement,
            "is_depot": is_depot,
        }


# ----------------------------------------------------------------------
# 3. System-Decoded Greedy Dispatcher
# ----------------------------------------------------------------------
class SystemDecodedGreedyDispatcher:
    """
    Chooses actions strictly permitted by env.action_masks():
    1. Executes immediate local administrative actions (loading, consolidate) if flagged valid.
    2. Prioritizes delivery movements over returning to the depot.
    3. Picks the movement destination that minimizes true network travel distance.
    """

    def __init__(self, resolver: SystemActionResolver, depot_idx: int = 0):
        self.resolver = resolver
        self.depot_idx = depot_idx
        self.land_matrix = resolver.land_matrix

    def select_action(self, mask: np.ndarray) -> int:
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            return 0
        if len(valid_actions) == 1:
            return int(valid_actions[0])

        current_loc = self.resolver.get_active_vehicle_location()

        admin_actions = []
        customer_movements = []
        depot_movements = []

        for a in valid_actions:
            meta = self.resolver.decode_agent_action(int(a))

            if not meta["is_movement"]:
                admin_actions.append(a)
            elif meta["is_depot"]:
                depot_movements.append((a, meta["target_node"]))
            else:
                customer_movements.append((a, meta["target_node"]))

        # Priority 1: Zero-cost administrative/load actions enabled by the environment
        if len(admin_actions) > 0:
            return int(admin_actions[0])

        # Priority 2: Customer drop-offs prioritized over returning to the depot
        if len(customer_movements) > 0:
            candidates = customer_movements
        elif len(depot_movements) > 0:
            candidates = depot_movements
        else:
            return int(valid_actions[0])

        # Priority 3: Nearest Neighbor selection using distance matrix
        best_action = candidates[0][0]
        min_dist = float("inf")

        for act_idx, target_node in candidates:
            if target_node is not None and self.land_matrix is not None:
                dist = float(self.land_matrix[current_loc, target_node])
            else:
                dist = 1.0

            if dist < min_dist:
                min_dist = dist
                best_action = act_idx

        return int(best_action)


# ----------------------------------------------------------------------
# 4. Benchmark Runner
# ----------------------------------------------------------------------
def evaluate_greedy(env, resolver: SystemActionResolver, num_episodes: int = 10, seed: int = 42):
    dispatcher = SystemDecodedGreedyDispatcher(resolver)

    print(f"\n=======================================================")
    print(f"--- RUNNING SYSTEM-DECODED GREEDY BENCHMARK ---")
    print(f"Episodes: {num_episodes} | Seed: {seed}")
    print(f"Distance Matrix: {'LOADED' if resolver.land_matrix is not None else 'DEFAULT'}")
    print(f"=======================================================\n")

    returns = []
    makespans = []
    step_counts = []
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        final_info = {}

        while not (done or truncated):
            mask = env.action_masks()
            action = dispatcher.select_action(mask)
            step_res = env.step(action)

            if len(step_res) == 5:
                obs, reward, done, truncated, info = step_res
            else:
                obs, reward, done, info = step_res
                truncated = False

            ep_reward += reward
            steps += 1
            final_info = info

        returns.append(ep_reward)
        step_counts.append(steps)
        makespan = final_info.get("makespan", final_info.get("terminal_logs", {}).get("makespan", 0.0))
        makespans.append(makespan)

        print(
            f"Episode {ep:02d}/{num_episodes:02d} | "
            f"Reward: {ep_reward:8.2f} | "
            f"Steps: {steps:3d} | "
            f"Makespan: {makespan:6.1f}"
        )

    duration = time.time() - start_time
    print(f"\n=======================================================")
    print(f"--- GREEDY DISPATCHER SUMMARY ---")
    print(f"Mean Reward   : {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Mean Makespan : {np.mean(makespans):.2f} +/- {np.std(makespans):.2f}")
    print(f"Mean Steps    : {np.mean(step_counts):.1f}")
    print(f"Elapsed Time  : {duration:.2f}s")
    print(f"=======================================================\n")


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))

    vrp_instance_path = os.path.join(
        script_path,
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

    parser = argparse.ArgumentParser(description="System-Decoded Greedy Baseline")
    parser.add_argument("--episodes", type=int, default=10, help="Number of benchmark episodes")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    args = parser.parse_args()

    env = make_env(vrp_instance_path, sim_config, instance_name)
    resolver = SystemActionResolver(env, depot_idx=0)

    evaluate_greedy(
        env=env,
        resolver=resolver,
        num_episodes=args.episodes,
        seed=args.seed,
    )