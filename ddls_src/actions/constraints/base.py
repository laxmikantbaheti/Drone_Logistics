import numpy as np
from typing import Dict, Any, List, Tuple, Callable

# Import the SimulationAction enum
from ddls_src.actions.action_enums import SimulationAction


# Forward declaration for LogisticsSimulation to avoid circular dependency
# This will be replaced by an actual import once LogisticsSimulation is fully defined
class LogisticsSimulation:
    pass


# NO_OPERATION is now part of the SimulationAction enum
NO_OPERATION = SimulationAction.NO_OPERATION


class ActionMasker:
    """
    Generates a boolean mask for the action space based on the current simulation state
    and predefined constraints. This class ensures that the RL agent only selects
    valid actions.
    """

    def __init__(self, simulation_instance: 'LogisticsSimulation', action_space_size: int,
                 action_map: Dict[Tuple, int]):
        """
        Initializes the ActionMasker with a reference to the main simulation,
        the total size of the flattened action space, and the action mapping.

        Args:
            simulation_instance (LogisticsSimulation): Reference to the main simulation instance.
                                                       Used to query GlobalState and other managers.
            action_space_size (int): The total number of possible flattened actions.
            action_map (Dict[Tuple, int]): A mapping from action tuple (e.g., (SimulationAction.TRUCK_TO_NODE, 1, 2))
                                           to its flattened integer index.
        """
        self.simulation_instance = simulation_instance
        self.action_space_size = action_space_size
        self.action_map = action_map
        # A list of callable functions, each representing a specific constraint rule.
        # Each function takes the LogisticsSimulation instance and returns a boolean numpy array
        # of the same shape as the action space, where True means valid and False means invalid.
        self.constraint_rules: List[Callable[['LogisticsSimulation'], np.ndarray]] = []

        # Reverse action map for efficient lookup of action tuples by index
        self._reverse_action_map: Dict[int, Tuple] = {idx: act_tuple for act_tuple, idx in action_map.items()}

        # Register all constraint rule methods during initialization
        self._register_all_default_constraints()

        print("ActionMasker initialized.")

    def _register_all_default_constraints(self) -> None:
        """
        Registers all default constraint rule methods.
        This method should be updated as new constraint rules are implemented.
        """
        self._register_constraint(self._mask_order_actions)
        self._register_constraint(self._mask_assign_actions)
        self._register_constraint(self._mask_consolidate_actions)
        self._register_constraint(self._mask_reassign_actions)  # New: Reassign action masking

        self._register_constraint(self._mask_truck_movement_actions)
        self._register_constraint(self._mask_truck_load_unload_actions)

        self._register_constraint(self._mask_drone_launch_actions)
        self._register_constraint(self._mask_drone_load_unload_actions)
        self._register_constraint(self._mask_drone_landing_actions)
        self._register_constraint(self._mask_drone_charging_actions)
        self._register_constraint(
            self._mask_drone_to_charging_station_actions)  # New: Drone to charging station masking

        self._register_constraint(self._mask_micro_hub_actions)

        self._register_constraint(self._mask_maintenance_actions)

        self._register_constraint(self._mask_no_operation_action)

    def _register_constraint(self, constraint_func: Callable[['LogisticsSimulation'], np.ndarray]) -> None:
        """
        Adds a new constraint rule function to the list of active constraints.

        Args:
            constraint_func (Callable[[LogisticsSimulation], np.ndarray]): A function that takes
                                                                          the LogisticsSimulation instance
                                                                          and returns a boolean numpy array
                                                                          representing the mask for that rule.
        """
        self.constraint_rules.append(constraint_func)

    def _get_action_index(self, action_tuple: Tuple) -> int:
        """Helper to get the flattened index for an action tuple."""
        return self.action_map.get(action_tuple)

    def _set_action_validity(self, action_tuple: Tuple, is_valid: bool, current_mask: np.ndarray) -> None:
        """Helper to set a specific action's validity in the mask."""
        flattened_index = self._get_action_index(action_tuple)
        if flattened_index is not None and 0 <= flattened_index < self.action_space_size:
            current_mask[flattened_index] = is_valid

    def generate_mask(self) -> np.ndarray:
        """
        Computes and returns the composite boolean mask by applying all registered
        constraint rules. The final mask is the logical AND of all individual rule masks.

        Returns:
            np.ndarray: A boolean numpy array of shape (action_space_size,),
                        where True indicates a valid action and False an invalid one.
        """
        # Initialize a mask where all actions are initially considered valid
        composite_mask = np.ones(self.action_space_size, dtype=bool)

        # Apply each registered constraint rule
        for constraint_func in self.constraint_rules:
            rule_mask = constraint_func(self.simulation_instance)
            # Ensure rule_mask has the correct shape
            if rule_mask.shape != composite_mask.shape:
                raise ValueError(f"Constraint rule {constraint_func.__name__} returned a mask "
                                 f"of shape {rule_mask.shape}, expected {composite_mask.shape}.")
            composite_mask = np.logical_and(composite_mask, rule_mask)

        return composite_mask

    # --- Concrete Constraint Rule Functions (Implemented) ---

    def _mask_order_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks PRIORITIZE_ORDER, CANCEL_ORDER, FLAG_FOR_RE_DELIVERY if order_ID is invalid
        or not in expected status. ACCEPT_ORDER is valid for pending orders.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        for order_id, order in global_state.orders.items():
            # ACCEPT_ORDER: Only valid for 'pending' orders
            if order.status != "pending":
                self._set_action_validity((SimulationAction.ACCEPT_ORDER, order_id), False, mask)

            # PRIORITIZE_ORDER, CANCEL_ORDER, FLAG_FOR_RE_DELIVERY, REASSIGN_ORDER:
            # Not valid for 'delivered' or 'cancelled' orders
            if order.status in ["delivered", "cancelled"]:
                self._set_action_validity((SimulationAction.PRIORITIZE_ORDER, order_id, Any), False,
                                          mask)  # Mask all priorities
                self._set_action_validity((SimulationAction.CANCEL_ORDER, order_id), False, mask)
                self._set_action_validity((SimulationAction.FLAG_FOR_RE_DELIVERY, order_id), False, mask)

        # Mask actions for non-existent orders
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] in [SimulationAction.ACCEPT_ORDER, SimulationAction.PRIORITIZE_ORDER,
                                   SimulationAction.CANCEL_ORDER, SimulationAction.FLAG_FOR_RE_DELIVERY]:
                order_id = action_tuple[1]
                if order_id not in global_state.orders:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_assign_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks ASSIGN_ORDER_TO_TRUCK, ASSIGN_ORDER_TO_DRONE, ASSIGN_ORDER_TO_MICRO_HUB if
        vehicle/hub is unavailable, full, or order is already assigned/invalid status.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        for order_id, order in global_state.orders.items():
            if order.status not in ["pending", "accepted", "flagged_re_delivery"]:
                # Mask all assignment actions for orders not in assignable status
                for truck_id in global_state.trucks.keys():
                    self._set_action_validity((SimulationAction.ASSIGN_ORDER_TO_TRUCK, order_id, truck_id), False, mask)
                for drone_id in global_state.drones.keys():
                    self._set_action_validity((SimulationAction.ASSIGN_ORDER_TO_DRONE, order_id, drone_id), False, mask)
                for micro_hub_id in global_state.micro_hubs.keys():
                    self._set_action_validity((SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB, order_id, micro_hub_id),
                                              False, mask)
                continue  # No need to check vehicle availability if order status is wrong

            # Check vehicle/hub availability and capacity for assignment
            for truck_id, truck in global_state.trucks.items():
                if truck.status not in ["idle", "loading", "unloading"] or len(
                        truck.cargo_manifest) >= truck.max_payload_capacity:
                    self._set_action_validity((SimulationAction.ASSIGN_ORDER_TO_TRUCK, order_id, truck_id), False, mask)

            for drone_id, drone in global_state.drones.items():
                if drone.status not in ["idle", "loading", "unloading"] or len(
                        drone.cargo_manifest) >= drone.max_payload_capacity:
                    self._set_action_validity((SimulationAction.ASSIGN_ORDER_TO_DRONE, order_id, drone_id), False, mask)
                # Additional drone-specific check: battery level for assignment (e.g., must be above 20%)
                # if drone.battery_level < 0.2: # Example threshold
                #     self._set_action_validity((SimulationAction.ASSIGN_ORDER_TO_DRONE, order_id, drone_id), False, mask)

            for micro_hub_id, micro_hub in global_state.micro_hubs.items():
                if micro_hub.operational_status != "active" or micro_hub.is_package_transfer_unavailable:
                    self._set_action_validity((SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB, order_id, micro_hub_id),
                                              False, mask)

        # Mask actions for non-existent orders/vehicles/hubs
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] in [SimulationAction.ASSIGN_ORDER_TO_TRUCK, SimulationAction.ASSIGN_ORDER_TO_DRONE,
                                   SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB]:
                order_id = action_tuple[1]
                target_id = action_tuple[2]
                if order_id not in global_state.orders:
                    self._set_action_validity(action_tuple, False, mask)
                if action_tuple[0] == SimulationAction.ASSIGN_ORDER_TO_TRUCK and target_id not in global_state.trucks:
                    self._set_action_validity(action_tuple, False, mask)
                elif action_tuple[0] == SimulationAction.ASSIGN_ORDER_TO_DRONE and target_id not in global_state.drones:
                    self._set_action_validity(action_tuple, False, mask)
                elif action_tuple[
                    0] == SimulationAction.ASSIGN_ORDER_TO_MICRO_HUB and target_id not in global_state.micro_hubs:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_consolidate_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks CONSOLIDATE_FOR_TRUCK/DRONE if vehicle is not at a consolidation point
        (e.g., depot/micro-hub) or no eligible orders are present at that node, or vehicle is full.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        # Truck consolidation
        for truck_id, truck in global_state.trucks.items():
            if truck.current_node_id is None:  # Must be at a node
                self._set_action_validity((SimulationAction.CONSOLIDATE_FOR_TRUCK, truck_id), False, mask)
                continue

            current_node = global_state.get_entity("node", truck.current_node_id)
            # Must be a loadable node and have packages
            if not current_node.is_loadable or not current_node.packages_held:
                self._set_action_validity((SimulationAction.CONSOLIDATE_FOR_TRUCK, truck_id), False, mask)
                continue

            # Truck must not be full
            if len(truck.cargo_manifest) >= truck.max_payload_capacity:
                self._set_action_validity((SimulationAction.CONSOLIDATE_FOR_TRUCK, truck_id), False, mask)
                continue

            # Check if there are any eligible orders at the node
            eligible_orders_at_node = [
                o_id for o_id in current_node.packages_held
                if o_id in global_state.orders and global_state.orders[o_id].status in ["pending", "accepted",
                                                                                        "flagged_re_delivery"]
                   and global_state.orders[o_id].assigned_vehicle_id is None
            ]
            if not eligible_orders_at_node:
                self._set_action_validity((SimulationAction.CONSOLIDATE_FOR_TRUCK, truck_id), False, mask)

        # Drone consolidation
        for drone_id, drone in global_state.drones.items():
            if drone.current_node_id is None:  # Must be at a node
                self._set_action_validity((SimulationAction.CONSOLIDATE_FOR_DRONE, drone_id), False, mask)
                continue

            current_node = global_state.get_entity("node", drone.current_node_id)
            # Must be a loadable node and have packages
            if not current_node.is_loadable or not current_node.packages_held:
                self._set_action_validity((SimulationAction.CONSOLIDATE_FOR_DRONE, drone_id), False, mask)
                continue

            # Drone must not be full
            if len(drone.cargo_manifest) >= drone.max_payload_capacity:
                self._set_action_validity((SimulationAction.CONSOLIDATE_FOR_DRONE, drone_id), False, mask)
                continue

            # Check if there are any eligible orders at the node
            eligible_orders_at_node = [
                o_id for o_id in current_node.packages_held
                if o_id in global_state.orders and global_state.orders[o_id].status in ["pending", "accepted",
                                                                                        "flagged_re_delivery"]
                   and global_state.orders[o_id].assigned_vehicle_id is None
            ]
            if not eligible_orders_at_node:
                self._set_action_validity((SimulationAction.CONSOLIDATE_FOR_DRONE, drone_id), False, mask)

        # Mask actions for non-existent vehicles
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] == SimulationAction.CONSOLIDATE_FOR_TRUCK and action_tuple[1] not in global_state.trucks:
                self._set_action_validity(action_tuple, False, mask)
            elif action_tuple[0] == SimulationAction.CONSOLIDATE_FOR_DRONE and action_tuple[
                1] not in global_state.drones:
                self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_reassign_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks REASSIGN_ORDER if order is invalid, new vehicle is unavailable/full,
        or order is not in a state that allows reassignment.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] == SimulationAction.REASSIGN_ORDER:
                order_id = action_tuple[1]
                new_vehicle_id = action_tuple[2]

                if order_id not in global_state.orders:
                    self._set_action_validity(action_tuple, False, mask)
                    continue
                order = global_state.orders[order_id]

                # Order must not be delivered or cancelled
                if order.status in ["delivered", "cancelled"]:
                    self._set_action_validity(action_tuple, False, mask)
                    continue

                # New vehicle must exist
                new_vehicle = None
                if new_vehicle_id in global_state.trucks:
                    new_vehicle = global_state.trucks[new_vehicle_id]
                elif new_vehicle_id in global_state.drones:
                    new_vehicle = global_state.drones[new_vehicle_id]

                if new_vehicle is None:
                    self._set_action_validity(action_tuple, False, mask)
                    continue

                # New vehicle must be available and not full
                if new_vehicle.status not in ["idle", "loading", "unloading"] or \
                        len(new_vehicle.cargo_manifest) >= new_vehicle.max_payload_capacity:
                    self._set_action_validity(action_tuple, False, mask)
                    continue

                # If order is currently in transit with another vehicle, that vehicle must also be able to transfer
                # (This is a complex rule, might be simplified for MVP)
                # For now, we assume reassignment can happen if the new vehicle is ready.
        return mask

    def _mask_truck_movement_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks TRUCK_TO_NODE, RE_ROUTE_TRUCK_TO_NODE if truck is busy,
        under maintenance, broken down, or destination is invalid/unreachable.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()
        network = sim.network  # Access the Network graph

        for truck_id, truck in global_state.trucks.items():
            # Truck must not be busy, under maintenance, or broken down
            if truck.status in ["en_route", "loading", "unloading", "maintenance", "broken_down"]:
                for dest_node_id in global_state.nodes.keys():  # Mask all destinations for this truck
                    self._set_action_validity((SimulationAction.TRUCK_TO_NODE, truck_id, dest_node_id), False, mask)
                    self._set_action_validity((SimulationAction.RE_ROUTE_TRUCK_TO_NODE, truck_id, dest_node_id), False,
                                              mask)
                continue  # No need to check destinations if truck is unavailable

            # Check reachability for each destination node
            for dest_node_id in global_state.nodes.keys():
                # Determine start node for pathfinding
                start_node_for_path = truck.current_node_id
                if start_node_for_path is None:  # If mid-segment, path from next node
                    if truck.current_route and len(truck.current_route) > 1:
                        start_node_for_path = truck.current_route[1]
                    else:  # Truck is not at a node and has no valid route
                        self._set_action_validity((SimulationAction.TRUCK_TO_NODE, truck_id, dest_node_id), False, mask)
                        self._set_action_validity((SimulationAction.RE_ROUTE_TRUCK_TO_NODE, truck_id, dest_node_id),
                                                  False, mask)
                        continue

                # If start or end node does not exist, mask
                if start_node_for_path not in global_state.nodes or dest_node_id not in global_state.nodes:
                    self._set_action_validity((SimulationAction.TRUCK_TO_NODE, truck_id, dest_node_id), False, mask)
                    self._set_action_validity((SimulationAction.RE_ROUTE_TRUCK_TO_NODE, truck_id, dest_node_id), False,
                                              mask)
                    continue

                # Check if path exists using Network's calculate_shortest_path
                path_exists = bool(network.calculate_shortest_path(start_node_for_path, dest_node_id, 'truck'))
                if not path_exists:
                    self._set_action_validity((SimulationAction.TRUCK_TO_NODE, truck_id, dest_node_id), False, mask)
                    self._set_action_validity((SimulationAction.RE_ROUTE_TRUCK_TO_NODE, truck_id, dest_node_id), False,
                                              mask)

        # Mask actions for non-existent trucks or nodes
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] in [SimulationAction.TRUCK_TO_NODE, SimulationAction.RE_ROUTE_TRUCK_TO_NODE]:
                truck_id = action_tuple[1]
                dest_node_id = action_tuple[2]
                if truck_id not in global_state.trucks or dest_node_id not in global_state.nodes:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_truck_load_unload_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks LOAD_TRUCK_ACTION, UNLOAD_TRUCK_ACTION if truck not at valid load/unload node,
        or no cargo to unload/no space to load.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        for truck_id, truck in global_state.trucks.items():
            # Truck must be at a node and available for load/unload operations
            if truck.current_node_id is None or truck.status not in ["idle", "loading", "unloading"]:
                for order_id in global_state.orders.keys():  # Mask all orders for this truck
                    self._set_action_validity((SimulationAction.LOAD_TRUCK_ACTION, truck_id, order_id), False, mask)
                    self._set_action_validity((SimulationAction.UNLOAD_TRUCK_ACTION, truck_id, order_id), False, mask)
                continue

            current_node = global_state.get_entity("node", truck.current_node_id)

            for order_id, order in global_state.orders.items():
                # LOAD_TRUCK_ACTION
                if not current_node.is_loadable or \
                        order_id not in current_node.packages_held or \
                        len(truck.cargo_manifest) >= truck.max_payload_capacity or \
                        order.status in ["in_transit", "delivered",
                                         "cancelled"]:  # Order already on vehicle or delivered/cancelled
                    self._set_action_validity((SimulationAction.LOAD_TRUCK_ACTION, truck_id, order_id), False, mask)

                # UNLOAD_TRUCK_ACTION
                if not current_node.is_unloadable or \
                        order_id not in truck.cargo_manifest:  # Order not in truck's cargo
                    self._set_action_validity((SimulationAction.UNLOAD_TRUCK_ACTION, truck_id, order_id), False, mask)

        # Mask actions for non-existent trucks/orders
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] in [SimulationAction.LOAD_TRUCK_ACTION, SimulationAction.UNLOAD_TRUCK_ACTION]:
                truck_id = action_tuple[1]
                order_id = action_tuple[2]
                if truck_id not in global_state.trucks or order_id not in global_state.orders:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_drone_launch_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks LAUNCH_DRONE if drone not at base, base inactive/blocked,
        drone battery low, or no order assigned/order not in cargo.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()
        network = sim.network

        for drone_id, drone in global_state.drones.items():
            # Drone must be at a node and available
            if drone.current_node_id is None or drone.status not in ["idle", "loading", "unloading", "charging"]:
                for order_id in global_state.orders.keys():  # Mask all orders for this drone
                    self._set_action_validity((SimulationAction.LAUNCH_DRONE, drone_id, order_id), False, mask)
                continue

            # Check if drone has cargo to deliver
            if not drone.cargo_manifest:
                for order_id in global_state.orders.keys():
                    self._set_action_validity((SimulationAction.LAUNCH_DRONE, drone_id, order_id), False, mask)
                continue

            current_node = global_state.get_entity("node", drone.current_node_id)

            # Check if launch node is valid (e.g., depot or active micro-hub)
            if current_node.type == 'micro_hub':
                micro_hub = global_state.get_entity("micro_hub", current_node.id)
                if micro_hub.operational_status != "active" or micro_hub.is_blocked_for_launches:
                    for order_id in global_state.orders.keys():
                        self._set_action_validity((SimulationAction.LAUNCH_DRONE, drone_id, order_id), False, mask)
                    continue

            # Check drone battery level for flight
            # Assuming a minimum battery percentage (e.g., 20%)
            if drone.battery_level < 0.2:  # This threshold should be configurable
                for order_id in global_state.orders.keys():
                    self._set_action_validity((SimulationAction.LAUNCH_DRONE, drone_id, order_id), False, mask)
                continue

            # Check if the order to be launched is actually in the drone's cargo and has a valid customer node
            for order_id, order in global_state.orders.items():
                if order_id not in drone.cargo_manifest or \
                        order.customer_node_id not in global_state.nodes or \
                        not bool(
                            network.calculate_shortest_path(drone.current_node_id, order.customer_node_id, 'drone')):
                    self._set_action_validity((SimulationAction.LAUNCH_DRONE, drone_id, order_id), False, mask)

        # Mask actions for non-existent drones/orders
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] == SimulationAction.LAUNCH_DRONE:
                drone_id = action_tuple[1]
                order_id = action_tuple[2]
                if drone_id not in global_state.drones or order_id not in global_state.orders:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_drone_load_unload_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks DRONE_LOAD_ACTION, DRONE_UNLOAD_ACTION if drone not at valid load/unload point,
        or no cargo to unload/no space to load.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        for drone_id, drone in global_state.drones.items():
            # Drone must be at a node and available for load/unload operations
            if drone.current_node_id is None or drone.status not in ["idle", "loading", "unloading", "charging"]:
                for order_id in global_state.orders.keys():
                    self._set_action_validity((SimulationAction.DRONE_LOAD_ACTION, drone_id, order_id), False, mask)
                    self._set_action_validity((SimulationAction.DRONE_UNLOAD_ACTION, drone_id, order_id), False, mask)
                continue

            current_node = global_state.get_entity("node", drone.current_node_id)

            for order_id, order in global_state.orders.items():
                # DRONE_LOAD_ACTION
                if not current_node.is_loadable or \
                        order_id not in current_node.packages_held or \
                        len(drone.cargo_manifest) >= drone.max_payload_capacity or \
                        order.status in ["in_transit", "delivered", "cancelled"]:
                    self._set_action_validity((SimulationAction.DRONE_LOAD_ACTION, drone_id, order_id), False, mask)

                # DRONE_UNLOAD_ACTION
                if not current_node.is_unloadable or \
                        order_id not in drone.cargo_manifest:
                    self._set_action_validity((SimulationAction.DRONE_UNLOAD_ACTION, drone_id, order_id), False, mask)

        # Mask actions for non-existent drones/orders
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] in [SimulationAction.DRONE_LOAD_ACTION, SimulationAction.DRONE_UNLOAD_ACTION]:
                drone_id = action_tuple[1]
                order_id = action_tuple[2]
                if drone_id not in global_state.drones or order_id not in global_state.orders:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_drone_landing_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks DRONE_LANDING_ACTION if drone is not en-route or at a valid landing zone.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        for drone_id, drone in global_state.drones.items():
            # Drone must be en-route or already at a node
            if drone.status not in ["en_route", "idle", "loading", "unloading", "charging"]:
                self._set_action_validity((SimulationAction.DRONE_LANDING_ACTION, drone_id), False, mask)
                continue

            if drone.current_node_id is None:  # Mid-flight, check if next node in route is valid landing
                if not drone.current_route or len(drone.current_route) < 2:
                    self._set_action_validity((SimulationAction.DRONE_LANDING_ACTION, drone_id), False, mask)
                    continue
                landing_node_id = drone.current_route[1]  # Next node in planned route
            else:  # Already at a node
                landing_node_id = drone.current_node_id

            if landing_node_id not in global_state.nodes:  # Target landing node must exist
                self._set_action_validity((SimulationAction.DRONE_LANDING_ACTION, drone_id), False, mask)
                continue

            landing_node = global_state.get_entity("node", landing_node_id)

            # Check if landing node is a valid recovery point (e.g., depot or active micro-hub)
            if landing_node.type == 'micro_hub':
                micro_hub = global_state.get_entity("micro_hub", landing_node.id)
                if micro_hub.operational_status != "active" or micro_hub.is_blocked_for_recoveries:
                    self._set_action_validity((SimulationAction.DRONE_LANDING_ACTION, drone_id), False, mask)
                    continue

            # If drone has cargo, ensure it's landing at the customer node for delivery
            # or a micro-hub for transfer. If not, it might be an invalid landing.
            # This is a policy decision, not a strict constraint for MVP.
            # For now, if it's at a valid node or has a valid next node, allow landing.

        # Mask actions for non-existent drones
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] == SimulationAction.DRONE_LANDING_ACTION:
                drone_id = action_tuple[1]
                if drone_id not in global_state.drones:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_drone_to_charging_station_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks DRONE_TO_CHARGING_STATION if drone is not available or charging station is invalid/unreachable.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()
        network = sim.network

        for drone_id, drone in global_state.drones.items():
            # Drone must be available for new route (not broken down, not already charging)
            if drone.status in ["broken_down", "charging", "loading", "unloading"]:
                for station_id in global_state.nodes.keys():
                    self._set_action_validity((SimulationAction.DRONE_TO_CHARGING_STATION, drone_id, station_id), False,
                                              mask)
                continue

            # Check reachability to charging stations
            for station_id in global_state.nodes.keys():
                if station_id not in global_state.nodes:  # Station node must exist
                    self._set_action_validity((SimulationAction.DRONE_TO_CHARGING_STATION, drone_id, station_id), False,
                                              mask)
                    continue

                station_node = global_state.get_entity("node", station_id)
                if not station_node.is_charging_station:  # Must be a charging station
                    self._set_action_validity((SimulationAction.DRONE_TO_CHARGING_STATION, drone_id, station_id), False,
                                              mask)
                    continue

                if station_node.type == 'micro_hub':  # If micro-hub, check operational status
                    micro_hub = global_state.get_entity("micro_hub", station_node.id)
                    if micro_hub.operational_status != "active":
                        self._set_action_validity((SimulationAction.DRONE_TO_CHARGING_STATION, drone_id, station_id),
                                                  False, mask)
                        continue

                # Check if path exists
                start_node_for_path = drone.current_node_id
                if start_node_for_path is None:  # If mid-flight, path from next node
                    if drone.current_route and len(drone.current_route) > 1:
                        start_node_for_path = drone.current_route[1]
                    else:  # Drone is not at a node and has no valid route
                        self._set_action_validity((SimulationAction.DRONE_TO_CHARGING_STATION, drone_id, station_id),
                                                  False, mask)
                        continue

                path_exists = bool(network.calculate_shortest_path(start_node_for_path, station_id, 'drone'))
                if not path_exists:
                    self._set_action_validity((SimulationAction.DRONE_TO_CHARGING_STATION, drone_id, station_id), False,
                                              mask)

        # Mask actions for non-existent drones or nodes
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] == SimulationAction.DRONE_TO_CHARGING_STATION:
                drone_id = action_tuple[1]
                station_id = action_tuple[2]
                if drone_id not in global_state.drones or station_id not in global_state.nodes:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_drone_charging_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks DRONE_CHARGE_ACTION if drone is not at a charging station or no available slot,
        or battery is full.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        for drone_id, drone in global_state.drones.items():
            # Drone must be at a node
            if drone.current_node_id is None:
                for duration in self.action_map.keys():  # Assuming duration is the param for CHARGE_ACTION
                    if isinstance(duration, (int, float)):  # Check if it's a duration parameter
                        self._set_action_validity((SimulationAction.DRONE_CHARGE_ACTION, drone_id, duration), False,
                                                  mask)
                continue

            current_node = global_state.get_entity("node", drone.current_node_id)

            # Node must be a charging station
            if not current_node.is_charging_station:
                for duration in self.action_map.keys():
                    if isinstance(duration, (int, float)):
                        self._set_action_validity((SimulationAction.DRONE_CHARGE_ACTION, drone_id, duration), False,
                                                  mask)
                continue

            # If MicroHub, check operational status and slot availability
            if current_node.type == 'micro_hub':
                micro_hub = global_state.get_entity("micro_hub", current_node.id)
                if micro_hub.operational_status != "active" or not micro_hub.get_available_charging_slots():
                    for duration in self.action_map.keys():
                        if isinstance(duration, (int, float)):
                            self._set_action_validity((SimulationAction.DRONE_CHARGE_ACTION, drone_id, duration), False,
                                                      mask)
                    continue
                # Also check if the drone is *assigned* to a slot in that micro_hub
                is_assigned_slot = False
                for slot_id, assigned_drone in micro_hub.charging_slots.items():
                    if assigned_drone == drone_id:
                        is_assigned_slot = True
                        break
                if not is_assigned_slot:
                    # Drone must be assigned a slot to charge
                    for duration in self.action_map.keys():
                        if isinstance(duration, (int, float)):
                            self._set_action_validity((SimulationAction.DRONE_CHARGE_ACTION, drone_id, duration), False,
                                                      mask)
                    continue

            # Drone battery must not be full
            if drone.battery_level >= drone.max_battery_capacity:
                for duration in self.action_map.keys():
                    if isinstance(duration, (int, float)):
                        self._set_action_validity((SimulationAction.DRONE_CHARGE_ACTION, drone_id, duration), False,
                                                  mask)
                continue

            # Drone status should be 'charging' to take this action (or can be set by the action)
            # If the action itself sets the status, this check might be relaxed.
            if drone.status != "charging":
                for duration in self.action_map.keys():
                    if isinstance(duration, (int, float)):
                        self._set_action_validity((SimulationAction.DRONE_CHARGE_ACTION, drone_id, duration), False,
                                                  mask)
                continue

        # Mask actions for non-existent drones
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] == SimulationAction.DRONE_CHARGE_ACTION:
                drone_id = action_tuple[1]
                if drone_id not in global_state.drones:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_micro_hub_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks ACTIVATE/DEACTIVATE_MICRO_HUB, ADD_TO_CHARGING_QUEUE based on hub status/capacity.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        for micro_hub_id, micro_hub in global_state.micro_hubs.items():
            # ACTIVATE_MICRO_HUB
            if micro_hub.operational_status == "active":
                self._set_action_validity((SimulationAction.ACTIVATE_MICRO_HUB, micro_hub_id), False, mask)

            # DEACTIVATE_MICRO_HUB
            if micro_hub.operational_status == "inactive":
                self._set_action_validity((SimulationAction.DEACTIVATE_MICRO_HUB, micro_hub_id), False, mask)

            # ADD_TO_CHARGING_QUEUE
            if micro_hub.operational_status != "active" or not micro_hub.get_available_charging_slots():
                for drone_id in global_state.drones.keys():  # Mask all drones for this hub
                    self._set_action_validity((SimulationAction.ADD_TO_CHARGING_QUEUE, micro_hub_id, drone_id), False,
                                              mask)
            else:  # If hub is active and has slots, check drone availability
                for drone_id, drone in global_state.drones.items():
                    # Drone must be at this micro-hub, not charging, and not broken down
                    if drone.current_node_id != micro_hub_id or \
                            drone.status == "charging" or \
                            drone.status == "broken_down":
                        self._set_action_validity((SimulationAction.ADD_TO_CHARGING_QUEUE, micro_hub_id, drone_id),
                                                  False, mask)
                    # Drone battery must not be full
                    if drone.battery_level >= drone.max_battery_capacity:
                        self._set_action_validity((SimulationAction.ADD_TO_CHARGING_QUEUE, micro_hub_id, drone_id),
                                                  False, mask)

        # Mask actions for non-existent micro-hubs or drones
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] in [SimulationAction.ACTIVATE_MICRO_HUB, SimulationAction.DEACTIVATE_MICRO_HUB]:
                if action_tuple[1] not in global_state.micro_hubs:
                    self._set_action_validity(action_tuple, False, mask)
            elif action_tuple[0] == SimulationAction.ADD_TO_CHARGING_QUEUE:
                hub_id = action_tuple[1]
                drone_id = action_tuple[2]
                if hub_id not in global_state.micro_hubs or drone_id not in global_state.drones:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_maintenance_actions(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Masks FLAG_VEHICLE_FOR_MAINTENANCE, FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB
        based on current status.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        global_state = sim.get_current_global_state()

        # FLAG_VEHICLE_FOR_MAINTENANCE
        for vehicle_id in list(global_state.trucks.keys()) + list(global_state.drones.keys()):
            vehicle = None
            if vehicle_id in global_state.trucks:
                vehicle = global_state.trucks[vehicle_id]
            elif vehicle_id in global_state.drones:
                vehicle = global_state.drones[vehicle_id]

            if vehicle and (vehicle.status == "maintenance" or vehicle.status == "broken_down"):
                self._set_action_validity((SimulationAction.FLAG_VEHICLE_FOR_MAINTENANCE, vehicle_id), False, mask)

            # Also mask if vehicle is currently en_route and cannot be immediately pulled for maintenance
            if vehicle and vehicle.status == "en_route":
                self._set_action_validity((SimulationAction.FLAG_VEHICLE_FOR_MAINTENANCE, vehicle_id), False, mask)

        # FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB
        for micro_hub_id, micro_hub in global_state.micro_hubs.items():
            # Check each service type
            _RESOURCE_TYPES = [
                SimulationAction.RESOURCE_CHARGING_SLOT,
                SimulationAction.RESOURCE_PACKAGE_SORTING_SERVICE,
                SimulationAction.RESOURCE_LAUNCHES,
                SimulationAction.RESOURCE_RECOVERIES
            ]
            for service_type in _RESOURCE_TYPES:
                is_unavailable = False
                if service_type == SimulationAction.RESOURCE_CHARGING_SLOT:
                    # If no slots available or hub inactive, it's implicitly unavailable
                    if not micro_hub.get_available_charging_slots() or micro_hub.operational_status != "active":
                        is_unavailable = True
                elif service_type == SimulationAction.RESOURCE_PACKAGE_SORTING_SERVICE:
                    is_unavailable = micro_hub.is_package_transfer_unavailable
                elif service_type == SimulationAction.RESOURCE_LAUNCHES:
                    is_unavailable = micro_hub.is_blocked_for_launches
                elif service_type == SimulationAction.RESOURCE_RECOVERIES:
                    is_unavailable = micro_hub.is_blocked_for_recoveries

                if is_unavailable:
                    self._set_action_validity(
                        (SimulationAction.FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB, micro_hub_id, service_type),
                        False, mask)

        # Mask actions for non-existent entities
        for action_tuple, idx in self.action_map.items():
            if action_tuple[0] == SimulationAction.FLAG_VEHICLE_FOR_MAINTENANCE:
                vehicle_id = action_tuple[1]
                if vehicle_id not in global_state.trucks and vehicle_id not in global_state.drones:
                    self._set_action_validity(action_tuple, False, mask)
            elif action_tuple[0] == SimulationAction.FLAG_UNAVAILABILITY_OF_SERVICE_AT_MICRO_HUB:
                hub_id = action_tuple[1]
                if hub_id not in global_state.micro_hubs:
                    self._set_action_validity(action_tuple, False, mask)
        return mask

    def _mask_no_operation_action(self, sim: 'LogisticsSimulation') -> np.ndarray:
        """
        Ensures the NO_OPERATION action is always valid.
        """
        mask = np.ones(self.action_space_size, dtype=bool)
        no_op_index = self._get_action_index(NO_OPERATION)
        if no_op_index is not None:
            mask[no_op_index] = True  # Explicitly ensure NO_OPERATION is True
        return mask

