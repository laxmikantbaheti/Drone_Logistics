# ddls_src/functions/data_manager.py
import pandas as pd
from ddls_src.entities.vehicles.base import Vehicle
from ddls_src.entities.order import Order


class EventLogger:
    """
    Centralized event-driven logger. Listens to entity state changes
    and categorizes the logs per entity type for clean CSV exports.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clears all stored data. Called at the start of a new simulation."""
        self.logs = {
            "vehicles": [],
            "orders": [],
            "micro_hubs": []
        }
        self.recorded_events_count = 0

    def handle_entity_state_change(self, p_event_id, p_event_object):
        """
        The universal listener. Caught whenever an entity calls self.raise_state_change_event().
        """
        entity = p_event_object.get_raising_object()

        # Safety check: ensure entity has a reference to global state and time
        if not hasattr(entity, 'global_state') or entity.global_state is None:
            return

        current_time = entity.global_state.current_time

        # --- 1. Log Vehicle Events ---
        if isinstance(entity, Vehicle):
            # STRICTLY WATCHING THE CARGO MANIFEST ONLY
            manifest_ids = [f"{o.get_id()}-{o.get_pickup_node_id(), o.get_delivery_node_id()}-{o.size}" for o in entity.cargo_manifest]

            energy = getattr(entity, 'battery_level', getattr(entity, 'fuel_level', None))

            self.logs["vehicles"].append({
                'time': current_time,
                'vehicle_id': f"{entity.get_id()}-{entity.max_payload_capacity}",
                'vehicle_type': entity.C_NAME,
                'status': entity.get_state_value_by_dim_name(entity.C_DIM_TRIP_STATE[0]),
                'current_node': entity.get_current_node(),
                "current_coords": entity.current_location_coords,
                'energy_level': energy,
                "cargo_capacity": entity.get_cargo_capacity(),
                'cargo_size': entity.get_current_cargo_size(),
                'cargo_manifest': str(manifest_ids)  # <--- Exactly what is in the truck/drone right now
            })
            self.recorded_events_count += 1

        # --- 2. Log Order Events ---
        elif isinstance(entity, Order):
            self.logs["orders"].append({
                'time': current_time,
                'order_id': f"{entity.get_id()}-{entity.get_pickup_node_id(), entity.get_delivery_node_id()}",
                'status': entity.get_state_value_by_dim_name(entity.C_DIM_DELIVERY_STATUS[0]),
                'current_node': getattr(entity, 'current_node_id', 'Unknown'),
                'pickup_node': entity.get_pickup_node_id(),
                'delivery_node': entity.get_delivery_node_id()
            })
            self.recorded_events_count += 1

    def export_reports(self, base_filepath: str = 'scenario_report', logs=None):
        """Converts categorized logs to DataFrames, sorts by Entity ID, and exports to CSV."""
        print(f"EventLogger: Compiling {self.recorded_events_count} events into reports...")

        if logs is None:
            # Export Vehicles
            if self.logs["vehicles"]:
                df_vehicles = pd.DataFrame(self.logs["vehicles"])
                # Sort by vehicle_id to group entities, then chronologically
                df_vehicles = df_vehicles.sort_values(by=['vehicle_id', 'time'])
                v_path = f"{base_filepath}_vehicles.csv"
                df_vehicles.to_csv(v_path, index=False)
                print(f" - Exported {len(self.logs['vehicles'])} vehicle events to {v_path}")

            # Export Orders
            if self.logs["orders"]:
                df_orders = pd.DataFrame(self.logs["orders"])
                # Sort by order_id to group entities, then chronologically
                df_orders = df_orders.sort_values(by=['order_id', 'time'])
                o_path = f"{base_filepath}_orders.csv"
                df_orders.to_csv(o_path, index=False)
                print(f" - Exported {len(self.logs['orders'])} order events to {o_path}")
        else:
            if logs["vehicles"]:
                df_vehicles = pd.DataFrame(logs["vehicles"])
                # Sort by vehicle_id to group entities, then chronologically
                df_vehicles = df_vehicles.sort_values(by=['vehicle_id', 'time'])
                v_path = f"{base_filepath}_vehicles.csv"
                df_vehicles.to_csv(v_path, index=False)
                print(f" - Exported {len(logs['vehicles'])} vehicle events to {v_path}")

            # Export Orders
            if logs["orders"]:
                df_orders = pd.DataFrame(logs["orders"])
                # Sort by order_id to group entities, then chronologically
                df_orders = df_orders.sort_values(by=['order_id', 'time'])
                o_path = f"{base_filepath}_orders.csv"
                df_orders.to_csv(o_path, index=False)
                print(f" - Exported {len(logs['orders'])} order events to {o_path}")