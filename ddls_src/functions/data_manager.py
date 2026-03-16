# ddls_src/managers/data_manager.py
import pandas as pd
import os


class DataManager:
    """
    Centralized data storage for the simulation.
    Uses fast-append lists for $O(1)$ insertions during the step loop,
    and compiles them into DataFrames/CSVs at the end of the simulation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clears all stored data. Call this at the start of a new RL episode."""
        self.vehicle_state_logs = []
        self.order_timeline_logs = []

    def log_vehicle_state(self, current_time: float, vehicle_id: str, node_id: int, status: str,
                          battery_level: float = None, num_picked_up: int = 0, num_delivered: int = 0,
                          pickup_orders: list = None, delivery_orders: list = None):
        """Logs the micro-events, movements, and current order manifest of vehicles."""
        self.vehicle_state_logs.append({
            'time': current_time,
            'vehicle_id': vehicle_id,
            'node_id': node_id,
            'status': status,
            'battery': battery_level,
            'num_pickup_tasks': num_picked_up,
            'num_delivery_tasks': num_delivered,
            # We convert lists to strings or keep them as lists depending on how you want the CSV to look
            'pickup_orders': str(pickup_orders) if pickup_orders else "[]",
            'delivery_orders': str(delivery_orders) if delivery_orders else "[]"
        })

    def log_order_event(self, current_time: float, order_id: str, event_type: str, vehicle_id: str = None):
        """
        Logs order lifecycle events (e.g., 'created', 'picked_up', 'delivered').
        """
        self.order_timeline_logs.append({
            'time': current_time,
            'order_id': order_id,
            'event_type': event_type,  # e.g., 'pickup', 'delivery'
            'vehicle_id': vehicle_id
        })

    def export_reports(self, base_filepath: str = 'scenario_report'):
        """Converts logs to DataFrames and exports them efficiently."""
        print("DataManager: Compiling and exporting reports...")

        # 1. Export Vehicle States
        if self.vehicle_state_logs:
            df_vehicles = pd.DataFrame(self.vehicle_state_logs)
            df_vehicles.to_csv(f"{base_filepath}_vehicles.csv", index=False)

        # 2. Export Order Timelines
        if self.order_timeline_logs:
            df_orders = pd.DataFrame(self.order_timeline_logs)

            # Pivot to match your previous timeline format (Start Time, End Time)
            # This is vastly faster using Pandas than nested Python loops
            df_pivot = df_orders.pivot(index='order_id', columns='event_type', values='time').reset_index()
            df_pivot.to_csv(f"{base_filepath}_order_timeline.csv", index=False)

        print(f"DataManager: Export complete. to: {base_filepath}")
        return df_vehicles if self.vehicle_state_logs else pd.DataFrame(), \
            df_orders if self.order_timeline_logs else pd.DataFrame()