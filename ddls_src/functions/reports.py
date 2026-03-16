# ddls_src/functions/reports.py
import pandas as pd
from typing import Any, Dict, Tuple


def export_simulation_reports(global_state: Any, output_format: str = 'csv', base_filepath: str = 'scenario_report') -> \
Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    print(f"Generating single-row order reports with cross-referenced actual node tracking...")

    # 1. First, prepare Vehicle Data for cross-referencing
    all_vehicles = {**getattr(global_state, 'trucks', {}), **getattr(global_state, 'drones', {})}
    vehicle_histories = {}
    master_vehicle_logs = []

    for v_id, vehicle_obj in all_vehicles.items():
        if hasattr(vehicle_obj, 'state_history') and vehicle_obj.state_history:
            vehicle_histories[v_id] = vehicle_obj.state_history
            master_vehicle_logs.extend(vehicle_obj.state_history)

    # Save standard vehicle report
    df_veh_out = pd.DataFrame(master_vehicle_logs)
    if not df_veh_out.empty:
        df_veh_out.to_csv(f"{base_filepath}_vehicles.csv", index=False)

    # 2. Extract Order Information (One Row Per Order)
    all_orders = getattr(global_state, 'orders', {})
    master_order_rows = []

    for o_id, order_obj in all_orders.items():
        oid_str = str(order_obj.get_id())

        # Base row data
        row = {
            'order_id': oid_str,
            'planned_pickup': order_obj.pickup_node_id,
            'planned_delivery': order_obj.delivery_node_id,
            'final_status': order_obj.status,
            'time_placed': order_obj.time_received,
            'actual_pickup_node': None,
            'time_picked_up': None,
            'actual_delivery_node': None,
            'time_delivered': None,
            'assigned_vehicle': order_obj.assigned_vehicle_id
        }

        # --- SEARCH LOGIC ---
        # We look through every vehicle history to find exactly when this order appeared/disappeared
        for v_id, history in vehicle_histories.items():
            for entry in history:
                # Check if our order was in the vehicle's delivery_orders list at this time
                # Note: We assume the history stores delivery_orders as a list of IDs
                cargo = entry.get('delivery_orders', [])

                # If order just appeared in cargo -> that's the ACTUAL PICKUP
                if oid_str in str(cargo) and row['time_picked_up'] is None:
                    row['time_picked_up'] = entry['time']
                    row['actual_pickup_node'] = entry['node_id']
                    row['assigned_vehicle'] = v_id

                # If order was in cargo and now is gone -> that's the ACTUAL DELIVERY
                if row['time_picked_up'] is not None and oid_str not in str(cargo) and row['time_delivered'] is None:
                    # We check the previous entry to see where it was dropped off
                    row['time_delivered'] = entry['time']
                    row['actual_delivery_node'] = entry['node_id']

        master_order_rows.append(row)

    df_orders = pd.DataFrame(master_order_rows)
    if not df_orders.empty:
        df_orders = df_orders.sort_values(by=['order_id'])
        df_orders.to_csv(f"{base_filepath}_order_report.csv", index=False)

    print("Export complete. CSV generated with one row per order and actual nodes.")
    return df_veh_out, df_orders, {}