import csv
import json
from typing import Any, Dict, Tuple


def export_simulation_reports(global_state: Any, output_format: str = 'csv', base_filepath: str = 'scenario_report') -> \
Tuple[Dict, list, Dict]:
    """
    Extracts:
    1. The sequential node visits per vehicle, capturing 0-duration stops and mapping orders.
    2. Detailed order assignment/coordination records.
    3. A timeline of completed deliveries per vehicle.
    """
    all_vehicles = {
        **getattr(global_state, 'trucks', {}),
        **getattr(global_state, 'drones', {})
    }

    nodes_report_data = {}
    order_records = []
    timeline_report_data = {}

    if not all_vehicles:
        print("No vehicles found for reporting.")
        return nodes_report_data, order_records, timeline_report_data

    for vehicle_id, vehicle_obj in all_vehicles.items():

        # --- 1. Process Sequential Node Visits (FIXED MICRO-EVENT PARSING) ---
        state_data = vehicle_obj.data_storage.get(vehicle_obj.C_DATA_FRAME_VEH_STATES, {})
        node_visits = []
        current_visit = None

        if state_data:
            sorted_events = sorted(state_data.items())
            for current_time, micro_events in sorted_events:
                # Iterate through ALL events recorded at this exact timestep
                # This prevents overwriting the initial node if the vehicle leaves at t=0
                for event in micro_events:
                    try:
                        node = event[1]
                    except (IndexError, TypeError):
                        continue

                    if node is not None:
                        # Vehicle is at a node
                        if current_visit is None or current_visit['node'] != node:
                            if current_visit is not None:
                                current_visit['departure_time'] = current_time
                                node_visits.append(current_visit)
                            current_visit = {
                                'node': node,
                                'arrival_time': current_time,
                                'departure_time': global_state.current_time,  # Fallback to current time
                                'pickups': [],
                                'deliveries': []
                            }
                    elif node is None and current_visit is not None:
                        # Vehicle departed the node (Node becomes None when En Route)
                        current_visit['departure_time'] = current_time
                        node_visits.append(current_visit)
                        current_visit = None

            # Append the final visit if the simulation ends while the vehicle is at a node
            if current_visit is not None:
                node_visits.append(current_visit)

        # --- 2. Process Order Assignments & Map to Nodes ---
        timeline_data = vehicle_obj.data_storage.get(vehicle_obj.C_DATA_FRAME_VEH_TIMELINE, {})
        vehicle_deliveries = []

        for order_id, times in timeline_data.items():
            if not times:
                continue

            clean_id = str(order_id).strip()
            start_time = times[0][0]

            # Map Pickup to the correct Node Visit
            for visit in node_visits:
                # Includes 0-duration windows where arrival_time == departure_time
                if visit['arrival_time'] <= start_time <= visit['departure_time']:
                    visit['pickups'].append({'order_id': clean_id, 'time': start_time})
                    break

            if len(times) >= 2:
                end_time = times[1][0]
                status = "Delivered"

                # Map Delivery to the correct Node Visit
                for visit in node_visits:
                    if visit['arrival_time'] <= end_time <= visit['departure_time']:
                        visit['deliveries'].append({'order_id': clean_id, 'time': end_time})
                        break

                # Add to flattened delivery timeline report
                time_window_str = f"({start_time}, {end_time})"
                vehicle_deliveries.append({
                    "Order ID": clean_id,
                    "Delivery Time (s)": end_time,
                    "Time Window": time_window_str
                })
            else:
                end_time = global_state.current_time
                status = "In Transit"

            # Parse coordinated order details
            base_id = clean_id.split("_")[0]
            leg_info = "Direct Delivery"
            if "_1" in clean_id:
                leg_info = "Leg 1 (Hub/Transfer)"
            elif "_2" in clean_id:
                leg_info = "Leg 2 (Final Delivery)"

            pickup_node = "Unknown"
            delivery_node = "Unknown"
            try:
                order_obj = global_state.get_entity("order", int(clean_id) if clean_id.isdigit() else clean_id)
                if order_obj:
                    pickup_node = order_obj.get_pickup_node_id()
                    delivery_node = order_obj.get_delivery_node_id()
            except Exception:
                pass

            order_records.append({
                "Base Order ID": base_id,
                "Order ID (Leg)": clean_id,
                "Coordination Type": leg_info,
                "Assigned Vehicle": vehicle_id,
                "Pickup Node": pickup_node,
                "Delivery Node": delivery_node,
                "Start Time (s)": start_time,
                "End Time (s)": end_time,
                "Duration (s)": end_time - start_time,
                "Status": status
            })

        # Finalize data structures for this vehicle
        nodes_report_data[vehicle_id] = node_visits
        vehicle_deliveries.sort(key=lambda x: x["Delivery Time (s)"])
        timeline_report_data[vehicle_id] = vehicle_deliveries

    # Sort master order records
    order_records.sort(key=lambda x: (x["Base Order ID"], x["Start Time (s)"]))

    # --- 3. Export Logic ---
    if output_format.lower() == 'csv':

        # Export Node Sequence with Pickup/Delivery Details
        nodes_path = f"{base_filepath}_nodes.csv"
        with open(nodes_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Vehicle ID", "Sequence Step", "Node ID",
                "Arrival Time (s)", "Departure Time (s)",
                "Pickup Count", "Picked Up Orders (ID @ Time)",
                "Delivery Count", "Delivered Orders (ID @ Time)"
            ])
            for v_id, visits in nodes_report_data.items():
                for idx, visit in enumerate(visits):
                    pickups_str = ", ".join([f"{p['order_id']} @ {p['time']}s" for p in visit['pickups']])
                    deliveries_str = ", ".join([f"{d['order_id']} @ {d['time']}s" for d in visit['deliveries']])

                    writer.writerow([
                        v_id,
                        idx + 1,
                        visit['node'],
                        visit['arrival_time'],
                        visit['departure_time'],
                        len(visit['pickups']),
                        pickups_str,
                        len(visit['deliveries']),
                        deliveries_str
                    ])

        # Export Orders CSV
        orders_path = f"{base_filepath}_orders.csv"
        with open(orders_path, mode='w', newline='') as file:
            if order_records:
                writer = csv.DictWriter(file, fieldnames=order_records[0].keys())
                writer.writeheader()
                writer.writerows(order_records)

        # Export Delivery Timeline CSV (Flattened)
        timeline_path = f"{base_filepath}_delivery_timeline.csv"
        with open(timeline_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            max_deliveries = max([len(deliveries) for deliveries in timeline_report_data.values()], default=0)

            header = ["Vehicle ID"]
            for i in range(1, max_deliveries + 1):
                header.extend([f"Order {i} ID", f"Order {i} (Pickup, Delivery)"])
            writer.writerow(header)

            for v_id, deliveries in timeline_report_data.items():
                row = [v_id]
                for delivery in deliveries:
                    row.extend([delivery["Order ID"], delivery["Time Window"]])
                writer.writerow(row)

        print(f"Successfully exported CSV reports to:\n - {nodes_path}\n - {orders_path}\n - {timeline_path}")

    elif output_format.lower() == 'json':
        full_path = f"{base_filepath}.json"
        combined_data = {
            "node_visits": nodes_report_data,
            "order_assignments": order_records,
            "delivery_timeline": timeline_report_data
        }
        with open(full_path, 'w') as file:
            json.dump(combined_data, file, indent=4)
        print(f"Successfully exported combined JSON report to {full_path}")

    return nodes_report_data, order_records, timeline_report_data