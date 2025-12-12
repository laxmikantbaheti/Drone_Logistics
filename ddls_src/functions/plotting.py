# Complete Python script for plotting vehicle timelines and states.

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, Any, List

from numpy.ma.extras import apply_over_axes


# --- Consolidated Mocks for a Runnable Example ---

class MockVehicle:
    """A mock Vehicle class to simulate your updated data structure."""
    C_DATA_FRAME_VEH_STATES = "VEH_STATES"
    C_DATA_FRAME_VEH_TIMELINE = "VEH_TIMELINE"

    def __init__(self, vehicle_id: str, state_data: Dict = None, order_data: Dict = None):
        self.id = vehicle_id
        self.data_storage = {
            self.C_DATA_FRAME_VEH_STATES: state_data or {},
            self.C_DATA_FRAME_VEH_TIMELINE: order_data or {}
        }


class MockGlobalState:
    """A mock GlobalState class with all necessary attributes."""

    def __init__(self, trucks: Dict[str, MockVehicle], drones: Dict[str, MockVehicle], current_time: float):
        self.trucks = trucks
        self.drones = drones
        self.current_time = current_time


# --- Plotting Function 1: Vehicle Order Timelines (Unchanged) ---

def plot_vehicle_gantt_chart(global_state: Any, show_order_labels: bool = True) -> None:
    """
    Generates a Gantt chart for ALL vehicles, showing stacked order timelines.
    Includes unused vehicles.
    """
    truck_data = getattr(global_state, 'trucks', {})
    drone_data = getattr(global_state, 'drones', {})
    all_vehicle_ids = sorted(list(truck_data.keys()) + list(drone_data.keys()))

    if not all_vehicle_ids:
        print("No vehicles found.")
        return

    processed_vehicles = {}
    for vehicle_id in all_vehicle_ids:
        vehicle_obj = truck_data.get(vehicle_id) or drone_data.get(vehicle_id)
        timeline_data = vehicle_obj.data_storage.get(vehicle_obj.C_DATA_FRAME_VEH_TIMELINE, {})

        orders = []
        for order_id, times in timeline_data.items():
            if len(times) == 2 and times[1][0] > times[0][0]:
                orders.append({'id': order_id, 'start': times[0][0], 'end': times[1][0]})

        if not orders:
            continue

        orders.sort(key=lambda x: x['start'])
        lanes = []
        for order in orders:
            placed = False
            for lane in lanes:
                if order['start'] >= lane[-1]['end']:
                    lane.append(order)
                    placed = True
                    break
            if not placed:
                lanes.append([order])

        vehicle_type = 'truck' if vehicle_id in truck_data else 'drone'
        processed_vehicles[vehicle_id] = {'lanes': lanes, 'type': vehicle_type}

    total_lanes = sum(len(v['lanes']) for v in processed_vehicles.values())
    num_unused = len(all_vehicle_ids) - len(processed_vehicles)
    fig_height = total_lanes + num_unused + (len(all_vehicle_ids) * 0.5)
    fig, ax = plt.subplots(figsize=(16, max(6, fig_height * 0.6)))

    color_map = {'truck': 'tab:blue', 'drone': 'tab:green'}
    y_ticks, y_tick_labels, y_pos = [], [], 0

    for vehicle_id in all_vehicle_ids:
        if vehicle_id in processed_vehicles:
            data = processed_vehicles[vehicle_id]
            lanes, num_lanes = data['lanes'], len(data['lanes'])
            y_ticks.append(y_pos + (num_lanes - 1) / 2.0)
            y_tick_labels.append(vehicle_id)
            for lane in lanes:
                bar_ranges = [(o['start'], o['end'] - o['start']) for o in lane]
                ax.broken_barh(bar_ranges, (y_pos - 0.2, 0.4), facecolors=color_map[data['type']], edgecolor='black', alpha = 0.85)
                if show_order_labels:
                    for order in lane:
                        start, duration = order['start'], order['end'] - order['start']
                        ax.text(start + duration / 2, y_pos, order['id'], ha='center', va='center', color='black',
                                fontsize=11)
                y_pos += 1
        else:
            y_ticks.append(y_pos)
            y_tick_labels.append(f"{vehicle_id} (Unused)")
            y_pos += 1

        ax.axhline(y_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
        y_pos += 0.5

    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Vehicle ID")
    ax.set_title("Complete Vehicle Order Timeline")
    ax.legend(handles=[Patch(facecolor=c, label=l.capitalize()) for l, c in color_map.items()], loc='best')
    plt.tight_layout()
    plt.show()


# --- Plotting Function 2: Vehicle State Timelines (MODIFIED COLORING) ---

def plot_vehicle_states(global_state: Any) -> None:
    """
    Generates a Gantt chart of vehicle states from time-keyed data.
    Handles instantaneous events as "milestones."
    """
    all_vehicles = {**getattr(global_state, 'trucks', {}), **getattr(global_state, 'drones', {})}
    if not all_vehicles:
        print("No vehicles found.")
        return

    processed_data = {}
    unique_states = set()

    # 1. Parse the time-keyed structure for each vehicle
    for vehicle_id, vehicle_obj in all_vehicles.items():
        state_data = vehicle_obj.data_storage.get(vehicle_obj.C_DATA_FRAME_VEH_STATES)
        if not state_data:
            continue

        sorted_events = sorted(state_data.items())
        intervals, milestones = [], []

        # 2. Separate events into intervals and milestones
        for i in range(len(sorted_events)):
            current_time, current_event = sorted_events[i]
            current_state = current_event[-1][0]
            current_node = current_event[-1][1]

            next_time = sorted_events[i + 1][0] if i + 1 < len(sorted_events) else global_state.current_time

            if next_time > current_time:
                intervals.append({'state': current_state, 'start': current_time, 'duration': next_time - current_time, 'node':current_node})
                unique_states.add(current_state)
            else:
                milestones.append({'time': current_time, 'label': current_state, "node":current_node})

        processed_data[vehicle_id] = {'intervals': intervals, 'milestones': milestones}

    if not unique_states:
        print("No state data with duration found to plot.")
        return

    # 3. Plotting
    # --- MODIFICATION: Use a categorical colormap for more distinct colors ---
    cmap = plt.get_cmap('tab10')
    colors = cmap.colors
    state_colors = {name: colors[i % len(colors)] for i, name in enumerate(sorted(list(unique_states)))}
    # --- END OF MODIFICATION ---

    fig, ax = plt.subplots(figsize=(16, len(all_vehicles) * 0.8 + 2))

    vehicle_ids = sorted(all_vehicles.keys())
    for i, vehicle_id in enumerate(vehicle_ids):
        data = processed_data.get(vehicle_id)
        if not data: continue

        # Plot state bars
        if data['intervals']:
            bar_ranges = [(iv['start'], iv['duration']) for iv in data['intervals']]
            bar_colors = [state_colors.get(iv['state'], 'gray') for iv in data['intervals']]
            ax.broken_barh(bar_ranges, (i - 0.1, 0.2), facecolors=bar_colors)

            # --- NEW: Add text labels for node IDs to bars ---
            for iv in data['intervals']:
                center_x = iv['start'] + iv['duration'] / 2
                ax.text(center_x, i, iv['node'], ha='center', va='center', color='white', fontsize=9, fontweight='bold')

        # Plot milestones as diamonds
        # if data['milestones']:
        #     milestone_times = [m['time'] for m in data['milestones']]
        #     y_vals = [i] * len(milestone_times)
        #     ax.plot(milestone_times, y_vals, 'D', markersize=8, color='black', label='Milestone' if i == 0 else "")
        #
        #     # --- NEW: Add text labels for node IDs to milestones ---
        #     for m in data['milestones']:
        #         ax.text(m['time'], i - 0.05, m['node'], ha='center', va='bottom', color='black', fontsize=7)

    ax.set_yticks(range(len(vehicle_ids)))
    ax.set_yticklabels(vehicle_ids)
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Vehicle ID")
    ax.set_title("Vehicle State Timeline")
    ax.grid(axis='x', linestyle=':', alpha=0.7)

    legend_elements = [Patch(facecolor=c, label=s) for s, c in state_colors.items()]
    if any(p.get('milestones') for p in processed_data.values()):
        legend_elements.append(
            plt.Line2D([0], [0], marker='D', color='w', label='Milestone', markerfacecolor='black', markersize=10))
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import math


def plot_vehicle_cargo_history(global_state):
    """
    Plots cargo statistics for each vehicle in separate subplots within the same figure.
    Includes a reference dashed line for the vehicle's maximum payload capacity.
    Ensures the timeline starts at t=0 with zero cargo if no initial data exists.

    Args:
        global_state: The GlobalState object containing 'trucks', 'drones', and 'current_time'.
    """
    # 1. Gather all vehicles (Trucks and Drones)
    all_vehicles = list(global_state.trucks.values()) + list(global_state.drones.values())

    # 2. Filter for vehicles that actually have cargo data
    active_vehicles = [
        v for v in all_vehicles
        if hasattr(v, 'cargo_stats') and v.cargo_stats
    ]

    num_vehicles = len(active_vehicles)

    if num_vehicles == 0:
        print("No cargo statistics available to plot.")
        return

    # 3. Setup the subplots (Vertical stack)
    # Share x-axis to easily compare timelines across vehicles
    fig, axes = plt.subplots(nrows=math.ceil(num_vehicles/2), ncols=2, sharex=True)
    axes = axes.flatten()
    # Handle the case where there is only 1 vehicle (axes is not a list)
    if num_vehicles == 1:
        axes = [axes]

    for i, vehicle in enumerate(active_vehicles):
        ax = axes[i]

        # --- Prepare Data ---
        sorted_times = sorted(vehicle.cargo_stats.keys())
        loads = [vehicle.cargo_stats[t] for t in sorted_times]

        # [NEW LOGIC] Prepend zero if the timeline doesn't start at 0.0
        # This handles the "no cargo in the beginning" duration.
        if sorted_times and sorted_times[0] > 0.0:
            sorted_times.insert(0, 0.0)
            loads.insert(0, 0)

        # Extend line to current simulation time (holds the last known value)
        if sorted_times and sorted_times[-1] < global_state.current_time:
            sorted_times.append(global_state.current_time)
            loads.append(loads[-1])

        # --- Plot Cargo Load (Step Chart) ---
        label_str = f"{vehicle.C_NAME} {vehicle.get_id()}"
        ax.step(sorted_times, loads, where='post', label='Current Load', linewidth=2)

        # --- Plot Capacity Reference Line ---
        capacity = getattr(vehicle, 'max_payload_capacity', None)
        if capacity is not None:
            ax.axhline(y=capacity, color='r', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Max Capacity ({capacity})')

        # --- Formatting ---
        ax.set_ylabel("Cargo Units")
        ax.set_title(label_str, loc='left', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize='small')

        # Only set the X-label for the bottom-most subplot
        if i == num_vehicles - 1:
            ax.set_xlabel("Simulation Time (s)")

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Any


def plot_invalid_delivery_gantt_chart(global_state: Any, show_order_labels: bool = True) -> None:
    """
    Generates a Gantt chart for ALL vehicles to visualize delivery sequences.

    Highlights invalid deliveries with a red hatched pattern (//).
    A delivery is INVALID if Leg 2 starts before Leg 1 ends (Start_2 < End_1),
    regardless of which vehicle carries which leg.
    """
    truck_data = getattr(global_state, 'trucks', {})
    drone_data = getattr(global_state, 'drones', {})
    all_vehicle_ids = sorted(list(truck_data.keys()) + list(drone_data.keys()))

    if not all_vehicle_ids:
        print("No vehicles found.")
        return

    # --- Step 1: Global Order Registry ---
    # Collect start/end times for EVERY order from EVERY vehicle.
    # This enables us to cross-reference Leg 1 on a Truck with Leg 2 on a Drone.
    all_orders_map = {}

    for vehicle_id in all_vehicle_ids:
        vehicle_obj = truck_data.get(vehicle_id) or drone_data.get(vehicle_id)
        timeline_data = vehicle_obj.data_storage.get(vehicle_obj.C_DATA_FRAME_VEH_TIMELINE, {})

        for order_id, times in timeline_data.items():
            if not times: continue

            # Start time is always the first event
            start_time = times[0][0]

            # End time is the second event, OR current_time if the order is still in transit
            if len(times) >= 2:
                end_time = times[1][0]
            else:
                end_time = global_state.current_time

            if end_time > start_time:
                clean_id = str(order_id).split(" - ")[0]
                all_orders_map[clean_id] = {'start': start_time, 'end': end_time}

    # --- Step 2: Plotting Loop ---
    processed_vehicles = {}

    # Pre-process lanes for plotting
    for vehicle_id in all_vehicle_ids:
        vehicle_obj = truck_data.get(vehicle_id) or drone_data.get(vehicle_id)
        timeline_data = vehicle_obj.data_storage.get(vehicle_obj.C_DATA_FRAME_VEH_TIMELINE, {})

        orders = []
        for order_id, times in timeline_data.items():
            if not times: continue

            start_time = times[0][0]
            if len(times) >= 2:
                end_time = times[1][0]
            else:
                end_time = global_state.current_time

            if end_time > start_time:
                orders.append({'id': str(order_id).strip(), 'start': start_time, 'end': end_time})

        if not orders:
            continue

        orders.sort(key=lambda x: x['start'])
        lanes = []
        for order in orders:
            placed = False
            for lane in lanes:
                if order['start'] >= lane[-1]['end']:
                    lane.append(order)
                    placed = True
                    break
            if not placed:
                lanes.append([order])

        vehicle_type = 'truck' if vehicle_id in truck_data else 'drone'
        processed_vehicles[vehicle_id] = {'lanes': lanes, 'type': vehicle_type}

    # Setup Figure
    total_lanes = sum(len(v['lanes']) for v in processed_vehicles.values())
    num_unused = len(all_vehicle_ids) - len(processed_vehicles)
    fig_height = total_lanes + num_unused + (len(all_vehicle_ids) * 0.5)

    fig, ax = plt.subplots(figsize=(16, max(6, fig_height * 0.6)))
    color_map = {'truck': 'tab:blue', 'drone': 'tab:green'}
    y_ticks, y_tick_labels, y_pos = [], [], 0

    for vehicle_id in all_vehicle_ids:
        if vehicle_id in processed_vehicles:
            data = processed_vehicles[vehicle_id]
            lanes, num_lanes = data['lanes'], len(data['lanes'])

            y_ticks.append(y_pos + (num_lanes - 1) / 2.0)
            y_tick_labels.append(vehicle_id)

            for lane in lanes:
                normal_ranges = []
                invalid_ranges = []

                for order in lane:
                    is_invalid = False
                    order_id = order['id']

                    # --- VALIDATION LOGIC ---
                    # We only check validity if we are "Leg 2" (_2)
                    if "_2" in order_id:
                        base_id = order_id.split("_")[0]
                        sibling_id = f"{base_id}_1"

                        # We are Leg 2. Start time is s2.
                        s2 = order['start']

                        # Look up Leg 1 in the GLOBAL map (could be on any vehicle)
                        if sibling_id in all_orders_map:
                            e1 = all_orders_map[sibling_id]['end']

                            # INVALID CONDITION: Leg 2 starts before Leg 1 ends
                            if s2 < e1:
                                is_invalid = True

                    if "_1" in order_id:
                        base_id = order_id.split("_")[0]
                        sibling_id = f"{base_id}_2"

                        # We are Leg 2. Start time is s2.
                        e1 = order['end']

                        # Look up Leg 1 in the GLOBAL map (could be on any vehicle)
                        if sibling_id in all_orders_map:
                            s2 = all_orders_map[sibling_id]['start']

                            # INVALID CONDITION: Leg 2 starts before Leg 1 ends
                            if s2 < e1:
                                is_invalid = True
                    # ------------------------

                    duration = order['end'] - order['start']
                    if is_invalid:
                        invalid_ranges.append((order['start'], duration))
                    else:
                        normal_ranges.append((order['start'], duration))

                # Draw Valid Bars
                if normal_ranges:
                    ax.broken_barh(normal_ranges, (y_pos - 0.2, 0.4),
                                   facecolors=color_map[data['type']],
                                   edgecolor='black', alpha=0.85)

                # Draw Invalid Bars (Red + Hatched)
                if invalid_ranges:
                    ax.broken_barh(invalid_ranges, (y_pos - 0.2, 0.4),
                                   facecolors="white",
                                   edgecolor='gray', hatch='//', linewidth=1.0, alpha=0.5)

                if show_order_labels:
                    for order in lane:
                        start, duration = order['start'], order['end'] - order['start']
                        ax.text(start + duration / 2, y_pos, order['id'],
                                ha='center', va='center', color='black', fontsize=11, clip_on=True)
                y_pos += 1
        else:
            y_ticks.append(y_pos)
            y_tick_labels.append(f"{vehicle_id} (Unused)")
            y_pos += 1

        ax.axhline(y_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
        y_pos += 0.5

    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=11)

    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Vehicle ID")
    ax.set_title("Vehicle Order Delivery Timeline")

    legend_elements = [Patch(facecolor=c, label=l.capitalize()) for l, c in color_map.items()]
    legend_elements.append(Patch(facecolor='white', edgecolor='gray', hatch='//', label='Invalid Sequence'))

    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()


# --- Example Usage ---

if __name__ == '__main__':
    C_TRIP_STATE_IDLE = "Idle"
    C_TRIP_STATE_EN_ROUTE = "En Route"
    C_TRIP_STATE_LOADING = "Loading"
    C_TRIP_STATE_UNLOADING = "Unloading"

    mock_trucks = {
        "Truck-Alpha": MockVehicle("Truck-Alpha",
                                   order_data={"Order-A": [10, 50]},
                                   state_data={
                                       0: [C_TRIP_STATE_IDLE, 'Depot'],
                                       10: [C_TRIP_STATE_EN_ROUTE, 'n1'],
                                       50: [C_TRIP_STATE_LOADING, 'p1'],  # This will become a Milestone
                                       50.0: [C_TRIP_STATE_UNLOADING, 'p1'],  # This will be the state with duration
                                       55: [C_TRIP_STATE_IDLE, 'd1']
                                   }
                                   ),
    }

    mock_drones = {"Drone-X": MockVehicle("Drone-X")}

    mock_state = MockGlobalState(trucks=mock_trucks, drones=mock_drones, current_time=60)

    print("--- Generating Plot 1: Vehicle Order Timelines ---")
    plot_vehicle_gantt_chart(mock_state)

    print("\n--- Generating Plot 2: Vehicle State Timelines ---")
    plot_vehicle_states(mock_state)