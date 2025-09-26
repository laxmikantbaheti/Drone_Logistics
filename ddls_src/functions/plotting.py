import matplotlib.pyplot as plt
from typing import Dict, Any, List


# --- Mocks to simulate your environment for a runnable example ---
class MockVehicle:
    """A mock Vehicle class to simulate your data structure."""

    def __init__(self, vehicle_id: str, data: Dict[str, List[float]]):
        self.id = vehicle_id
        # data_storage format: {'order_id': [pickup_time, delivery_time]}
        self.data_storage = data


class MockGlobalState:
    """A mock GlobalState class with separate truck and drone dicts."""

    def __init__(self, trucks: Dict[str, MockVehicle], drones: Dict[str, MockVehicle]):
        self.trucks = trucks
        self.drones = drones


# --- The Updated Plotting Function ---
def plot_vehicle_gantt_chart(global_state: Any, show_order_labels: bool = True) -> None:
    """
    Generates a Gantt chart for ALL vehicles, including unused ones.

    For each vehicle, it calculates concurrent orders and stacks them.
    Vehicles with no orders are displayed with an "(Unused)" label.

    Args:
        global_state: The main state object containing 'trucks' and 'drones' dicts.
        show_order_labels (bool): If True, adds the order ID text to each bar.
    """
    truck_data = getattr(global_state, 'trucks', {})
    drone_data = getattr(global_state, 'drones', {})

    # 1. Get a complete list of all vehicle IDs from the start
    all_vehicle_ids = sorted(list(truck_data.keys()) + list(drone_data.keys()))

    if not all_vehicle_ids:
        print("No vehicles found in global_state.")
        return

    processed_vehicles = {}
    # 2. Process orders only for vehicles that have them
    for vehicle_id in all_vehicle_ids:
        vehicle_obj = truck_data.get(vehicle_id) or drone_data.get(vehicle_id)
        orders = []
        for order_id, times in vehicle_obj.data_storage.items():
            if len(times) == 2 and times[1] > times[0]:
                orders.append({'id': order_id, 'start': times[0], 'end': times[1]})

        if not orders:
            continue

        # Arrange orders into non-overlapping "lanes"
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

    # 3. Calculate figure height based on total lanes AND unused vehicles
    total_lanes = sum(len(v['lanes']) for v in processed_vehicles.values())
    num_unused = len(all_vehicle_ids) - len(processed_vehicles)
    # Each lane/unused vehicle takes 1 unit, plus 0.5 for padding per vehicle
    fig_height_units = total_lanes + num_unused + (len(all_vehicle_ids) * 0.5)
    fig, ax = plt.subplots(figsize=(16, max(6, fig_height_units * 0.6)))

    color_map = {'truck': 'tab:blue', 'drone': 'tab:green'}
    y_ticks, y_tick_labels, y_pos = [], [], 0

    # 4. Main plotting loop: Iterate through ALL vehicles
    for vehicle_id in all_vehicle_ids:
        # CASE A: The vehicle was used and has processed data
        if vehicle_id in processed_vehicles:
            data = processed_vehicles[vehicle_id]
            lanes, num_lanes = data['lanes'], len(data['lanes'])

            y_ticks.append(y_pos + (num_lanes - 1) / 2.0)
            y_tick_labels.append(vehicle_id)

            for lane in lanes:
                bar_ranges = [(o['start'], o['end'] - o['start']) for o in lane]
                ax.broken_barh(bar_ranges, (y_pos - 0.2, 0.4),
                               facecolors=color_map[data['type']], edgecolor='black')
                if show_order_labels:
                    for order in lane:
                        start, duration = order['start'], order['end'] - order['start']
                        ax.text(start + duration / 2, y_pos, order['id'],
                                ha='center', va='center', color='white', fontsize=9)
                y_pos += 1

        # CASE B: The vehicle is unused
        else:
            y_ticks.append(y_pos)
            y_tick_labels.append(f"{vehicle_id} (Unused)")
            y_pos += 1  # Reserve one empty lane for it

        ax.axhline(y_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
        y_pos += 0.5  # Add padding between vehicles

    # 5. Format the final chart
    if y_ticks:  # Only set ticks if there are vehicles
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Vehicle ID", fontsize=12)
    ax.set_title("Vehicle Timeline", fontsize=16, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.8)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tab:blue', label='Truck'),
                       Patch(facecolor='tab:green', label='Drone')]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.show()


# --- Example Usage with an Unused Vehicle ---
if __name__ == '__main__':
    mock_trucks = {
        "Truck-Alpha": MockVehicle("Truck-Alpha", {
            "Order-A": [5, 20],  # Lane 1
            "Order-B": [10, 25],  # Lane 2 (overlaps A)
            "Order-C": [26, 40],  # Lane 1 (after A/B)
        }),
        "Truck-Beta": MockVehicle("Truck-Beta", {}),  # This truck is UNUSED
    }

    mock_drones = {
        "Drone-X": MockVehicle("Drone-X", {
            "Order-F": [8, 18],  # Lane 1
            "Order-G": [10, 22],  # Lane 2
        }),
        "Drone-Y": MockVehicle("Drone-Y", {"Order-H": [30, 45]}),  # Used
    }

    mock_state = MockGlobalState(trucks=mock_trucks, drones=mock_drones)
    plot_vehicle_gantt_chart(mock_state)