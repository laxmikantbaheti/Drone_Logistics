import json
import math


def create_instance():
    # 1. Define Nodes (50 Nodes: 1 Depot, 2 Micro-Hubs, 47 Customers)
    nodes = [
        {"id": 0, "coords": [50, 50], "type": "depot", "is_loadable": True, "is_unloadable": True,
         "is_charging_station": True},
        # Customers 1-10
        {"id": 1, "coords": [12, 18], "type": "customer"}, {"id": 2, "coords": [92, 14], "type": "customer"},
        {"id": 3, "coords": [33, 82], "type": "customer"}, {"id": 4, "coords": [15, 45], "type": "customer"},
        {"id": 5, "coords": [67, 22], "type": "customer"}, {"id": 6, "coords": [88, 88], "type": "customer"},
        {"id": 7, "coords": [10, 90], "type": "customer"}, {"id": 8, "coords": [45, 12], "type": "customer"},
        {"id": 9, "coords": [72, 75], "type": "customer"}, {"id": 10, "coords": [25, 25], "type": "customer"},
        # Customers 11-20
        {"id": 11, "coords": [55, 60], "type": "customer"}, {"id": 12, "coords": [80, 40], "type": "customer"},
        {"id": 13, "coords": [30, 70], "type": "customer"}, {"id": 14, "coords": [95, 5], "type": "customer"},
        {"id": 15, "coords": [5, 95], "type": "customer"}, {"id": 16, "coords": [60, 85], "type": "customer"},
        {"id": 17, "coords": [22, 55], "type": "customer"}, {"id": 18, "coords": [40, 40], "type": "customer"},
        {"id": 19, "coords": [77, 12], "type": "customer"}, {"id": 20, "coords": [90, 50], "type": "customer"},
        # Customers 21-30
        {"id": 21, "coords": [18, 33], "type": "customer"}, {"id": 22, "coords": [52, 92], "type": "customer"},
        {"id": 23, "coords": [85, 65], "type": "customer"}, {"id": 24, "coords": [8, 8], "type": "customer"},
        {"id": 25, "coords": [35, 35], "type": "customer"}, {"id": 26, "coords": [65, 35], "type": "customer"},
        {"id": 27, "coords": [35, 65], "type": "customer"}, {"id": 28, "coords": [65, 65], "type": "customer"},
        {"id": 29, "coords": [98, 98], "type": "customer"}, {"id": 30, "coords": [2, 2], "type": "customer"},
        # Customers 31-40
        {"id": 31, "coords": [42, 58], "type": "customer"}, {"id": 32, "coords": [58, 42], "type": "customer"},
        {"id": 33, "coords": [14, 76], "type": "customer"}, {"id": 34, "coords": [82, 28], "type": "customer"},
        {"id": 35, "coords": [29, 11], "type": "customer"}, {"id": 36, "coords": [91, 79], "type": "customer"},
        {"id": 37, "coords": [50, 90], "type": "customer"}, {"id": 38, "coords": [50, 10], "type": "customer"},
        {"id": 39, "coords": [10, 50], "type": "customer"}, {"id": 40, "coords": [90, 50], "type": "customer"},
        # Customers 41-47
        {"id": 41, "coords": [20, 80], "type": "customer"}, {"id": 42, "coords": [80, 20], "type": "customer"},
        {"id": 43, "coords": [44, 44], "type": "customer"}, {"id": 44, "coords": [66, 66], "type": "customer"},
        {"id": 45, "coords": [11, 66], "type": "customer"}, {"id": 46, "coords": [77, 33], "type": "customer"},
        {"id": 47, "coords": [33, 77], "type": "customer"},
        # Micro Hubs
        {"id": 48, "coords": [25, 75], "type": "micro_hub"},
        {"id": 49, "coords": [75, 25], "type": "micro_hub"}
    ]

    # 2. Define Fleet
    trucks = [
        {"id": 101, "start_node_id": 0, "max_payload_capacity": 4, "max_speed": 60.0},
        {"id": 102, "start_node_id": 0, "max_payload_capacity": 4, "max_speed": 60.0},
        {"id": 103, "start_node_id": 0, "max_payload_capacity": 4, "max_speed": 60.0},
        {"id": 104, "start_node_id": 0, "max_payload_capacity": 4, "max_speed": 60.0},
        {"id": 105, "start_node_id": 0, "max_payload_capacity": 2, "max_speed": 50.0},
        {"id": 106, "start_node_id": 0, "max_payload_capacity": 2, "max_speed": 50.0},
        {"id": 107, "start_node_id": 0, "max_payload_capacity": 2, "max_speed": 50.0},
        {"id": 108, "start_node_id": 0, "max_payload_capacity": 2, "max_speed": 50.0}
    ]

    drones = [
        {"id": 201, "start_node_id": 0, "max_payload_capacity": 2, "max_speed": 75.0},
        {"id": 202, "start_node_id": 0, "max_payload_capacity": 2, "max_speed": 80.0},
        {"id": 203, "start_node_id": 0, "max_payload_capacity": 2, "max_speed": 80.0}
    ]

    # 3. Define Orders
    orders = [
        {"id": 1001, "p_pickup_node_id": 0, "p_delivery_node_id": 1},
        {"id": 1002, "p_pickup_node_id": 1, "p_delivery_node_id": 5},
        {"id": 1003, "p_pickup_node_id": 2, "p_delivery_node_id": 4},
        {"id": 1004, "p_pickup_node_id": 0, "p_delivery_node_id": 6},
        {"id": 1005, "p_pickup_node_id": 3, "p_delivery_node_id": 7},
        {"id": 1006, "p_pickup_node_id": 5, "p_delivery_node_id": 8},
        {"id": 1007, "p_pickup_node_id": 9, "p_delivery_node_id": 10},
        {"id": 1008, "p_pickup_node_id": 0, "p_delivery_node_id": 11},
        {"id": 1009, "p_pickup_node_id": 12, "p_delivery_node_id": 0},
        {"id": 1010, "p_pickup_node_id": 4, "p_delivery_node_id": 3},
        {"id": 1011, "p_pickup_node_id": 13, "p_delivery_node_id": 14},
        {"id": 1012, "p_pickup_node_id": 15, "p_delivery_node_id": 16},
        {"id": 1013, "p_pickup_node_id": 17, "p_delivery_node_id": 18},
        {"id": 1014, "p_pickup_node_id": 19, "p_delivery_node_id": 20},
        {"id": 1015, "p_pickup_node_id": 21, "p_delivery_node_id": 22},
        {"id": 1016, "p_pickup_node_id": 0, "p_delivery_node_id": 23},
        {"id": 1017, "p_pickup_node_id": 24, "p_delivery_node_id": 0},
        {"id": 1018, "p_pickup_node_id": 25, "p_delivery_node_id": 26},
        {"id": 1019, "p_pickup_node_id": 27, "p_delivery_node_id": 28},
        {"id": 1020, "p_pickup_node_id": 29, "p_delivery_node_id": 30},
        {"id": 1021, "p_pickup_node_id": 31, "p_delivery_node_id": 32},
        {"id": 1022, "p_pickup_node_id": 33, "p_delivery_node_id": 34},
        {"id": 1023, "p_pickup_node_id": 35, "p_delivery_node_id": 36},
        {"id": 1024, "p_pickup_node_id": 37, "p_delivery_node_id": 48},
        {"id": 1025, "p_pickup_node_id": 38, "p_delivery_node_id": 49},
        {"id": 1026, "p_pickup_node_id": 48, "p_delivery_node_id": 39},
        {"id": 1027, "p_pickup_node_id": 49, "p_delivery_node_id": 40},
        {"id": 1028, "p_pickup_node_id": 41, "p_delivery_node_id": 42},
        {"id": 1029, "p_pickup_node_id": 43, "p_delivery_node_id": 44},
        {"id": 1030, "p_pickup_node_id": 45, "p_delivery_node_id": 46},
        {"id": 1031, "p_pickup_node_id": 47, "p_delivery_node_id": 1},
        {"id": 1032, "p_pickup_node_id": 2, "p_delivery_node_id": 3},
        {"id": 1033, "p_pickup_node_id": 4, "p_delivery_node_id": 5},
        {"id": 1034, "p_pickup_node_id": 0, "p_delivery_node_id": 49},
        {"id": 1035, "p_pickup_node_id": 48, "p_delivery_node_id": 0}
    ]

    # 4. Generate Distance Matrices
    # Air = Euclidean
    # Ground = Manhattan (City Block)
    air_dist_matrix = {}
    ground_dist_matrix = {}

    for i_node in nodes:
        i_id = str(i_node["id"])
        air_dist_matrix[i_id] = {}
        ground_dist_matrix[i_id] = {}

        ix, iy = i_node["coords"]

        for j_node in nodes:
            j_id = str(j_node["id"])
            jx, jy = j_node["coords"]

            # Euclidean
            dist_air = math.sqrt((ix - jx) ** 2 + (iy - jy) ** 2)

            # Manhattan
            dist_ground = abs(ix - jx) + abs(iy - jy)

            air_dist_matrix[i_id][j_id] = round(dist_air, 1)
            ground_dist_matrix[i_id][j_id] = round(float(dist_ground), 1)

    # 5. Compile Final JSON Structure
    data = {
        "nodes": nodes,
        "edges": [],
        "trucks": trucks,
        "drones": drones,
        "micro_hubs": [],
        "orders": orders,
        "initial_time": 0.0,
        "air_distance_matrix": air_dist_matrix,
        "ground_distance_matrix": ground_dist_matrix
    }

    # 6. Write to File
    filename = "ddls_src/config/large_instance.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"File '{filename}' has been created successfully.")


if __name__ == "__main__":
    create_instance()
