import numpy as np
from sklearn.cluster import DBSCAN
import pulp
import math
import vrplib
import os


def calc_dist(p1, p2):
    """Calculates the 2D Euclidean distance between two coordinate points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_candidate_hubs(b2b_customers, eps, min_samples=3):
    coords = np.array(b2b_customers)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    candidate_hubs = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_points = coords[labels == cluster_id]
        centroid = tuple(cluster_points.mean(axis=0))
        candidate_hubs.append(centroid)

    return candidate_hubs


def optimize_hub_locations(b2b_customers, candidate_hubs, warehouse, R_max, alpha=1, beta=1, penalty=10000):
    prob = pulp.LpProblem("AI4Drone_Facility_Location", pulp.LpMinimize)

    C = range(len(b2b_customers))
    M = range(len(candidate_hubs))

    h = pulp.LpVariable.dicts("Hub_Open", M, cat='Binary')
    a = pulp.LpVariable.dicts("Assign_Hub", (M, C), 0, 1, cat='Continuous')
    b = pulp.LpVariable.dicts("Assign_Warehouse", (range(1), C), cat='Binary')
    p = pulp.LpVariable.dicts("Penalty_Unassigned", C, cat='Binary')

    objective = []
    for j in M:
        objective.append(alpha * calc_dist(warehouse, candidate_hubs[j]) * h[j])

    for k in C:
        objective.append(beta * calc_dist(warehouse, b2b_customers[k]) * b[0][k])
        for j in M:
            objective.append(beta * calc_dist(candidate_hubs[j], b2b_customers[k]) * a[j][k])
        objective.append(penalty * p[k])

    prob += pulp.lpSum(objective)

    for k in C:
        prob += pulp.lpSum([a[j][k] for j in M]) + b[0][k] + p[k] == 1

        if calc_dist(warehouse, b2b_customers[k]) > R_max:
            prob += b[0][k] == 0

        for j in M:
            prob += a[j][k] <= h[j]
            if calc_dist(candidate_hubs[j], b2b_customers[k]) > R_max:
                prob += a[j][k] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    active_hubs = [candidate_hubs[j] for j in M if h[j].varValue > 0.5]

    return len(active_hubs), active_hubs


def process_and_generate_files(vrp_filename, eps_grid=15.0, r_max_grid=30.0):
    try:
        # 1. Parse Instance
        instance = vrplib.read_instance(vrp_filename)
        coords = instance['node_coord']
        name = instance.get('name', os.path.basename(vrp_filename))
        dimension = instance.get('dimension', len(coords))

        # In Augerat, Node 1 (index 0) is the warehouse
        warehouse = tuple(coords[0])
        b2b_customers = [tuple(c) for c in coords[1:]]

        # 2. Optimize Hubs
        candidate_hubs = generate_candidate_hubs(b2b_customers, eps=eps_grid, min_samples=3)
        num_hubs, microhubs = optimize_hub_locations(b2b_customers, candidate_hubs, warehouse, r_max_grid)

        # 3. File Setup
        base_name = os.path.splitext(vrp_filename)[0]
        dist_output_filename = f"{base_name}.dist"
        mh_output_filename = f"{base_name}.mh"
        w = 8

        # 4. Generate .dist File
        with open(dist_output_filename, 'w') as out_f:
            out_f.write(f"NAME : {name}\n")
            out_f.write(f"DIMENSION : {dimension}\n\n")

            def write_labeled_matrix(label, matrix_type):
                out_f.write(f"{label}\n")

                # Column headers (1-indexed for standard VRP format)
                header_row = " " * w
                for i in range(dimension):
                    header_row += f"{str(i + 1):>{w}}"
                out_f.write(header_row + "\n")

                for i in range(dimension):
                    row_str = f"{str(i + 1):>{w}}"
                    for j in range(dimension):
                        x1, y1 = coords[i]
                        x2, y2 = coords[j]
                        if matrix_type == "AIR":
                            val = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                        else:
                            val = abs(x1 - x2) + abs(y1 - y2)
                        row_str += f"{round(val, 2):>{w}}"
                    out_f.write(row_str + "\n")
                out_f.write("\n")

            write_labeled_matrix("AIR_DISTANCE_MATRIX (EUCLIDEAN)", "AIR")
            write_labeled_matrix("GROUND_DISTANCE_MATRIX (MANHATTAN)", "GROUND")

        # 5. Generate .mh File
        with open(mh_output_filename, 'w') as mh_f:
            mh_f.write(f"NAME : {name}_MICROHUBS\n")
            mh_f.write(f"NUM_MICROHUBS : {num_hubs}\n")
            mh_f.write("MICROHUB_COORD_SECTION\n")
            for idx, (hx, hy) in enumerate(microhubs, start=1):
                mh_f.write(f"MH_{idx} {round(hx, 2)} {round(hy, 2)}\n")
            mh_f.write("EOF\n")

        print(f"[{name}] Generated {num_hubs} hubs -> {dist_output_filename} & {mh_output_filename}")

    except Exception as e:
        print(f"Error processing {vrp_filename}: {e}")


if __name__ == "__main__":
    target_directory = 'ddls_src/scenarios/vrp_d_instances/VRP-D/'

    # Check if target directory exists, otherwise fallback to current directory
    if not os.path.exists(target_directory):
        target_directory = os.getcwd()

    vrp_files = [os.path.join(target_directory, f) for f in os.listdir(target_directory) if f.lower().endswith('.vrp')]

    if not vrp_files:
        print(f"No .vrp files found in {target_directory}.")
    else:
        print(f"Found {len(vrp_files)} .vrp files. Starting processing...\n")
        for file in vrp_files:
            # Adjust eps_grid and r_max_grid as needed
            process_and_generate_files(file, eps_grid=15.0, r_max_grid=30.0)
        print("\nAll files processed.")