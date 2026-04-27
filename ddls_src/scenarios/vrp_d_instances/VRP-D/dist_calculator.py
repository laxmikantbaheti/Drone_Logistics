import math
import os

def generate_dist_file_with_headers(vrp_filename):
    nodes = {}
    dimension = 0
    name = ""

    try:
        with open(vrp_filename, 'r') as f:
            lines = f.readlines()

            coord_section = False
            for line in lines:
                line = line.strip()
                if not line: continue

                if line.startswith("NAME"):
                    name = line.split(":")[-1].strip()
                elif line.startswith("DIMENSION"):
                    dimension = int(line.split(":")[-1].strip())
                elif line.startswith("NODE_COORD_SECTION"):
                    coord_section = True
                    continue
                elif line.startswith("DEMAND_SECTION") or line.startswith("DEPOT_SECTION") or line == "EOF":
                    coord_section = False
                    continue

                if coord_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        nodes[node_id] = (x, y)

        node_ids = sorted(nodes.keys())
        n = len(node_ids)
        base_name = os.path.splitext(os.path.basename(vrp_filename))[0]
        output_filename = f"{base_name}.dist"

        # Width for formatting alignment
        w = 8

        with open(output_filename, 'w') as out_f:
            out_f.write(f"NAME : {name}\n")
            out_f.write(f"DIMENSION : {dimension}\n\n")

            # Function to write a labeled matrix
            def write_labeled_matrix(label, matrix_type):
                out_f.write(f"{label}\n")

                # Write Column Headers
                header_row = " " * w  # Empty corner space
                for nid in node_ids:
                    header_row += f"{str(nid):>{w}}"
                out_f.write(header_row + "\n")

                # Write Rows with Row Headers
                for i, row_id in enumerate(node_ids):
                    row_str = f"{str(row_id):>{w}}"  # Row index
                    for j, col_id in enumerate(node_ids):
                        x1, y1 = nodes[row_id]
                        x2, y2 = nodes[col_id]

                        if matrix_type == "AIR":
                            val = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                        else:
                            val = abs(x1 - x2) + abs(y1 - y2)

                        row_str += f"{round(val, 2):>{w}}"
                    out_f.write(row_str + "\n")
                out_f.write("\n")

            write_labeled_matrix("AIR_DISTANCE_MATRIX (EUCLIDEAN)", "AIR")
            write_labeled_matrix("GROUND_DISTANCE_MATRIX (MANHATTAN)", "GROUND")

        print(f"Successfully generated labeled file: {output_filename}")

    except Exception as e:
        print(f"Error processing {vrp_filename}: {e}")

if __name__ == "__main__":
    # Get all .vrp files in the current directory
    current_directory = os.getcwd()
    vrp_files = [f for f in os.listdir(current_directory) if f.lower().endswith('.vrp')]

    if not vrp_files:
        print("No .vrp files found in the current folder.")
    else:
        print(f"Found {len(vrp_files)} .vrp files. Starting processing...\n")
        for file in vrp_files:
            generate_dist_file_with_headers(file)
        print("\nAll files processed.")