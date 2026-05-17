import os
import re
from ddls_src.functions.plotting import SimulationPlotter


def plot_training_episode_routes(full_training_name, episode_num, root_dir=".", save_to_disk=False):
    """
    Plots logs for a specific training run where the folder has a timestamp
    but the episode files do not.

    Args:
        full_training_name (str): Folder name (e.g., 'MaskablePPO_A_n69_k9_20260427_134341')
        episode_num (int): The episode number
        root_dir (str): Base directory
        save_to_disk (bool): Whether to save the plots to disk instead of showing them
    """

    # 1. Extract core name (strip timestamp _YYYYMMDD_HHMMSS)
    parts = full_training_name.split('_')
    if len(parts) > 2 and parts[-1].isdigit() and parts[-2].isdigit():
        core_name = "_".join(parts[:-2])
    else:
        core_name = full_training_name

    # 2. Construct paths
    episode_folder = os.path.join(root_dir, full_training_name, "episodes")
    filename_prefix = f"{core_name}_ep_{episode_num}"
    target_base_path = os.path.join(episode_folder, filename_prefix)

    # 3. Verification & Fallback
    v_path = f"{target_base_path}_vehicles.csv"
    if not os.path.exists(v_path):
        if os.path.exists(f"{filename_prefix}_vehicles.csv"):
            target_base_path = filename_prefix
        else:
            print(f"Error: File not found at {v_path}")
            return

    # 4. Initialize Plotter and Generate
    plotter = SimulationPlotter(base_filepath=target_base_path)

    print(f"Training Folder: {full_training_name}")
    print(f"File Prefix:     {filename_prefix}")

    # Determine output directory dynamically if saving
    out_dir = episode_folder if save_to_disk else "."

    # plotter.generate_plot('state_timeline', save_to_disk=save_to_disk, output_dir=out_dir)
    # plotter.generate_plot('cargo_gantt', save_to_disk=save_to_disk, output_dir=out_dir)

    # --- NEW PLOT CALL ---
    plotter.generate_plot('2d_routes', save_to_disk=save_to_disk, output_dir=out_dir)


# Usage:
plot_training_episode_routes("MaskablePPO_A_n32_k5_20260517_105758", 13198)