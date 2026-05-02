import os
import re
from ddls_src.functions.plotting import SimulationPlotter

def plot_training_episode(full_training_name, episode_num, root_dir="."):
    """
    Plots logs for a specific training run where the folder has a timestamp
    but the episode files do not.

    Args:
        full_training_name (str): Folder name (e.g., 'MaskablePPO_A_n69_k9_20260427_134341')
        episode_num (int): The episode number
        root_dir (str): Base directory
    """

    # 1. Extract core name (strip timestamp _YYYYMMDD_HHMMSS)
    # This splits from the right and takes everything before the last two segments
    parts = full_training_name.split('_')
    if len(parts) > 2 and parts[-1].isdigit() and parts[-2].isdigit():
        core_name = "_".join(parts[:-2])
    else:
        core_name = full_training_name

    # 2. Construct paths
    # Folder: root/MaskablePPO_A_n69_k9_20260427_134341/episode/
    # File:   MaskablePPO_A_n69_k9_ep_53_vehicles.csv
    episode_folder = os.path.join(root_dir, full_training_name, "episodes")
    filename_prefix = f"{core_name}_ep_{episode_num}"
    target_base_path = os.path.join(episode_folder, filename_prefix)

    # 3. Verification & Fallback
    v_path = f"{target_base_path}_vehicles.csv"
    if not os.path.exists(v_path):
        # Fallback to current directory for the uploaded sample file
        if os.path.exists(f"{filename_prefix}_vehicles.csv"):
            target_base_path = filename_prefix
        else:
            print(f"Error: File not found at {v_path}")
            return

    # 4. Initialize Plotter and Generate
    plotter = SimulationPlotter(base_filepath=target_base_path)

    print(f"Training Folder: {full_training_name}")
    print(f"File Prefix:     {filename_prefix}")

    plotter.generate_plot('state_timeline')
    plotter.generate_plot('cargo_gantt')

# Usage:
plot_training_episode("MaskablePPO_A_n69_k9_20260428_131949", 1)