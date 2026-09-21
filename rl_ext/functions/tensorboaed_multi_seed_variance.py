import os
import glob
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_interpolate_runs(
        csv_paths: List[str],
        num_grid_points: int = 1000,
        smooth_weight: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads TensorBoard CSVs, aligns steps onto a unified evaluation grid,
    and returns (common_steps, seed_matrix).

    seed_matrix shape: (n_seeds, num_grid_points)
    """
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # Normalize column names for TensorBoard exports
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        step_col = next((c for c in df.columns if "step" in c), "step")
        # Match 'result', 'value', or the scalar metric column
        val_col = next((c for c in df.columns if c in ["result", "value", "scalar"] or "reward" in c), df.columns[-1])

        df = df[[step_col, val_col]].dropna().sort_values(by=step_col)

        # Optional exponential moving average per seed
        if smooth_weight > 0.0:
            df[val_col] = df[val_col].ewm(alpha=(1.0 - smooth_weight)).mean()

        dfs.append((df[step_col].values, df[val_col].values))

    # Determine common step boundaries across all 5 runs
    min_step = max(steps[0] for steps, _ in dfs)
    max_step = min(steps[-1] for steps, _ in dfs)
    common_steps = np.linspace(min_step, max_step, num_grid_points)

    # 1D linear interpolation to align seeds onto the identical step array
    interpolated_runs = []
    for steps, values in dfs:
        interp_vals = np.interp(common_steps, steps, values)
        interpolated_runs.append(interp_vals)

    return common_steps, np.array(interpolated_runs)


def plot_multi_seed_curve(
        csv_paths: List[str],
        output_filename: str = "multi_seed_training_curve.png",
        metric_name: str = "Episodic Return / Reward",
        smooth_weight: float = 0.6
):
    """
    Plots the mean curve with standard deviation fill band across all seed CSVs.
    """
    common_steps, seed_matrix = load_and_interpolate_runs(
        csv_paths=csv_paths,
        num_grid_points=1000,
        smooth_weight=smooth_weight
    )

    # Calculate statistics across runs (axis 0 = seeds)
    mean_curve = np.mean(seed_matrix, axis=0)
    std_curve = np.std(seed_matrix, axis=0)
    upper_bound = mean_curve + std_curve
    lower_bound = mean_curve - std_curve

    # Figure styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    # 1. Plot individual seed runs faintly in the background
    for i in range(seed_matrix.shape[0]):
        ax.plot(common_steps, seed_matrix[i], color="#1f77b4", alpha=0.15, lw=1.0)

    # 2. Plot aggregate mean curve
    ax.plot(common_steps, mean_curve, color="#1f77b4", lw=2.2, label=f"Mean (5 Seeds)")

    # 3. Plot aggregate variance fill band (+-1 standard deviation)
    ax.fill_between(
        common_steps,
        lower_bound,
        upper_bound,
        color="#1f77b4",
        alpha=0.22,
        edgecolor="none",
        label=r"$\pm 1\ \sigma$ Variance Band"
    )

    # Formatting axes and labels
    ax.set_title("Maskable PPO Performance (Fixed Sim Instance, 5 Model Seeds)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Timesteps", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_xlim(common_steps[0], common_steps[-1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"[Plot Saved] Figure saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    # Provide the 5 CSV paths downloaded from TensorBoard
    csv_files = [
        "run_seed_42.csv",
        "run_seed_101.csv",
        "run_seed_202.csv",
        "run_seed_303.csv",
        "run_seed_404.csv"
    ]

    # Verify files exist before plotting
    valid_files = [f for f in csv_files if os.path.exists(f)]

    if len(valid_files) != 5:
        print(f"[Warning] Found {len(valid_files)}/5 files. Check paths if files are missing:")
        for f in csv_files:
            print(f" - {f}: {'FOUND' if os.path.exists(f) else 'MISSING'}")

    if valid_files:
        plot_multi_seed_curve(
            csv_paths=valid_files,
            output_filename="ppo_multi_seed_variance.png",
            metric_name="Evaluation Reward",
            smooth_weight=0.6  # Adjust between 0.0 (no smoothing) and 0.9 (heavy smoothing)
        )