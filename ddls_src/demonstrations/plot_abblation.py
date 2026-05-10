import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_dual_axis_excel(path, x_axis_name, y_axes_config):
    """
    Plots Excel data with selectable plot types and custom alpha (transparency).
    """
    df = pd.read_excel(path)
    x_data = df[x_axis_name].astype(str)
    x_indices = np.arange(len(x_data))

    fig, ax1 = plt.subplots(figsize=(14, 8))

    all_handles = []
    all_labels = []
    axis_titles = list(y_axes_config.keys())

    def draw_plots(ax, title, config, color_map, is_secondary=False):
        plot_type = config.get("type", "line").lower()
        columns = config.get("columns", [])
        # Get alpha from config, or use defaults (0.7 for bars, 1.0 for others)
        alpha_val = config.get("alpha", 0.7 if plot_type == 'bar' else 1.0)

        num_cols = len(columns)
        width = 0.8 / num_cols if plot_type == 'bar' else 0

        handles = []
        for i, col in enumerate(columns):
            label = col
            # Use specific colors for primary (Blues) and secondary (Reds)
            color = color_map(0.5 + (i / (num_cols + 1)) * 0.5)

            if plot_type == 'bar':
                offset = (i - (num_cols - 1) / 2) * width
                h = ax.bar(x_indices + offset, df[col], width=width,
                           label=label, alpha=alpha_val, color=color)
                handles.append(h)
            elif plot_type == 'point':
                h, = ax.plot(x_indices, df[col], marker='o', linestyle='None',
                             label=label, alpha=alpha_val, color=color)
                handles.append(h)
            else:  # Line
                h, = ax.plot(x_indices, df[col], marker='o', linestyle='-',
                             label=label, alpha=alpha_val, color=color)
                handles.append(h)

        ax.set_ylabel(title, fontweight='bold', color='tab:blue' if not is_secondary else 'tab:red')
        return handles, columns

    # 1. Primary Axis (Left)
    h1, l1 = draw_plots(ax1, axis_titles[0], y_axes_config[axis_titles[0]], plt.cm.Blues)
    all_handles.extend(h1)
    all_labels.extend(l1)

    # 2. Secondary Axis (Right)
    if len(axis_titles) > 1:
        ax2 = ax1.twinx()
        h2, l2 = draw_plots(ax2, axis_titles[1], y_axes_config[axis_titles[1]], plt.cm.Reds, is_secondary=True)
        all_handles.extend(h2)
        all_labels.extend(l2)
        ax2.spines['right'].set_color('tab:red')
        ax1.spines['left'].set_color('tab:blue')

    # Formatting
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_data, rotation=90)
    ax1.set_xlabel(x_axis_name, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Legend
    ax1.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    plt.title(f"Instance Analysis: {', '.join(axis_titles)}", pad=25, fontsize=14)

    plt.tight_layout()
    plt.show()


# --- CONFIGURATION WITH ALPHA ---
# y_config = {
#     "Distance": {
#         "columns": ["Multi-Use (RL) (2 D, 2 M)", "Multi-Use (RL) (0 D, 0 M)", "Best Known (Single Trip)"],
#         "type": "bar",
#         "alpha": 1  # Set transparency for this group
#     },
#     "% Improvement": {
#         "columns": ["% Improvement"],
#         "type": "line",
#         "alpha": 0.7  # Fully opaque
#     }
# }

# y_config = {
#     "Distance": {
#         "columns": ["Multi-Use (RL) (2 D, 2 M)"],
#         "type": "bar",
#         "alpha": 1  # Set transparency for this group
#     },
#     "Computation Time": {
#         "columns": ["Computation Time"],
#         "type": "line",
#         "alpha": 0.7  # Fully opaque
#     }
# }

y_config = {
    "Distance": {
        "columns": ["Multi-Use (RL) (2 D, 2 M)", "Multi-Use (RL) (0 D, 0 M)", "Best Known (Single Trip)"],
        "type": "bar",
        "alpha": 1  # Set transparency for this group
    },
}

# Example usage:
plot_dual_axis_excel("C:\\Users\\Goldscheid\\Desktop\\Trial Results.xlsx", "Instance", y_config)