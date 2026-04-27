import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_reward_history(file_path, window_size=50):
    """
    Reads a CSV file containing 'episode' and 'reward' columns and plots
    the reward over episodes with a smoothing convolution filter.

    Args:
        file_path (str): Path to the CSV file.
        window_size (int): The size of the window for the smoothing filter.
    """
    # Load the data
    df = pd.read_csv(file_path)

    episodes = df['episode'].values
    rewards = df['reward'].values

    # Calculate the moving average using convolution
    # We use a boxcar filter (an array of ones divided by the window size)
    weights = np.ones(window_size) / window_size
    smoothed_rewards = np.convolve(rewards, weights, mode='valid')

    # Adjust episode array to match the 'valid' convolution output size
    # This aligns the smoothed value with the end of its respective window
    smoothed_episodes = episodes[window_size - 1:]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, color='skyblue', alpha=0.3, label='Original Reward')
    plt.plot(smoothed_episodes, smoothed_rewards, color='darkblue', linewidth=2,
             label=f'Smoothed (Window={window_size})')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward over Episodes')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Example usage:
plot_reward_history('C:\Baheti\Developments\Drone_Logistics\\results\MaskablePPO_20260425_000507\MaskablePPO_episode_log.csv', window_size=800)