#!/usr/bin/env python3
"""Script to plot p_wcom (CoM position in world frame) from stepping dataset"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

def plot_p_wcom(filepath, episode_idx=0, plot_3d=True):
    """
    Plot the Center of Mass position in world frame.

    Args:
        filepath: Path to the .npz file
        episode_idx: Which episode to plot (default: 0 for first episode)
        plot_3d: Whether to include 3D trajectory plot
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found!")
        return

    # Load data
    data = np.load(filepath)

    if 'p_wcom' not in data.files:
        print("Error: 'p_wcom' not found in dataset!")
        data.close()
        return

    p_wcom = data['p_wcom']
    traj_indices = data['traj'] if 'traj' in data.files else None
    traj_dt = data['traj_dt'].item() if 'traj_dt' in data.files else 0.01

    print(f"p_wcom shape: {p_wcom.shape}")
    print(f"Number of timesteps: {p_wcom.shape[0]}")
    if len(p_wcom.shape) > 1:
        print(f"CoM dimension: {p_wcom.shape[1]}")

    # Determine episode range
    if traj_indices is not None:
        num_episodes = len(traj_indices)
        print(f"Number of episodes: {num_episodes}")

        if episode_idx >= num_episodes:
            print(f"Warning: Episode {episode_idx} not found. Using episode 0.")
            episode_idx = 0

        start_idx = traj_indices[episode_idx]
        if episode_idx < num_episodes - 1:
            end_idx = traj_indices[episode_idx + 1]
        else:
            end_idx = len(p_wcom)

        p_wcom_episode = p_wcom[start_idx:end_idx]
        print(f"\nPlotting episode {episode_idx} (timesteps {start_idx} to {end_idx})")
    else:
        p_wcom_episode = p_wcom
        print("\nPlotting all timesteps")

    # Create time array
    n_timesteps = len(p_wcom_episode)
    time = np.arange(n_timesteps) * traj_dt

    # Create figure with subplots
    if plot_3d:
        fig = plt.figure(figsize=(15, 10))

        # XYZ components over time
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(time, p_wcom_episode[:, 0], 'r-', label='X', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('X position (m)')
        ax1.set_title('CoM X Position')
        ax1.grid(True)
        ax1.legend()

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(time, p_wcom_episode[:, 1], 'g-', label='Y', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Y position (m)')
        ax2.set_title('CoM Y Position')
        ax2.grid(True)
        ax2.legend()

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(time, p_wcom_episode[:, 2], 'b-', label='Z', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Z position (m)')
        ax3.set_title('CoM Z Position')
        ax3.grid(True)
        ax3.legend()

        # 3D trajectory
        ax4 = plt.subplot(2, 2, 4, projection='3d')

        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, n_timesteps))

        # Plot trajectory with color gradient
        for i in range(n_timesteps - 1):
            ax4.plot(p_wcom_episode[i:i+2, 0],
                    p_wcom_episode[i:i+2, 1],
                    p_wcom_episode[i:i+2, 2],
                    color=colors[i], linewidth=2)

        # Mark start and end
        ax4.scatter(p_wcom_episode[0, 0], p_wcom_episode[0, 1], p_wcom_episode[0, 2],
                   c='green', s=100, marker='o', label='Start')
        ax4.scatter(p_wcom_episode[-1, 0], p_wcom_episode[-1, 1], p_wcom_episode[-1, 2],
                   c='red', s=100, marker='x', label='End')

        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('3D CoM Trajectory')
        ax4.legend()

        # Set equal aspect ratio
        max_range = np.array([
            p_wcom_episode[:, 0].max() - p_wcom_episode[:, 0].min(),
            p_wcom_episode[:, 1].max() - p_wcom_episode[:, 1].min(),
            p_wcom_episode[:, 2].max() - p_wcom_episode[:, 2].min()
        ]).max() / 2.0

        mid_x = (p_wcom_episode[:, 0].max() + p_wcom_episode[:, 0].min()) * 0.5
        mid_y = (p_wcom_episode[:, 1].max() + p_wcom_episode[:, 1].min()) * 0.5
        mid_z = (p_wcom_episode[:, 2].max() + p_wcom_episode[:, 2].min()) * 0.5

        ax4.set_xlim(mid_x - max_range, mid_x + max_range)
        ax4.set_ylim(mid_y - max_range, mid_y + max_range)
        ax4.set_zlim(mid_z - max_range, mid_z + max_range)

    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(time, p_wcom_episode[:, 0], 'r-', linewidth=2)
        axes[0].set_ylabel('X position (m)')
        axes[0].set_title('CoM X Position')
        axes[0].grid(True)

        axes[1].plot(time, p_wcom_episode[:, 1], 'g-', linewidth=2)
        axes[1].set_ylabel('Y position (m)')
        axes[1].set_title('CoM Y Position')
        axes[1].grid(True)

        axes[2].plot(time, p_wcom_episode[:, 2], 'b-', linewidth=2)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Z position (m)')
        axes[2].set_title('CoM Z Position')
        axes[2].grid(True)

    plt.tight_layout()

    # Print statistics
    print(f"\nCoM Position Statistics:")
    print(f"X: min={p_wcom_episode[:, 0].min():.3f}, max={p_wcom_episode[:, 0].max():.3f}, mean={p_wcom_episode[:, 0].mean():.3f}")
    print(f"Y: min={p_wcom_episode[:, 1].min():.3f}, max={p_wcom_episode[:, 1].max():.3f}, mean={p_wcom_episode[:, 1].mean():.3f}")
    print(f"Z: min={p_wcom_episode[:, 2].min():.3f}, max={p_wcom_episode[:, 2].max():.3f}, mean={p_wcom_episode[:, 2].mean():.3f}")
    print(f"\nTotal displacement: {np.linalg.norm(p_wcom_episode[-1] - p_wcom_episode[0]):.3f} m")

    data.close()

    # Save the plot
    output_filename = f"p_wcom_episode_{episode_idx}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_filename}")
    plt.close()

if __name__ == "__main__":
    # Default file
    default_file = "stepping_dataset_11.19_test.npz"

    # Parse arguments
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = default_file

    if len(sys.argv) > 2:
        episode_idx = int(sys.argv[2])
    else:
        episode_idx = 0

    plot_p_wcom(filepath, episode_idx=episode_idx, plot_3d=True)
