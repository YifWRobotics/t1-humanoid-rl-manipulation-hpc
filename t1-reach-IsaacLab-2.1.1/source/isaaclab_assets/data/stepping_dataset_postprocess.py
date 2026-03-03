#!/usr/bin/env python3
"""Script to inspect the shapes and contents of stepping_dataset.npz"""

import numpy as np
import os

def trim_dataset(input_file, output_file, target_timesteps=20):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        return

    data = np.load(input_file)

    if 'traj' not in data.files:
        print("Error: 'traj' key not found in dataset!")
        data.close()
        return

    traj_indices = data['traj']
    num_episodes = len(traj_indices)

    episode_lengths = []
    for i in range(num_episodes):
        start_idx = traj_indices[i]
        if i < num_episodes - 1:
            end_idx = traj_indices[i + 1]
        else:
            end_idx = len(data['q'])
        episode_lengths.append(end_idx - start_idx)

    trimmed_data = {}
    time_indexed_keys = ['q', 'qd', 'T_blf', 'T_brf', 'T_stsw', 'p_wcom', 'T_wbase', 'cmd_footstep', 'cmd_stance']
    non_time_keys = ['traj_dt']

    for key in time_indexed_keys:
        if key not in data.files:
            continue

        original_array = data[key]
        trimmed_segments = []

        for i in range(num_episodes):
            start_idx = traj_indices[i]
            available_middle = episode_lengths[i] - 2
            timesteps_to_take = min(target_timesteps, available_middle)
            segment_start = start_idx + 1
            segment_end = segment_start + timesteps_to_take
            segment = original_array[segment_start:segment_end]
            trimmed_segments.append(segment)

        trimmed_data[key] = np.concatenate(trimmed_segments, axis=0)

    new_traj_indices = []
    current_idx = 0
    for i in range(num_episodes):
        new_traj_indices.append(current_idx)
        available_middle = episode_lengths[i] - 2
        timesteps_to_take = min(target_timesteps, available_middle)
        current_idx += timesteps_to_take

    trimmed_data['traj'] = np.array(new_traj_indices, dtype=traj_indices.dtype)

    for key in non_time_keys:
        if key in data.files:
            trimmed_data[key] = data[key]

    all_processed_keys = set(time_indexed_keys + non_time_keys + ['traj'])
    for key in data.files:
        if key not in all_processed_keys:
            trimmed_data[key] = data[key]

    data.close()
    np.savez(output_file, **trimmed_data)

def pad_dataset(input_file, output_file, front_pad=25, back_pad=25):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        return

    data = np.load(input_file)

    if 'traj' not in data.files:
        print("Error: 'traj' key not found in dataset!")
        data.close()
        return

    traj_indices = data['traj']
    num_episodes = len(traj_indices)

    episode_lengths = []
    for i in range(num_episodes):
        start_idx = traj_indices[i]
        if i < num_episodes - 1:
            end_idx = traj_indices[i + 1]
        else:
            end_idx = len(data['q'])
        episode_lengths.append(end_idx - start_idx)

    padded_data = {}
    state_keys = ['q', 'qd', 'T_blf', 'T_brf', 'T_stsw', 'p_wcom', 'T_wbase']
    command_keys = ['cmd_footstep', 'cmd_stance']
    non_time_keys = ['traj_dt']

    for key in state_keys:
        if key not in data.files:
            continue

        original_array = data[key]
        padded_segments = []

        for i in range(num_episodes):
            start_idx = traj_indices[i]
            end_idx = start_idx + episode_lengths[i]
            segment = original_array[start_idx:end_idx]
            front_padding = np.repeat(segment[0:1], front_pad, axis=0)
            back_padding = np.repeat(segment[-1:], back_pad, axis=0)
            padded_segment = np.concatenate([front_padding, segment, back_padding], axis=0)
            padded_segments.append(padded_segment)

        padded_data[key] = np.concatenate(padded_segments, axis=0)

    for key in command_keys:
        if key not in data.files:
            continue

        original_array = data[key]
        padded_segments = []

        for i in range(num_episodes):
            start_idx = traj_indices[i]
            end_idx = start_idx + episode_lengths[i]
            segment = original_array[start_idx:end_idx]

            if key == 'cmd_stance':
                random_stance = np.random.choice([0, 1])
                front_padding = np.full((front_pad,) + segment.shape[1:], random_stance, dtype=segment.dtype)
                back_padding = np.zeros((back_pad,) + segment.shape[1:], dtype=segment.dtype)
            else:
                front_padding = np.zeros((front_pad,) + segment.shape[1:], dtype=segment.dtype)
                back_padding = np.zeros((back_pad,) + segment.shape[1:], dtype=segment.dtype)

            padded_segment = np.concatenate([front_padding, segment, back_padding], axis=0)
            padded_segments.append(padded_segment)

        padded_data[key] = np.concatenate(padded_segments, axis=0)

    new_traj_indices = []
    current_idx = 0
    new_episode_length = episode_lengths[0] + front_pad + back_pad

    for i in range(num_episodes):
        new_traj_indices.append(current_idx)
        current_idx += new_episode_length

    padded_data['traj'] = np.array(new_traj_indices, dtype=traj_indices.dtype)

    for key in non_time_keys:
        if key in data.files:
            padded_data[key] = data[key]

    all_processed_keys = set(state_keys + command_keys + non_time_keys + ['traj'])
    for key in data.files:
        if key not in all_processed_keys:
            padded_data[key] = data[key]

    data.close()
    np.savez(output_file, **padded_data)

def inspect_npz(filepath):
    """Inspect the contents of an NPZ file and print detailed information."""

    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found!")
        return

    print(f"Inspecting: {filepath}")
    print("=" * 80)

    # Load the NPZ file
    data = np.load(filepath)

    print(f"\nFile contains {len(data.files)} arrays/values\n")

    # Expected structure
    expected_keys = {
        'q': 'joint positions with base',
        'qd': 'joint velocities',
        'T_blf': 'body frame to left foot frame transform (x, y, z, yaw)',
        'T_brf': 'body frame to right foot frame transform (x, y, z, yaw)',
        'T_stsw': 'stance foot to swing foot transform (x, y, z, yaw)',
        'p_wcom': 'CoM position in world frame',
        'T_wbase': 'base transform in world frame (x, y, z, qw, qx, qy, qz)',
        'cmd_footstep': '[x, y, sin(yaw), cos(yaw)] in stance foot frame',
        'cmd_stance': '0=left stance, 1=right stance',
        'traj': 'starting indices of each trajectory',
        'traj_dt': 'time step between frames'
    }

    # Print details for each array
    print("Contents:")
    print("-" * 80)

    for key in data.files:
        value = data[key]
        description = expected_keys.get(key, 'Unknown')

        print(f"\n{key}:")
        print(f"  Description: {description}")
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")

        # Handle scalar values
        if value.shape == ():
            print(f"  Value: {value.item()}")
        else:
            print(f"  Min: {np.min(value):.6f}")
            print(f"  Max: {np.max(value):.6f}")
            print(f"  Mean: {np.mean(value):.6f}")
            print(f"  First few values: {value.flat[:5]}")

    print("\n" + "=" * 80)

    # Check for missing expected keys
    missing_keys = set(expected_keys.keys()) - set(data.files)
    if missing_keys:
        print(f"\nWarning: Missing expected keys: {missing_keys}")

    # Check for unexpected keys
    unexpected_keys = set(data.files) - set(expected_keys.keys())
    if unexpected_keys:
        print(f"\nNote: Found unexpected keys: {unexpected_keys}")

    # Print summary statistics
    print("\nSummary:")
    print("-" * 80)

    # Determine n (number of frames)
    n_samples = None
    for key in ['q', 'qd', 'T_blf', 'T_brf', 'T_stsw', 'p_wcom', 'T_wbase', 'cmd_footstep', 'cmd_stance']:
        if key in data.files and len(data[key].shape) > 0:
            n_samples = data[key].shape[0]
            break

    if n_samples:
        print(f"Number of frames (n): {n_samples}")

    if 'q' in data.files:
        nq = data['q'].shape[1] if len(data['q'].shape) > 1 else 'N/A'
        print(f"Number of joint positions (nq): {nq}")

    if 'qd' in data.files:
        nv = data['qd'].shape[1] if len(data['qd'].shape) > 1 else 'N/A'
        print(f"Number of joint velocities (nv): {nv}")

    if 'traj' in data.files:
        k = data['traj'].shape[0]
        print(f"Number of trajectories (k): {k}")
        print(f"Trajectory start indices: {data['traj']}")

    if 'traj_dt' in data.files:
        dt = data['traj_dt'].item() if data['traj_dt'].shape == () else data['traj_dt'][0]
        print(f"Time step (traj_dt): {dt} seconds")
        if n_samples and dt:
            print(f"Total duration: {n_samples * dt:.2f} seconds")

    if 'traj' in data.files:
        # Determine episode length from first episode
        traj_indices = data['traj']
        if len(traj_indices) > 1:
            first_episode_length = traj_indices[1] - traj_indices[0]
        else:
            first_episode_length = len(data['q']) if 'q' in data.files else 37

        print(f"\nFirst episode data (timesteps 0-{first_episode_length}):")
        print("-" * 80)
        if 'T_blf' in data.files:
            print(f"T_blf: {data['T_blf'][:first_episode_length]}")
        if 'T_brf' in data.files:
            print(f"T_brf: {data['T_brf'][:first_episode_length]}")
        if 'T_stsw' in data.files:
            print(f"T_stsw: {data['T_stsw'][:first_episode_length]}")
        if 'p_wcom' in data.files:
            print(f"p_wcom: {data['p_wcom'][:first_episode_length]}")
        if 'T_wbase' in data.files:
            print(f"T_wbase: {data['T_wbase'][:first_episode_length]}")
        if 'cmd_footstep' in data.files:
            print(f"cmd_footstep: {data['cmd_footstep'][:first_episode_length]}")
        if 'cmd_stance' in data.files:
            print(f"cmd_stance: {data['cmd_stance'][:first_episode_length]}")
        print(f"traj: {data['traj']}")
        if 'traj_dt' in data.files:
            dt_value = data['traj_dt'].item() if data['traj_dt'].shape == () else data['traj_dt']
            print(f"traj_dt: {dt_value}")

    data.close()
    print()

if __name__ == "__main__":
    import sys
    import os

    # Default file paths
    default_input = "stepping_dataset_11.11_rand.npz"
    default_trimmed = "stepping_dataset_trimmed.npz"
    default_output = "stepping_dataset_final.npz"
    default_trim_timesteps = 20
    default_front_pad = 25
    default_back_pad = 25

    # Parse arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = default_output

    if len(sys.argv) > 3:
        trim_timesteps = int(sys.argv[3])
    else:
        trim_timesteps = default_trim_timesteps

    if len(sys.argv) > 4:
        front_pad = int(sys.argv[4])
    else:
        front_pad = default_front_pad

    if len(sys.argv) > 5:
        back_pad = int(sys.argv[5])
    else:
        back_pad = default_back_pad
    inspect_npz(default_input)
    # trimmed_file = default_trimmed

    # trim_dataset(input_file, trimmed_file, trim_timesteps)
    # pad_dataset(trimmed_file, output_file, front_pad, back_pad)

    # if os.path.exists(trimmed_file):
    #     os.remove(trimmed_file)
