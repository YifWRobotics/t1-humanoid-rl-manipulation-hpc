#!/bin/bash

# Remote and local paths
REMOTE_USER="zgu78"
REMOTE_HOST="login-ice.pace.gatech.edu"
REMOTE_DIR="/home/hice1/zgu78/scratch/t1_rl/IsaacLab/isaaclab/logs/rsl_rl/t1_foot_track"
LOCAL_DIR="/home/zimengchai/Documents/IsaacLab/logs/rsl_rl/t1_foot_track"

# List of folders to copy
FOLDERS=(
    "lessqposbaseline_data12.09less_AddObsNoise_20251215_1559_ice_h100x1"
    "lessqposbaseline_data12.09less_avgYaw0.1_20251215_1552_ice_h100x1"
    "lessqposbaseline_data12.09less_avgYaw1_20251215_1551_ice_h100x1"
    "lessqposbaseline_data12.09less_avgYaw_20251215_1547_ice_h100x1"
    "lessqposbaseline_data12.09less_avgYaw4_20251215_1553_ice_h100x1"
    "lessqposbaseline_data12.09less_noJointRand_20251215_1607_ice_h100x1"
    "lessqposbaseline_data12.09less_noSurvival_20251215_1604_ice_h100x1"
    "lessqposbaseline_data1215Yaw_20251215_1951_ice_h100x1"
    "lessqposbaseline_data1215Yaw_6s_randHeading_20251215_2143_ice_h100x1"
    "lessqposbaseline_data1215Yaw_randHeading_linvel.5_20251215_2244_ice_h100x1"
)

echo "Starting to copy folders from remote to local..."
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo "Local: ${LOCAL_DIR}"
echo ""

# Create local directory if it doesn't exist
mkdir -p "${LOCAL_DIR}"

# Copy each folder using rsync
for folder in "${FOLDERS[@]}"; do
    echo "Copying ${folder}..."
    rsync -avz --progress \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${folder}/" \
        "${LOCAL_DIR}/${folder}/"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully copied ${folder}"
    else
        echo "✗ Failed to copy ${folder}"
    fi
    echo ""
done

echo "All transfers completed!"
