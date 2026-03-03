#!/bin/bash
# Helper script to run Python scripts with Isaac Sim's Python environment
# This script deactivates conda and uses Isaac Sim's Python directly

# Deactivate conda if active
if [ -n "$CONDA_PREFIX" ]; then
    echo "[INFO] Deactivating conda environment..."
    # Deactivate all conda environments
    while [ -n "$CONDA_PREFIX" ]; do
        conda deactivate 2>/dev/null || break
    done
fi

# Set paths
ISAAC_SIM_PATH="/home/lidar/isaacsim"
ISAACLAB_PATH="/home/lidar/Documents/AB_Git/IsaacLab"
WORKSPACE_PATH="/home/lidar/Documents/AB_Git/t1_dp_tasks"

# Add all necessary paths to PYTHONPATH
export PYTHONPATH="${WORKSPACE_PATH}/source:${PYTHONPATH}"
export PYTHONPATH="${WORKSPACE_PATH}/source/t1_dp_locomotion_manager_based:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab_tasks:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab_assets:${PYTHONPATH}"

# Unset conda variables
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE _CE_M _CE_CONDA

# Run the script with Isaac Sim's Python
exec "${ISAAC_SIM_PATH}/python.sh" "$@"

