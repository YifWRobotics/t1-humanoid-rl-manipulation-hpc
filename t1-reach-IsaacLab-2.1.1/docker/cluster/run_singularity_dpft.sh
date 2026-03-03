#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.dpft
source $SCRIPT_DIR/../.env.base

# make sure that all directories exists in cache directory
setup_directories
# copy all cache files
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
# $1 = /storage/home/hcoda1/4/zgu78/r-yzhao301-0/Research/IsaacLab/isaaclab_20251002_223000
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1") # isaaclab_20251002_223000

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# create a persistant overlay using apptainer with fakeroot
if [ ! -f "$CLUSTER_ISAACLAB_DIR/$dir_name.img" ] && [[ ! ( "$2" == "isaac-lab-dpft" || "$2" == "isaac-lab-dpftcloud" ) ]]; then
    apptainer overlay create --size 2048 $CLUSTER_ISAACLAB_DIR/$dir_name.img
fi

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
if [[ "$2" == "isaac-lab-base" ]]; then
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    -B $CLUSTER_ISAACLAB_DIR/datasets:/workspace/isaaclab/datasets:rw \
    --overlay $CLUSTER_ISAACLAB_DIR/$dir_name.img \
    --nv --containall $TMPDIR/$2.sif \
    bash -c "export ISAACLAB_PATH=/workspace/isaaclab && export WANDB_API_KEY=$(cat ~/.wandb_api_key) && cd /workspace/isaaclab && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}"
elif [[ "$2" == "isaac-lab-dpft" ]] || [[ "$2" == "isaac-lab-dpftcloud" ]]; then
# create ckpts and outputs directories in advance to populate mount points
mkdir -p $TMPDIR/$dir_name/diffusion_humanoid/ckpts
mkdir -p $TMPDIR/$dir_name/diffusion_humanoid/outputs

CLUSTER_PYTHON_EXECUTABLE=scripts/$3
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_OUTPUT_DIR:/workspace/isaaclab/diffusion_humanoid/outputs:rw \
    -B $CLUSTER_CHECKPOINTS_DIR:/workspace/isaaclab/diffusion_humanoid/ckpts:ro \
    --nv --containall --writable-tmpfs $TMPDIR/$2.sif \
    bash -c "export ISAACLAB_PATH=/workspace/isaaclab && cd /workspace/isaaclab/diffusion_humanoid && export WANDB_API_KEY=${WANDB_KEY} && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:4}"
else
    echo "Error: Invalid container profile: $2"
    exit 1
fi 

# copy resulting cache files back to host
# rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

# remove the temporary image file
if $REMOVE_OVERLAY_AFTER_JOB; then
    rm -f $CLUSTER_ISAACLAB_DIR/$dir_name.img
fi

echo "(run_singularity.py): Return"
