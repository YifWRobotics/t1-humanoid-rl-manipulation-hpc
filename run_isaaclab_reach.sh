#!/usr/bin/env bash
set -euo pipefail

SIF="$HOME/scratch/isaaclab_sif/isaac-lab_2.1.0.sif"
REPO="$HOME/scratch/t1-humanoid-rl-manipulation-hpc/t1-reach-IsaacLab-2.1.1"

mkdir -p "$HOME/scratch/docker/isaac-sim/cache"/{kit,ov,pip,glcache,computecache}
mkdir -p "$HOME/scratch/docker/isaac-sim"/{logs,data,documents,kit-data,kit-logs}
mkdir -p "$HOME/scratch/isaaclab"/{logs,data_storage}

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  $0 scripts/.../train.py [args...]"
  echo "  $0 bash -lc 'commands...'"
  exit 2
fi

CMD="$1"
shift || true

ISAACLAB_PYTHONPATH="/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_rl:/workspace/isaaclab/source/isaaclab_tasks"

APPTAINER_BASE=(
  apptainer exec --cleanenv --nv
  --pwd /workspace/isaaclab
  --env ACCEPT_EULA=Y
  --env PRIVACY_CONSENT=Y
  --env WARP_DISABLE_CUDA=1
  --env DISPLAY=
  --env OMNI_KIT_DISABLE_WINDOWING=1
  --env KIT_DISABLE_WINDOWING=1
  --env "PYTHONPATH=${ISAACLAB_PYTHONPATH}"
  --bind "$REPO":/workspace/isaaclab:rw
  --bind "$HOME/scratch/isaaclab/logs":/workspace/isaaclab/logs:rw
  --bind "$HOME/scratch/isaaclab/data_storage":/workspace/isaaclab/data_storage:rw
  --bind "$HOME/scratch/docker/isaac-sim/cache/kit":/isaac-sim/kit/cache:rw
  --bind "$HOME/scratch/docker/isaac-sim/kit-data":/isaac-sim/kit/data:rw
  --bind "$HOME/scratch/docker/isaac-sim/kit-logs":/isaac-sim/kit/logs:rw
  --bind "$HOME/scratch/docker/isaac-sim/cache/ov":/root/.cache/ov:rw
  --bind "$HOME/scratch/docker/isaac-sim/cache/pip":/root/.cache/pip:rw
  --bind "$HOME/scratch/docker/isaac-sim/cache/glcache":/root/.cache/nvidia/GLCache:rw
  --bind "$HOME/scratch/docker/isaac-sim/cache/computecache":/root/.nv/ComputeCache:rw
  --bind "$HOME/scratch/docker/isaac-sim/logs":/root/.nvidia-omniverse/logs:rw
  --bind "$HOME/scratch/docker/isaac-sim/data":/root/.local/share/ov/data:rw
  --bind "$HOME/scratch/docker/isaac-sim/documents":/root/Documents:rw
  "$SIF"
)

if [[ "$CMD" == "bash" ]]; then
  "${APPTAINER_BASE[@]}" bash "$@"
else
  "${APPTAINER_BASE[@]}" \
    /isaac-sim/python.sh \
    "/workspace/isaaclab/$CMD" \
    "$@"
fi