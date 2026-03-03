#!/usr/bin/env bash
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=pace_logs/T1_H200_%j.log
#SBATCH -N1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zgu78@gatech.edu
#SBATCH --job-name="DPFT-$(date +"%Y%m%d_%H%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT
cat < job.sh
sbatch < job.sh
rm job.sh