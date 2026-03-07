# Train Manipulation for Booster T1

YouTube demo: https://www.youtube.com/watch?v=mQGBrfK2c9Q

## 1. Set Up Isaac Lab 2.1.1 For PACE HPC Clusters

### 1.1. Recommended Directory Layout on PACE

```bash
scratch/
├── t1-humanoid-rl-manipulation-hpc/          (this repo)
├── isaaclab_sif/                             (container image)
├── docker/isaac-sim/                         (cache)
├── isaaclab/logs/                            (training logs)
```

### 1.2. Build the SIF from NGC

From the scratch directory:
```bash
mkdir -p isaaclab_sif && cd isaaclab_sif
apptainer build isaac-lab_2.1.0.sif docker://nvcr.io/nvidia/isaac-lab:2.1.0
```

### 1.3. Create Persistent Cache and Logs Directories

```bash
mkdir -p ~/scratch/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache}
mkdir -p ~/scratch/docker/isaac-sim/{logs,data,documents,kit-data,kit-logs}
mkdir -p ~/scratch/isaaclab/{logs,data_storage}
```

### 1.4. Clone the T1 Humanoid Reinforcement Learning for Manipulation Repository

```bash
cd ~/scratch
git clone https://github.com/YifWRobotics/t1-humanoid-rl-manipulation-hpc.git
cd t1-humanoid-rl-manipulation-hpc
git lfs install
git lfs pull
```

## 2. Training and Deployment

### 2.1. Training

```bash
chmod +x ~/scratch/t1-humanoid-rl-manipulation-hpc/run_isaaclab_reach.sh
```

```bash
~/scratch/t1-humanoid-rl-manipulation-hpc/run_isaaclab_reach.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task FK-Tracking-T1-v0 \
  --num_envs 4096 \
  --max_iterations 150001 \
  --name t1_reach \
  --run_name PACE \
  --logger wandb \
  --log_project_name isaac_t1_reach \
  --seed 42 \
  --headless \
  --video \
  --video_length 500 \
  --video_interval 5000 \
  env.commands.both_hand_pose.rel_random=0.0 \
  env.commands.both_hand_pose.rel_fk=1.0 \
  env.commands.both_hand_pose.rel_default=0.0
```

### 2.2. Deployment

Run this command on a PC, not a HPC cluster:
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task FK-Tracking-T1-Play-v0 \
  --num_envs 4 \
  --checkpoint /absolute/path/to/model_150000.pt \
  env.commands.both_hand_pose.rel_random=0.0 \
  env.commands.both_hand_pose.rel_fk=1.0 \
  env.commands.both_hand_pose.rel_default=0.0
```