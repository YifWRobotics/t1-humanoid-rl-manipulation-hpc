# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# wandb API import
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# Add wandb run id argument
parser.add_argument("--wandb_run_id", type=str, default=None, help="Wandb run id to automatically download latest policy.")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (user or org).")
parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import time
import torch
import numpy as np
# wandb API import
import wandb

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config, register_task_to_hydra
import carb
# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    # Select checkpoint: wandb, pretrained, or local
    resume_path = None
    wandb_run = None
    run_name = None
    step_str = None
    if args_cli.wandb_run_id:
        print(f"[INFO] Downloading latest policy from wandb run: {args_cli.wandb_run_id}")
        api = wandb.Api()
        run_path = args_cli.wandb_run_id
        # Accept full run path: entity/project/run
        if "/" not in run_path:
            # if entity and project flags are provided, use them
            if args_cli.wandb_entity and args_cli.wandb_project:
                run_path = f"{args_cli.wandb_entity}/{args_cli.wandb_project}/{run_path}"
            else:
                # try to use environment variable or fail
                env_path = os.environ.get("WANDB_RUN_PATH")
                if env_path:
                    run_path = os.path.join(env_path, run_path)
                else:
                    print(
                        "[ERROR] wandb run id does not include project/entity. Provide '--wandb_entity' and '--wandb_project', or a full 'entity/project/run' path, or set WANDB_RUN_PATH env var."
                    )
                    return
        try:
            run = api.run(run_path)
        except Exception as e:
            print(f"[ERROR] Could not access wandb run '{run_path}': {e}")
            return
        wandb_run = run
        # Try to find policy.pt or policy.onnx first
        policy_files = [f for f in run.files() if f.name.endswith("policy.pt") or f.name.endswith("policy.onnx")]
        # If not found, look for model_*.pt files or any .pt
        if not policy_files:
            model_files = [f for f in run.files() if f.name.endswith(".pt")]
            if not model_files:
                print("[ERROR] No .pt model file found in wandb run.")
                return
            # Sort by step number if possible, else by created_at
            import re

            def extract_step(fname):
                m = re.search(r"model_(\d+)\.pt", fname)
                return int(m.group(1)) if m else None

            # Filter only files with a valid step number
            model_files_with_step = [(f, extract_step(f.name)) for f in model_files]
            model_files_with_step = [item for item in model_files_with_step if item[1] is not None]
            if model_files_with_step:
                # Sort by step number descending
                model_files_sorted = sorted(model_files_with_step, key=lambda x: x[1], reverse=True)
                latest_policy = model_files_sorted[0][0]
                step_str = str(model_files_sorted[0][1])
                print(f"[INFO] Selected model file: {latest_policy.name} (step {step_str})")
            else:
                # Fallback: sort by creation time
                model_files_sorted = sorted(model_files, key=lambda f: f.created_at, reverse=True)
                latest_policy = model_files_sorted[0]
                # create a timestamp-based step string
                step_str = latest_policy.created_at.isoformat().replace(":", "").replace("-", "").split("+")[0]
                print(f"[INFO] Selected model file by creation time: {latest_policy.name}")
        else:
            latest_policy = sorted(policy_files, key=lambda f: f.created_at, reverse=True)[0]
            print(f"[INFO] Selected policy file: {latest_policy.name}")
        # Prepare download location and filenames
        import shutil

        run_name = getattr(run, "name", None) or "wandb_run"
        # sanitize run name
        run_name = str(run_name).replace(" ", "_")
        base_name, ext = os.path.splitext(latest_policy.name)
        if step_str is None:
            # try to extract step from filename
            import re

            m = re.search(r"model_(\d+)", latest_policy.name)
            step_str = m.group(1) if m else latest_policy.created_at.isoformat().replace(":", "").replace("-", "").split("+")[0]

        renamed_filename = f"{run_name}_model_{step_str}{ext}"
        download_dir = os.path.join("wandb_downloads", run_name)
        os.makedirs(download_dir, exist_ok=True)
        original_path = os.path.join(download_dir, latest_policy.name)
        renamed_path = os.path.join(download_dir, renamed_filename)
        print(f"[INFO] Downloading {latest_policy.name} to {original_path}")
        latest_policy.download(replace=True, root=download_dir)
        # Rename the file to include run name and step
        try:
            # If download saved to a different place, try to move it
            if os.path.exists(original_path):
                shutil.move(original_path, renamed_path)
            else:
                # try to find file in download_dir
                found = False
                for f in os.listdir(download_dir):
                    if f == latest_policy.name:
                        shutil.move(os.path.join(download_dir, f), renamed_path)
                        found = True
                        break
                if not found:
                    # as last resort, copy from remote (download API may have placed it under a different name)
                    print(f"[WARNING] Could not find downloaded file {original_path}, attempting to copy with expected name.")
            print(f"[INFO] Renamed to {renamed_path}")
        except Exception as e:
            print(f"[WARNING] Could not rename file: {e}")
        resume_path = renamed_path
    # Determine task name: prefer explicit CLI task, else try to infer from wandb run config
    if args_cli.task:
        task_name = args_cli.task.split(":")[-1]
    else:
        task_name = None
        if wandb_run is not None:
            # common keys used to store env/task in run.config
            cfg = getattr(wandb_run, "config", {}) or {}
            for key in ("task", "task_name", "env", "env_name", "environment"):
                if key in cfg and cfg[key]:
                    task_name = str(cfg[key])
                    break
        if task_name is None:
            print("[ERROR] No task specified and could not infer task from wandb run. Please pass --task or ensure the wandb run config contains 'task' or 'env'.")
            return
    print(f"[INFO] Number of environments: (to be set by Hydra)")
    """Play with RSL-RL agent."""
    train_task_name = task_name.replace("-Play", "")

    # Now register and run Hydra for the discovered task
    # hydra_task_config returns a decorator that registers task and executes hydra.main
    @hydra_task_config(task_name, "rsl_rl_cfg_entry_point")
    def _inner(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        log_dir = os.path.dirname(resume_path) if resume_path is not None else log_root_path

        # create isaac environment
        env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if resume_path is None:
            print("[ERROR] No checkpoint specified to load.")
            return

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # extract the neural network module
        # we do this in a try-except to maintain backwards compatibility.
        try:
            # version 2.3 onwards
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = ppo_runner.alg.actor_critic

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        os.makedirs(export_model_dir, exist_ok=True)
        safe_run_name = run_name or "exported_policy"
        pt_filename = f"{safe_run_name}_model_{step_str}.pt"
        onnx_filename = f"{safe_run_name}_model_{step_str}.onnx"
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename=pt_filename)
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename=onnx_filename
        )

        dt = env.unwrapped.step_dt

        # reset environment
        obs, _ = env.get_observations()
        timestep = 0

        max_steps = args_cli.video_length  # Maximum number of steps to record
        time_data = np.zeros(max_steps)

        # base_acc_data_list = []
        # base_vel_data_list = []
        # base_pose_data_list = []

        step_list = []
        step_count = 0

        total_base_pose_data = []
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                t1 = time.time()
                actions = policy(obs)
                # env stepping
                obs, _, _, infos = env.step(actions)
                # time_data[step_count] = timestep * dt
                step_count += 1
                step_list.append(step_count)

                # base_acc_data = env.unwrapped.scene["robot"].data.body_acc_w[:, 0, :].cpu().numpy()
                # base_acc_data_list.append(base_acc_data)

                # base_vel_data = env.unwrapped.scene["robot"].data.body_vel_w[:, 0, :].cpu().numpy()
                # base_vel_data_list.append(base_vel_data)

                # base_pose_data = env.unwrapped.scene["robot"].data.body_pose_w[:, 0, :].cpu().numpy()
                # base_pose_data_list.append(base_pose_data)

            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            # print(f"sleep_time: {sleep_time:.4f}")
            # if args_cli.real_time and sleep_time > 0:
            #     time.sleep(sleep_time)

        # close the simulator
        env.close()

    # call the inner hydra-wrapped function
    _inner()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()