# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for downloading checkpoints from Weights & Biases."""

from pathlib import Path


def get_wandb_checkpoint_path(log_root_path: str, wandb_run_path: str) -> str:
    """Download checkpoint from a W&B run.

    Args:
        log_root_path: Root path for logs directory.
        wandb_run_path: W&B run path in format 'entity/project/run_id'.

    Returns:
        Path to the downloaded checkpoint file (latest model_*.pt).

    Raises:
        ImportError: If wandb is not installed.
        ValueError: If no checkpoint file is found in the run.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is not installed. Please install it with: pip install wandb"
        )

    # Initialize wandb API
    api = wandb.Api()

    # Get the run
    print(f"[INFO] Fetching W&B run: {wandb_run_path}")
    run = api.run(wandb_run_path)

    # Extract run_id from path (e.g., "entity/project/run_id" -> "run_id")
    run_id = wandb_run_path.split("/")[-1]
    download_dir = Path(log_root_path) / "wandb_checkpoints" / run_id

    # Query wandb API to find checkpoint files
    files = [file.name for file in run.files() if "model" in file.name and file.name.endswith(".pt")]

    if not files:
        raise ValueError(
            f"No model checkpoint files found in W&B run: {wandb_run_path}"
        )

    # Find the latest checkpoint by parsing iteration number (model_XXXX.pt)
    def get_iteration(filename):
        try:
            # Extract number from "model_12345.pt" -> 12345
            return int(filename.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1

    checkpoint_file = max(files, key=get_iteration)
    checkpoint_path = download_dir / checkpoint_file

    # If this checkpoint is not cached locally, download it
    if not checkpoint_path.exists():
        print(f"[INFO] Downloading checkpoint: {checkpoint_file}")
        download_dir.mkdir(parents=True, exist_ok=True)
        wandb_file = run.file(checkpoint_file)
        wandb_file.download(str(download_dir), replace=True)
    else:
        print(f"[INFO] Using cached checkpoint: {checkpoint_path}")

    print(f"[INFO] Checkpoint ready at: {checkpoint_path}")
    return str(checkpoint_path)
